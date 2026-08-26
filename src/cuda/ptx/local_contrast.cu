// SPDX-License-Identifier: GPL-3.0-or-later
// He–Sun–Tang guided filter (self-guided), device side.
//
// Mirrors src/local_contrast.rs. The CPU builds four global f64
// summed-area tables; that *formulation* is wrong for a GPU, because a
// box filter never needed a global prefix sum at all — only window sums,
// and those are separable. Each statistic is computed in two passes,
// horizontal window sums then vertical, with every output element
// produced by ONE thread accumulating at most (2r+1) values sequentially
// in a fixed order. No cross-thread reduction exists anywhere, so the
// result is bit-reproducible on any launch geometry.
//
// Dropping the global prefix sum did NOT license dropping the f64, and an
// earlier version of this file argued that it did: "the accumulation is
// over at most tens of values, where f32 is comfortably sufficient". That
// reasoning counts the number of terms and ignores their magnitude and
// the cancellation downstream. It missed the committed bound by 51.8x on
// a dark field with a 1e4 highlight. The L/L² partial sums and the
// coefficient sums are f64; see box_h_l_l2.
//
// Window clamping reproduces integral.rs::window_sum exactly:
// rows [max(0, y-r), min(h-1, y+r)], cols likewise, area = their
// product. The 0/0 -> 0 convention and the negative-variance clamp
// match the CPU code line for line.
//
// Kernel graph (two f64 L/L² buffers, four f32 ones):
//   box_h_l_l2:   L            -> hsum_l, hsum_l2      (row windows)
//   coeff_ab:     hsum_l, hsum_l2, eps -> a, b         (column windows + model)
//   box_h_ab:     a, b         -> hsum_a, hsum_b       (row windows; may alias
//                                                       the first two buffers)
//   final_out:    hsum_a, hsum_b, L, strength -> out   (column windows + blend)

// The L and L² partial sums are f64, matching the host's summed-area
// tables. They were Kahan-compensated f32, and compensation is the wrong
// tool twice over here.
//
// First, magnitude. L² of a specular highlight is enormous: at L = 1e4 a
// 17-wide window sums to ~1e9, and f32 carries seven digits, so the low
// bits are gone *in storage* before any arithmetic happens.
//
// Second, and worse, the variance is `mean(L²) − mean(L)²` — a
// subtraction of two nearly equal large numbers, where the true variance
// may be many orders smaller than either. Kahan compensates a *sum*; it
// does nothing whatever for a cancelling difference. The host acknowledges
// exactly this at src/local_contrast.rs:150 and answers it with f64.
//
// Measured before: 51.8x local_contrast's committed (1e-4, 1e-6) bound on
// a dark field with a 1e4 highlight.
extern "C" __global__ void box_h_l_l2(const float* __restrict__ input,
                                      double* __restrict__ hsum_l,
                                      double* __restrict__ hsum_l2,
                                      int h, int w, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);
    const float* row = input + (long long)y * w;

    double sum = 0.0, sum2 = 0.0;
    for (int i = x1; i <= x2; ++i) {
        double v = (double)row[i];
        sum += v;
        sum2 += v * v;
    }
    long long idx = (long long)y * w + x;
    hsum_l[idx] = sum;
    hsum_l2[idx] = sum2;
}

extern "C" __global__ void coeff_ab(const double* __restrict__ hsum_l,
                                    const double* __restrict__ hsum_l2,
                                    float* __restrict__ a_out,
                                    float* __restrict__ b_out,
                                    int h, int w, int r, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int y1 = max(0, y - r);
    int y2 = min(h - 1, y + r);
    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);

    double sum = 0.0, sum2 = 0.0;
    for (int j = y1; j <= y2; ++j) {
        long long idx = (long long)j * w + x;
        sum += hsum_l[idx];
        sum2 += hsum_l2[idx];
    }
    double area = (double)((y2 - y1 + 1) * (x2 - x1 + 1));
    double mean_l = sum / area;
    double mean_l2 = sum2 / area;
    // Same clamp, same 0/0 -> 0 convention as the CPU (see
    // src/local_contrast.rs::guided_filter). The subtraction below is the
    // cancelling one, and is the reason everything above it is f64.
    double var_l = fmax(mean_l2 - mean_l * mean_l, 0.0);
    double eps_d = (double)eps;
    double a = (var_l + eps_d > 0.0) ? var_l / (var_l + eps_d) : 0.0;

    long long idx = (long long)y * w + x;
    a_out[idx] = (float)a;
    b_out[idx] = (float)(mean_l * (1.0 - a));
}

// f32 here, deliberately. Unlike the L/L² pass these sums feed no
// cancelling difference: `a` is confined to [0, 1] and `b` is used
// linearly, never squared. f64 would cost the same 6x it costs above and
// buy nothing measurable -- checked, not assumed.
//
// Kahan-compensated, though, and that part is NOT optional: a caller may
// pass a radius covering the whole image, and then this sums every pixel
// in it rather than a handful. Dropping the compensation while switching
// these back to f32 put r=9999 at 1.15x the bound.
extern "C" __global__ void box_h_ab(const float* __restrict__ a_in,
                                    const float* __restrict__ b_in,
                                    float* __restrict__ hsum_a,
                                    float* __restrict__ hsum_b,
                                    int h, int w, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);
    const float* row_a = a_in + (long long)y * w;
    const float* row_b = b_in + (long long)y * w;

    float sum_a = 0.0f, ca = 0.0f, sum_b = 0.0f, cb = 0.0f;
    for (int i = x1; i <= x2; ++i) {
        float va = row_a[i];
        float ya = va - ca;      float ta = sum_a + ya;
        ca = (ta - sum_a) - ya;  sum_a = ta;
        float vb = row_b[i];
        float yb = vb - cb;      float tb = sum_b + yb;
        cb = (tb - sum_b) - yb;  sum_b = tb;
    }
    long long idx = (long long)y * w + x;
    hsum_a[idx] = sum_a;
    hsum_b[idx] = sum_b;
}

extern "C" __global__ void final_out(const float* __restrict__ hsum_a,
                                     const float* __restrict__ hsum_b,
                                     const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int h, int w, int r, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int y1 = max(0, y - r);
    int y2 = min(h - 1, y + r);
    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);

    float sum_a = 0.0f, ca = 0.0f, sum_b = 0.0f, cb = 0.0f;
    for (int j = y1; j <= y2; ++j) {
        long long idx = (long long)j * w + x;
        float va = hsum_a[idx];
        float ya = va - ca;      float ta = sum_a + ya;
        ca = (ta - sum_a) - ya;  sum_a = ta;
        float vb = hsum_b[idx];
        float yb = vb - cb;      float tb = sum_b + yb;
        cb = (tb - sum_b) - yb;  sum_b = tb;
    }
    float area = (float)((y2 - y1 + 1) * (x2 - x1 + 1));
    float mean_a = sum_a / area;
    float mean_b = sum_b / area;

    long long idx = (long long)y * w + x;
    float l = input[idx];
    float q = mean_a * l + mean_b;
    output[idx] = l + strength * (l - q);
}
