// SPDX-License-Identifier: GPL-3.0-or-later
// He–Sun–Tang guided filter (self-guided), device side.
//
// Mirrors src/local_contrast.rs. The CPU builds four global f64
// summed-area tables; that formulation is wrong for a GPU — consumer
// cards run f64 at 1/64 rate, and a global f32 prefix sum over 24 M
// samples loses the low bits the variance is computed from. But a box
// filter never needed a global sum: only window sums, and those are
// separable. Each statistic is computed in two passes — horizontal
// window sums, then vertical — with every output element produced by
// ONE thread accumulating at most (2r+1) values sequentially in a fixed
// order. No cross-thread reduction exists anywhere, so the result is
// bit-reproducible on any launch geometry, and the accumulation is over
// at most tens of values, where f32 is comfortably sufficient (the
// window sum of a 17-element row is not the 24-million-element prefix
// sum the CPU's f64 tables protect).
//
// Window clamping reproduces integral.rs::window_sum exactly:
// rows [max(0, y-r), min(h-1, y+r)], cols likewise, area = their
// product. The 0/0 -> 0 convention and the negative-variance clamp
// match the CPU code line for line.
//
// Kernel graph (five full-resolution f32 buffers):
//   box_h_l_l2:   L            -> hsum_l, hsum_l2      (row windows)
//   coeff_ab:     hsum_l, hsum_l2, eps -> a, b         (column windows + model)
//   box_h_ab:     a, b         -> hsum_a, hsum_b       (row windows; may alias
//                                                       the first two buffers)
//   final_out:    hsum_a, hsum_b, L, strength -> out   (column windows + blend)

extern "C" __global__ void box_h_l_l2(const float* __restrict__ input,
                                      float* __restrict__ hsum_l,
                                      float* __restrict__ hsum_l2,
                                      int h, int w, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);
    const float* row = input + (long long)y * w;

    // Kahan-compensated: windows are usually tiny, but a caller may pass
    // a radius covering the whole image, and compensation makes the long
    // case accurate for two extra flops in a bandwidth-bound loop.
    float sum = 0.0f, c1 = 0.0f, sum2 = 0.0f, c2 = 0.0f;
    for (int i = x1; i <= x2; ++i) {
        float v = row[i];
        float y1v = v - c1;      float t1 = sum + y1v;
        c1 = (t1 - sum) - y1v;   sum = t1;
        float v2 = v * v;
        float y2v = v2 - c2;     float t2 = sum2 + y2v;
        c2 = (t2 - sum2) - y2v;  sum2 = t2;
    }
    long long idx = (long long)y * w + x;
    hsum_l[idx] = sum;
    hsum_l2[idx] = sum2;
}

extern "C" __global__ void coeff_ab(const float* __restrict__ hsum_l,
                                    const float* __restrict__ hsum_l2,
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

    float sum = 0.0f, c1 = 0.0f, sum2 = 0.0f, c2 = 0.0f;
    for (int j = y1; j <= y2; ++j) {
        long long idx = (long long)j * w + x;
        float v = hsum_l[idx];
        float y1v = v - c1;      float t1 = sum + y1v;
        c1 = (t1 - sum) - y1v;   sum = t1;
        float v2 = hsum_l2[idx];
        float y2v = v2 - c2;     float t2 = sum2 + y2v;
        c2 = (t2 - sum2) - y2v;  sum2 = t2;
    }
    float area = (float)((y2 - y1 + 1) * (x2 - x1 + 1));
    float mean_l = sum / area;
    float mean_l2 = sum2 / area;
    // Same clamp, same 0/0 -> 0 convention as the CPU (see
    // src/local_contrast.rs::guided_filter).
    float var_l = fmaxf(mean_l2 - mean_l * mean_l, 0.0f);
    float a = (var_l + eps > 0.0f) ? var_l / (var_l + eps) : 0.0f;

    long long idx = (long long)y * w + x;
    a_out[idx] = a;
    b_out[idx] = mean_l * (1.0f - a);
}

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
