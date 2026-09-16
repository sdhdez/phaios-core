// SPDX-License-Identifier: GPL-3.0-or-later
// Geometry, device side. Mirrors src/geometry.rs. Four kernels:
// crop_kernel, orient_kernel, resample_kernel (resize) and
// straighten_kernel. All four are bit-exact against the CPU. The first
// two are pure index permutations, with no arithmetic on pixel values.
// The other two resample, but their filters are polynomial.

extern "C" __global__ void crop_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int in_w, int c,
                                       int x0, int y0,
                                       int out_h, int out_w) {
    long long n = (long long)out_h * out_w * c;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    long long pix = i / c;
    int ox = (int)(pix % out_w);
    int oy = (int)(pix / out_w);
    long long src = ((long long)(oy + y0) * in_w + (ox + x0)) * c + ch;
    output[i] = input[src];
}

// The orientation as (transpose, flip_y, flip_x) applied to source
// coordinates — the same decomposition Orientation::flags() defines on
// the host, so the two backends share one definition and cannot drift.
extern "C" __global__ void orient_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int in_h, int in_w, int c,
                                         int transpose, int flip_y, int flip_x) {
    int out_h = transpose ? in_w : in_h;
    int out_w = transpose ? in_h : in_w;
    long long n = (long long)out_h * out_w * c;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    long long pix = i / c;
    int ox = (int)(pix % out_w);
    int oy = (int)(pix / out_w);

    // Order matters and mirrors the CPU exactly: the flips act on the
    // OUTPUT axes of the (possibly transposed) view, then the transpose
    // swaps back into source coordinates. Flipping input axes instead
    // yields the inverse rotation for the quarter-turns — the
    // conformance suite caught precisely that.
    int a = flip_y ? (out_h - 1 - oy) : oy;
    int b = flip_x ? (out_w - 1 - ox) : ox;
    int iy = transpose ? b : a;
    int ix = transpose ? a : b;

    output[i] = input[((long long)iy * in_w + ix) * c + ch];
}

// ── Resampling (resize, straighten) ──────────────────────────────────────────
// Operation-for-operation transcriptions of src/geometry.rs. The
// filters are polynomial, evaluation and accumulation order match the
// CPU exactly, and -fmad=false keeps every mul/add correctly rounded —
// so these are bit-exact against the CPU like the rest of geometry.

__device__ __forceinline__ float filter_eval(int filter, float t) {
    t = fabsf(t);
    if (filter == 0) {                 // Area (box profile when upscaling)
        return (t <= 0.5f) ? 1.0f : 0.0f;
    } else if (filter == 1) {          // Bilinear
        return fmaxf(1.0f - t, 0.0f);
    } else {                           // CatmullRom
        if (t <= 1.0f) {
            return ((1.5f * t - 2.5f) * t) * t + 1.0f;
        } else if (t < 2.0f) {
            return ((-0.5f * t + 2.5f) * t - 4.0f) * t + 2.0f;
        }
        return 0.0f;
    }
}

// One thread per output element; the tap loop mirrors resample_axis1.
// axis = 0: resample along width (input rows × in_len × c).
// axis = 1: resample along height (columns walked with stride).
extern "C" __global__ void resample_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int rows, int in_len, int out_len, int c,
                                           float scale, float support,
                                           int filter, int area_minify,
                                           int axis) {
    long long n = (long long)rows * out_len * c;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int ch = (int)(idx % c);
    long long lane = idx / c;
    int i = (int)(lane % out_len);
    int row = (int)(lane / out_len);

    float centre = ((float)i + 0.5f) * scale - 0.5f;
    float denom = fmaxf(scale, 1.0f);
    long long k0 = (long long)floorf(centre - support);
    long long k1 = (long long)ceilf(centre + support);

    float acc = 0.0f;
    float wsum = 0.0f;
    for (long long k = k0; k <= k1; ++k) {
        float w;
        if (area_minify) {
            float lo = fmaxf((float)k - 0.5f, centre - scale * 0.5f);
            float hi = fminf((float)k + 0.5f, centre + scale * 0.5f);
            w = fmaxf(hi - lo, 0.0f);
        } else {
            w = filter_eval(filter, ((float)k - centre) / denom);
        }
        if (w != 0.0f) {
            long long kc = k < 0 ? 0 : (k > in_len - 1 ? in_len - 1 : k);
            long long src = (axis == 0)
                ? ((long long)row * in_len + kc) * c + ch
                : (kc * rows + row) * c + ch;
            acc += w * input[src];
            wsum += w;
        }
    }
    long long dst = (axis == 0)
        ? ((long long)row * out_len + i) * c + ch
        : ((long long)i * rows + row) * c + ch;
    output[dst] = (wsum != 0.0f) ? acc / wsum : 0.0f;
}

// Straighten: inverse-rotate each output pixel centre into the source
// and sample 16-tap Catmull-Rom in the CPU's exact j-then-i order.
extern "C" __global__ void straighten_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int in_h, int in_w, int c,
                                             int out_h, int out_w,
                                             float sin_a, float cos_a) {
    long long n = (long long)out_h * out_w * c;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int ch = (int)(idx % c);
    long long lane = idx / c;
    int ox = (int)(lane % out_w);
    int oy = (int)(lane / out_w);

    float cx_out = (float)out_w * 0.5f;
    float cy_out = (float)out_h * 0.5f;
    float cx_in = (float)in_w * 0.5f;
    float cy_in = (float)in_h * 0.5f;

    float dx = (float)ox + 0.5f - cx_out;
    float dy = (float)oy + 0.5f - cy_out;
    float sx = cos_a * dx + sin_a * dy + cx_in - 0.5f;
    float sy = -sin_a * dx + cos_a * dy + cy_in - 0.5f;

    float fx = floorf(sx);
    float fy = floorf(sy);
    float tx = sx - fx;
    float ty = sy - fy;
    long long ix = (long long)fx;
    long long iy = (long long)fy;

    float wx[4] = {
        filter_eval(2, tx + 1.0f),
        filter_eval(2, tx),
        filter_eval(2, 1.0f - tx),
        filter_eval(2, 2.0f - tx),
    };
    float wy[4] = {
        filter_eval(2, ty + 1.0f),
        filter_eval(2, ty),
        filter_eval(2, 1.0f - ty),
        filter_eval(2, 2.0f - ty),
    };

    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        long long yj = iy - 1 + j;
        yj = yj < 0 ? 0 : (yj > in_h - 1 ? in_h - 1 : yj);
        float row_acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            long long xi = ix - 1 + i;
            xi = xi < 0 ? 0 : (xi > in_w - 1 ? in_w - 1 : xi);
            row_acc += wx[i] * input[(yj * in_w + xi) * c + ch];
        }
        acc += wy[j] * row_acc;
    }
    output[idx] = acc;
}
