// SPDX-License-Identifier: GPL-3.0-or-later
// Separable Gaussian blur, device side. Mirrors src/blur.rs: a direct
// convolution below sigma 4, three box passes at or above it.
//
// Bounded, not bit-exact, and deliberately so. The CPU accumulates each
// 1-D pass in f64; doing the same here would run at 1/64 rate on a
// consumer card, so both kernels accumulate in f32 with Kahan
// compensation instead -- the same trade local_contrast.cu already makes,
// with the same reasoning. The committed bound is in docs/ffi.md section 6.
//
// The axis flag follows resample_kernel in geometry.cu:
//   axis = 0: filter along width  (index (row * len + i) * c + ch)
//   axis = 1: filter along height (index (i * rows + row) * c + ch)
// so a two-pass separable filter never transposes the image.

__device__ __forceinline__ long long idx_of(int axis, long long row, long long i,
                                            long long rows, long long len,
                                            long long c, long long ch) {
    return (axis == 0) ? ((row * len + i) * c + ch)
                       : ((i * rows + row) * c + ch);
}

// Direct convolution against caller-supplied weights, clamped at the
// borders. One thread per output element; the tap loop mirrors
// convolve_row in src/blur.rs, including its clamped index arithmetic.
extern "C" __global__ void blur_conv_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const float* __restrict__ weights,
                                            int taps,
                                            int rows, int len, int c,
                                            int axis,
                                            long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    long long ch  = t % c;
    long long rem = t / c;
    long long i   = rem % len;
    long long row = rem / len;

    int radius = (taps - 1) / 2;
    long long last = (long long)len - 1;

    float acc = 0.0f, comp = 0.0f;
    for (int k = 0; k < taps; ++k) {
        long long src_i = i + k - radius;
        if (src_i < 0) src_i = 0;
        if (src_i > last) src_i = last;
        float v = weights[k] * input[idx_of(axis, row, src_i, rows, len, c, ch)];
        // Kahan: a wide kernel at sigma just under the crossover is 33
        // taps, and compensation costs two flops in a bandwidth-bound loop.
        float y = v - comp;
        float s = acc + y;
        comp = (s - acc) - y;
        acc = s;
    }
    output[idx_of(axis, row, i, rows, len, c, ch)] = acc;
}

// Box filter with a sliding window, clamped at the borders. One thread
// per (row, channel) lane: the running sum is serial along the lane, and
// a 24 MP frame still gives thousands of lanes.
//
// Recomputed rather than incremental at the ends: src/blur.rs adds the
// entering sample and subtracts the leaving one, and doing the same here
// keeps the two in step.
extern "C" __global__ void blur_box_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int radius,
                                           int rows, int len, int c,
                                           int axis,
                                           long long lanes) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= lanes) return;

    long long ch  = t % c;
    long long row = t / c;
    long long last = (long long)len - 1;
    float width = (float)(2 * radius + 1);

    // Leading window, clamped at both ends.
    float acc = 0.0f, comp = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        long long s = k;
        if (s < 0) s = 0;
        if (s > last) s = last;
        float v = input[idx_of(axis, row, s, rows, len, c, ch)];
        float y = v - comp;
        float sum = acc + y;
        comp = (sum - acc) - y;
        acc = sum;
    }
    output[idx_of(axis, row, 0, rows, len, c, ch)] = acc / width;

    for (long long i = 1; i < len; ++i) {
        long long enter = i + radius;
        if (enter > last) enter = last;
        long long leave = i - radius - 1;
        if (leave < 0) leave = 0;
        acc += input[idx_of(axis, row, enter, rows, len, c, ch)];
        acc -= input[idx_of(axis, row, leave, rows, len, c, ch)];
        output[idx_of(axis, row, i, rows, len, c, ch)] = acc / width;
    }
}
