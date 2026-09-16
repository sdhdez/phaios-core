// SPDX-License-Identifier: GPL-3.0-or-later
// Separable Gaussian blur, device side. Mirrors src/blur.rs: a direct
// convolution below BOX_CROSSOVER_SIGMA (6), three box passes at or
// above it.
//
// Bounded, not bit-exact, and deliberately so: the CPU accumulates every
// 1-D pass in f64, and the two paths here answer that differently.
//
//   - The direct convolution accumulates in f32 with Kahan compensation,
//     the same trade local_contrast.cu makes. That loop is well
//     conditioned -- every weight is positive and nothing is ever
//     subtracted -- so compensation is enough, and f64 would buy nothing
//     for a 1/64 rate.
//   - The box pass accumulates in f64, matching box_lane exactly. A
//     sliding window *subtracts*, and no amount of compensation survives
//     that on high-dynamic-range input. See the comment on that kernel.
//
// The committed bound is in docs/ffi.md section 6.
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

// Box filter with a sliding window, clamped at the borders.
//
// Segmented, not one-thread-per-lane. A 24 MP frame has only about five
// thousand rows or columns, so a thread per lane leaves a modern device
// ~95% idle and was measured at 40 ms against the direct path's 10 ms
// for the same blur. Each lane is instead cut into `seg_len` chunks and
// one thread takes each: the sliding sum stays O(1) per output, and the
// thread count rises by the segment factor.
//
// The price is that every segment recomputes its own leading window,
// O(radius) work repeated `segments` times per lane. The host picks the
// segment count so that overhead stays well under the sliding work it
// buys parallelism for.
//
// Accumulation is f64, matching box_lane on the host exactly.
//
// It was Kahan-compensated f32, on the reasoning that applies to the
// direct path above. That reasoning does not survive a sliding window.
// Kahan bounds the error by 2*eps*sum|x_i|, which a bright sample
// dominates, so on linear scene-referred input -- a dark field with a
// specular highlight, which is the data this crate exists to process --
// the accumulator holds ~1e8 while the samples entering it are ~1e-4 and
// are annihilated on contact. When the bright sample leaves the window
// what they contributed is simply gone. Measured against an exact oracle:
// 164% relative error at a 1e4 highlight, and up to 2.8e7 times the
// committed cross-backend bound. Compensation cannot fix this; only
// carrying the magnitude can.
//
// The cost is nearly nil because this kernel is bandwidth-bound, not
// compute-bound: two loads and a store per output against three f64 ops,
// so even at a consumer card's 1/64 f64 rate the arithmetic hides under
// the memory traffic.
extern "C" __global__ void blur_box_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int radius,
                                           int rows, int len, int c,
                                           int axis,
                                           int seg_len,
                                           long long threads) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;

    long long segments = ((long long)len + seg_len - 1) / seg_len;
    long long seg = t % segments;
    long long lane = t / segments;
    long long ch = lane % c;
    long long row = lane / c;

    long long begin = seg * seg_len;
    long long end = begin + seg_len;
    if (end > len) end = len;
    if (begin >= end) return;

    long long last = (long long)len - 1;
    double width = (double)(2 * radius + 1);

    // Leading window for this segment's first output.
    double acc = 0.0;
    for (long long k = begin - radius; k <= begin + radius; ++k) {
        long long s = k < 0 ? 0 : (k > last ? last : k);
        acc += (double)input[idx_of(axis, row, s, rows, len, c, ch)];
    }
    output[idx_of(axis, row, begin, rows, len, c, ch)] = (float)(acc / width);

    for (long long i = begin + 1; i < end; ++i) {
        long long enter = i + radius;
        if (enter > last) enter = last;
        long long leave = i - radius - 1;
        if (leave < 0) leave = 0;
        acc += (double)input[idx_of(axis, row, enter, rows, len, c, ch)];
        acc -= (double)input[idx_of(axis, row, leave, rows, len, c, ch)];
        output[idx_of(axis, row, i, rows, len, c, ch)] = (float)(acc / width);
    }
}
