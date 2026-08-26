// SPDX-License-Identifier: GPL-3.0-or-later
// The two element-wise halves of glow; the spread between them is
// blur_device, so this file holds no filtering of its own. Mirrors
// src/glow.rs.
//
// Both kernels are bit-exact against their CPU counterparts -- subtract,
// compare, multiply, add. The composed result is bounded rather than
// exact only because the blur between them is.

// weight = max(in - threshold, 0). Written with the positive test on the
// difference so a NaN sample yields 0 here exactly as it does on the
// host, rather than propagating into the blur and spreading across the
// frame.
extern "C" __global__ void glow_weight_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              float threshold,
                                              long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float excess = input[i] - threshold;
    output[i] = (excess > 0.0f) ? excess : 0.0f;
}

// out = in + amount * spread. Not clamped: the pipeline carries headroom
// to highlight_rolloff.
extern "C" __global__ void glow_add_kernel(const float* __restrict__ input,
                                           const float* __restrict__ spread,
                                           float* __restrict__ output,
                                           float amount,
                                           long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = input[i] + amount * spread[i];
}
