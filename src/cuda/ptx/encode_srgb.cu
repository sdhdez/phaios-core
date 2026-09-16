// SPDX-License-Identifier: GPL-3.0-or-later
// IEC 61966-2-1 sRGB transfer, device side. Mirrors src/encode.rs.
// inv_gamma is computed on the host as 1.0f/2.4f — the same f32 bits the
// CPU kernel uses. powf is the one op here that may differ from libm by
// a few ULP; everything else is correctly rounded on both sides.

extern "C" __global__ void encode_srgb_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              float inv_gamma,
                                              long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = input[i];
        output[i] = (v <= 0.0031308f) ? 12.92f * v
                                      : 1.055f * powf(v, inv_gamma) - 0.055f;
    }
}
