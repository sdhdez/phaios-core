// SPDX-License-Identifier: GPL-3.0-or-later
// IEC 61966-2-1 sRGB transfer, device side. Mirrors src/encode.rs.
// inv_gamma is computed on the host as 1.0f/2.4f — the same f32 bits the
// CPU kernel uses. Compiled with fast math (build.rs PRECISION): powf is
// approximate (ex2/lg2.approx), 1.055 * pow - 0.055 is contracted into an
// FMA and subnormal inputs and results flush to zero, so agreement is
// bounded
// (rtol 1e-5, atol 1e-7) and a driver update can move the last bits.

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
