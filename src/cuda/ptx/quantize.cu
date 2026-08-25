// SPDX-License-Identifier: GPL-3.0-or-later
// Dithered quantisation to integer codes, device side. Mirrors
// src/quantize.rs.
//
// Bit-exact against the CPU kernel. The dither is exact 64-bit integer
// arithmetic (the same splitmix64 the grain kernel uses, asserted over
// 2^20 coordinates there) followed by multiplies and adds on values with
// 24-bit mantissas, and the rounding is floorf(v + 0.5f) rather than a
// library rounding routine -- floor is exact and the add is correctly
// rounded, so the composition cannot drift.
//
// The comparisons are written negated (`!(v > 0.0f)`) so that a NaN
// sample takes the zero branch on both backends rather than reaching an
// undefined float-to-integer conversion.

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long pixel_hash(unsigned long long seed,
                                                         unsigned long long x,
                                                         unsigned long long y) {
    unsigned long long key = x * 0x9E3779B97F4A7C15ULL ^ y * 0xC2B2AE3D27D4EB4FULL;
    return splitmix64(seed ^ splitmix64(key));
}

// Two independent 24-bit uniforms, centred and summed: a triangular
// deviate on [-1, 1]. Matches `triangular_dither` in src/quantize.rs.
__device__ __forceinline__ float triangular_dither(unsigned long long bits) {
    const float SCALE = 1.0f / 16777216.0f; // 2^-24
    float u1 = ((float)(bits >> 40)) * SCALE;
    float u2 = ((float)((bits >> 16) & 0x00FFFFFFULL)) * SCALE;
    return (u1 - 0.5f) + (u2 - 0.5f);
}

__device__ __forceinline__ float quantize_sample(float value, float dither, float max_code) {
    float v = value * max_code + dither;
    if (!(v > 0.0f)) return 0.0f;      // negative, zero, or NaN
    if (v >= max_code) return max_code;
    float rounded = floorf(v + 0.5f);
    return rounded > max_code ? max_code : rounded;
}

// `dithered` is 0 or 1; `c` is the channel count, needed to recover the
// (x, y, channel) triple from the flat index so the hash key matches the
// CPU's `Zip::indexed` exactly.
__device__ __forceinline__ float dither_for(long long i, int w, int c,
                                            unsigned long long seed, int dithered) {
    if (!dithered) return 0.0f;
    long long ch = i % c;
    long long pixel = i / c;
    long long x = pixel % w;
    long long y = pixel / w;
    unsigned long long key = seed ^ ((unsigned long long)ch * 0x9E3779B97F4A7C15ULL);
    return triangular_dither(pixel_hash(key, (unsigned long long)x, (unsigned long long)y));
}

extern "C" __global__ void quantize_u8_kernel(const float* __restrict__ input,
                                              unsigned char* __restrict__ output,
                                              int w, int c,
                                              unsigned long long seed, int dithered,
                                              long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = dither_for(i, w, c, seed, dithered);
    output[i] = (unsigned char)quantize_sample(input[i], d, 255.0f);
}

extern "C" __global__ void quantize_u16_kernel(const float* __restrict__ input,
                                               unsigned short* __restrict__ output,
                                               int w, int c,
                                               unsigned long long seed, int dithered,
                                               long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = dither_for(i, w, c, seed, dithered);
    output[i] = (unsigned short)quantize_sample(input[i], d, 65535.0f);
}
