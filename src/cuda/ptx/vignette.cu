// SPDX-License-Identifier: GPL-3.0-or-later
// Radial vignette, device side. Mirrors src/vignette.rs line for line,
// including the degenerate feather == 0 step and pixel-centre mapping.
// Every operation here is correctly rounded (mul, add, div, sqrt,
// min/max) and the build disables FMA contraction, so this kernel is
// bit-exact against the CPU.

extern "C" __global__ void vignette_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int h, int w, int c,
                                           float amount, float inner,
                                           float roundness) {
    long long n = (long long)h * w * c;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int pix = (int)(i / c);
    int y = pix / w;
    int x = pix % w;

    float half_h = (float)h / 2.0f;
    float half_w = (float)w / 2.0f;
    float ny = ((float)y + 0.5f - half_h) / half_h;
    float nx = ((float)x + 0.5f - half_w) / half_w;

    float circular = sqrtf(nx * nx + ny * ny) * 0.70710678f;
    float rect = fmaxf(fabsf(nx), fabsf(ny));
    float d = circular + (rect - circular) * roundness;

    float falloff;
    if (1.0f <= inner) {
        falloff = (d < inner) ? 0.0f : 1.0f;
    } else {
        float t = fminf(fmaxf((d - inner) / (1.0f - inner), 0.0f), 1.0f);
        falloff = t * t * (3.0f - 2.0f * t);
    }
    output[i] = fmaxf(input[i] * (1.0f - amount * falloff), 0.0f);
}
