// SPDX-License-Identifier: GPL-3.0-or-later
// HSL-weighted B&W conversion, device side. Mirrors src/bw.rs::hsl_bw
// line for line: hexagonal hue, chroma ratio (max-min)/max, circular
// Gaussian blend over eight fixed band centres. fmodf has the same
// sign-of-dividend semantics as Rust's % on f32. expf is the one
// implementation-defined operation; everything else is correctly
// rounded (no FMA contraction at build time).

__device__ const float BAND[8] = {0.0f, 30.0f, 60.0f, 120.0f, 180.0f, 240.0f, 270.0f, 300.0f};

extern "C" __global__ void hsl_bw_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         float wr, float wg, float wb,
                                         float w0, float w1, float w2, float w3,
                                         float w4, float w5, float w6, float w7,
                                         float two_sigma_sq,
                                         long long npix) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;

    const float* p = input + i * 3;
    float r = p[0], g = p[1], b = p[2];
    float base = wr * r + wg * g + wb * b;

    // hue_and_chroma_ratio, as in src/bw.rs.
    float rc = fmaxf(r, 0.0f), gc = fmaxf(g, 0.0f), bc = fmaxf(b, 0.0f);
    float mx = fmaxf(fmaxf(rc, gc), bc);
    float mn = fminf(fminf(rc, gc), bc);
    float delta = mx - mn;

    float hue = 0.0f, chroma = 0.0f;
    if (delta > 0.0f && mx > 0.0f) {
        if (mx == rc) {
            hue = 60.0f * fmodf((gc - bc) / delta, 6.0f);
        } else if (mx == gc) {
            hue = 60.0f * ((bc - rc) / delta + 2.0f);
        } else {
            hue = 60.0f * ((rc - gc) / delta + 4.0f);
        }
        if (hue < 0.0f) hue += 360.0f;
        chroma = delta / mx;
    }

    float weights[8] = {w0, w1, w2, w3, w4, w5, w6, w7};
    float multiplier = 0.0f;
    for (int k = 0; k < 8; ++k) {
        // hue_distance_deg: circular distance.
        float d = fmodf(fabsf(hue - BAND[k]), 360.0f);
        if (d > 180.0f) d = 360.0f - d;
        multiplier += weights[k] * expf(-d * d / two_sigma_sq);
    }

    output[i] = fmaxf(base * (1.0f + multiplier * chroma), 0.0f);
}
