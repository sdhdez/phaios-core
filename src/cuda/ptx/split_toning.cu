// SPDX-License-Identifier: GPL-3.0-or-later
// Split-toning in OKLab, device side. Mirrors src/split_toning.rs:
// neutral-lightness shortcut (one cbrt), smoothstep crossfade of the
// two tints, OKLab -> linear sRGB (matrix + cubes). The host
// precomputes edge0/edge1 and the neutral row-sum exactly as the CPU
// does, so cbrtf is the only implementation-defined operation here.
// (H, W, 1) in, (H, W, 3) out.

extern "C" __global__ void split_toning_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               float row_sum,
                                               float edge0, float edge1,
                                               float shadow_a, float shadow_b,
                                               float highlight_a, float highlight_b,
                                               long long npix) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;

    float y = input[i];
    float big_l = cbrtf(y) * row_sum;   // neutral_oklab_lightness

    // smoothstep(edge0, edge1, big_l), degenerate window included.
    float mix;
    if (edge1 <= edge0) {
        mix = (big_l < edge0) ? 0.0f : 1.0f;
    } else {
        float t = fminf(fmaxf((big_l - edge0) / (edge1 - edge0), 0.0f), 1.0f);
        mix = t * t * (3.0f - 2.0f * t);
    }

    float a = shadow_a + (highlight_a - shadow_a) * mix;
    float b = shadow_b + (highlight_b - shadow_b) * mix;

    // oklab_to_linear_srgb, constants identical to src/split_toning.rs.
    float l_ = big_l + 0.39633778f * a + 0.21580376f * b;
    float m_ = big_l - 0.105561346f * a - 0.06385417f * b;
    float s_ = big_l - 0.08948418f * a - 1.2914855f * b;
    float l3 = l_ * l_ * l_;
    float m3 = m_ * m_ * m_;
    float s3 = s_ * s_ * s_;

    float* out = output + i * 3;
    out[0] = 4.0767417f * l3 - 3.3077116f * m3 + 0.23096994f * s3;
    out[1] = -1.268438f * l3 + 2.6097574f * m3 - 0.34131938f * s3;
    out[2] = -0.0041960863f * l3 - 0.7034186f * m3 + 1.7076147f * s3;
}
