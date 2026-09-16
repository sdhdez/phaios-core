// SPDX-License-Identifier: GPL-3.0-or-later
// 1-D lookup table with linear interpolation, device side. Mirrors
// `lut_sample` in src/lut.rs statement for statement.
//
// Bit-exact against the CPU kernel: subtract, divide, multiply,
// truncation and a linear interpolation, every one exact or correctly
// rounded, with -fmad=false stopping the compiler from contracting the
// interpolation's multiply-add. No transcendental, so no libm to
// disagree with.
//
// NaN is tested for explicitly and propagated, rather than being allowed
// to fall through the comparisons into an end entry -- the same
// reasoning as highlight_rolloff, and the same divergence class the v0.2
// audit found in hsl_bw.

extern "C" __global__ void lut_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ lut,
                                      int lut_len,
                                      float min, float max,
                                      long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float value = input[i];
    if (isnan(value)) {
        output[i] = value;
        return;
    }

    int last = lut_len - 1;
    float t = (value - min) / (max - min) * (float)last;

    if (!(t > 0.0f)) {
        output[i] = lut[0];
        return;
    }
    if (t >= (float)last) {
        output[i] = lut[last];
        return;
    }

    int idx = (int)t; // truncation toward zero, and t is in (0, last)
    float frac = t - (float)idx;
    // Convex combination, matching src/lut.rs. The difference form
    // `a + frac*(b-a)` overflows when adjacent entries are more than
    // FLT_MAX apart; this one is bounded by max(|a|,|b|) and still hits
    // both knots exactly.
    output[i] = (1.0f - frac) * lut[idx] + frac * lut[idx + 1];
}
