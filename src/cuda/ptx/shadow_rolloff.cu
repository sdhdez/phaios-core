// SPDX-License-Identifier: GPL-3.0-or-later
// Cubic Hermite shadow toe, device side. Mirrors `shadow_sample` in
// src/shadow_rolloff.rs statement for statement, including the Horner
// grouping -- a different association would round differently and cost
// bit-exactness.
//
// Bit-exact against the CPU kernel: multiply, add, subtract and one
// divide, all correctly rounded by IEEE-754-2008 section 5.4.1, with
// -fmad=false stopping the compiler contracting any multiply-add. No
// transcendental, so no libm to disagree with.
//
// The comparisons are negated (`!(x < knee)`) rather than positive so a
// NaN sample passes through on both backends instead of entering the
// polynomial on one and not the other -- the divergence class the v0.2
// audit found in hsl_bw.

extern "C" __global__ void shadow_rolloff_kernel(const float* __restrict__ input,
                                                 float* __restrict__ output,
                                                 float knee, float strength,
                                                 long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    if (!(strength > 0.0f) || !(knee > 0.0f) || !(x < knee)) {
        output[i] = x;
        return;
    }

    float s = 1.0f - strength;
    if (x <= 0.0f) {
        // -INF must be returned before the multiply: at strength == 1.0
        // the slope s is exactly zero and 0.0f * -INF is NaN, which would
        // turn an infinity into a NaN -- and into a different NaN payload
        // than the host produces. Mirrors the guard in src/shadow_rolloff.rs.
        if (isinf(x)) {
            output[i] = x;
            return;
        }
        // C1 continuation below zero, at the curve's slope at the origin.
        output[i] = s * x;
        return;
    }

    float u = x / knee;
    output[i] = knee * (u * (s + u * (2.0f * (1.0f - s) + u * (s - 1.0f))));
}
