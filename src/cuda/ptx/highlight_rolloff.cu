// SPDX-License-Identifier: GPL-3.0-or-later
// Quadratic Bezier highlight shoulder, device side. Mirrors
// `rolloff_sample` in src/highlight_rolloff.rs statement for statement.
//
// Bit-exact against the CPU kernel: the evaluation uses only add,
// subtract, multiply, divide and square root, all of which IEEE-754-2008
// section 5.4.1 requires to be correctly rounded, and the build passes
// -fmad=false so no multiply-add is contracted. No transcendental is
// involved, so there is no libm to disagree with.
//
// The first test is written negated (`!(x > knee)`) rather than as
// `x <= knee` so that a NaN sample takes the pass-through branch on both
// backends. The positive form would send NaN into the solve on one side
// and not the other -- exactly the divergence hsl_bw had.

extern "C" __global__ void highlight_rolloff_kernel(const float* __restrict__ input,
                                                    float* __restrict__ output,
                                                    float knee, float white,
                                                    long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = input[i];
    if (!(x > knee)) {
        output[i] = x;
        return;
    }
    if (x >= white) {
        output[i] = 1.0f;
        return;
    }

    // x(t) = (W + k - 2)t^2 + 2(1 - k)t + k, solved for t at this x.
    float a = white + knee - 2.0f;
    float b = 2.0f * (1.0f - knee);
    float c = knee - x;

    float t;
    if (a == 0.0f) {
        // W = 2 - k: the quadratic degenerates to a line (the classic
        // parabolic shoulder). A legal configuration, not an edge case.
        t = -c / b;
    } else {
        float disc = b * b - 4.0f * a * c;
        t = (-b + sqrtf(fmaxf(disc, 0.0f))) / (2.0f * a);
    }

    float one_minus_t = 1.0f - t;
    output[i] = 1.0f - (1.0f - knee) * one_minus_t * one_minus_t;
}
