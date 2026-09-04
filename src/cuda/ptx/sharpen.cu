// SPDX-License-Identifier: GPL-3.0-or-later
// The pointwise half of sharpen -- subtract, gate, combine. The blur
// `detail` is measured against is blur_device, so this file holds no
// filtering of its own. Mirrors src/sharpen.rs's `soft_gate` plus the
// final combine in `sharpen`.
//
// No transcendentals -- subtract, divide, multiply, add, and the
// min/max pair a clamp lowers to. Bit-exact against the CPU: `threshold`
// stands in directly for `soft_gate`'s `span`, because
// `(SOFT_KNEE_SPAN - 1.0) * threshold` is exactly `threshold` when
// SOFT_KNEE_SPAN == 2.0 -- multiplying by the exactly-representable 1.0
// rounds nowhere -- so recomputing that product here would only be a
// slower way to write the same bits. The composed result is bounded
// rather than exact only because the blur underneath it is.

// t == 1 whenever threshold == 0: the identity gate, special-cased
// rather than computed because the ramp would otherwise have zero width
// (see soft_gate's own doc comment). Otherwise a Hermite smoothstep of
// |d|, ramping from 0 at `threshold` to 1 at `2 * threshold`.
extern "C" __global__ void sharpen_apply_kernel(const float* __restrict__ input,
                                                const float* __restrict__ blurred,
                                                float* __restrict__ output,
                                                float amount,
                                                float threshold,
                                                long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = input[i];
    float d = v - blurred[i];
    float t = 1.0f;
    if (threshold != 0.0f) {
        float u = fminf(fmaxf((fabsf(d) - threshold) / threshold, 0.0f), 1.0f);
        t = u * u * (3.0f - 2.0f * u);
    }
    output[i] = v + amount * t * d;
}
