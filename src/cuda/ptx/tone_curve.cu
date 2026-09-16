// SPDX-License-Identifier: GPL-3.0-or-later
// ASC CDL slope/offset/power, device side. Mirrors src/tone.rs.
// unit_power skips powf exactly as the CPU fast path does, so the
// power == 1 configuration is bit-exact; the powf path differs from
// libm by a few ULP at most.

extern "C" __global__ void tone_curve_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             float slope, float offset,
                                             float power, int unit_power,
                                             long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = fmaxf(input[i] * slope + offset, 0.0f);
        output[i] = unit_power ? t : powf(t, power);
    }
}
