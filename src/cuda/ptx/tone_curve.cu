// SPDX-License-Identifier: GPL-3.0-or-later
// ASC CDL slope/offset/power, device side. Mirrors src/tone.rs.
// unit_power skips powf exactly as the CPU fast path does.
// Compiled with fast math (build.rs PRECISION): powf is approximate
// (ex2/lg2.approx) and bounded (rtol 1e-5, atol 1e-7). The affine step is
// pinned to the CPU's two roundings with __fmul_rn/__fadd_rn, which fast
// math never contracts: fused into one FMA, slope*x + offset kept a
// residue where the CPU cancels to exactly 0, and powf at power < 1 lifted
// it by up to 8 8-bit codes. So t matches the CPU except that subnormal
// values flush to zero, and the power == 1 path agrees to the bit unless
// an input or the affine result is subnormal. A driver update can move
// the last bits of the powf path.

extern "C" __global__ void tone_curve_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             float slope, float offset,
                                             float power, int unit_power,
                                             long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = fmaxf(__fadd_rn(__fmul_rn(input[i], slope), offset), 0.0f);
        output[i] = unit_power ? t : powf(t, power);
    }
}
