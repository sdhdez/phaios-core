// SPDX-License-Identifier: GPL-3.0-or-later
// Exposure compensation, device side.
//
// Mirrors src/exposure.rs exactly: out = in * gain, where the host has
// already computed gain = 2^stops. The kernel is one IEEE-754 multiply,
// which CUDA rounds correctly — so this is the one kernel whose GPU
// output is asserted BIT-IDENTICAL to the CPU's, with no tolerance.
// That property is what makes it the Stage A vertical slice: any
// disagreement is a bug in the stack, never in the numerics.
//
// The element count is passed as long long: 24 MP x 3 channels is 75 M
// elements, comfortably inside i32, but a 200 MP scan is not, and the
// CPU path has no such limit to inherit.

extern "C" __global__ void exposure_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           float gain,
                                           long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * gain;
    }
}
