// SPDX-License-Identifier: GPL-3.0-or-later
// Standard-luminance B&W conversion, device side. Mirrors src/bw.rs.
// The dot product evaluates left to right exactly as the Rust
// expression does, and with FMA contraction disabled each mul and add
// is correctly rounded — bit-exact against the CPU.

extern "C" __global__ void luminance_bw_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               float wr, float wg, float wb,
                                               long long npix) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        const float* p = input + i * 3;
        output[i] = wr * p[0] + wg * p[1] + wb * p[2];
    }
}
