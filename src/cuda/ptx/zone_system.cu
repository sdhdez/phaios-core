// SPDX-License-Identifier: GPL-3.0-or-later
// Adams/Archer Zone System, device side. Mirrors src/tone.rs.
//
// The offsets arrive as a DENSE 11-entry array in zone order (absent
// zones are 0.0). This preserves the CPU's ordered-reduction guarantee
// mechanically: adding an exact +0.0 term is the identity in IEEE-754,
// so the dense ascending loop computes bit-for-bit the same sum as the
// CPU's sparse sorted iteration. log2f/expf/powf are the
// implementation-defined operations; the bound covers them.

extern "C" __global__ void zone_system_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              const float* __restrict__ offsets11,
                                              long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float MIDDLE_GREY = 0.18f;
    const float TWO_SIGMA_SQ = 2.0f * 0.8f * 0.8f;
    const float L_EPSILON = 1e-10f;

    float l = input[i];
    float l_pos = fmaxf(l, L_EPSILON);
    float zone_pos = 5.0f + log2f(l_pos / MIDDLE_GREY);

    float total = 0.0f;
    for (int z = 0; z <= 10; ++z) {
        float d = zone_pos - (float)z;
        total += offsets11[z] * expf(-d * d / TWO_SIGMA_SQ);
    }
    output[i] = l * powf(2.0f, total);
}
