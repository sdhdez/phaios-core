// SPDX-License-Identifier: GPL-3.0-or-later
// Per-channel histogram, device side. Mirrors src/histogram.rs.
//
// Bit-identical to the CPU by construction rather than by tolerance:
// the only arithmetic on pixel values is the bin assignment in
// `slot_of`, which is a subtract, a divide, a multiply and a truncation
// -- all exact or correctly rounded -- and everything after it is
// integer counting. Integer addition commutes, so the atomics'
// completion order cannot change the totals. This is the one reduction
// in the crate that is deterministic for free.
//
// Two kernels rather than one. The shared-memory version privatises the
// histogram per block, which removes almost all global contention, but
// it needs c*(bins+3) counters in shared memory. That is 3 KB for the
// 256-bin display case and 786 KB for a 65536-bin analysis pass, so the
// host picks the global-atomic version when the table will not fit.

__device__ __forceinline__ int slot_of(float value, float min, float max, int bins) {
    if (isnan(value)) return bins + 2;
    if (value < min)  return bins;
    if (value > max)  return bins + 1;
    // `max` itself belongs in the last bin, not in `above`.
    float t = (value - min) / (max - min);
    int idx = (int)(t * (float)bins);
    return idx >= bins ? bins - 1 : idx;
}

// Shared-memory privatised histogram. Launch with
// shared_mem_bytes = c * (bins + 3) * sizeof(unsigned int).
//
// The per-block counters are 32-bit: a block processes at most a few
// hundred thousand samples under the host's grid sizing, far below the
// 4-billion point where that would wrap.
extern "C" __global__ void histogram_shared_kernel(const float* __restrict__ input,
                                                   unsigned long long* __restrict__ output,
                                                   int c, int bins,
                                                   float min, float max,
                                                   long long n) {
    extern __shared__ unsigned int block_counts[];
    const int slots = c * (bins + 3);

    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        block_counts[i] = 0u;
    }
    __syncthreads();

    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int ch = (int)(i % c);
        atomicAdd(&block_counts[ch * (bins + 3) + slot_of(input[i], min, max, bins)], 1u);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < slots; i += blockDim.x) {
        unsigned int v = block_counts[i];
        if (v != 0u) {
            atomicAdd(&output[i], (unsigned long long)v);
        }
    }
}

// Global-atomic fallback for bin counts too large to privatise.
extern "C" __global__ void histogram_global_kernel(const float* __restrict__ input,
                                                   unsigned long long* __restrict__ output,
                                                   int c, int bins,
                                                   float min, float max,
                                                   long long n) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int ch = (int)(i % c);
        atomicAdd(&output[ch * (bins + 3) + slot_of(input[i], min, max, bins)], 1ULL);
    }
}
