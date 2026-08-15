// SPDX-License-Identifier: GPL-3.0-or-later
// Procedural film grain, device side. Mirrors src/film_grain.rs.
//
// The integer half — splitmix64 over (seed, x, y) — is exact 64-bit
// arithmetic and MUST match the CPU bit for bit; the conformance suite
// asserts it over 2^20 coordinates via grain_hash_kernel. The
// Box-Muller half (logf, sqrtf, cosf) cannot be exact, and the
// band-pass reuses the separable Kahan-compensated box-filter approach
// proven in local_contrast.cu. Border clamping and the 0-area
// conventions match integral.rs::window_sum exactly.

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ unsigned long long pixel_hash(unsigned long long seed,
                                                         unsigned long long x,
                                                         unsigned long long y) {
    unsigned long long key = x * 0x9E3779B97F4A7C15ULL ^ y * 0xC2B2AE3D27D4EB4FULL;
    return splitmix64(seed ^ splitmix64(key));
}

__device__ __forceinline__ float standard_normal(unsigned long long bits) {
    const float SCALE = 1.0f / 16777216.0f; // 2^-24
    float u1 = ((float)(bits >> 40) + 0.5f) * SCALE;
    float u2 = ((float)((bits >> 16) & 0xFFFFFFULL)) * SCALE;
    float radius = sqrtf(-2.0f * logf(u1));
    return radius * cosf(6.2831855f * u2); // TAU as f32, same bits as Rust's consts::TAU
}

// Debug/conformance entry: raw hashes, for the bit-exactness assertion.
extern "C" __global__ void grain_hash_kernel(unsigned long long* __restrict__ out,
                                             unsigned long long seed,
                                             int w, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (unsigned long long)(i % w);
    unsigned long long y = (unsigned long long)(i / w);
    out[i] = pixel_hash(seed, x, y);
}

extern "C" __global__ void grain_noise(float* __restrict__ noise,
                                       unsigned long long seed,
                                       int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    noise[(long long)y * w + x] =
        standard_normal(pixel_hash(seed, (unsigned long long)x, (unsigned long long)y));
}

// Horizontal window sums at BOTH radii in one pass over the noise.
extern "C" __global__ void grain_h(const float* __restrict__ noise,
                                   float* __restrict__ hsum_in,
                                   float* __restrict__ hsum_out,
                                   int h, int w, int r_in, int r_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float* row = noise + (long long)y * w;
    int xo1 = max(0, x - r_out);
    int xo2 = min(w - 1, x + r_out);
    int xi1 = max(0, x - r_in);
    int xi2 = min(w - 1, x + r_in);

    float so = 0.0f, co = 0.0f, si = 0.0f, ci = 0.0f;
    for (int i = xo1; i <= xo2; ++i) {
        float v = row[i];
        float yo = v - co; float to = so + yo;
        co = (to - so) - yo; so = to;
        if (i >= xi1 && i <= xi2) {
            float yi = v - ci; float ti = si + yi;
            ci = (ti - si) - yi; si = ti;
        }
    }
    long long idx = (long long)y * w + x;
    hsum_in[idx] = si;
    hsum_out[idx] = so;
}

// Vertical windows, means, band-pass, envelope, blend.
extern "C" __global__ void grain_final(const float* __restrict__ hsum_in,
                                       const float* __restrict__ hsum_out,
                                       const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int h, int w, int r_in, int r_out,
                                       float intensity, float normalisation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int yo1 = max(0, y - r_out);
    int yo2 = min(h - 1, y + r_out);
    int yi1 = max(0, y - r_in);
    int yi2 = min(h - 1, y + r_in);
    int xo1 = max(0, x - r_out);
    int xo2 = min(w - 1, x + r_out);
    int xi1 = max(0, x - r_in);
    int xi2 = min(w - 1, x + r_in);

    float so = 0.0f, co = 0.0f, si = 0.0f, ci = 0.0f;
    for (int j = yo1; j <= yo2; ++j) {
        long long idx = (long long)j * w + x;
        float vo = hsum_out[idx];
        float yo = vo - co; float to = so + yo;
        co = (to - so) - yo; so = to;
        if (j >= yi1 && j <= yi2) {
            float vi = hsum_in[idx];
            float yi = vi - ci; float ti = si + yi;
            ci = (ti - si) - yi; si = ti;
        }
    }
    float area_in = (float)((yi2 - yi1 + 1) * (xi2 - xi1 + 1));
    float area_out = (float)((yo2 - yo1 + 1) * (xo2 - xo1 + 1));
    float bandpass = si / area_in - so / area_out;

    long long idx = (long long)y * w + x;
    float l = input[idx];
    float t = fminf(fmaxf(l, 0.0f), 1.0f);
    float envelope = 4.0f * t * (1.0f - t);
    output[idx] = fmaxf(l + intensity * envelope * normalisation * bandpass, 0.0f);
}
