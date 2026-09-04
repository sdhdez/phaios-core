// SPDX-License-Identifier: GPL-3.0-or-later
// Cross-guided (C = 3) half of the guided-filter denoise, device side.
// Mirrors src/denoise.rs's `cross_guided`. The self-guided half (C = 1,
// and every other channel count, per channel) needs no new kernel at
// all -- src/cuda/kernels/denoise.rs calls
// `local_contrast::local_contrast_device` directly, the same reasoning
// src/denoise.rs documents for its own self-guided/local_contrast alias
// (Q1 in .cache/scratch/denoise/PLAN.md: negating one multiply operand
// is an exact sign flip, so `p + (-amount)*(p-q)` rounds identically to
// `l + strength*(l-q)` at `strength = -amount`).
//
// This file adds exactly three kernels and reuses four more unchanged
// from local_contrast.cu (box_h_l_l2, box_h_ab) and luminance_bw.cu
// (luminance_bw_kernel) -- see src/cuda/kernels/denoise.rs for the
// 14-launch sequence that wires them together. Nothing here duplicates
// those three kernels' bodies.
//
// Cross-guided needs, per pixel: mean(I), mean(I^2) (shared across all
// three channels) and, per channel, mean(p_c), mean(I*p_c), from which
// var(I), cov(I,p_c), a_c = cov/(var+eps) and b_c = mean(p_c) -
// a_c*mean(I) follow -- the same 0/0 -> 0 convention and the same
// var >= 0 clamp as local_contrast.cu's `coeff_ab`, generalised from
// var(I) alone (self-guided: I = p) to cov(I, p_c) between two
// genuinely different arrays.
//
// Channel access, not extraction. `box_h_l_l2`/`box_h_ab` assume a
// contiguous single-channel (H, W) buffer, which a channel of the
// interleaved (H, W, 3) image is not -- but rather than add a device
// kernel to extract one (the plan authorises exactly three new kernels,
// all below), `box_h_cross` and `final_out_cross` -- the two that need
// per-channel pixel data at all -- read and write the ORIGINAL
// interleaved buffer directly, indexed by a `channel`/`n_channels`
// stride pair. `coeff_ab_cross` needs no such access: its four inputs
// (hsum_i, hsum_i2, hsum_p, hsum_ip) are already standalone (H, W)
// buffers produced by box_h_l_l2/box_h_cross, so it is channel-agnostic
// exactly like `coeff_ab`.
//
// Precision. The L/L^2-style shared sums (hsum_i, hsum_i2, reused from
// box_h_l_l2) and the new p_c/I*p_c sums (box_h_cross) are f64,
// matching box_h_l_l2's own reasoning: var(I) and cov(I,p_c) are each a
// cancelling subtraction of a mean-of-squares/products against a
// product-of-means, and Kahan compensation (which the a/b stage below
// uses instead) does nothing for a cancelling difference -- only wider
// storage does. `I*p_c` is formed by casting EACH factor to double
// before multiplying (not: multiply in float, then widen the product),
// the same style box_h_l_l2 uses for `v*v`. Since
// `.cache/scratch/denoise/PLAN.md`'s "Decision 2" (the CPU cancellation
// fix), src/denoise.rs's own CPU `cross_guided` sums each window
// directly too, the same shape as this file's kernels, and casts each
// factor to f64 before multiplying for the identical reason -- the two
// implementations no longer differ in this respect. (An earlier version
// of this CPU kernel formed `I*p_c` in f32 before widening the product,
// forced by reusing the single-array `integral::sat` helper it no
// longer calls.)
// `box_h_ab` (reused) and `final_out_cross`'s own box-sum of a_c/b_c
// stay Kahan-compensated f32, matching `coeff_ab`/`final_out`: nothing
// downstream of a_c/b_c is squared or subtracted, so f32 is enough,
// exactly as local_contrast.cu's own comment on `box_h_ab` argues.
//
// No transcendentals anywhere in this file.

// Row (horizontal) window sums of p_c and I*p_c, f64 accumulation
// exactly like box_h_l_l2 -- generalises it from one array (L, summed
// against itself) to a guide/channel PAIR. `img` is the original
// interleaved (H, W, n_channels) buffer; `channel` picks one channel
// out of it directly (row-major, channel innermost), so no
// channel-extraction pass is needed anywhere in this file.
extern "C" __global__ void box_h_cross(const float* __restrict__ guide,
                                       const float* __restrict__ img,
                                       int channel, int n_channels,
                                       double* __restrict__ hsum_p,
                                       double* __restrict__ hsum_ip,
                                       int h, int w, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);
    const float* row_g = guide + (long long)y * w;
    long long row_base = (long long)y * w * n_channels + channel;

    double sum_p = 0.0, sum_ip = 0.0;
    for (int i = x1; i <= x2; ++i) {
        double gv = (double)row_g[i];
        double pv = (double)img[row_base + (long long)i * n_channels];
        sum_p += pv;
        sum_ip += gv * pv;
    }
    long long idx = (long long)y * w + x;
    hsum_p[idx] = sum_p;
    hsum_ip[idx] = sum_ip;
}

// Column pass -> the model coefficients a_c, b_c. Channel-agnostic: its
// four inputs are already standalone (H, W) buffers (hsum_i/hsum_i2
// from the shared box_h_l_l2 call, hsum_p/hsum_ip from box_h_cross
// above), so no channel/stride argument is needed here at all --
// mirrors `coeff_ab` in local_contrast.cu almost exactly, with
// var(I)/mean(I) kept and a second, independent signal (p_c) added.
extern "C" __global__ void coeff_ab_cross(const double* __restrict__ hsum_i,
                                          const double* __restrict__ hsum_i2,
                                          const double* __restrict__ hsum_p,
                                          const double* __restrict__ hsum_ip,
                                          float* __restrict__ a_out,
                                          float* __restrict__ b_out,
                                          int h, int w, int r, float eps) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int y1 = max(0, y - r);
    int y2 = min(h - 1, y + r);
    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);

    double sum_i = 0.0, sum_i2 = 0.0, sum_p = 0.0, sum_ip = 0.0;
    for (int j = y1; j <= y2; ++j) {
        long long idx = (long long)j * w + x;
        sum_i  += hsum_i[idx];
        sum_i2 += hsum_i2[idx];
        sum_p  += hsum_p[idx];
        sum_ip += hsum_ip[idx];
    }
    double area = (double)((y2 - y1 + 1) * (x2 - x1 + 1));
    double mean_i  = sum_i / area;
    double mean_i2 = sum_i2 / area;
    double mean_p  = sum_p / area;
    double mean_ip = sum_ip / area;

    // Same clamp, same 0/0 -> 0 convention as `coeff_ab` and the CPU's
    // `cross_guided` (src/denoise.rs). var(I) alone can drive the
    // cancelling subtraction below zero on a large-magnitude guide;
    // cov(I, p_c) has no such floor -- it is a genuine covariance, free
    // to be negative -- so only var_i is clamped, matching the CPU.
    double var_i = fmax(mean_i2 - mean_i * mean_i, 0.0);
    double cov = mean_ip - mean_i * mean_p;
    double eps_d = (double)eps;
    double a = (var_i + eps_d > 0.0) ? cov / (var_i + eps_d) : 0.0;
    double b = mean_p - a * mean_i;

    long long idx = (long long)y * w + x;
    a_out[idx] = (float)a;
    b_out[idx] = (float)b;
}

// q_c = mean(a_c)*I + mean(b_c) (a_c/b_c box-summed exactly like
// `final_out`, Kahan f32); out_c = p_c + strength*(p_c - q_c), the same
// operation order as `final_out` and as the CPU's `cross_guided`
// (`strength` here is the caller's `-amount`, matching
// `local_contrast_device`'s own convention for the self-guided reuse).
// Unlike `final_out`, which conflates the guide and the signal (both
// are `L` in the self-guided case), this kernel takes the guide `I` and
// the channel `p_c` as two separate inputs -- `img`/`output` are the
// interleaved (H, W, n_channels) buffers, read/written one channel at a
// time via the same channel/stride indexing box_h_cross uses.
extern "C" __global__ void final_out_cross(const float* __restrict__ hsum_a,
                                           const float* __restrict__ hsum_b,
                                           const float* __restrict__ guide,
                                           const float* __restrict__ img,
                                           float* __restrict__ output,
                                           int channel, int n_channels,
                                           int h, int w, int r, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int y1 = max(0, y - r);
    int y2 = min(h - 1, y + r);
    int x1 = max(0, x - r);
    int x2 = min(w - 1, x + r);

    float sum_a = 0.0f, ca = 0.0f, sum_b = 0.0f, cb = 0.0f;
    for (int j = y1; j <= y2; ++j) {
        long long idx = (long long)j * w + x;
        float va = hsum_a[idx];
        float ya = va - ca;      float ta = sum_a + ya;
        ca = (ta - sum_a) - ya;  sum_a = ta;
        float vb = hsum_b[idx];
        float yb = vb - cb;      float tb = sum_b + yb;
        cb = (tb - sum_b) - yb;  sum_b = tb;
    }
    float area = (float)((y2 - y1 + 1) * (x2 - x1 + 1));
    float mean_a = sum_a / area;
    float mean_b = sum_b / area;

    long long pix = (long long)y * w + x;
    float iv = guide[pix];
    float pv = img[pix * n_channels + channel];
    float q = mean_a * iv + mean_b;
    output[pix * n_channels + channel] = pv + strength * (pv - q);
}
