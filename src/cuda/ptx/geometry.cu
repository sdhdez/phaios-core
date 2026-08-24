// SPDX-License-Identifier: GPL-3.0-or-later
// Crop and dihedral orientation, device side. Mirrors src/geometry.rs.
// Pure index permutations — no arithmetic on pixel values — so both
// kernels are bit-exact against the CPU by construction.

extern "C" __global__ void crop_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int in_w, int c,
                                       int x0, int y0,
                                       int out_h, int out_w) {
    long long n = (long long)out_h * out_w * c;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    long long pix = i / c;
    int ox = (int)(pix % out_w);
    int oy = (int)(pix / out_w);
    long long src = ((long long)(oy + y0) * in_w + (ox + x0)) * c + ch;
    output[i] = input[src];
}

// The orientation as (transpose, flip_y, flip_x) applied to source
// coordinates — the same decomposition Orientation::flags() defines on
// the host, so the two backends share one definition and cannot drift.
extern "C" __global__ void orient_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int in_h, int in_w, int c,
                                         int transpose, int flip_y, int flip_x) {
    int out_h = transpose ? in_w : in_h;
    int out_w = transpose ? in_h : in_w;
    long long n = (long long)out_h * out_w * c;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    long long pix = i / c;
    int ox = (int)(pix % out_w);
    int oy = (int)(pix / out_w);

    // Order matters and mirrors the CPU exactly: the flips act on the
    // OUTPUT axes of the (possibly transposed) view, then the transpose
    // swaps back into source coordinates. Flipping input axes instead
    // yields the inverse rotation for the quarter-turns — the
    // conformance suite caught precisely that.
    int a = flip_y ? (out_h - 1 - oy) : oy;
    int b = flip_x ? (out_w - 1 - ox) : ox;
    int iy = transpose ? b : a;
    int ix = transpose ? a : b;

    output[i] = input[((long long)iy * in_w + ix) * c + ch];
}
