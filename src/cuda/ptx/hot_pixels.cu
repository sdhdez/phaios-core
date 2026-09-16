// SPDX-License-Identifier: GPL-3.0-or-later
// Conditional (switching) 3x3 median hot-pixel filter, device side.
//
// Mirrors src/hot_pixels.rs exactly: the same clamped 3x3 gather in the
// same row-major order, the same 19-comparator sorting network run
// pair-by-pair with fminf/fmaxf (IEEE-754-2008 minNum/maxNum, matching
// Rust's f32::min/f32::max bit for bit -- see that file's module doc for
// why), and the same two-term criterion. No transcendentals and no
// shared state, so with -fmad=false (build.rs)
// every operation here is correctly rounded on both sides and the whole
// kernel is bit-exact against the CPU.

// Sort a pair through fminf/fmaxf, reading both inputs before writing
// either -- mirrors sort2 in src/hot_pixels.rs, whose
// `(v[i].min(v[j]), v[i].max(v[j]))` tuple likewise reads both original
// values before either write. Writing `v[i] = fminf(v[i], v[j])` in
// place (skipping the local copies) would corrupt the second line's read
// of `v[i]` once the first line has overwritten it.
__device__ __forceinline__ void sort2(float* v, int i, int j) {
    float a = v[i];
    float b = v[j];
    v[i] = fminf(a, b);
    v[j] = fmaxf(a, b);
}

// The median of nine values via the fixed 19-comparator network
// transcribed verbatim, pair by pair, from `median9` in
// src/hot_pixels.rs (the S. M. Smith (1996) / Devillard `opt_med9`
// network) -- identical order, identical indices, so host and device
// execute the same sequence of comparisons for any input, including one
// that holds a NaN: `fminf`/`fmaxf` implement `minNum`/`maxNum`, so the
// first comparator that touches a NaN's slot replaces it with a copy of
// whatever it was compared against, exactly as `f32::min`/`f32::max` do
// on the host.
__device__ __forceinline__ float median9(float* v) {
    // ── Comparator network (19 pairs) — DO NOT REORDER ──────────────────
    sort2(v, 1, 2);
    sort2(v, 4, 5);
    sort2(v, 7, 8);
    sort2(v, 0, 1);
    sort2(v, 3, 4);
    sort2(v, 6, 7);
    sort2(v, 1, 2);
    sort2(v, 4, 5);
    sort2(v, 7, 8);
    sort2(v, 0, 3);
    sort2(v, 5, 8);
    sort2(v, 4, 7);
    sort2(v, 3, 6);
    sort2(v, 1, 4);
    sort2(v, 2, 5);
    sort2(v, 4, 7);
    sort2(v, 4, 2);
    sort2(v, 6, 4);
    sort2(v, 4, 2);
    // ── End comparator network ───────────────────────────────────────────
    return v[4];
}

// One thread per output element, deriving (y, x, ch) from its linear
// index the same way blur_conv_kernel derives (row, i, ch) -- see
// blur.cu -- but over both spatial axes at once, since the 3x3 window is
// not separable. Layout is the crate's row-major (H, W, C):
// t = (y * w + x) * c + ch.
extern "C" __global__ void hot_pixels_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             float threshold,
                                             float relative,
                                             int h, int w, int c,
                                             long long n) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    long long cc = (long long)c;
    long long ww = (long long)w;

    long long ch  = t % cc;
    long long rem = t / cc;
    long long x   = rem % ww;
    long long y   = rem / ww;

    // Each of the four +/-1 offsets clamped independently -- duplicate-
    // edge padding, a fixed nine-input window even at the corners.
    // Matches blur_conv_kernel's clamp idiom (blur.cu) and the
    // y0/y2/x0/x2 computed in src/hot_pixels.rs exactly.
    long long last_y = (long long)h - 1;
    long long last_x = ww - 1;
    long long y0 = y - 1; if (y0 < 0) y0 = 0;
    long long y2 = y + 1; if (y2 > last_y) y2 = last_y;
    long long x0 = x - 1; if (x0 < 0) x0 = 0;
    long long x2 = x + 1; if (x2 > last_x) x2 = last_x;

    // Gathered in the same row-major window order as the host: top row
    // left-to-right, middle row left-to-right, bottom row left-to-right.
    float v[9];
    v[0] = input[(y0 * ww + x0) * cc + ch];
    v[1] = input[(y0 * ww + x ) * cc + ch];
    v[2] = input[(y0 * ww + x2) * cc + ch];
    v[3] = input[(y  * ww + x0) * cc + ch];
    v[4] = input[(y  * ww + x ) * cc + ch];
    v[5] = input[(y  * ww + x2) * cc + ch];
    v[6] = input[(y2 * ww + x0) * cc + ch];
    v[7] = input[(y2 * ww + x ) * cc + ch];
    v[8] = input[(y2 * ww + x2) * cc + ch];

    // Read separately from the window array, mirroring `let p =
    // img[[y, x, ch]];` in src/hot_pixels.rs, which reads the centre
    // sample directly rather than reusing `window[4]` -- same memory
    // location either way, since median9 has not run yet.
    float p = input[(y * ww + x) * cc + ch];
    float m = median9(v);
    float limit = threshold + relative * fabsf(m);
    output[t] = (fabsf(p - m) > limit) ? m : p;
}
