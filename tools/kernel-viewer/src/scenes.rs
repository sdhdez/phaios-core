// SPDX-License-Identifier: GPL-3.0-or-later
//! Built-in test scenes (all generated linear, deterministic, no assets)
//! and the linear-space downscaler.

use ndarray::Array3;

/// The synthetic "photo": 1536×1024, built to exercise the kernels the
/// flat Macbeth patches show poorly — smooth gradients for vignette and
/// zone reading, >1.0 highlights for exposure/tone headroom, fine
/// texture for grain and local contrast, colour patches for the HSL
/// bands, deep shadow detail for the low zones.
pub fn synthetic_photo() -> Array3<f32> {
    const W: usize = 1536;
    const H: usize = 1024;
    let mut img = Array3::<f32>::zeros((H, W, 3));

    // Deterministic noise: same generator and seed as benches/kernels.rs.
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || -> f32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0
    };
    // Pre-generate the noise patch so pixel order can't drift.
    let noise: Vec<f32> = (0..340 * 256).map(|_| next()).collect();

    for y in 0..H {
        for x in 0..W {
            let px: [f32; 3] = if y < 340 {
                // Sky: vertical gradient, horizon → zenith.
                let t = y as f32 / 339.0; // 0 zenith-ward at top? invert:
                let t = 1.0 - t; // t = 1 at top (zenith), 0 at horizon
                let lerp = |a: f32, b: f32| a + (b - a) * t;
                [lerp(0.30, 0.10), lerp(0.42, 0.16), lerp(0.65, 0.45)]
            } else if y < 680 {
                // Texture band: six 256-px patches.
                let ty = y - 340;
                match x / 256 {
                    0 => {
                        // 1-px checker.
                        let v = if (x + y) % 2 == 0 { 0.18 } else { 0.62 };
                        [v, v, v]
                    }
                    1 => {
                        let v = 0.1 + 0.6 * noise[ty * 256 + (x - 256)];
                        [v, v, v]
                    }
                    2 => {
                        // Smooth horizontal ramp 0 → 1 across the patch.
                        let v = (x - 512) as f32 / 255.0;
                        [v, v, v]
                    }
                    3 => [0.35, 0.22, 0.17],    // skin
                    4 => [0.149, 0.200, 0.086], // foliage (Macbeth 04)
                    _ => [0.183, 0.267, 0.467], // blue sky (Macbeth 03)
                }
            } else {
                // Shadow band: gradient with faint sinusoidal detail.
                let t = (y - 680) as f32 / (H - 680 - 1) as f32;
                let base = 0.020 + (0.002 - 0.020) * t;
                let detail = 1.0 + 0.15 * (x as f32 * std::f32::consts::TAU / 24.0).sin();
                let v = base * detail;
                [v, v, v]
            };
            for c in 0..3 {
                img[[y, x, c]] = px[c];
            }
        }
    }

    // Sun: additive Gaussian clipping well above 1.0 — proves the
    // display clamp and gives exposure/tone something to recover.
    let (sx, sy, sigma, peak) = (1200.0_f32, 90.0_f32, 45.0_f32, 4.0_f32);
    for y in 0..340 {
        for x in 0..W {
            let d2 = (x as f32 - sx).powi(2) + (y as f32 - sy).powi(2);
            let add = peak * (-d2 / (2.0 * sigma * sigma)).exp();
            if add > 1e-3 {
                for c in 0..3 {
                    img[[y, x, c]] += add;
                }
            }
        }
    }
    img
}

/// The 24-patch Macbeth chart at 128-px patches (768×512).
///
/// Patch values duplicated from `examples/shared/mod.rs` (BabelColor D65
/// averages, sRGB transfer removed). Duplicated rather than imported:
/// `examples/shared` is `#[path]`-included test scaffolding, not part of
/// the phaios-core public API, and a const table beats exporting it.
pub fn macbeth() -> Array3<f32> {
    const PATCHES: [[f32; 3]; 24] = [
        [0.400, 0.225, 0.155],
        [0.763, 0.488, 0.349],
        [0.183, 0.267, 0.467],
        [0.149, 0.200, 0.086],
        [0.341, 0.345, 0.625],
        [0.148, 0.604, 0.531],
        [0.763, 0.325, 0.031],
        [0.102, 0.145, 0.510],
        [0.631, 0.165, 0.165],
        [0.082, 0.045, 0.122],
        [0.416, 0.620, 0.059],
        [0.749, 0.514, 0.012],
        [0.027, 0.063, 0.416],
        [0.090, 0.306, 0.094],
        [0.502, 0.039, 0.031],
        [0.714, 0.620, 0.008],
        [0.565, 0.122, 0.404],
        [0.012, 0.353, 0.502],
        [0.914, 0.914, 0.914],
        [0.573, 0.573, 0.573],
        [0.353, 0.353, 0.353],
        [0.188, 0.188, 0.188],
        [0.086, 0.086, 0.086],
        [0.031, 0.031, 0.031],
    ];
    const P: usize = 128;
    let mut img = Array3::<f32>::zeros((4 * P, 6 * P, 3));
    for (i, patch) in PATCHES.iter().enumerate() {
        let (row, col) = (i / 6, i % 6);
        for y in 0..P {
            for x in 0..P {
                for c in 0..3 {
                    img[[row * P + y, col * P + x, c]] = patch[c];
                }
            }
        }
    }
    img
}

/// Downscale so the longer edge is at most `max_edge`, by plain area
/// averaging over LINEAR values.
///
/// Deliberately not `image::imageops`: this must provably run in linear
/// light (after any decode), and 25 lines of arithmetic is easier to
/// trust than a resampling framework's colour assumptions.
pub fn downscale_max_edge(img: &Array3<f32>, max_edge: usize) -> Array3<f32> {
    let (h, w, c) = img.dim();
    let edge = h.max(w);
    if edge <= max_edge {
        return img.clone();
    }
    let factor = edge.div_ceil(max_edge); // integer box size
    let (nh, nw) = (h / factor, w / factor);
    let mut out = Array3::<f32>::zeros((nh, nw, c));
    let norm = 1.0 / (factor * factor) as f32;
    for y in 0..nh {
        for x in 0..nw {
            for ch in 0..c {
                let mut acc = 0.0_f32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        acc += img[[y * factor + dy, x * factor + dx, ch]];
                    }
                }
                out[[y, x, ch]] = acc * norm;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenes_are_deterministic() {
        let hash = |a: &Array3<f32>| -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for v in a.iter() {
                h ^= v.to_bits() as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            h
        };
        assert_eq!(hash(&synthetic_photo()), hash(&synthetic_photo()));
        assert_eq!(hash(&macbeth()), hash(&macbeth()));
    }

    #[test]
    fn downscaler_preserves_the_mean() {
        let img = synthetic_photo();
        let small = downscale_max_edge(&img, 400);
        // Means agree closely (edge cropping from integer division allowed).
        let mean = |a: &Array3<f32>| a.iter().sum::<f32>() / a.len() as f32;
        assert!((mean(&img) - mean(&small)).abs() < 0.01);
        assert!(small.dim().0.max(small.dim().1) <= 400);
    }

    #[test]
    fn sun_clips_above_one() {
        let img = synthetic_photo();
        let max = img.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max > 2.0, "sun should clip well above 1.0, got {max}");
    }
}
