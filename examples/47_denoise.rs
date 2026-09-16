// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 47 — Guided-filter noise reduction, cross-guided on RGB.
//!
//! Demonstrates `denoise` on the RGB Macbeth chart, cross-guided
//! (`C == 3`): a shared luminance guide `I` provides the edges, and each
//! channel is smoothed according to *its own* covariance with that guide
//! rather than its own local variance.
//!
//! ```text
//! I     = luminance_bw(img)
//! a_c   = cov(I, p_c) / (var(I) + noise_sigma^2)
//! b_c   = mean(p_c) - a_c * mean(I)
//! q_c   = mean(a_c) * I + mean(b_c)
//! out_c = p_c - amount * (p_c - q_c)
//! ```
//!
//! Fixed-seed per-channel noise is added to the chart in code, via
//! [`phaios_core::film_grain::splitmix64`] — no image assets, ever, and
//! reproducible bit-for-bit on every run.
//!
//! Three files are written: `47_none` (the clean chart), `47_noisy`
//! (the same chart with the noise added) and `47_denoised`.
//!
//! What to look for:
//! - `47_noisy` against `47_denoised`: the noise inside every patch is
//!   visibly quieter while the patch boundaries stay as legible as they
//!   are in `47_none` — the printed variance and edge-height tables
//!   below make both effects numeric.
//! - The third table plants a synthetic RGB probe (not from the chart):
//!   a channel with no structure of its own is smoothed at essentially
//!   the *same* rate whether or not the *other* channels' shared
//!   luminance guide has a strong edge nearby — proof that the guide,
//!   not the channel's own signal, drives the smoothing.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::denoise::{DenoiseParams, denoise};
use phaios_core::film_grain::splitmix64;

/// Denoise parameters used for the main chart rendering: `noise_sigma`
/// tuned to roughly match the injected noise's own standard deviation.
const RADIUS: u32 = 4;
const NOISE_SIGMA: f32 = 0.04;

/// The "Neutral 5" (~18% grey) patch, used for the flat-area noise
/// measurement: patch index 21 (0-based), grid row 3 col 3.
const GREY_PATCH: (usize, usize) = (192, 192); // (y0, x0)

/// The White (idx 18) / Neutral 8 (idx 19) boundary, used for the edge
/// measurement: adjacent patches, a real ~0.34 step at x = 64.
const EDGE_ROW: usize = 220;
const EDGE_X: usize = 64;

fn write(name: &str, img: ndarray::ArrayView3<f32>) {
    let path = format!("examples/output/47_{name}.ppm");
    shared::write_ppm_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
}

/// Sample variance over an interior sub-window of a patch, well clear of
/// its own borders (`margin` pixels in from each edge) so the denoise
/// radius never reaches outside the patch while sampling.
fn patch_variance(img: &Array3<f32>, (y0, x0): (usize, usize), margin: usize, ch: usize) -> f64 {
    let mut vals = Vec::new();
    for y in (y0 + margin)..(y0 + shared::PATCH_SIZE - margin) {
        for x in (x0 + margin)..(x0 + shared::PATCH_SIZE - margin) {
            vals.push(img[[y, x, ch]] as f64);
        }
    }
    let n = vals.len() as f64;
    let mean = vals.iter().sum::<f64>() / n;
    vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let none = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    write("none", none.view());

    // ── Fixed-seed per-channel noise, uncorrelated across channels ────
    // (real sensor read noise is roughly independent per channel too),
    // via splitmix64 -- no RNG dependency, reproducible bit-for-bit.
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    let noisy = Array3::from_shape_fn(none.raw_dim(), |(_, _, _)| {
        state = splitmix64(state);
        let bits24 = (state >> 40) as i32; // 0..=0xFF_FFFF
        (bits24 - 0x0080_0000) as f32 / 8_388_608.0 * 0.05 // +/- 0.05 amplitude
    }) + &none;
    write("noisy", noisy.view());

    let params = DenoiseParams::new(RADIUS, NOISE_SIGMA, 1.0, LuminanceStandard::Bt709);
    let denoised = denoise(noisy.view(), &params).unwrap();
    write("denoised", denoised.view());

    println!("wrote examples/output/47_*.ppm");

    // ── Noise gone in a flat patch interior ────────────────────────────
    println!(
        "\nvariance inside a flat patch (Neutral 5, ~18% grey), radius={RADIUS} margin clear:"
    );
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "channel", "noisy var", "denoised var", "ratio"
    );
    for ch in 0..3 {
        let margin = 2 * RADIUS as usize;
        let v_noisy = patch_variance(&noisy, GREY_PATCH, margin, ch);
        let v_denoised = patch_variance(&denoised, GREY_PATCH, margin, ch);
        println!(
            "{ch:>10} {v_noisy:>12.6} {v_denoised:>12.6} {:>10.4}",
            v_denoised / v_noisy.max(1e-12)
        );
    }

    // ── Edges kept: a real patch boundary survives denoising ──────────
    println!("\na real patch boundary (White -> Neutral 8, x={EDGE_X}), channel 0:");
    let half = 2 * RADIUS as usize;
    let step =
        |img: &Array3<f32>| img[[EDGE_ROW, EDGE_X + half, 0]] - img[[EDGE_ROW, EDGE_X - half, 0]];
    println!("  true (no noise):  {:.4}", step(&none));
    println!("  noisy:            {:.4}", step(&noisy));
    println!(
        "  denoised:         {:.4}  (most of the true step survives)",
        step(&denoised)
    );

    // ── A channel smoothed by the guide, not by its own signal ────────
    // A dedicated synthetic probe, not drawn from the chart: R and G
    // share a strong step (a strong shared luminance edge); B is flat
    // plus a small checkerboard, uncorrelated with that step.
    println!("\nsynthetic RGB probe: does blue's own noise care where the *guide's* edge is?");
    const PROBE_RADIUS: u32 = 4;
    let (ph, pw) = (24_usize, 48_usize);
    let mid = pw / 2;
    let probe = Array3::from_shape_fn((ph, pw, 3), |(y, x, c)| match c {
        0 | 1 => {
            if x < mid {
                0.2_f32
            } else {
                0.8_f32
            }
        }
        _ => {
            0.5 + if (x + y) % 2 == 0 {
                0.05_f32
            } else {
                -0.05_f32
            }
        }
    });
    let probe_params = DenoiseParams::new(PROBE_RADIUS, 0.1, 1.0, LuminanceStandard::Bt709);
    let probe_out = denoise(probe.view(), &probe_params).unwrap();
    let reach = 2 * PROBE_RADIUS as usize;
    let checker_p2p = |col: usize| -> f32 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for y in reach..(ph - reach) {
            let v = probe_out[[y, col, 2]];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };
    let near_edge = checker_p2p(mid);
    let far_from_edge = checker_p2p(reach + 1);
    println!(
        "  blue's checker amplitude near the guide's edge: {near_edge:.4}, far from it: {far_from_edge:.4}"
    );
    println!(
        "  (close to each other -- blue is smoothed at the same rate regardless of the\n   \
         guide's own structure nearby, because a_blue comes from cov(I, blue), not\n   \
         var(blue); red, which *shares* the guide's edge, keeps it: left={:.4} right={:.4})",
        probe_out[[ph / 2, mid - 1, 0]],
        probe_out[[ph / 2, mid, 0]]
    );
}
