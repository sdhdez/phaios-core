// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 43 — Unsharp masking with a soft threshold.
//!
//! Demonstrates `sharpen`:
//!
//! ```text
//! blurred = blur_σ(img)
//! detail  = img − blurred
//! out     = img + amount · soft_gate(detail, threshold) · detail
//! ```
//!
//! Three renderings of the B&W Macbeth chart, all at the same σ:
//!
//! 1. `43_none` — the reference, unsharpened.
//! 2. `43_plain` — `threshold = 0`: every pixel's detail is amplified,
//!    including whatever faint low-level ripple sits inside a flat
//!    patch — the textbook unsharp mask, Gonzalez & Woods §3.6.
//! 3. `43_thresholded` — same `amount` and `sigma`, `threshold > 0`:
//!    patch interiors (small `|detail|`) are left alone; patch edges
//!    (large `|detail|`) are sharpened exactly as in image 2.
//!
//! What to look for:
//! - Patch edges in images 2 and 3 carry a halo — the light side
//!   overshoots past white, the dark side undershoots past black. That
//!   is Gonzalez & Woods' own ringing, not a bug: `sharpen` never
//!   clamps.
//! - Inside a patch, image 2 shows faint mottling that image 3 does
//!   not — the threshold has gated the low-level detail out.
//! - The two printed tables below make both effects numeric: a step
//!   edge gets measurably steeper (and stops getting steeper once a
//!   threshold covers it), and gating is by *detail* magnitude at each
//!   position, not by the pixel's own value. The line above them checks
//!   that `amount = 0` is the exact identity, bit for bit.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::sharpen::{SharpenParams, sharpen};

fn write(name: &str, img: ndarray::ArrayView3<f32>) {
    let path = format!("examples/output/43_{name}.ppm");
    shared::write_ppm_grey_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // 1. Reference: unsharpened.
    write("none", bw.view());

    // 2. Plain unsharp mask — no threshold, every pixel's detail amplified.
    let plain_params = SharpenParams::new(1.5, 2.0, 0.0);
    let plain = sharpen(bw.view(), &plain_params).unwrap();
    write("plain", plain.view());

    // 3. Thresholded — same amount/sigma, small detail gated out.
    let gated_params = SharpenParams::new(1.5, 2.0, 0.03);
    let thresholded = sharpen(bw.view(), &gated_params).unwrap();
    write("thresholded", thresholded.view());

    println!("wrote examples/output/43_*.ppm");

    // ── amount = 0 is the exact identity ──────────────────────────────
    let identity_params = SharpenParams::new(0.0, 2.0, 0.03);
    let identity = sharpen(bw.view(), &identity_params).unwrap();
    let same_bits = identity
        .iter()
        .zip(bw.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    println!(
        "\namount=0 is the exact identity: {} ({same_bits} of {} elements bit-identical)",
        identity == bw,
        bw.len()
    );

    // ── A step edge gets steeper, and a threshold can shut that off ──
    println!("\na step edge gets steeper at threshold=0, unchanged once threshold covers it:");
    let n = 48_usize;
    let mut step = Array3::<f32>::zeros((1, n, 1));
    for x in (n / 2)..n {
        step[[0, x, 0]] = 0.6;
    }
    let sigma = 3.0_f32;
    let half = 6_usize;
    let slope = |img: &Array3<f32>| img[[0, n / 2 + half, 0]] - img[[0, n / 2 - half, 0]];
    let input_slope = slope(&step);
    println!("{:>28} {:>10}", "threshold", "slope across the window");
    println!("{:>28} {input_slope:>10.4}", "input (unsharpened)");
    for threshold in [0.0_f32, 0.004, 0.008, 0.012, 5.0] {
        let params = SharpenParams::new(2.0, sigma, threshold);
        let out = sharpen(step.view(), &params).unwrap();
        println!("{threshold:>28.3} {:>10.4}", slope(&out));
    }
    println!(
        "  slope falls back toward the input's 0.6000 as threshold rises past the\n  \
         |detail| this window sees, and threshold 5.0 exceeds every |detail| the step\n  \
         produces anywhere, so the gate stays shut and the slope matches the input\n  \
         exactly — the identity property proved algebraically in src/sharpen.rs\n  \
         (threshold >= max|detail| ⇒ identity), not just measured here."
    );

    // ── What the threshold buys: gating by DETAIL, not by pixel value ─
    println!("\nthreshold gates by |detail| at each position, not by the pixel's own value:");
    println!(
        "a flat ripple (small |detail|) beside a step (large |detail|), sigma 1.5, amount 2.0:"
    );
    // A flat buffer (10..20) separates the ripple from the step by more
    // than the blur's reach, so each region's |detail| reflects only
    // its own content, not the other's.
    let mut mixed = Array3::<f32>::from_elem((1, 40, 1), 0.2_f32);
    for x in 0..10 {
        // A faint ripple — the kind of low-level detail a threshold
        // exists to leave alone.
        mixed[[0, x, 0]] += if x % 2 == 0 { 0.01 } else { -0.01 };
    }
    for x in 20..40 {
        // A hard step — large |detail| at the transition.
        mixed[[0, x, 0]] = 1.4;
    }
    for threshold in [0.0_f32, 0.05, 0.2] {
        let params = SharpenParams::new(2.0, 1.5, threshold);
        let out = sharpen(mixed.view(), &params).unwrap();
        let ripple_amp = (2..8)
            .map(|x| (out[[0, x, 0]] - mixed[[0, x, 0]]).abs())
            .fold(0.0_f32, f32::max);
        let step_amp = (17..24)
            .map(|x| (out[[0, x, 0]] - mixed[[0, x, 0]]).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "  threshold {threshold:>4.2}: max |added| in the ripple {ripple_amp:.4}, at the step {step_amp:.4}"
        );
    }
    println!(
        "  (as threshold rises the ripple's contribution disappears while the step's\n   \
         does not — the ripple's |detail| is far below the step's)"
    );
}
