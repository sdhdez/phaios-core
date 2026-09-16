// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 20 — Gaussian blur, and the two paths behind it.
//!
//! Demonstrates `blur`, which exists mostly so that halation, diffusion
//! and veiling glare can be written on top of it: all three are *blur,
//! weighted, added back*.
//!
//! Five renderings of the checker at increasing σ, plus a printed table
//! showing what the kernel actually does at each one.
//!
//! What to look for:
//! - The impulse response table. Below σ = 6 the transfer is a direct
//!   convolution and matches a sampled true Gaussian to within about a
//!   part in a million; at or above it, three box passes take over and
//!   the profile departs by a few parts in ten thousand of the impulse's
//!   total energy, less as σ grows — invisible in a picture, and the
//!   price of a cost that no longer grows with radius.
//! - The peak column. It is the impulse's centre sample, which falls as
//!   σ rises because the same unit of light is spread over more pixels.
//!   Read it beside the deviation column: a peak that matched a true
//!   Gaussian while the tails did not would still be a wrong filter.
//! - Energy. A blur redistributes light rather than creating it, so the
//!   impulse sums to one — until the kernel is wide enough to overrun
//!   the frame, at which point clamped borders lose the tail. Both cases
//!   are printed.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::blur::{BlurParams, BlurShape, blur};
use phaios_core::bw::{LuminanceStandard, luminance_bw};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    for sigma in [0.0_f32, 1.0, 3.0, 8.0, 24.0] {
        let out = blur(bw.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap();
        let path = format!("examples/output/20_blur_sigma{sigma}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }
    println!("wrote examples/output/20_blur_sigma*.ppm");

    // ── What the kernel is actually doing ────────────────────────────
    println!("\nimpulse response, measured against a true Gaussian:");
    println!(
        "{:>7} {:>8} {:>12} {:>14} {:>10}",
        "sigma", "path", "peak", "max deviation", "energy"
    );
    for sigma in [1.0_f32, 3.0, 5.9, 6.0, 12.0, 24.0] {
        // A frame comfortably wider than the kernel, so nothing leaks.
        let n = (4.0 * sigma).ceil() as usize * 2 + 9;
        let mut imp = Array3::<f32>::zeros((n, n, 1));
        imp[[n / 2, n / 2, 0]] = 1.0;
        let out = blur(imp.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap();

        // The continuous Gaussian this should match, sampled and normalised.
        let two_s2 = 2.0 * f64::from(sigma) * f64::from(sigma);
        let mut ref_img = vec![0.0_f64; n * n];
        let mut total = 0.0_f64;
        for y in 0..n {
            for x in 0..n {
                let dy = y as f64 - (n / 2) as f64;
                let dx = x as f64 - (n / 2) as f64;
                let v = (-(dx * dx + dy * dy) / two_s2).exp();
                ref_img[y * n + x] = v;
                total += v;
            }
        }
        let deviation = (0..n)
            .flat_map(|y| (0..n).map(move |x| (y, x)))
            .map(|(y, x)| (f64::from(out[[y, x, 0]]) - ref_img[y * n + x] / total).abs())
            .fold(0.0_f64, f64::max);
        let energy: f32 = out.iter().sum();
        let path = if sigma < 6.0 { "direct" } else { "box" };
        println!(
            "{sigma:>7.1} {path:>8} {:>12.6} {deviation:>14.2e} {energy:>10.5}",
            out[[n / 2, n / 2, 0]]
        );
    }

    // ── And the border case, stated rather than hidden ───────────────
    let mut small = Array3::<f32>::zeros((16, 16, 1));
    small[[8, 8, 0]] = 1.0;
    let leaked: f32 = blur(small.view(), &BlurParams::new(5.0, BlurShape::Gaussian))
        .unwrap()
        .iter()
        .sum();
    println!(
        "\nthe same impulse in a 16x16 frame at sigma 5: energy {leaked:.4}\n  \
         (borders clamp, so a kernel wider than the frame loses its tails —\n   \
         that is what clamping means, not a defect)"
    );
}
