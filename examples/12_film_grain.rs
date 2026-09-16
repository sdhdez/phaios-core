// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 12 — Procedural film grain.
//!
//! Demonstrates `film_grain` across intensity, size and seed:
//!
//! 1. Reference — `intensity = 0`, the identity.
//! 2. Fine grain — `intensity 0.15, size 1.0`.
//! 3. Coarse grain — the same intensity at `size 4.0`.
//! 4. Heavy — `intensity 0.4, size 2.0`.
//! 5. The same as 4 with a different seed: same character, different
//!    dice.
//!
//! What to look for:
//! - The neutral ramp in row 4 shows the envelope directly. The white
//!   and black patches at either end are clean; the mid-grey patches
//!   carry the most grain. That is `4·t·(1−t)`, and it is why grain
//!   reads as film rather than as sensor noise.
//! - Images 2 and 3 have the same amount of grain, not the same
//!   coarseness — the analytic normalisation keeps the intensity slider
//!   meaning one thing at every size. Without it, coarse grain would
//!   come out much weaker.
//! - Images 4 and 5 differ only in seed, so their grain is statistically
//!   identical but positioned differently.
//!
//! This example also checks determinism directly: it renders
//! configuration 4 twice and compares the bytes. The table under that
//! line measures the envelope rather than describing it, printing the
//! grain's standard deviation on flat patches from black to white.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::film_grain::{GrainParams, film_grain};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, GrainParams); 5] = [
        ("reference", GrainParams::new(0.0, 1.0, 20_260_815)),
        ("fine", GrainParams::new(0.15, 1.0, 20_260_815)),
        ("coarse", GrainParams::new(0.15, 4.0, 20_260_815)),
        ("heavy", GrainParams::new(0.4, 2.0, 20_260_815)),
        ("heavy_other_seed", GrainParams::new(0.4, 2.0, 1_234_567)),
    ];

    for (name, params) in configs {
        let out = film_grain(bw.view(), &params).unwrap();
        let path = format!("examples/output/12_grain_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    // Determinism is the property the kernel is built around, so the
    // example demonstrates it rather than asserting it in prose.
    let params = GrainParams::new(0.4, 2.0, 20_260_815);
    let first = film_grain(bw.view(), &params).unwrap();
    let second = film_grain(bw.view(), &params).unwrap();
    let identical = first == second;

    // The envelope, measured on flat patches rather than described.
    println!("wrote examples/output/12_grain_*.ppm");
    println!("same seed renders identically: {identical}");
    println!("grain spread by luminance (intensity 0.4, size 2):");
    for level in [0.0_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let patch = Array3::from_elem((64, 64, 1), level);
        let out = film_grain(patch.view(), &params).unwrap();
        let n = out.len() as f32;
        let mean = out.iter().sum::<f32>() / n;
        let std = (out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n).sqrt();
        let bar = "#".repeat((std * 200.0).round() as usize);
        println!("  L = {level:<4} sigma = {std:.4}  {bar}");
    }
}
