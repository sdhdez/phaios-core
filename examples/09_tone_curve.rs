// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 09 — Parametric tone curve (ASC CDL).
//!
//! Demonstrates `tone_curve`, the slope/offset/power primary correction,
//! isolating one control at a time so each one's signature is visible:
//!
//! 1. Reference — the identity `(1.0, 0.0, 1.0)`.
//! 2. Slope 1.4 — gain. Scales everything about black.
//! 3. Offset +0.05 — lift. Moves black itself, which is what makes the
//!    shadows go milky.
//! 4. Power 0.7 — gamma. Opens the midtones while pinning 0 and 1.
//! 5. A combined grade: slight lift, gentle gain, midtone bend.
//!
//! What to look for:
//! - In image 2 the darkest patches barely move while the light ones
//!   run out of range: a gain pivots on black, so it costs highlights
//!   first. Compare with image 4, where the same visual midtone lift
//!   leaves the white patch exactly where it was.
//! - Image 3 is the one to study for what *not* to do accidentally: the
//!   black patch (#24) is no longer black, and no later stage can put it
//!   back — the lift is applied before the clamp.
//! - The curve is monotonic in all five, so no pair of patches ever
//!   swaps order. That is the property a spline editor would have to be
//!   constrained to preserve, and it comes free here.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::tone::{ToneCurveParams, tone_curve};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, ToneCurveParams); 5] = [
        ("reference", ToneCurveParams::new(1.0, 0.0, 1.0)),
        ("slope_gain", ToneCurveParams::new(1.4, 0.0, 1.0)),
        ("offset_lift", ToneCurveParams::new(1.0, 0.05, 1.0)),
        ("power_gamma", ToneCurveParams::new(1.0, 0.0, 0.7)),
        ("combined", ToneCurveParams::new(1.15, 0.01, 0.85)),
    ];

    for (name, params) in configs {
        let out = tone_curve(bw.view(), &params).unwrap();
        let path = format!("examples/output/09_curve_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    // Show the shape of the combined curve at the eleven zone anchors.
    let combined = ToneCurveParams::new(1.15, 0.01, 0.85);
    println!("wrote examples/output/09_curve_*.ppm");
    println!("combined grade, sampled at the zone anchors:");
    for zone in 0..=10 {
        let linear = 0.18_f32 * 2.0_f32.powi(zone - 5);
        let sample = Array3::from_elem((1, 1, 1), linear);
        let out = tone_curve(sample.view(), &combined).unwrap()[[0, 0, 0]];
        println!("  zone {zone:>2}: {linear:>9.5} -> {out:>9.5}");
    }
}
