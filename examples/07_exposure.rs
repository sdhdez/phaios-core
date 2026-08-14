// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 07 — Exposure compensation.
//!
//! Demonstrates `exposure` at −2, 0 and +2 EV on the synthetic Macbeth
//! chart, before any B&W conversion — its position in the pipeline.
//!
//! What to look for:
//! - Each stop is a factor of two in *linear* light, but the outputs are
//!   sRGB-encoded for display, so the visible steps look far smaller
//!   than 2×. That gap between linear arithmetic and perceived
//!   brightness is the whole reason the pipeline is linear until the
//!   final encode.
//! - At +2 EV the white patch (#19) and the light neutrals clip to 255
//!   in the 8-bit file, but the underlying f32 values are above 1.0 and
//!   still recoverable: the kernel does not clamp. Pull the same array
//!   back with −2 EV and the original values return exactly.
//! - At −2 EV the dark patches compress towards black but keep their
//!   ordering — no crushing, because nothing is clipped at the bottom
//!   either.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::exposure::exposure;

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    for (name, stops) in [("minus2", -2.0_f32), ("reference", 0.0), ("plus2", 2.0)] {
        let out = exposure(rgb.view(), stops).unwrap();
        let path = format!("examples/output/07_exposure_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    // Round-trip: +2 EV then −2 EV must return the original values.
    let up = exposure(rgb.view(), 2.0).unwrap();
    let back = exposure(up.view(), -2.0).unwrap();
    let max_err = rgb
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    println!("wrote examples/output/07_exposure_{{minus2,reference,plus2}}.ppm");
    println!("+2 EV then -2 EV round-trip: max error {max_err:.3e}");
}
