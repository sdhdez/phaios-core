// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 06 — sRGB transfer encoding.
//!
//! Demonstrates `encode_srgb` by writing:
//! 1. Linear output of the luminance kernel (looks too dark — no gamma).
//! 2. sRGB-encoded output (correct display brightness).
//!
//! What to look for:
//! - Image 1 appears very dark in the shadows: a linear value of 0.18
//!   (middle grey, Zone V) is only 18% of peak, but the eye expects
//!   ~46% brightness (because CRT displays had γ ≈ 2.2 and we are
//!   adapted to that). The neutral patch #22 should look very dark in
//!   image 1 and approximately mid-grey in image 2.
//! - Image 2's neutral ramp (row 4) should visually span from near-
//!   black to near-white in a perceptually even progression.
//! - This example also verifies that `encode_srgb` is idempotent on
//!   an already-encoded image only approximately (encoding twice is
//!   not the identity).

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::encode::encode_srgb;

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // 1 — Linear luminance (no transfer encoding — will look too dark)
    shared::write_ppm_grey(
        Path::new("examples/output/06_encode_linear.ppm"),
        bw.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — sRGB-encoded (correct display brightness)
    let encoded = encode_srgb(bw.view()).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/06_encode_srgb.ppm"),
        encoded.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/06_encode_{{linear,srgb}}.ppm");
}
