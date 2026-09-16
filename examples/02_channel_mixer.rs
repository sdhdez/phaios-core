// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 02 — Channel mixer B&W conversion.
//!
//! Demonstrates `channel_mixer_bw` by writing three output images:
//! 1. `02_mixer_bt709.ppm` — the BT.709 reference, taken through
//!    `luminance_bw` so the weights are the exact BT.709 constants
//!    (0.2126, 0.7152, 0.0722) rather than a rounded triple typed in by
//!    hand.
//! 2. `02_mixer_red_only.ppm` — red-boosted weights (1.0, 0.0, 0.0),
//!    infrared-like: reds bright, greens and blues very dark.
//! 3. `02_mixer_green_blue.ppm` — custom weights (0.0, 0.5, 0.5), equal
//!    green+blue, no red.
//!
//! What to look for:
//! - Image 2 inverts the tonal relationship between red (#15) and green
//!   (#14) patches relative to image 1.
//! - Negative weights (not shown here but valid) produce inverted tones.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, channel_mixer_bw, luminance_bw};
use std::path::Path;

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    // 1 — BT.709 reference (via luminance_bw for exact BT.709 weights)
    let bw_bt709 = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/02_mixer_bt709.ppm"),
        bw_bt709.view(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — Red channel only (infrared-like)
    let bw_red = channel_mixer_bw(rgb.view(), [1.0, 0.0, 0.0]).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/02_mixer_red_only.ppm"),
        bw_red.view(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 3 — Equal green+blue, no red
    let bw_gb = channel_mixer_bw(rgb.view(), [0.0, 0.5, 0.5]).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/02_mixer_green_blue.ppm"),
        bw_gb.view(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/02_mixer_{{bt709,red_only,green_blue}}.ppm");
}
