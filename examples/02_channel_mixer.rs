// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 02 — Channel mixer B&W conversion.
//!
//! Demonstrates `channel_mixer_bw` by writing three output images:
//! 1. Standard BT.709 weights (0.21, 0.72, 0.07) as a reference.
//! 2. Red-boosted weights (1.0, 0.0, 0.0) — infrared-like: reds bright,
//!    greens and blues very dark.
//! 3. Custom weights (0.0, 0.5, 0.5) — equal green+blue, no red.
//!
//! What to look for:
//! - Image 2 inverts the tonal relationship between red (#15) and green
//!   (#14) patches relative to image 1.
//! - Negative weights (not shown here but valid) produce inverted tones.

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
    shared::write_ppm_grey(
        Path::new("examples/output/02_mixer_bt709.ppm"),
        bw_bt709.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — Red channel only (infrared-like)
    let bw_red = channel_mixer_bw(rgb.view(), [1.0, 0.0, 0.0]).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/02_mixer_red_only.ppm"),
        bw_red.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 3 — Equal green+blue, no red
    let bw_gb = channel_mixer_bw(rgb.view(), [0.0, 0.5, 0.5]).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/02_mixer_green_blue.ppm"),
        bw_gb.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/02_mixer_{{bt709,red_only,green_blue}}.ppm");
}
