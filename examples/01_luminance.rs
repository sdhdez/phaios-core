// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 01 — Standard luminance B&W conversion.
//!
//! Demonstrates `luminance_bw` with the default BT.709 weights on the
//! synthetic Macbeth chart. What to look for in the output:
//!
//! - The neutral patches (row 4) should form a smooth grey ramp.
//! - The green patch (#11, yellow-green) will render brighter than the
//!   red patch (#15) despite both being vivid — this is the green
//!   dominance of the BT.709 luminance weights (0.71 vs 0.21).
//! - Compare with examples 02 and 03 to see how different conversion
//!   methods change relative tonal values.

#[path = "shared/mod.rs"]
mod shared;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use std::path::Path;

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    shared::write_ppm_grey(
        Path::new("examples/output/01_luminance_bt709.ppm"),
        bw.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/01_luminance_bt709.ppm");
}
