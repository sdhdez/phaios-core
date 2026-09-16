// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 08 — HSL-weighted B&W conversion.
//!
//! Demonstrates `hsl_bw`, which scales the luminance of each pixel by a
//! Gaussian blend of eight per-hue-band weights, modulated by how
//! saturated the pixel is. Four configurations:
//!
//! 1. All weights zero — identical to `luminance_bw` (the reference).
//! 2. Blue −0.8 — the classic darkened sky, without touching anything
//!    else. Compare with example 03's red filter, which reaches the same
//!    place by attenuating whole channels and drags the greens down too.
//! 3. Yellow +0.8, green +0.4 — foliage separation.
//! 4. Red −0.9 — skin and reds pushed down, a look that no Wratten
//!    filter produces because a real filter cannot subtract.
//!
//! What to look for:
//! - The neutral patches (row 4) are identical in all four images. Zero
//!   chroma means zero modulation, whatever the weights say: the method
//!   only moves colours, never greys.
//! - In image 2 the blue-sky patch (#03) darkens sharply while the
//!   bluish-green patch (#06) moves much less — its hue is about 70°
//!   from the blue band centre, where the default 30° sigma has fallen
//!   to roughly a fifteenth of peak.
//! - In image 4 the red patch (#15) goes nearly black while the orange
//!   patch (#07) only dims: bands overlap, they do not partition.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{HslWeightedParams, LuminanceStandard, hsl_bw};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    // Band order: red, orange, yellow, green, aqua, blue, purple, magenta.
    let configs: [(&str, [f32; 8]); 4] = [
        ("reference", [0.0; 8]),
        ("blue_down", [0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0, 0.0]),
        ("foliage", [0.0, 0.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0]),
        ("red_down", [-0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    for (name, weights) in configs {
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        let out = hsl_bw(rgb.view(), &params).unwrap();
        let path = format!("examples/output/08_hsl_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    println!("wrote examples/output/08_hsl_{{reference,blue_down,foliage,red_down}}.ppm");
}
