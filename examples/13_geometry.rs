// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 13 — Exact geometry: orientation and crop.
//!
//! Demonstrates `orient` (all eight Exif transforms) and `crop`, and the
//! property that makes their pipeline position load-bearing: vignetting
//! a crop is not the same image as cropping a vignette.
//!
//! What to look for:
//! - The eight orientation outputs of the chart: four keep 6×4 patches,
//!   four are transposed to 4×6. Patch #01 (dark skin, top-left) tracks
//!   the corner through every transform.
//! - `13_geo_crop_then_vignette.ppm` darkens toward the corners of the
//!   *crop*; `13_geo_vignette_then_crop.ppm` shows the off-centre
//!   falloff of the original frame. Geometry runs first in the pipeline
//!   precisely so the first behaviour is what a sidecar reproduces.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::geometry::{CropParams, Orientation, crop, orient};
use phaios_core::vignette::{VignetteParams, vignette};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    for value in 1..=8_u16 {
        let o = Orientation::from_exif(value).unwrap();
        let out = orient(rgb.view(), o).unwrap();
        let (h, w, _) = out.dim();
        let path = format!("examples/output/13_geo_orient_{value}.ppm");
        shared::write_ppm_display(Path::new(&path), out.view(), w, h);
    }

    // The order property, made visible.
    let cp = CropParams::new(0, 0, shared::WIDTH as u32 / 2, shared::HEIGHT as u32);
    let vg = VignetteParams::new(0.7, 1.0, 0.0);

    let a = vignette(crop(rgb.view(), &cp).unwrap().view(), &vg).unwrap();
    let (h, w, _) = a.dim();
    shared::write_ppm_display(
        Path::new("examples/output/13_geo_crop_then_vignette.ppm"),
        a.view(),
        w,
        h,
    );

    let b = crop(vignette(rgb.view(), &vg).unwrap().view(), &cp).unwrap();
    shared::write_ppm_display(
        Path::new("examples/output/13_geo_vignette_then_crop.ppm"),
        b.view(),
        w,
        h,
    );

    println!(
        "wrote examples/output/13_geo_{{orient_1..8,crop_then_vignette,vignette_then_crop}}.ppm"
    );
}
