// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 14 — Resampling: resize and straighten.
//!
//! Demonstrates `resize` at three filters and `straighten` at two
//! angles on the synthetic Macbeth chart.
//!
//! What to look for:
//! - `14_resize_area_down.ppm` (½ size) keeps every patch's tone: area
//!   averaging preserves the mean. `14_resize_catmull_up.ppm` (2×)
//!   keeps the patch borders crisp without ringing overshoot on the
//!   flat chart.
//! - `14_resize_bilinear_down.ppm` is the same reduction as
//!   `14_resize_area_down.ppm` through a two-tap filter: on a chart of
//!   flat patches the two are hard to tell apart, which is the point —
//!   the difference between them lives in fine detail this image has
//!   none of.
//! - The straighten outputs are smaller than the input — the largest
//!   inscribed rectangle — and their patch edges stay smooth
//!   (Catmull-Rom), not staircased. Their dimensions are printed, one
//!   line per angle, as `14_straighten_3.ppm` and
//!   `14_straighten_-12.ppm` are written.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::geometry::{ResizeFilter, ResizeParams, StraightenParams, resize, straighten};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let (w, h) = (shared::WIDTH as u32, shared::HEIGHT as u32);

    let configs: [(&str, ResizeParams); 3] = [
        (
            "area_down",
            ResizeParams::new(w / 2, h / 2, ResizeFilter::Area),
        ),
        (
            "bilinear_down",
            ResizeParams::new(w / 2, h / 2, ResizeFilter::Bilinear),
        ),
        (
            "catmull_up",
            ResizeParams::new(w * 2, h * 2, ResizeFilter::CatmullRom),
        ),
    ];
    for (name, params) in configs {
        let out = resize(rgb.view(), &params).unwrap();
        let (oh, ow, _) = out.dim();
        let path = format!("examples/output/14_resize_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), out.view(), ow, oh);
    }

    for degrees in [3.0_f32, -12.0] {
        let out = straighten(rgb.view(), &StraightenParams::new(degrees)).unwrap();
        let (oh, ow, _) = out.dim();
        let path = format!("examples/output/14_straighten_{degrees}.ppm");
        shared::write_ppm_display(Path::new(&path), out.view(), ow, oh);
        println!(
            "straighten {degrees:>6}: {}x{} -> {ow}x{oh}",
            shared::WIDTH,
            shared::HEIGHT
        );
    }

    println!("wrote examples/output/14_{{resize_*,straighten_*}}.ppm");
}
