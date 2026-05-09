// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 03 — Colour filter B&W simulation.
//!
//! Applies all six Wratten-style presets (None, Yellow #8 K2, Orange #21,
//! Red #25 A, Green #11 X1, Blue #47 C5) and writes one PPM per preset.
//!
//! What to look for:
//! - Red #25 A makes the blue patch (#03, blue sky) very dark and
//!   lightens reds (#09, #15). Classic infrared / dramatic sky effect.
//! - Green #11 X1 brightens the foliage patch (#04, #11, #14) and
//!   darkens reds and blues — natural landscape look.
//! - Blue #47 C5 effectively inverts what Red does: sky is bright, reds
//!   are dark. Used for atmospheric haze enhancement.

#[path = "shared/mod.rs"]
mod shared;

use ndarray::Array3;
use phaios_core::bw::{ColorFilter, LuminanceStandard, color_filter_bw};
use std::path::Path;

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    let presets: &[(&str, ColorFilter)] = &[
        ("none", ColorFilter::NoFilter),
        ("yellow", ColorFilter::Yellow8K2),
        ("orange", ColorFilter::Orange21),
        ("red", ColorFilter::Red25A),
        ("green", ColorFilter::Green11X1),
        ("blue", ColorFilter::Blue47C5),
    ];

    for (name, filter) in presets {
        let bw = color_filter_bw(rgb.view(), *filter, LuminanceStandard::Bt709).unwrap();
        let out = format!("examples/output/03_filter_{name}.ppm");
        shared::write_ppm_grey(
            Path::new(&out),
            bw.as_slice().unwrap(),
            shared::WIDTH,
            shared::HEIGHT,
        );
        println!("wrote {out}");
    }
}
