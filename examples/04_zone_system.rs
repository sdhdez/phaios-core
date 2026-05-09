// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 04 — Adams/Archer Zone System tone curve.
//!
//! Demonstrates `zone_system` with three configurations:
//! 1. No-op (empty params) — output == input.
//! 2. "Pull" Zone V: offset Zone V by −1 stop (darker mid-tones).
//! 3. "Push" Zone VII by +1 stop, "deepen" Zone III by −0.5 stops
//!    — a classic landscape / darkroom interpretation curve.
//!
//! What to look for:
//! - Configuration 1 and 3 are written side by side; compare the grey
//!   ramp in row 4 of the chart.
//! - The Gaussian blending means a Zone V offset also affects Zones IV
//!   and VI (σ = 0.8 zones) — the influence is gradual, not a sharp step.
//! - Patches near the target zone show the most change.

#[path = "shared/mod.rs"]
mod shared;

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::tone::{ZoneParams, zone_system};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // 1 — No-op reference
    let params_noop = ZoneParams::default();
    let out_ref = zone_system(bw.view(), &params_noop).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/04_zone_reference.ppm"),
        out_ref.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — Pull Zone V (mid-tones) down by 1 stop
    let mut offsets_pull = HashMap::new();
    offsets_pull.insert(5_i32, -1.0_f32);
    let out_pull = zone_system(bw.view(), &ZoneParams::new(offsets_pull)).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/04_zone_pull_v.ppm"),
        out_pull.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 3 — Push Zone VII (upper mid-tones/highlights) up, deepen Zone III (shadows)
    let mut offsets_push = HashMap::new();
    offsets_push.insert(7_i32, 1.0_f32);
    offsets_push.insert(3_i32, -0.5_f32);
    let out_push = zone_system(bw.view(), &ZoneParams::new(offsets_push)).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/04_zone_push_vii.ppm"),
        out_push.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/04_zone_{{reference,pull_v,push_vii}}.ppm");
}
