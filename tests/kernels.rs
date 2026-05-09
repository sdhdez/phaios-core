// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for phaios-core numerical correctness.
//!
//! Each test verifies a specific property stated in the specification.
//! Tests are added per kernel as Step 1 progresses.

use std::collections::HashMap;

use ndarray::array;
use phaios_core::bw::{
    ColorFilter, LuminanceStandard, channel_mixer_bw, color_filter_bw, luminance_bw,
};
use phaios_core::tone::{ZoneParams, zone_system};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn rgb(r: f32, g: f32, b: f32) -> ndarray::Array3<f32> {
    array![[[r, g, b]]]
}

// ── B&W luminance tests ───────────────────────────────────────────────────────

/// BT.709 luminance of pure red must equal 0.2126 ± 1e-6.
///
/// Source: ITU-R BT.709-6 (2015), Table 1.
#[test]
fn bw_709_red() {
    let img = rgb(1.0, 0.0, 0.0);
    let out = luminance_bw(img.view(), LuminanceStandard::Bt709).unwrap();
    let y = out[[0, 0, 0]];
    assert!(
        (y - 0.2126_f32).abs() < 1e-6,
        "BT.709 red luminance: expected 0.2126, got {y}"
    );
}

/// BT.709 luminance of pure green must equal 0.7152 ± 1e-6.
#[test]
fn bw_709_green() {
    let img = rgb(0.0, 1.0, 0.0);
    let out = luminance_bw(img.view(), LuminanceStandard::Bt709).unwrap();
    let y = out[[0, 0, 0]];
    assert!(
        (y - 0.7152_f32).abs() < 1e-6,
        "BT.709 green luminance: expected 0.7152, got {y}"
    );
}

/// Channel mixer with weights (1, 0, 0) must return the red channel exactly.
#[test]
fn channel_mixer_red_only() {
    let img = rgb(0.7, 0.3, 0.5);
    let out = channel_mixer_bw(img.view(), [1.0, 0.0, 0.0]).unwrap();
    let y = out[[0, 0, 0]];
    assert!(
        (y - 0.7_f32).abs() < 1e-7,
        "channel mixer [1,0,0]: expected 0.7 (red channel), got {y}"
    );
}

/// Red filter (#25 A) on pure blue input must produce output < 0.05.
///
/// Red filter transmission on blue channel is 0.02; BT.709 blue weight
/// is 0.0722. Combined: 0.02 × 0.0722 = 0.001444 — far below 0.05.
#[test]
fn red_filter_blue_input() {
    let img = rgb(0.0, 0.0, 1.0);
    let out = color_filter_bw(img.view(), ColorFilter::Red25A, LuminanceStandard::Bt709).unwrap();
    let y = out[[0, 0, 0]];
    assert!(
        y < 0.05,
        "red filter + pure blue should be nearly black, got {y}"
    );
}

// ── Zone System tests ─────────────────────────────────────────────────────────

/// A +1-stop offset on Zone V must approximately double middle-grey (0.18).
///
/// At zone_pos = 5.0 (exactly Zone V), the Gaussian peaks at 1.0, so the
/// total offset is exactly 1.0 and the output is 0.18 × 2^1 = 0.36.
#[test]
fn zone_v_plus_one_stop() {
    let img = array![[[0.18_f32]]];
    let mut offsets = HashMap::new();
    offsets.insert(5_i32, 1.0_f32);
    let params = ZoneParams::new(offsets);
    let out = zone_system(img.view(), &params).unwrap();
    let ratio = out[[0, 0, 0]] / 0.18_f32;
    assert!(
        (ratio - 2.0).abs() < 0.05,
        "Zone V +1 stop should give ~2× middle-grey, got ratio {ratio:.4}"
    );
}
