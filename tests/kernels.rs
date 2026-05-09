// SPDX-License-Identifier: GPL-3.0-or-later
//! Integration tests for phaios-core numerical correctness.
//!
//! Each test verifies a specific property stated in the specification.
//! Tests are added per kernel as Step 1 progresses.

use ndarray::array;
use phaios_core::bw::{
    ColorFilter, LuminanceStandard, channel_mixer_bw, color_filter_bw, luminance_bw,
};

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
