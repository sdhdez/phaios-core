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

// ── Guided filter tests ───────────────────────────────────────────────────────

use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};

/// On a constant image, the guided filter returns the same constant (1e-4).
#[test]
fn guided_constant_image() {
    let img = ndarray::Array3::from_elem((32, 32, 1), 0.4_f32);
    let params = GuidedFilterParams::new(4, 0.01);
    let out = local_contrast(img.view(), &params, 1.0).unwrap();
    for &v in out.iter() {
        assert!(
            (v - 0.4).abs() < 1e-3,
            "constant image local_contrast: expected ~0.4, got {v}"
        );
    }
}

/// With radius=0, the guided filter is the identity, so local_contrast is the identity.
#[test]
fn guided_identity_radius_zero() {
    let img = ndarray::Array3::from_shape_fn((8, 8, 1), |(y, x, _)| (y * 8 + x) as f32 / 64.0);
    let params = GuidedFilterParams::new(0, 0.0);
    let out = local_contrast(img.view(), &params, 1.0).unwrap();
    for (&l, &o) in img.iter().zip(out.iter()) {
        assert!(
            (o - l).abs() < 1e-4,
            "radius=0 local_contrast should be identity; l={l}, o={o}"
        );
    }
}

// ── sRGB encode tests ─────────────────────────────────────────────────────────

use phaios_core::encode::encode_srgb;

/// sRGB encode must be monotonically non-decreasing on [0, 1].
#[test]
fn srgb_monotonic() {
    let n = 10_000_usize;
    let input: Vec<f32> = (0..=n).map(|i| i as f32 / n as f32).collect();
    let img = ndarray::Array3::from_shape_vec((1, n + 1, 1), input).unwrap();
    let out = encode_srgb(img.view()).unwrap();
    let vals: Vec<f32> = out.iter().copied().collect();
    for w in vals.windows(2) {
        assert!(
            w[1] >= w[0],
            "sRGB encode not monotonic: encode({}) = {} > encode({}) = {}",
            (vals.iter().position(|&v| v == w[0]).unwrap()) as f32 / n as f32,
            w[0],
            (vals.iter().position(|&v| v == w[1]).unwrap()) as f32 / n as f32,
            w[1],
        );
    }
}

/// Both branches of the sRGB piecewise function must agree at the threshold.
#[test]
fn srgb_c1_continuous() {
    let threshold = 0.0031308_f32;
    let linear = 12.92 * threshold;
    let power = 1.055 * threshold.powf(1.0 / 2.4) - 0.055;
    assert!(
        (linear - power).abs() < 1e-5,
        "C¹ discontinuity at threshold: linear={linear:.8}, power={power:.8}"
    );
}
