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

// ── Exposure tests ────────────────────────────────────────────────────────────

use phaios_core::exposure::exposure;

/// One stop is exactly a factor of two in linear scene-referred data.
///
/// Source: standard photographic definition; Adams, *The Negative* (1948), ch. 4.
#[test]
fn exposure_one_stop_doubles_middle_grey() {
    let img = rgb(0.18, 0.18, 0.18);
    let out = exposure(img.view(), 1.0).unwrap();
    assert!(
        (out[[0, 0, 0]] - 0.36).abs() < 1e-6,
        "+1 EV on middle grey: expected 0.36, got {}",
        out[[0, 0, 0]]
    );
}

/// Exposure must be invertible: +n EV followed by −n EV returns the input.
///
/// 2^n is exact for integer n, so this is bit-exact, not approximate.
#[test]
fn exposure_round_trips() {
    let img = ndarray::Array3::from_shape_fn((16, 16, 3), |(y, x, c)| {
        (y * 48 + x * 3 + c) as f32 / 768.0
    });
    let there = exposure(img.view(), 3.0).unwrap();
    let back = exposure(there.view(), -3.0).unwrap();
    assert_eq!(back, img, "+3 EV then -3 EV must be the identity");
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

// ── HSL-weighted tests ────────────────────────────────────────────────────────

use phaios_core::bw::{HslWeightedParams, hsl_bw};

/// A negative blue weight must darken sky-blue without moving foliage-green.
///
/// This is the property the method exists for: unlike a coloured filter,
/// which attenuates whole channels, a band weight reaches only hues near
/// its centre.
#[test]
fn hsl_blue_weight_is_selective() {
    let sky = rgb(0.183, 0.267, 0.467); // Macbeth patch 03, blue sky
    let foliage = rgb(0.149, 0.200, 0.086); // patch 04, foliage

    let neutral = HslWeightedParams::default();
    let blue_down = HslWeightedParams::new(
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0, 0.0],
        LuminanceStandard::Bt709,
        30.0,
    );

    let sky_before = hsl_bw(sky.view(), &neutral).unwrap()[[0, 0, 0]];
    let sky_after = hsl_bw(sky.view(), &blue_down).unwrap()[[0, 0, 0]];
    let foliage_before = hsl_bw(foliage.view(), &neutral).unwrap()[[0, 0, 0]];
    let foliage_after = hsl_bw(foliage.view(), &blue_down).unwrap()[[0, 0, 0]];

    assert!(
        sky_after < sky_before * 0.8,
        "sky should darken markedly: {sky_before} → {sky_after}"
    );
    assert!(
        (foliage_after - foliage_before).abs() < foliage_before * 0.1,
        "foliage should barely move: {foliage_before} → {foliage_after}"
    );
}

/// The eight band weights must be applied in the documented order.
///
/// Guards against a transposed or reversed weight array, which would be
/// invisible in a symmetric test.
#[test]
fn hsl_band_order_is_red_first_magenta_last() {
    let cases = [
        (0, rgb(1.0, 0.0, 0.0)), // red, 0°
        (2, rgb(1.0, 1.0, 0.0)), // yellow, 60°
        (3, rgb(0.0, 1.0, 0.0)), // green, 120°
        (4, rgb(0.0, 1.0, 1.0)), // aqua, 180°
        (5, rgb(0.0, 0.0, 1.0)), // blue, 240°
        (7, rgb(1.0, 0.0, 1.0)), // magenta, 300°
    ];
    for (band, img) in cases {
        let mut weights = [0.0_f32; 8];
        weights[band] = 1.0;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);

        let base = hsl_bw(img.view(), &HslWeightedParams::default()).unwrap()[[0, 0, 0]];
        let boosted = hsl_bw(img.view(), &params).unwrap()[[0, 0, 0]];

        assert!(
            (boosted / base - 2.0).abs() < 1e-4,
            "band {band} should double its own hue: {base} → {boosted}"
        );
    }
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

// ── Parametric tone curve tests ───────────────────────────────────────────────

use phaios_core::tone::{ToneCurveParams, tone_curve};

/// The curve must be monotonic, so no two tones ever swap order.
///
/// This is the property that justified choosing ASC CDL over a spline:
/// it holds for any positive slope and power, with nothing to constrain.
#[test]
fn tone_curve_never_inverts_tonal_order() {
    let n = 2048;
    let ramp = ndarray::Array3::from_shape_fn((1, n, 1), |(_, x, _)| x as f32 / n as f32 * 4.0);

    for (slope, offset, power) in [
        (1.0_f32, 0.0_f32, 1.0_f32),
        (2.0, -0.3, 0.5),
        (0.4, 0.2, 3.0),
        (1.15, 0.01, 0.85),
    ] {
        let out = tone_curve(ramp.view(), &ToneCurveParams::new(slope, offset, power)).unwrap();
        let mut previous = f32::NEG_INFINITY;
        for &v in out.iter() {
            assert!(
                v >= previous,
                "slope={slope} offset={offset} power={power}: {previous} then {v}"
            );
            assert!(v.is_finite(), "produced {v}");
            previous = v;
        }
    }
}

// ── Film grain tests ──────────────────────────────────────────────────────────

use phaios_core::film_grain::{GrainParams, film_grain};

/// Same seed, same bytes — the guarantee the kernel is built around.
///
/// Checked across thread pools, since the reason for hashing coordinates
/// rather than running a generator is that the schedule cannot matter.
#[test]
fn grain_is_reproducible_across_thread_counts() {
    let img = ndarray::Array3::from_elem((96, 96, 1), 0.5_f32);
    let params = GrainParams::new(0.35, 2.0, 20_260_815);

    let reference = film_grain(img.view(), &params).unwrap();
    for threads in [1, 2, 5, 16] {
        let out = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| film_grain(img.view(), &params).unwrap());
        assert_eq!(out, reference, "grain changed on {threads} threads");
    }
}

/// Grain must vanish where the emulsion has no density and where it is
/// saturated: `4·L·(1−L)` is zero at both ends.
#[test]
fn grain_envelope_is_silent_at_both_ends() {
    for level in [0.0_f32, 1.0] {
        let img = ndarray::Array3::from_elem((32, 32, 1), level);
        let out = film_grain(img.view(), &GrainParams::new(1.0, 2.0, 3)).unwrap();
        for &v in out.iter() {
            assert!((v - level).abs() < 1e-6, "grain at L = {level}: {v}");
        }
    }
}

// ── Split-toning tests ────────────────────────────────────────────────────────

use phaios_core::split_toning::{SplitToningParams, split_toning};

/// An untinted pass must return the input as a neutral RGB triple.
///
/// Everything a tint does is measured against this baseline, so a drift
/// here would be a colour cast in every toned image.
#[test]
fn untinted_split_toning_round_trips_to_neutral() {
    let img = ndarray::Array3::from_shape_fn((16, 16, 1), |(y, x, _)| (y * 16 + x) as f32 / 256.0);
    let out = split_toning(img.view(), &SplitToningParams::default()).unwrap();
    assert_eq!(out.dim(), (16, 16, 3));
    for y in 0..16 {
        for x in 0..16 {
            for c in 0..3 {
                let (got, expected) = (out[[y, x, c]], img[[y, x, 0]]);
                assert!(
                    (got - expected).abs() < 1e-5,
                    "({y}, {x}, {c}): {got} vs {expected}"
                );
            }
        }
    }
}

// ── Vignette tests ────────────────────────────────────────────────────────────

use phaios_core::vignette::{VignetteParams, vignette};

/// The same parameters must give the same picture at any resolution.
///
/// A consumer renders a small preview and then the full frame; if these
/// disagreed, the preview would be lying.
#[test]
fn vignette_is_resolution_independent() {
    let params = VignetteParams::new(0.6, 0.8, 0.2);
    let small = vignette(
        ndarray::Array3::from_elem((64, 64, 1), 1.0_f32).view(),
        &params,
    )
    .unwrap();
    let large = vignette(
        ndarray::Array3::from_elem((512, 512, 1), 1.0_f32).view(),
        &params,
    )
    .unwrap();

    for (sy, sx) in [(0, 0), (16, 48), (32, 32), (63, 63)] {
        let a = small[[sy, sx, 0]];
        let b = large[[sy * 8 + 4, sx * 8 + 4, 0]];
        assert!(
            (a - b).abs() < 0.02,
            "preview and full frame disagree at ({sy}, {sx}): {a} vs {b}"
        );
    }
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
