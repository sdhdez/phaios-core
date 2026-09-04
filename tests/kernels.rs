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
    // Bands 1 and 6 were absent until the v0.2 audit: swapping the orange
    // and purple centres (or their weight slots) passed the whole suite.
    let cases = [
        (0, rgb(1.0, 0.0, 0.0)), // red, 0°
        (1, rgb(1.0, 0.5, 0.0)), // orange, 30°
        (2, rgb(1.0, 1.0, 0.0)), // yellow, 60°
        (3, rgb(0.0, 1.0, 0.0)), // green, 120°
        (4, rgb(0.0, 1.0, 1.0)), // aqua, 180°
        (5, rgb(0.0, 0.0, 1.0)), // blue, 240°
        (6, rgb(0.5, 0.0, 1.0)), // purple, 270°
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

// ── Geometry tests ────────────────────────────────────────────────────────────

use phaios_core::geometry::{CropParams, Orientation, crop, orient};

/// The full dihedral group: every orientation composed with its inverse
/// is the identity, and the four quarter-turn compositions close the
/// group. This is the property that makes sidecar-recorded orientation
/// reproducible.
#[test]
fn geometry_orientations_form_the_dihedral_group() {
    let img = ndarray::Array3::from_shape_fn((9, 13, 3), |(y, x, c)| (y * 39 + x * 3 + c) as f32);
    for value in 1..=8_u16 {
        let o = Orientation::from_exif(value).unwrap();
        let back = orient(orient(img.view(), o).unwrap().view(), o.inverse()).unwrap();
        assert_eq!(back, img, "{o:?}");
    }
    // Two quarter turns are a half turn.
    let two = orient(
        orient(img.view(), Orientation::Rotate90).unwrap().view(),
        Orientation::Rotate90,
    )
    .unwrap();
    assert_eq!(two, orient(img.view(), Orientation::Rotate180).unwrap());
}

/// Crop then vignette must equal vignette centred on the crop — i.e.
/// geometry-first ordering is what the vignette centre means.
#[test]
fn crop_defines_the_vignette_centre() {
    let img = ndarray::Array3::from_elem((64, 64, 1), 1.0_f32);
    let params = CropParams::new(0, 0, 32, 64); // left half
    let vg = phaios_core::vignette::VignetteParams::new(0.8, 1.0, 0.0);

    let cropped_then_vignetted =
        phaios_core::vignette::vignette(crop(img.view(), &params).unwrap().view(), &vg).unwrap();
    // The brightest point must be the centre of the CROP (16, 32), not
    // the centre of the original frame (32, 32) clipped to the half.
    let mut best = (0usize, 0usize);
    let mut best_v = 0.0_f32;
    for y in 0..64 {
        for x in 0..32 {
            let v = cropped_then_vignetted[[y, x, 0]];
            if v > best_v {
                best_v = v;
                best = (y, x);
            }
        }
    }
    assert!(
        (best.0 as i64 - 32).abs() <= 1 && (best.1 as i64 - 16).abs() <= 1,
        "vignette centre is at {best:?}, expected ~(32, 16) — the crop's centre"
    );
}

use phaios_core::geometry::{ResizeFilter, ResizeParams, StraightenParams, resize, straighten};

/// Downscaling must preserve total light (energy): the area filter is a
/// weighted average, so the mean survives — the property that makes
/// preview renders photometrically faithful to exports.
#[test]
fn resize_preserves_the_mean() {
    let img = ndarray::Array3::from_shape_fn((96, 64, 1), |(y, x, _)| {
        ((y * 64 + x) % 199) as f32 / 199.0
    });
    let out = resize(img.view(), &ResizeParams::new(23, 41, ResizeFilter::Area)).unwrap();
    let mean = |a: &ndarray::Array3<f32>| a.iter().sum::<f32>() / a.len() as f32;
    assert!((mean(&img) - mean(&out)).abs() < 5e-3);
}

/// Straightening by +θ then −θ returns the surviving region close to
/// the original (two resamplings soften, but must not shift): the
/// centre pixel neighbourhood correlates strongly with the source.
#[test]
fn straighten_round_trip_stays_registered() {
    let img =
        ndarray::Array3::from_shape_fn((128, 128, 1), |(y, x, _)| (((x / 8) + (y / 8)) % 2) as f32);
    let there = straighten(img.view(), &StraightenParams::new(10.0)).unwrap();
    let back = straighten(there.view(), &StraightenParams::new(-10.0)).unwrap();
    let (bh, bw, _) = back.dim();
    // Compare the central quarter against the same region of the source.
    let (oy, ox) = ((128 - bh) / 2, (128 - bw) / 2);
    let mut worst = 0.0_f32;
    for y in bh / 4..3 * bh / 4 {
        for x in bw / 4..3 * bw / 4 {
            let d = (back[[y, x, 0]] - img[[y + oy, x + ox, 0]]).abs();
            worst = worst.max(d);
        }
    }
    assert!(
        worst < 0.6,
        "round-trip straighten lost registration: worst {worst}"
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

/// The guided filter must lift fine texture *upwards* and leave a step
/// edge un-haloed — the two behaviours it exists for.
///
/// Audit findings F01, F14, F15 and F25. Before this test every
/// value-level exercise of `local_contrast`, here and in the unit tests,
/// fed it either a constant image or radius 0. On a constant image the
/// detail term `L − q` is identically zero, and at radius 0 the filter is
/// the identity, so the whole edge-preserving model could be arbitrarily
/// wrong and the suite still passed. Four mutations of
/// `src/local_contrast.rs` were each confirmed to change the output and
/// each left the suite green: dropping the `− mean(L)²` term from the
/// variance (F01), flipping the sign of the detail term (F14), forcing
/// `a = 0` so the filter collapses to a box mean (F15), and returning the
/// input unchanged (F25).
///
/// Measured on the image below at radius 4, ε = 0.01, strength 1.0, over
/// the plateau interiors (`—` where an earlier assertion fires first and
/// the halo is never reached):
///
/// | source | texture p2p | signed lift at a peak | block-mean halo |
/// |---|---|---|---|
/// | input  | 0.100000024 | —            | 0.0         |
/// | HEAD   | 0.179990250 | +0.039995134 | 0.030549347 |
/// | F01    | 0.119045120 | +0.009522557 | —           |
/// | F14    | 0.020009756 | −0.039995120 | —           |
/// | F15    | 0.199984760 | +0.049992383 | 0.237037120 |
/// | F25    | 0.099999994 |  0.0         | —           |
///
/// F14 is why the texture assertion is signed rather than a magnitude:
/// under the sign flip `max|out − in|` is 0.047312379, bit-identical to
/// HEAD, so a test measuring size alone is blind to it by construction.
/// F15 is why the halo assertion is here: a box-mean unsharp mask
/// amplifies texture *more* than the guided filter does, and only edge
/// preservation tells the two apart.
///
/// Reference: Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image
/// Filtering", *ECCV 2010*, LNCS 6311, pp. 1–14.
#[test]
fn guided_filter_lifts_texture_and_preserves_the_edge() {
    const N: usize = 64;
    const PLATEAU_LO: f32 = 0.2;
    const PLATEAU_HI: f32 = 0.8;
    const CHECKER: f32 = 0.05;
    const RADIUS: u32 = 4;

    // A step edge down the middle carrying a ±0.05 checkerboard: an edge
    // to preserve and a texture to lift, in one image. None of 0.2, 0.8
    // and 0.05 is exactly representable in binary32, which matters — an
    // earlier attempt to break this kernel by hand was a no-op because
    // the test input happened to be exact in f32.
    let img = ndarray::Array3::from_shape_fn((N, N, 1), |(y, x, _)| {
        let plateau = if x < N / 2 { PLATEAU_LO } else { PLATEAU_HI };
        let checker = if (x + y) % 2 == 0 { CHECKER } else { -CHECKER };
        plateau + checker
    });
    let params = GuidedFilterParams::new(RADIUS, 0.01);
    let out = local_contrast(img.view(), &params, 1.0).unwrap();

    // Interiors of the two plateaus. The coefficients a and b are averaged
    // through a second window pass, so a pixel draws on input up to 2r
    // away; these columns and rows are 2r clear of both the step and the
    // image border, and so are free of edge and window-clamping effects.
    let reach = 2 * RADIUS as usize;
    let interiors = [reach..(N / 2 - reach), (N / 2 + reach)..(N - reach)];
    let input_p2p = 2.0 * CHECKER;

    for cols in interiors {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut least_lift, mut least_drop) = (f32::INFINITY, f32::INFINITY);
        for y in reach..(N - reach) {
            for x in cols.clone() {
                let (i, o) = (img[[y, x, 0]], out[[y, x, 0]]);
                lo = lo.min(o);
                hi = hi.max(o);
                // Signed and per pixel: a checker cell that is a local
                // maximum must move up, a local minimum down. This is the
                // assertion the detail-sign flip cannot survive.
                if (x + y) % 2 == 0 {
                    least_lift = least_lift.min(o - i);
                } else {
                    least_drop = least_drop.min(i - o);
                }
            }
        }
        assert!(
            least_lift > 0.015,
            "every checker peak must be pushed up; the smallest lift was {least_lift}"
        );
        assert!(
            least_drop > 0.015,
            "and every trough pulled down; the smallest drop was {least_drop}"
        );
        // And the texture as a whole gains amplitude: HEAD reaches 1.80×,
        // where the three mutations that survive the sign test land at
        // 1.19× (F01), 0.20× (F14) and 1.00× (F25).
        assert!(
            hi - lo > 1.5 * input_p2p,
            "strength 1.0 must amplify the ±{CHECKER} texture past 1.5×: \
             {input_p2p} in, {} out",
            hi - lo
        );
    }

    // Edge preservation. A 2×2 block holds two checker peaks and two
    // troughs, so its mean cancels the texture and leaves the step alone:
    // every block mean of the input is 0.2 or 0.8. A guided filter keeps
    // them near that (HEAD strays 0.031); a box-mean unsharp mask rings
    // instead, and with a = 0 the halo reaches 0.237 — past both plateaus
    // and, at the dark side, out of [0, 1] altogether.
    let mut worst_halo = 0.0_f32;
    for y in (0..N).step_by(2) {
        for x in (0..N).step_by(2) {
            let mean =
                (out[[y, x, 0]] + out[[y, x + 1, 0]] + out[[y + 1, x, 0]] + out[[y + 1, x + 1, 0]])
                    / 4.0;
            worst_halo = worst_halo.max(mean - PLATEAU_HI).max(PLATEAU_LO - mean);
        }
    }
    assert!(
        worst_halo < 0.1,
        "the step edge must not halo: a 2×2 block mean overshoots a plateau by {worst_halo}"
    );

    // Strength is a plain scalar on the residual, so twice the strength is
    // twice the residual. The bound covers the f32 rounding of the final
    // add only; the worst deviation measured at HEAD is 3.0e-8.
    let doubled = local_contrast(img.view(), &params, 2.0).unwrap();
    let mut worst_nonlinearity = 0.0_f32;
    for ((&i, &o), &d) in img.iter().zip(out.iter()).zip(doubled.iter()) {
        worst_nonlinearity = worst_nonlinearity.max(((d - i) - 2.0 * (o - i)).abs());
    }
    assert!(
        worst_nonlinearity < 1e-6,
        "strength 2.0 must apply exactly twice the residual of strength 1.0; \
         worst deviation {worst_nonlinearity}"
    );
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

/// The sRGB transfer is C⁰ at the threshold but **not** C¹: the two
/// branches meet in value and part in slope (12.920 below, 12.703 above).
/// Both halves are asserted through the kernel — the previous version of
/// this test evaluated the two branch formulas and compared them to each
/// other, so it passed with `encode_srgb` deleted.
#[test]
fn srgb_is_c0_but_not_c1_at_the_threshold() {
    let t = 0.0031308_f32;
    let at = |x: f32| encode_srgb(array![[[x]]].view()).unwrap()[[0, 0, 0]];

    // C⁰: the kernel's value at the threshold satisfies *both* branch
    // formulas, which is what continuity there means.
    let linear = 12.92 * t;
    let power = 1.055 * t.powf(1.0 / 2.4) - 0.055;
    let v = at(t);
    assert!(
        (v - linear).abs() < 1e-6 && (v - power).abs() < 1e-6,
        "not continuous at the threshold: kernel={v:.9}, linear={linear:.9}, power={power:.9}"
    );

    // Not C¹: one-sided slopes, measured on the kernel, differ by ~1.8%.
    let h = 1e-5_f32;
    let slope_below = (at(t) - at(t - h)) / h;
    let slope_above = (at(t + h) - at(t)) / h;
    assert!(
        (slope_below - 12.92).abs() < 0.05,
        "slope below the threshold should be 12.92, measured {slope_below:.4}"
    );
    assert!(
        slope_above < slope_below * 0.995,
        "the transfer is not C¹ — the slope must drop across the threshold, \
         measured {slope_below:.4} below and {slope_above:.4} above"
    );
}

/// `channel_mixer_bw` documents negative weights as an infrared-like
/// inversion, and nothing pinned it: a regression clamping the output to
/// zero (as `hsl_bw` legitimately does) passed every CPU test.
#[test]
fn channel_mixer_negative_weights_subtract() {
    let img = rgb(0.2, 0.6, 0.4);
    let out = channel_mixer_bw(img.view(), [1.0, -1.0, 0.5]).unwrap()[[0, 0, 0]];
    let expect = 0.2 - 0.6 + 0.5 * 0.4;
    assert!(
        (out - expect).abs() < 1e-6,
        "weights must apply verbatim, including negatives: got {out}, want {expect}"
    );
    assert!(out < 0.0, "this combination is genuinely negative: {out}");
}

#[test]
fn channel_mixer_stays_linear_beyond_the_conventional_range() {
    // `channel_mixer_bw`'s doc offers -2..+2 as "conventional, enabling
    // infrared-like inversions", but the kernel is an unconstrained dot
    // product and nothing pinned what happens outside the values the
    // suite happens to use. Every weight triple anywhere in the crate --
    // Rust tests, ffi.py, ffi_gpu.py, conformance, benches, examples --
    // had |w| <= 1.2, so a clamp introduced at |2| would go unnoticed:
    // the outputs inside that band are identical either way.
    //
    // It would not be caught cross-backend either, unless the clamp
    // landed on only one side; the GPU computes the same dot product
    // from its own kernel, so a shared misconception stays invisible.
    let img = rgb(0.7, 0.4, 0.25);
    for w in [
        [3.0_f32, 0.0, 0.0],
        [3.0, -2.5, 0.4],
        [-2.0, 2.0, 0.0],
        [0.0, 0.0, -5.0],
        [10.0, -10.0, 10.0],
    ] {
        let got = channel_mixer_bw(img.view(), w).unwrap()[[0, 0, 0]];
        let want = w[0] * 0.7 + w[1] * 0.4 + w[2] * 0.25;
        assert!(
            (got - want).abs() < 1e-6,
            "weights {w:?} must apply verbatim: got {got}, want {want}"
        );
    }
}

// ── Highlight roll-off tests ─────────────────────────────────────────────────

use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};

/// The property the kernel exists for: highlight *detail* survives.
///
/// A hard clip maps every value above white onto the same code, so two
/// distinguishable highlights become one flat patch. The shoulder must
/// keep them apart all the way to the white point — that is the whole
/// difference between "clipped" and "rolled off", and a shape-only test
/// would pass on a kernel that just clipped.
#[test]
fn shoulder_preserves_highlight_separation_that_clipping_destroys() {
    // Six scene values spanning one stop either side of nominal white.
    let vals: Vec<f32> = (0..6).map(|i| 1.0 + i as f32 * 0.5).collect();
    let img = ndarray::Array3::from_shape_vec((1, 6, 1), vals.clone()).unwrap();

    // Hard clip (the default): everything collapses onto 1.0.
    let clipped = highlight_rolloff(img.view(), &RolloffParams::default()).unwrap();
    let distinct_clipped = {
        let mut v: Vec<u32> = clipped.iter().map(|x| x.to_bits()).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    assert_eq!(
        distinct_clipped, 1,
        "the default must clip: all six highlights should share one value"
    );

    // Shoulder to 4.0: every input stays distinguishable and ordered.
    let rolled = highlight_rolloff(img.view(), &RolloffParams::new(0.7, 4.0)).unwrap();
    let out: Vec<f32> = rolled.iter().copied().collect();
    for w in out.windows(2) {
        assert!(
            w[1] > w[0],
            "the shoulder must keep highlights separated: {out:?}"
        );
    }
    assert!(
        out.iter().all(|v| *v <= 1.0),
        "and still inside the displayable range: {out:?}"
    );
}

/// Raising the white point must recover *more* highlight range, not less
/// — the parameter has to mean what it says.
#[test]
fn a_higher_white_point_holds_more_highlight() {
    let probe = ndarray::array![[[3.0_f32]]];
    let near = highlight_rolloff(probe.view(), &RolloffParams::new(0.7, 2.0)).unwrap()[[0, 0, 0]];
    let far = highlight_rolloff(probe.view(), &RolloffParams::new(0.7, 8.0)).unwrap()[[0, 0, 0]];
    assert_eq!(near, 1.0, "white_point 2.0 puts 3.0 at pure white");
    assert!(
        far < 1.0,
        "white_point 8.0 must leave 3.0 short of white, got {far}"
    );
}

/// Determinism across thread counts, like every other kernel: the
/// per-element work is independent, so rayon's split must not matter.
#[test]
fn rolloff_is_thread_count_independent() {
    let img = ndarray::Array3::from_shape_fn((97, 131, 3), |(y, x, c)| {
        ((y * 131 + x + c) % 400) as f32 / 100.0
    });
    let params = RolloffParams::new(0.6, 5.0);

    // Calling twice in the ambient pool only asserts the kernel is a
    // pure function; the thread count never varied, so the name was a
    // claim the body did not make. Build the pools, as the grain tests
    // already do.
    let reference = highlight_rolloff(img.view(), &params).unwrap();
    for threads in [1, 2, 5, 16] {
        let out = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| highlight_rolloff(img.view(), &params).unwrap());
        assert_eq!(out, reference, "rolloff changed on {threads} threads");
    }
}

// ── Quantisation tests ───────────────────────────────────────────────────────

use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};

/// The property the kernel exists for: on a ramp too shallow to resolve
/// at the target depth, plain rounding produces flat plateaux — bands —
/// while dither disperses the step into a mixture that still carries the
/// gradient. A test that only checked dtype and shape would pass on a
/// kernel that ignored the dither setting entirely.
#[test]
fn dither_converts_banding_into_noise() {
    // 512 pixels spanning barely two 8-bit codes.
    let img =
        ndarray::Array3::from_shape_fn((1, 512, 1), |(_, x, _)| (60.0 + x as f32 / 511.0) / 255.0);

    let plain = quantize_u8(img.view(), &QuantizeParams::default()).unwrap();
    let dithered = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 2026)).unwrap();

    let transitions = |a: &ndarray::Array3<u8>| {
        a.iter()
            .zip(a.iter().skip(1))
            .filter(|(x, y)| x != y)
            .count()
    };

    assert!(
        transitions(&plain) <= 1,
        "undithered, the ramp should be one hard step: {} transitions",
        transitions(&plain)
    );
    assert!(
        transitions(&dithered) > 50,
        "dithered, the step should be dispersed: {} transitions",
        transitions(&dithered)
    );

    // And it must not have changed the picture's brightness while doing
    // it — triangular dither is zero-mean.
    let mean =
        |a: &ndarray::Array3<u8>| a.iter().map(|v| f64::from(*v)).sum::<f64>() / a.len() as f64;
    assert!(
        (mean(&dithered) - mean(&plain)).abs() < 0.6,
        "dither shifted the mean: {} vs {}",
        mean(&dithered),
        mean(&plain)
    );
}

/// Sixteen bits must resolve what eight cannot — otherwise the two
/// entry points are not doing different work.
#[test]
fn sixteen_bits_resolves_what_eight_cannot() {
    let img =
        ndarray::Array3::from_shape_fn((1, 64, 1), |(_, x, _)| (60.0 + x as f32 / 63.0) / 255.0);
    let p = QuantizeParams::default();

    let distinct = |v: Vec<u32>| {
        let mut v = v;
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let n8 = distinct(
        quantize_u8(img.view(), &p)
            .unwrap()
            .iter()
            .map(|c| u32::from(*c))
            .collect(),
    );
    let n16 = distinct(
        quantize_u16(img.view(), &p)
            .unwrap()
            .iter()
            .map(|c| u32::from(*c))
            .collect(),
    );
    assert!(n8 <= 2, "8-bit should collapse this ramp, got {n8} codes");
    assert!(n16 > 50, "16-bit should resolve it, got {n16} codes");
}

/// Determinism: the seed is the whole contract, so the same seed must
/// give the same bytes and a different one must not.
#[test]
fn quantize_dither_is_reproducible_from_the_seed() {
    let img = ndarray::Array3::from_shape_fn((37, 53, 3), |(y, x, c)| {
        ((y * 53 + x + c) % 255) as f32 / 255.0
    });
    let a = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 77)).unwrap();
    let b = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 77)).unwrap();
    let c = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 78)).unwrap();
    assert_eq!(a, b, "same seed, same bytes");
    assert_ne!(a, c, "different seed, different pattern");
}

// ── Histogram + LUT: the composition that justifies the pair ─────────────────

use phaios_core::histogram::{HistogramParams, histogram};
use phaios_core::lut::{LutParams, apply_lut};

/// The claim behind adding these two primitives instead of an
/// `equalise` kernel: histogram → CDF → apply_lut *is* histogram
/// equalisation, with no third component. If this does not hold, the
/// factoring was wrong.
#[test]
fn histogram_and_lut_compose_into_equalisation() {
    // A low-contrast image: everything crammed into [0.4, 0.6].
    let img = ndarray::Array3::<f32>::from_shape_fn((64, 64, 1), |(y, x, _)| {
        0.4 + ((y * 64 + x) % 256) as f32 / 255.0 * 0.2
    });

    let params = HistogramParams::new(256, 0.0, 1.0);
    let before = histogram(img.view(), &params).unwrap();

    // The equalising transfer, aligned to how apply_lut places entries.
    let table = before.equalisation_lut().row(0).to_owned();
    let equalised = apply_lut(img.view(), table.view(), &LutParams::default()).unwrap();

    let after = histogram(equalised.view(), &params).unwrap();

    // Equalisation must spread the distribution across the range.
    let occupied = |h: &phaios_core::histogram::Histogram| {
        (0..h.bins).filter(|&b| h.counts()[[0, b]] > 0).count()
    };
    let span = |h: &phaios_core::histogram::Histogram| {
        let lo = (0..h.bins).find(|&b| h.counts()[[0, b]] > 0).unwrap();
        let hi = (0..h.bins).rev().find(|&b| h.counts()[[0, b]] > 0).unwrap();
        hi - lo
    };

    assert!(
        span(&after) > span(&before) * 3,
        "equalisation should widen the occupied range: {} -> {}",
        span(&before),
        span(&after)
    );
    assert!(
        occupied(&after) >= occupied(&before),
        "and should not lose distinct levels: {} -> {}",
        occupied(&before),
        occupied(&after)
    );
    assert_eq!(after.total(0), 64 * 64, "no samples invented or lost");
}

/// An identity LUT built from a *flat* histogram must be a near-identity
/// transfer — the sanity check that the CDF is oriented correctly and
/// not, say, inverted.
#[test]
fn equalising_an_already_flat_image_changes_little() {
    let img = ndarray::Array3::<f32>::from_shape_fn((256, 256, 1), |(y, x, _)| {
        ((y * 256 + x) % 256) as f32 / 255.0
    });
    let params = HistogramParams::new(256, 0.0, 1.0);
    let cdf = histogram(img.view(), &params).unwrap().cdf();
    let out = apply_lut(
        img.view(),
        cdf.row(0).to_owned().view(),
        &LutParams::default(),
    )
    .unwrap();

    let max_shift = img
        .iter()
        .zip(out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_shift < 0.02,
        "an already-uniform image should barely move, shifted {max_shift}"
    );
}

// ── Shadow roll-off and the characteristic curve ─────────────────────────────

use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};

/// The property the toe exists for: shadow *separation* shrinks while
/// the endpoints stay pinned. Checking only the endpoints would pass on
/// a kernel that did nothing between them.
#[test]
fn the_toe_compresses_shadow_separation() {
    // Eight evenly spaced samples inside the toe region.
    let vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.02).collect();
    let img = ndarray::Array3::from_shape_vec((1, 8, 1), vals).unwrap();

    // Measured next to black, which is where the toe acts. Across the
    // whole toe region the effect is diluted by the part near the knee,
    // where the curve has already returned to the identity — the span
    // from 0 to 0.14 barely moves, so measuring that would understate
    // the kernel and pass on a far weaker one.
    let near_black = |strength: f32| {
        let out = shadow_rolloff(img.view(), &ShadowRolloffParams::new(0.2, strength)).unwrap();
        out[[0, 1, 0]] - out[[0, 0, 0]]
    };

    let none = near_black(0.0);
    let some = near_black(0.5);
    let full = near_black(1.0);
    assert!(
        (none - 0.02).abs() < 1e-6,
        "strength 0 must leave the separation alone, got {none}"
    );
    assert!(
        some < none * 0.65,
        "half strength should compress clearly: {some} vs {none}"
    );
    assert!(
        full < none * 0.25,
        "full strength should compress about fivefold: {full} vs {none}"
    );
    assert!(
        full < some,
        "and more than half strength does: {full} vs {some}"
    );

    // And the order is never disturbed — a fold here would be a bug.
    let out = shadow_rolloff(img.view(), &ShadowRolloffParams::new(0.2, 1.0)).unwrap();
    for i in 1..8 {
        assert!(
            out[[0, i, 0]] > out[[0, i - 1, 0]],
            "ordering broken at {i}"
        );
    }
}

/// The toe darkens as it compresses. A version that lifted the shadows
/// instead would be a black-lift control, a different thing entirely,
/// and would still pass a separation-only test.
#[test]
fn the_toe_darkens_rather_than_lifting() {
    let img = ndarray::Array3::from_shape_fn((1, 64, 1), |(_, x, _)| x as f32 / 320.0);
    let out = shadow_rolloff(img.view(), &ShadowRolloffParams::new(0.2, 0.7)).unwrap();
    let lifted = img.iter().zip(out.iter()).filter(|(i, o)| o > i).count();
    assert_eq!(
        lifted, 0,
        "{lifted} samples were lifted instead of deepened"
    );
}

/// The composition this kernel was built to complete: toe, straight
/// section, shoulder. The signature of a characteristic curve is that
/// the slope is low at both ends and holds in the middle, so that is
/// what gets asserted — not merely that three calls run.
#[test]
fn toe_curve_and_shoulder_compose_a_characteristic_curve() {
    use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};
    use phaios_core::tone::{ToneCurveParams, tone_curve};

    let n = 4000_usize;
    let top = 4.0_f32;
    let img = ndarray::Array3::from_shape_fn((1, n, 1), |(_, x, _)| x as f32 / n as f32 * top);

    let out = highlight_rolloff(
        tone_curve(
            shadow_rolloff(img.view(), &ShadowRolloffParams::new(0.18, 0.8))
                .unwrap()
                .view(),
            &ToneCurveParams::new(1.2, 0.0, 1.0),
        )
        .unwrap()
        .view(),
        &RolloffParams::new(0.7, 3.0),
    )
    .unwrap();

    // Monotone, and inside the displayable range at the top.
    let mut prev = f32::NEG_INFINITY;
    for i in 0..n {
        let v = out[[0, i, 0]];
        assert!(v >= prev, "the composed curve dips at sample {i}");
        assert!(v <= 1.0 + 1e-6, "and must not exceed white: {v}");
        prev = v;
    }

    // Local slope at three places: deep shadow, midtone, near white.
    let slope_at = |x: f32| {
        let i = ((x / top) * n as f32) as usize;
        let (a, b) = (out[[0, i, 0]], out[[0, i + 1, 0]]);
        (b - a) / (top / n as f32)
    };
    let shadow = slope_at(0.01);
    let mid = slope_at(0.4);
    let highlight = slope_at(2.0);

    assert!(
        shadow < mid * 0.6,
        "the toe must hold less contrast than the midtones: {shadow} vs {mid}"
    );
    assert!(
        highlight < mid * 0.6,
        "and so must the shoulder: {highlight} vs {mid}"
    );
    assert!(
        mid > 1.0,
        "the straight section should carry the contrast the curve was given: {mid}"
    );
}

// ── Sharpen tests ──────────────────────────────────────────────────────────────

use phaios_core::blur::{BlurParams, BlurShape, blur};
use phaios_core::sharpen::{SharpenParams, sharpen};

/// A smooth, low-frequency field is close to its own blur, so a correct
/// unsharp mask changes it only a little — but deliberately not a
/// *constant* field: `blur` maps any constant to itself exactly, so a
/// constant input's `detail` is zero regardless of how `detail` is
/// computed, and CONTRIBUTING.md warns exactly about tests that pass by
/// luck for that reason. Here `detail` is measurably nonzero (the blur
/// genuinely moves the field), so a kernel that gated the blurred image
/// itself instead of the `img - blurred` residual diverges sharply from
/// one that doesn't. This pins `out == img + amount*(img - blurred)`
/// against a `blurred` computed independently, by calling `blur`
/// directly rather than trusting `sharpen`'s internals.
#[test]
fn a_low_frequency_field_matches_the_formula_computed_independently() {
    let (h, w) = (48, 64);
    let img = ndarray::Array3::from_shape_fn((h, w, 1), |(y, x, _)| {
        0.5 + 0.1 * ((x as f32 / w as f32) * std::f32::consts::TAU).sin()
            + 0.05 * ((y as f32 / h as f32) * std::f32::consts::TAU).cos()
    });
    let sigma = 3.0_f32;
    let amount = 0.6_f32;

    let blurred = blur(img.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap();
    let out = sharpen(img.view(), &SharpenParams::new(amount, sigma, 0.0)).unwrap();

    // Sanity: if the blur barely moved the field, this test could not
    // tell a correct `detail` from an incorrect one either.
    let max_detail = img
        .iter()
        .zip(blurred.iter())
        .map(|(v, b)| (v - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_detail > 1e-4,
        "field is too flat to distinguish detail from no detail: {max_detail}"
    );

    for ((v, b), o) in img.iter().zip(blurred.iter()).zip(out.iter()) {
        let want = v + amount * (v - b);
        assert!(
            (want - o).abs() < 1e-4,
            "expected {want}, got {o} (v={v}, blurred={b})"
        );
    }
}

/// Classic unsharp-mask behaviour: at a step, the local slope sampled a
/// few pixels either side of the jump — well inside the blur's radius,
/// so both samples feel the crossing — must come out *steeper* than the
/// same window on the untouched step, from the overshoot/undershoot the
/// gated detail adds back.
///
/// Mutation: flip the combining sign (`out = img - amount*T*detail`) and
/// the overshoot/undershoot inverts, pulling each side *toward* the
/// other instead of away from it — the sampled slope must come out
/// shallower than the original step's, not merely different.
#[test]
fn a_step_edge_gets_steeper_not_shallower() {
    let (h, w) = (5, 81);
    let (lo, hi) = (0.2_f32, 0.8_f32);
    let mid = w / 2;
    let img = ndarray::Array3::from_shape_fn((h, w, 1), |(_, x, _)| if x < mid { lo } else { hi });

    let sigma = 4.0_f32;
    let k = 2_usize; // sample offset, well inside the blur's radius

    let out = sharpen(img.view(), &SharpenParams::new(1.0, sigma, 0.0)).unwrap();

    let row = h / 2;
    let img_slope = img[[row, mid + k, 0]] - img[[row, mid - k, 0]];
    let out_slope = out[[row, mid + k, 0]] - out[[row, mid - k, 0]];

    assert!(
        (img_slope - (hi - lo)).abs() < 1e-6,
        "sanity: the untouched step must still be the plain jump here: {img_slope}"
    );
    assert!(
        out_slope > img_slope,
        "sharpened step must be steeper across the same window: {out_slope} vs {img_slope}"
    );
}

/// `amount = 0.0` and `sigma = 0.0` are each an exact identity — no
/// detail computation is needed to know a zero-strength or zero-radius
/// sharpen does nothing, so the fast path is pinned bit-for-bit.
#[test]
fn amount_zero_and_sigma_zero_are_bit_exact_identities() {
    let img = ndarray::Array3::from_shape_fn((11, 13, 2), |(y, x, c)| {
        ((y * 13 + x * 3 + c) % 17) as f32 / 16.0
    });
    for params in [
        SharpenParams::new(0.0, 2.0, 0.0),
        SharpenParams::new(0.0, 2.0, 0.3),
        SharpenParams::new(0.7, 0.0, 0.0),
        SharpenParams::new(0.7, 0.0, 0.3),
    ] {
        let out = sharpen(img.view(), &params).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "params={params:?}");
        }
    }
}

/// The Hermite band's *algebraic* zero: once `threshold` is at least as
/// large as the biggest `|detail|` anywhere in the frame, every pixel's
/// gate is exactly `0.0`, so sharpening is the identity regardless of
/// `amount`. This is the test the maintainer's mutation check names
/// directly: forcing `soft_gate` to a constant `1.0` must fail *this*
/// assertion while leaving `a_low_frequency_field_matches_the_formula_
/// computed_independently` (above) untouched — that test uses
/// `threshold = 0`, where `T ≡ 1` is already the correct value, so a
/// gate that is *always* `1` is indistinguishable from a correct one
/// there.
#[test]
fn threshold_covering_all_detail_is_the_identity() {
    let img = ndarray::Array3::from_shape_fn((17, 19, 1), |(y, x, _)| {
        0.4 + 0.3 * (x as f32 * 0.7 + y as f32 * 1.3).sin()
    });
    let sigma = 2.5_f32;

    let blurred_ref = blur(img.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap();
    let max_detail = img
        .iter()
        .zip(blurred_ref.iter())
        .map(|(v, b)| (v - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_detail > 1e-3, "test image is too smooth: {max_detail}");

    let threshold = max_detail * 1.001; // strictly above every |detail|
    let out = sharpen(img.view(), &SharpenParams::new(0.9, sigma, threshold)).unwrap();
    for (a, b) in img.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-6, "expected identity: {a} became {b}");
    }
}

/// Linear in `amount` for a *fixed* image: `T` and `detail` do not
/// depend on `amount`, only their product is scaled by it, so doubling
/// `amount` must exactly double the change from the input.
///
/// Mutation: compute the gate from `amount * detail` instead of `detail`
/// alone — doubling `amount` then also shifts pixels across the
/// soft-knee band, breaking the doubling relationship for whichever
/// pixels sit near it. `threshold` is set inside the image's achievable
/// `|detail|` range so some pixels do sit near the band; at
/// `threshold = 0` this mutation would be invisible (`T` is forced to 1
/// either way).
#[test]
fn sharpen_is_linear_in_amount() {
    let img = ndarray::Array3::from_shape_fn((23, 29, 1), |(y, x, _)| {
        0.5 + 0.4 * (x as f32 * 0.31 + y as f32 * 0.53).sin()
    });
    let sigma = 2.0_f32;
    let threshold = 0.02_f32;

    let out1 = sharpen(img.view(), &SharpenParams::new(0.25, sigma, threshold)).unwrap();
    let out2 = sharpen(img.view(), &SharpenParams::new(0.5, sigma, threshold)).unwrap();

    for ((v, o1), o2) in img.iter().zip(out1.iter()).zip(out2.iter()) {
        let (d1, d2) = (o1 - v, o2 - v);
        assert!(
            (d2 - 2.0 * d1).abs() < 1e-4,
            "not linear in amount: d(0.25)={d1}, d(0.5)={d2}"
        );
    }
}

/// Amplification magnitude is non-decreasing in `|detail|`, on *both*
/// signs — a bright spike and a dark dip of matching strength must
/// amplify equally. Because `blur` is linear and the background is
/// constant, `detail` at the centre of a fixed-shape spike scales
/// exactly with the spike's amplitude, so sweeping the amplitude sweeps
/// `|detail|` directly and predictably.
///
/// Mutation: drop `.abs()` before computing `u` — the ramp then reads
/// signed `detail`, so on the negative (dark-dip) side `d - threshold`
/// is always negative and `u` never leaves `0`, no matter how strong the
/// dip gets. The positive (bright-spike) side is unaffected, so only the
/// dark half of this test catches it.
#[test]
fn amplification_is_monotone_in_detail_magnitude_both_signs() {
    let n = 25;
    let base = 0.5_f32;
    let sigma = 3.0_f32;
    let threshold = 0.05_f32;

    let spot = |amplitude: f32| {
        ndarray::Array3::from_shape_fn((n, n, 1), |(y, x, _)| {
            let (dy, dx) = (y.abs_diff(n / 2), x.abs_diff(n / 2));
            if dy <= 1 && dx <= 1 {
                base + amplitude
            } else {
                base
            }
        })
    };

    let amplitudes = [0.02_f32, 0.06, 0.15, 0.35, 0.8];
    let mut bright_delta = Vec::new();
    let mut dark_delta = Vec::new();
    for &a in &amplitudes {
        let (bright, dark) = (spot(a), spot(-a));
        let params = SharpenParams::new(1.0, sigma, threshold);
        let out_b = sharpen(bright.view(), &params).unwrap();
        let out_d = sharpen(dark.view(), &params).unwrap();
        bright_delta.push((out_b[[n / 2, n / 2, 0]] - bright[[n / 2, n / 2, 0]]).abs());
        dark_delta.push((out_d[[n / 2, n / 2, 0]] - dark[[n / 2, n / 2, 0]]).abs());
    }

    for (name, deltas) in [("bright", &bright_delta), ("dark", &dark_delta)] {
        for w in deltas.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "{name} side not monotone: {deltas:?}");
        }
        assert!(
            *deltas.last().unwrap() > *deltas.first().unwrap() + 1e-3,
            "{name} side shows no real amplification growth: {deltas:?}"
        );
    }
}

// ── Hot-pixel tests ──────────────────────────────────────────────────────────

use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

/// Builds an `(h, w, 1)` image that is a smooth, strictly monotonic ramp
/// along `x` and constant along `y` — deliberately not a *flat* field, so
/// CONTRIBUTING.md's warning about constant-image tests does not apply:
/// a kernel that ignored its neighbourhood entirely could not pass these
/// tests by accident.
///
/// Because every row is identical, the 3x3 window around `(row, col)`
/// (any interior `row`) holds, before corruption, three copies each of
/// `ramp(col-1)`, `ramp(col)` and `ramp(col+1)`. Replacing the one centre
/// copy of `ramp(col)` with an outlier that ranks outside
/// `[ramp(col-1), ramp(col+1)]` leaves two surviving copies of
/// `ramp(col)` — still enough to occupy the sorted array's middle
/// (5th-of-9) slot regardless of which side the outlier ranks on, so the
/// window median is *exactly* `ramp(col)`, computable independently of
/// `hot_pixels` itself with no floating-point tolerance needed.
fn x_ramp(h: usize, w: usize) -> ndarray::Array3<f32> {
    ndarray::Array3::from_shape_fn((h, w, 1), |(_, x, _)| 0.1 + 0.02 * x as f32)
}

/// A finer-grained ramp than [`x_ramp`] (0.001/column instead of
/// 0.02/column, spanning roughly 0..0.1 over 101 columns), used where a
/// test needs both a "dark" and a "highlight" window on the same scale —
/// [`x_ramp`]'s coarser step would make a highlight-region deviation
/// hard to distinguish from the ramp's own local structure.
fn x_ramp_fine(h: usize, w: usize) -> ndarray::Array3<f32> {
    ndarray::Array3::from_shape_fn((h, w, 1), |(_, x, _)| 0.001 * x as f32)
}

/// A bright (bright/very-large) planted outlier, replaced bit-exactly by
/// the window median. Threshold and relative are both `0.0`
/// (unconditional replacement), so this isolates `median9` itself.
///
/// This is one of the two assertions the mutation gate's `median9 ->
/// window minimum` check targets: a minimum-based "median" is pulled
/// toward whichever of `ramp(col-1)`/`ramp(col+1)` is smaller rather than
/// staying at `ramp(col)`, or (if the outlier itself becomes the
/// reported minimum) leaves the outlier in place — either way this
/// assertion is expected to fail under that mutation. A *separate*
/// dark-outlier assertion (below) exists so a regression that breaks one
/// direction but not the other is still caught, and named individually.
#[test]
fn a_bright_outlier_on_a_ramp_is_replaced_by_the_window_median() {
    let (h, w) = (9, 15);
    let (row, col) = (4, 7);
    let mut img = x_ramp(h, w);
    let true_value = img[[row, col, 0]];
    assert_ne!(
        img[[row, col - 1, 0]].to_bits(),
        img[[row, col + 1, 0]].to_bits(),
        "sanity: the ramp must actually vary across this window"
    );
    img[[row, col, 0]] = 50.0; // far brighter than anything on the ramp

    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, 0.0)).unwrap();

    assert_eq!(
        out[[row, col, 0]].to_bits(),
        true_value.to_bits(),
        "bright outlier must be replaced by the window median (the ramp's \
         own value here): expected {true_value}, got {}",
        out[[row, col, 0]]
    );
}

/// The dark-outlier twin of the test above — see its documentation for
/// the shared construction. Named separately so a `median9 -> window
/// minimum`-style mutation, which plausibly affects the bright and dark
/// directions differently, is caught (and reported) on each side
/// independently rather than by one assertion that could pass for the
/// wrong reason.
#[test]
fn a_dark_outlier_on_a_ramp_is_replaced_by_the_window_median() {
    let (h, w) = (9, 15);
    let (row, col) = (4, 7);
    let mut img = x_ramp(h, w);
    let true_value = img[[row, col, 0]];
    img[[row, col, 0]] = -50.0; // far darker than anything on the ramp

    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, 0.0)).unwrap();

    assert_eq!(
        out[[row, col, 0]].to_bits(),
        true_value.to_bits(),
        "dark outlier must be replaced by the window median (the ramp's \
         own value here): expected {true_value}, got {}",
        out[[row, col, 0]]
    );
}

/// A plain two-level step is an exact identity under an unconditional
/// median (`threshold = relative = 0.0`), at *every* pixel including the
/// border rows/columns the index clamp produces: a step's own value is
/// always at least 5-of-9 in its clamped window, even at the transition
/// column (interior: a 3-6 or 6-3 split; at a border, the duplicated
/// edge value only reinforces whichever side already has the majority),
/// so the median always agrees with the pixel already there.
///
/// `assert_eq!` on bits, not a tolerance: this is provable exactly by
/// hand, not merely expected to be close. Mutation: swapping the median
/// for a mean would visibly soften the transition, since a mean of a
/// 3-6/6-3 split is not equal to either endpoint.
#[test]
fn a_step_edge_is_an_exact_identity_at_threshold_zero_including_the_borders() {
    let (h, w) = (7, 11);
    let (lo, hi) = (0.1_f32, 0.9_f32);
    let mid = w / 2;
    let img = ndarray::Array3::from_shape_fn((h, w, 1), |(_, x, _)| if x < mid { lo } else { hi });

    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, 0.0)).unwrap();

    for ((y, x, c), &v) in img.indexed_iter() {
        assert_eq!(
            out[[y, x, c]].to_bits(),
            v.to_bits(),
            "step edge must be an exact identity at threshold=0, including \
             the border rows/columns the index clamp produces: pixel \
             ({y},{x}) expected {v}, got {}",
            out[[y, x, c]]
        );
    }
}

/// A `threshold` set strictly above the actual `|p - m|` deviation keeps
/// the pixel unchanged — the mirror image of the two replacement tests
/// above, pinning the other side of the `>` comparison (an off-by-one
/// direction or an accidentally non-strict `>=` would flip this).
#[test]
fn threshold_strictly_above_the_planted_deviation_is_the_identity() {
    let (h, w) = (9, 15);
    let (row, col) = (4, 7);
    let mut img = x_ramp(h, w);
    let true_value = img[[row, col, 0]];
    let deviation = 0.1_f32; // far larger than the ramp's own 0.02 step
    img[[row, col, 0]] = true_value + deviation;

    let threshold = deviation + 0.05; // strictly above the planted deviation
    let out = hot_pixels(img.view(), &HotPixelParams::new(threshold, 0.0)).unwrap();

    for ((y, x, c), &v) in img.indexed_iter() {
        assert_eq!(
            out[[y, x, c]].to_bits(),
            v.to_bits(),
            "threshold above the deviation must be the identity: pixel \
             ({y},{x}) expected {v}, got {}",
            out[[y, x, c]]
        );
    }
}

/// With `threshold = 0.0`, the absolute-only form (`relative = 0.0`)
/// replaces *any* nonzero deviation — including this one, planted on a
/// bright (highlight) region of the ramp. Adding a large enough
/// `relative` term must keep it instead: `limit` grows with the local
/// median's own magnitude, and here it grows past the deviation.
///
/// Paired with the dark-median test below using the *same* deviation and
/// the *same* `relative`, so together they isolate the relative term's
/// effect (widens with brightness) from a kernel that is simply less
/// aggressive everywhere. Mutation: `limit = threshold` (ignoring
/// `relative` entirely) makes this assertion fail while the absolute
/// tests above stay green, since `relative` is the only field this test
/// varies.
#[test]
fn the_relative_term_keeps_a_highlight_deviation_the_absolute_form_would_replace() {
    let (h, w) = (9, 101);
    let (row, col) = (4, 95);
    let mut img = x_ramp_fine(h, w);
    let true_value = img[[row, col, 0]];
    let deviation = 0.05_f32;
    img[[row, col, 0]] = true_value + deviation;

    // Sanity: the absolute-only form (relative = 0) does replace this
    // deviation, which is the behaviour the relative term is meant to
    // change for a bright/highlight median.
    let absolute_only = hot_pixels(img.view(), &HotPixelParams::new(0.0, 0.0)).unwrap();
    assert_eq!(
        absolute_only[[row, col, 0]].to_bits(),
        true_value.to_bits(),
        "sanity: the absolute-only form must replace this deviation"
    );

    let relative = 1.0_f32; // limit = relative * |m| = 1.0 * ~0.095 > 0.05
    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, relative)).unwrap();
    let deviated = img[[row, col, 0]];
    assert_eq!(
        out[[row, col, 0]].to_bits(),
        deviated.to_bits(),
        "a highlight deviation within relative*|m| of the median must be \
         kept: expected {deviated} (unchanged), got {}",
        out[[row, col, 0]]
    );
}

/// The dark-median twin of the test above: the *same* absolute
/// deviation and the *same* `relative`, but planted on a low (dark)
/// region of the ramp, where `relative * |m|` stays small. Still
/// replaced — proving the previous test's identity was specifically
/// `relative` tracking brightness, not `relative` disabling the kernel
/// outright.
#[test]
fn the_relative_term_still_replaces_the_same_deviation_on_a_dark_median() {
    let (h, w) = (9, 101);
    let (row, col) = (4, 5);
    let mut img = x_ramp_fine(h, w);
    let true_value = img[[row, col, 0]];
    let deviation = 0.05_f32;
    img[[row, col, 0]] = true_value + deviation;

    let relative = 1.0_f32; // limit = relative * |m| = 1.0 * ~0.005 < 0.05
    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, relative)).unwrap();
    assert_eq!(
        out[[row, col, 0]].to_bits(),
        true_value.to_bits(),
        "the same deviation on a dark median must still be replaced: \
         expected {true_value}, got {}",
        out[[row, col, 0]]
    );
}

/// An outlier confined to one channel of an RGB-shaped ramp must be
/// corrected in that channel only — the other two, at every pixel, stay
/// bit-exact to the input. Channels carry deliberately different
/// ramps (not merely different constants) so a kernel that accidentally
/// read another channel's window would move a *distinguishable* amount,
/// not coincidentally match.
///
/// Mutation this catches: any window-gathering bug that reads across the
/// channel axis (e.g. a fixed channel index, or the wrong stride).
#[test]
fn hot_pixels_filters_channels_independently() {
    let (h, w) = (9, 15);
    let (row, col) = (4, 7);
    let bases = [0.1_f32, 0.3, 0.6];
    let slopes = [0.02_f32, 0.01, 0.005];
    let mut img =
        ndarray::Array3::from_shape_fn((h, w, 3), |(_, x, c)| bases[c] + slopes[c] * x as f32);
    let true_value = img[[row, col, 0]];
    img[[row, col, 0]] = 50.0; // outlier in channel 0 only

    let out = hot_pixels(img.view(), &HotPixelParams::new(0.0, 0.0)).unwrap();

    assert_eq!(
        out[[row, col, 0]].to_bits(),
        true_value.to_bits(),
        "the corrupted channel must still be corrected"
    );
    for ((y, x, c), &v) in img.indexed_iter() {
        if c == 0 {
            continue; // channel 0 is allowed to change; checked above
        }
        assert_eq!(
            out[[y, x, c]].to_bits(),
            v.to_bits(),
            "channel {c} must be untouched by channel 0's outlier: pixel \
             ({y},{x}) expected {v}, got {}",
            out[[y, x, c]]
        );
    }
}
