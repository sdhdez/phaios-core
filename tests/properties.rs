// SPDX-License-Identifier: GPL-3.0-or-later
//! Property-based tests over every public kernel's validation contract.
//!
//! `tests/kernels.rs` and the `#[cfg(test)]` modules beside each kernel
//! pin specific, hand-picked cases — a middle-grey pixel, a 45° angle, a
//! zone index of 11. What they cannot do, by construction, is cover the
//! *whole* domain of a `validate*` function: every finite `f32`, every
//! shape, every combination. Those validators are `pub(crate)` and shared
//! verbatim by the CPU and CUDA backends (CLAUDE.md §2, `docs/ffi.md`
//! throughout), so a gap here is a gap in both backends at once, and the
//! cross-backend conformance suite (`tests/cuda_conformance.rs`) cannot
//! see it: it compares two backends' answers to the same input, never
//! asks whether an input should have been rejected in the first place.
//!
//! This file exercises the validators through the public kernels (the
//! only way to reach a `pub(crate)` function from an integration test)
//! with `proptest`-generated inputs spanning each documented domain,
//! plus its complement. Six properties, applied per kernel where they
//! make sense and explicitly skipped with a reason where a kernel's
//! contract does not include them:
//!
//! - **P1** — valid input gives `Ok`, the documented output shape, and
//!   every output value finite.
//! - **P2** — invalid params or shape give `Err(Parameter | Shape)`,
//!   never a panic, and the message names the offending field.
//! - **P3** — validation depends only on shape and params, never on
//!   pixel values: two images of the same shape under the same params
//!   either both succeed or both fail with the identical message.
//! - **P4** — a caller-controlled output shape that would exceed the
//!   8 GiB single-allocation limit (`src/alloc.rs`) is refused *before*
//!   attempting the allocation, in well under the time a real allocation
//!   or a pixel pass would take. Tested on `resize`, the kernel whose
//!   output size is most directly caller-controlled.
//! - **P5** — layout-agnostic: the same logical array gives bit-identical
//!   output whatever its physical memory layout (CLAUDE.md §2,
//!   `docs/ffi.md` §2).
//! - **P6** — deterministic: the same input, params and (where relevant)
//!   seed give bit-identical output on repeated calls; for `film_grain`
//!   and dithered `quantize_*`, two different seeds give different
//!   output on a non-trivial image.
//!
//! # Reproducing a failure
//!
//! The RNG seed is fixed ([`SEED`]), so a failure here is byte-for-byte
//! reproducible: re-running `cargo test --test properties` regenerates
//! and shrinks the exact same sequence of cases. `failure_persistence` is
//! set to [`FileFailurePersistence::Off`] everywhere in this file, so no
//! `proptest-regressions/` directory is ever written — the fixed seed is
//! the only reproducibility mechanism, and it is enough, since nothing
//! here reads ambient files or other outside state to *generate* a case
//! (P4 alone reads the clock, and only to make an assertion about one).
//!
//! Images are not proptest-generated pixel by pixel. Each is a single
//! `u64` seed, drawn like any other proptest value (so it shrinks and
//! replays exactly like one), filled by [`image`] with an ordinary Rust
//! loop over the crate's own `splitmix64` hash — see that function's doc
//! comment for the full reasoning; in short, building one proptest
//! strategy-tree node per pixel, across thousands of generated cases, was
//! far more expensive than either generating or blurring the pixels ever
//! was.
//!
//! The default case count is 256 per property (proptest's own default),
//! reached everywhere in this file via [`cases`] rather than left to
//! `ProptestConfig::default()`, specifically so `PROPTEST_CASES` still
//! overrides it. One property is the deliberate exception:
//! `blur_valid_sigma_is_accepted` runs at a lower, separately-configured
//! count (see its own doc comment) because its cost is dominated by
//! `box_widths`'s search, not by anything this file generates.
//!
//! ```sh
//! PROPTEST_CASES=1000 cargo test --test properties
//! ```
//!
//! # A note on "valid" ranges
//!
//! Several validators check only `is_finite()` with no magnitude bound
//! (`zone_system`'s offsets, `tone_curve`'s slope/offset/power,
//! `channel_mixer_bw`'s weights, which are not validated at all). Taken
//! literally, "valid" therefore includes values like `1e30` — and for
//! several of these kernels a sufficiently extreme *finite* parameter
//! overflows the arithmetic to `±∞`, which would make this file's own
//! finiteness assertion (P1) fail on what is, per the validator, legal
//! input. That is a fact about floating-point magnitude, not a
//! validation defect, and it is not what this file is for. Where this
//! applies, the "valid" strategy samples a realistic-but-generous
//! sub-range of the validated domain rather than its full extent, and
//! says so at the point it does it.

use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use ndarray::{Array1, Array3, s};
use proptest::prelude::*;
use proptest::test_runner::{FileFailurePersistence, RngAlgorithm, RngSeed};

use phaios_core::blur::{self as blur_mod, BlurParams, BlurShape, blur};
use phaios_core::bw::{
    ColorFilter, HslWeightedParams, LuminanceStandard, channel_mixer_bw, color_filter_bw, hsl_bw,
    luminance_bw,
};
use phaios_core::encode::encode_srgb;
use phaios_core::error::PhaiosError;
use phaios_core::exposure::exposure;
use phaios_core::film_grain::{GrainParams, film_grain};
use phaios_core::geometry::{
    CropParams, Orientation, ResizeFilter, ResizeParams, StraightenParams, crop, orient, resize,
    straighten,
};
use phaios_core::glow::{GlowParams, glow};
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};
use phaios_core::histogram::{self as histogram_mod, HistogramParams, histogram};
use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};
use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};
use phaios_core::lut::{LutParams, apply_lut};
use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};
use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};
use phaios_core::sharpen::{SharpenParams, sharpen};
use phaios_core::split_toning::{SplitToningParams, split_toning};
use phaios_core::tone::{ToneCurveParams, ZoneParams, tone_curve, zone_system};
use phaios_core::vignette::{VignetteParams, vignette};

// ── Fixed seed and shared configuration ─────────────────────────────────────

/// Fixed across every property in this file. Dated rather than arbitrary,
/// so its provenance is obvious from the digits alone.
const SEED: u64 = 0x2026_0904_5EED_0001;

/// Cases per property. `PROPTEST_CASES` overrides this — read explicitly
/// here (rather than left to `ProptestConfig::default()`, whose own
/// built-in default is 256) so the crate's own moderate default applies
/// when the variable is unset.
fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// The `ProptestConfig` shared by every `proptest! { #![proptest_config(config())] ... }`
/// block in this file: fixed seed, fixed algorithm, persistence off.
fn config() -> ProptestConfig {
    ProptestConfig {
        cases: cases(),
        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
        rng_algorithm: RngAlgorithm::ChaCha,
        rng_seed: RngSeed::Fixed(SEED),
        ..ProptestConfig::default()
    }
}

/// The crate's single-allocation limit (`src/alloc.rs::MAX_ALLOCATION_BYTES`,
/// `docs/ffi.md` "Allocation is bounded"). Not reachable from this
/// integration-test crate (`alloc` is a private module), so restated here
/// from the constant's own documented value: 8 GiB.
const MAX_ALLOCATION_BYTES: u64 = 8 << 30;

// ── Shared strategies ────────────────────────────────────────────────────────

/// Image extent: H or W, within `1..=48` as specified, but weighted
/// heavily toward the small end.
///
/// An earlier version of this file generated pixels individually —
/// `prop::collection::vec(scene_value(), h * w * c)` — and *that* strategy
/// tree, not kernel execution, was the dominant cost (a single "valid
/// input" property at 128 cases took ~3.5 s regardless of the kernel or
/// the rayon thread count). [`image`] replaced it with a single `u64`
/// seed hashed per pixel, which is why pixel generation itself is no
/// longer the bottleneck — see its doc comment.
///
/// That leaves ordinary kernel cost, which still scales with `h * w * c`,
/// and matters for two more mundane reasons at the case counts this file
/// runs (256 per property, `cases`): raw wall time across ~70 properties,
/// and CPU contention — `cargo test` runs properties in parallel, and
/// `resize_oversized_target_is_refused_before_allocating`'s `< 100 ms`
/// wall-clock assertion (P4) was measured to flake under that contention
/// with images sampled uniformly up to 48×48. Biasing this distribution
/// toward small images while still reaching all the way to 48 keeps the
/// documented range reachable — including, occasionally, its extremes,
/// where `blur`'s and `local_contrast`'s neighbourhood operations and
/// `straighten`'s resampling get genuinely exercised — without either
/// cost on every one of the thousands of cases this file generates.
fn dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => 1usize..=6usize,
        2 => 7usize..=20usize,
        1 => 21usize..=48usize,
    ]
}

/// A channel count landing on 1 or 3 most of the time (what the "any
/// channel count" kernels see in the real pipeline) with an occasional
/// wider value, for the kernels that truly accept any `C`.
fn any_channels() -> impl Strategy<Value = usize> {
    prop_oneof![
        6 => Just(1usize),
        6 => Just(3usize),
        1 => 1usize..=8usize,
    ]
}

/// A channel count guaranteed to differ from `correct`, for exercising
/// shape rejection.
fn wrong_channels(correct: usize) -> impl Strategy<Value = usize> {
    (1usize..=8usize).prop_filter("must differ from the expected channel count", move |c| {
        *c != correct
    })
}

/// NaN, +∞ and −∞ — the three non-finite `f32` values, for building
/// invalid-parameter strategies.
fn non_finite_f32() -> impl Strategy<Value = f32> {
    prop_oneof![Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)]
}

/// Build an `(h, w, c)` image from a flat value vector. `expect` here
/// documents an invariant of this file's own strategies (the vector is
/// always sized `h * w * c` by construction), not caller input. Used only
/// where a test needs a *narrower* pixel range than [`image`]'s (the
/// seed-difference tests, which want a non-trivial midtone image rather
/// than the full scene-referred spread).
fn mk_image(h: usize, w: usize, c: usize, vals: Vec<f32>) -> Array3<f32> {
    Array3::from_shape_vec((h, w, c), vals)
        .expect("this file's strategies always build a vec of exactly h*w*c values")
}

/// One scene-referred pixel value, deterministically, from 64 hashed
/// bits: the same distribution the original `scene_value()` proptest
/// strategy sampled — the landmark values named in the task (0, a value
/// near zero, 18% grey, unity, a bright highlight, and their negatives —
/// white balance can push a channel slightly negative, and nothing here
/// validates pixel content) mixed with a continuous range wide enough to
/// exercise ordinary arithmetic without courting overflow when combined
/// with this file's parameter ranges — but selected by hashing an index
/// instead of drawing from `prop_oneof!`. 19 equally-likely buckets: 10
/// continuous, 1 each for the 9 landmarks, matching the original weights
/// exactly. See [`image`] for why this exists as a hash rather than a
/// `Strategy`.
fn scene_value_from_hash(bits: u64) -> f32 {
    const LANDMARKS: [f32; 9] = [0.0, 1e-6, -1e-6, 0.18, -0.18, 1.0, -1.0, 1.0e4, -1.0e4];
    let bucket = bits % 19;
    if bucket < 10 {
        // Independent of the bucket selector: the low bits (mod 19) pick
        // the bucket, the high 32 bits scale to -1000.0..1000.0.
        let unit = f64::from((bits >> 32) as u32) / f64::from(u32::MAX);
        (-1000.0 + unit * 2000.0) as f32
    } else {
        LANDMARKS[(bucket - 10) as usize]
    }
}

/// Build an `(h, w, c)` image whose pixels are drawn from a single `u64`
/// seed rather than individually from a proptest `Strategy`.
///
/// `prop::collection::vec(scene_value(), h * w * c)` — the original
/// approach — builds one shrinkable strategy-tree node per pixel: a
/// 48x48x3 image is 6912 of them, generated twice per case (paired
/// images) in every property in this file. That tree-building cost, not
/// kernel execution, was measured as the dominant cost of every property
/// here (see `dim`'s doc history in `CHANGELOG.md`). A single `u64`
/// strategy draw is O(1) regardless of image size; this function then
/// fills the array with an ordinary Rust loop — `splitmix64` twice per
/// pixel, hashing the seed together with the flat index — which costs
/// real but far smaller time, entirely outside proptest's own
/// bookkeeping.
///
/// Shrinking still works on what matters: `dim()` shrinks `h`/`w` toward
/// 1, kernel params shrink toward their own simplest values, and the
/// seed itself shrinks toward 0 (a perfectly ordinary seed, no special
/// casing needed) — proptest still finds and reports a minimal
/// `(shape, params, seed)` triple. What it no longer does is shrink
/// individual pixel values toward "simpler" ones, which was never what
/// distinguished a useful shrunk case here: every property in this file
/// cares about shape, params and the seed's *reproducibility*, not about
/// which particular finite value a given pixel happened to hold.
///
/// `splitmix64` is `phaios_core::film_grain::splitmix64` — `pub` already
/// (film_grain's own noise hash), reused here rather than duplicated.
fn image(h: usize, w: usize, c: usize, seed: u64) -> Array3<f32> {
    let n = h * w * c;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let bits = phaios_core::film_grain::splitmix64(
            seed ^ phaios_core::film_grain::splitmix64(i as u64),
        );
        data.push(scene_value_from_hash(bits));
    }
    Array3::from_shape_vec((h, w, c), data).expect("this loop always pushes exactly h*w*c values")
}

prop_compose! {
    /// Two independently-seeded images sharing one `(h, w, c)` shape, `c`
    /// fixed by the caller — the pair P3 needs to show validation does not
    /// depend on pixel content.
    fn fixed_c_image_pair(c: usize)(h in dim(), w in dim())
                                   (seed_a in any::<u64>(),
                                    seed_b in any::<u64>(),
                                    h in Just(h), w in Just(w))
                                   -> (Array3<f32>, Array3<f32>) {
        (image(h, w, c, seed_a), image(h, w, c, seed_b))
    }
}

prop_compose! {
    /// As [`fixed_c_image_pair`], but the channel count is itself drawn
    /// from [`any_channels`] — for the kernels that place no constraint on
    /// `C` at all.
    fn any_c_image_pair()(h in dim(), w in dim(), c in any_channels())
                         (seed_a in any::<u64>(),
                          seed_b in any::<u64>(),
                          h in Just(h), w in Just(w), c in Just(c))
                         -> (Array3<f32>, Array3<f32>) {
        (image(h, w, c, seed_a), image(h, w, c, seed_b))
    }
}

prop_compose! {
    /// A pair of images sharing a shape whose channel count is
    /// deliberately *wrong* for `correct_c` — for exercising the shape
    /// half of the validation contract.
    fn wrong_c_image_pair(correct_c: usize)(h in dim(), w in dim(), c in wrong_channels(correct_c))
                                           (seed_a in any::<u64>(),
                                            seed_b in any::<u64>(),
                                            h in Just(h), w in Just(w), c in Just(c))
                                           -> (Array3<f32>, Array3<f32>) {
        (image(h, w, c, seed_a), image(h, w, c, seed_b))
    }
}

// ── Shared assertion helpers ─────────────────────────────────────────────────

/// Turn a caught panic payload into a message. `catch_unwind` does not
/// hand back anything more structured than `Any`, and a `&'static str` or
/// `String` covers every panic this crate's own code can produce (`panic!`,
/// `assert!`, `.expect(...)`).
fn describe_panic(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "kernel panicked with a non-string payload".to_string())
}

/// Call `f`, converting a Rust panic into a plain `Err` message instead of
/// aborting the test case before proptest can shrink it. CLAUDE.md §2: no
/// kernel may panic on caller input, so a panic caught here already *is*
/// the property failure; this just gives it a message instead of a raw
/// unwind, and lets shrinking continue to find the minimal case.
fn catch_call<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    panic::catch_unwind(AssertUnwindSafe(f)).map_err(describe_panic)
}

/// P2 + P3 (the rejection half): assert `result` is `Err(Parameter |
/// Shape)`, not a panic and not `Ok`, and that the message contains
/// `expected_substr` — the offending field or dimension named, matching
/// the style already used throughout `tests/kernels.rs` and the
/// `#[cfg(test)]` modules. Returns the message so the caller can compare
/// it against a second image's (P3: validation must not depend on pixel
/// content).
fn expect_rejected_message<T>(
    result: Result<Result<T, PhaiosError>, String>,
    expected_substr: &str,
) -> Result<String, TestCaseError> {
    match result {
        Err(panic_msg) => Err(TestCaseError::fail(format!(
            "kernel panicked on invalid input instead of returning Err \
             (CLAUDE.md section 2: no panics on caller input): {panic_msg}"
        ))),
        Ok(Ok(_)) => Err(TestCaseError::fail(
            "expected Err for invalid input, kernel returned Ok",
        )),
        Ok(Err(e)) => {
            if !matches!(e, PhaiosError::Parameter(_) | PhaiosError::Shape(_)) {
                return Err(TestCaseError::fail(format!(
                    "expected a Parameter or Shape error, got {e:?}"
                )));
            }
            let msg = e.to_string();
            if !msg.contains(expected_substr) {
                return Err(TestCaseError::fail(format!(
                    "error message {msg:?} does not name the offending field \
                     (expected it to contain {expected_substr:?})"
                )));
            }
            Ok(msg)
        }
    }
}

/// The shape-rejection half of P2 + P3, shared by every kernel whose only
/// shape requirement is a fixed channel count: both images (of the same,
/// wrong, channel count) must be rejected with a message containing
/// "shape", and both messages must be identical.
fn assert_shape_rejected_pixel_independent<T>(
    result_a: Result<Result<T, PhaiosError>, String>,
    result_b: Result<Result<T, PhaiosError>, String>,
) -> Result<(), TestCaseError> {
    let msg_a = expect_rejected_message(result_a, "shape")?;
    let msg_b = expect_rejected_message(result_b, "shape")?;
    prop_assert_eq!(
        msg_a,
        msg_b,
        "shape-validation message depends on pixel content"
    );
    Ok(())
}

/// P5, layout-agnostic bit-exactness. `base` is reversed on its two
/// spatial axes (never the channel axis — reversing that would change
/// which physical channel each value represents, not just its layout) to
/// get a genuinely non-contiguous, negative-stride view; `.to_owned()`
/// materialises the identical logical array in standard C layout. `call`
/// is invoked on both and the results compared bit for bit — the same
/// pattern every kernel's own `accepts_any_layout` unit test already
/// uses, generalised over random input.
fn assert_layout_agnostic<T, F>(base: &Array3<f32>, call: F) -> Result<(), TestCaseError>
where
    T: PartialEq + std::fmt::Debug,
    F: Fn(ndarray::ArrayView3<f32>) -> Result<T, PhaiosError>,
{
    let reversed = base.slice(s![..;-1, ..;-1, ..]);
    let compact = reversed.to_owned();
    let a = call(reversed)?;
    let b = call(compact.view())?;
    prop_assert_eq!(a, b, "physical memory layout changed the output");
    Ok(())
}

/// Bit patterns of every element, for a stronger-than-`PartialEq`
/// comparison where two calls must be bit-identical (P5, P6) rather than
/// merely value-equal (which would treat two differently-signed NaNs, or
/// +0.0/-0.0, as the same when determinism promises they are not).
fn bits(a: &Array3<f32>) -> Vec<u32> {
    a.iter().map(|v| v.to_bits()).collect()
}

/// P6, determinism: two calls with identical input and params give
/// bit-identical output.
fn assert_deterministic_f32(
    call: impl Fn() -> Result<Array3<f32>, PhaiosError>,
) -> Result<(), TestCaseError> {
    let a = call()?;
    let b = call()?;
    prop_assert_eq!(a.dim(), b.dim());
    prop_assert_eq!(
        bits(&a),
        bits(&b),
        "same input and params produced different bytes on repeated calls"
    );
    Ok(())
}

// ── Geometry: crop ───────────────────────────────────────────────────────────

prop_compose! {
    /// A crop rectangle guaranteed to lie inside an `h x w` frame: pick
    /// width/height first, then x/y within the remaining room, so the
    /// constraint holds by construction rather than by filtering.
    fn valid_crop(h: usize, w: usize)(width in 0..=(w as u32), height in 0..=(h as u32))
                                     (x in 0..=(w as u32 - width),
                                      y in 0..=(h as u32 - height),
                                      width in Just(width), height in Just(height))
                                     -> CropParams {
        CropParams::new(x, y, width, height)
    }
}

prop_compose! {
    /// A crop rectangle guaranteed to exceed an `h x w` frame in exactly
    /// one dimension.
    fn invalid_crop(h: usize, w: usize)(overflow in 1u32..=10_000u32, bump_height in any::<bool>())
                                       -> CropParams {
        if bump_height {
            CropParams::new(0, 0, w as u32, h as u32 + overflow)
        } else {
            CropParams::new(0, 0, w as u32 + overflow, h as u32)
        }
    }
}

prop_compose! {
    /// A full valid crop case: shape, two independently-seeded images of
    /// that shape, and a rectangle guaranteed to lie inside it.
    fn valid_crop_case()(h in dim(), w in dim(), c in any_channels())
                        (params in valid_crop(h, w),
                         seed_a in any::<u64>(),
                         seed_b in any::<u64>(),
                         h in Just(h), w in Just(w), c in Just(c))
                        -> (Array3<f32>, Array3<f32>, CropParams) {
        (image(h, w, c, seed_a), image(h, w, c, seed_b), params)
    }
}

prop_compose! {
    /// A full invalid crop case: two independently-seeded images of a
    /// shape, and a rectangle guaranteed to exceed that shape.
    fn invalid_crop_case()(h in dim(), w in dim())
                          (params in invalid_crop(h, w),
                           seed_a in any::<u64>(),
                           seed_b in any::<u64>(),
                           h in Just(h), w in Just(w))
                          -> (Array3<f32>, Array3<f32>, CropParams) {
        (image(h, w, 3, seed_a), image(h, w, 3, seed_b), params)
    }
}

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6: a rectangle inside the frame is accepted
    /// regardless of pixel content, the output is exactly that rectangle,
    /// every value is finite (crop copies, it invents nothing), layout
    /// does not matter, and two calls agree.
    #[test]
    fn crop_valid_rectangle_is_accepted((img_a, img_b, params) in valid_crop_case()) {
        let c = img_a.dim().2;
        let out_a = crop(img_a.view(), &params)?;
        let out_b = crop(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), (params.height as usize, params.width as usize, c));
        prop_assert_eq!(out_b.dim(), (params.height as usize, params.width as usize, c));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| crop(v, &params))?;
        assert_deterministic_f32(|| crop(img_a.view(), &params))?;
    }

    /// P2 + P3b: a rectangle outside the frame is rejected regardless of
    /// pixel content, never panics, and names the frame it exceeded.
    #[test]
    fn crop_invalid_rectangle_is_rejected((img_a, img_b, params) in invalid_crop_case()) {
        let ra = catch_call(|| crop(img_a.view(), &params));
        let rb = catch_call(|| crop(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "frame")?;
        let msg_b = expect_rejected_message(rb, "frame")?;
        prop_assert_eq!(msg_a, msg_b, "crop rejection message depends on pixel content");
    }
}

// ── Geometry: orient ─────────────────────────────────────────────────────────

fn any_orientation() -> impl Strategy<Value = Orientation> {
    prop_oneof![
        Just(Orientation::Normal),
        Just(Orientation::FlipHorizontal),
        Just(Orientation::Rotate180),
        Just(Orientation::FlipVertical),
        Just(Orientation::Transpose),
        Just(Orientation::Rotate90),
        Just(Orientation::Transverse),
        Just(Orientation::Rotate270),
    ]
}

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6. `orient` has no `validate*` function at all —
    /// "every input and orientation is valid" is the doc comment on
    /// [`phaios_core::geometry::orient`] itself — so P2 does not apply: there
    /// is no invalid-parameter domain to construct (`Orientation` is a
    /// closed 8-variant enum with no invalid discriminant reachable from
    /// safe Rust), and no shape it rejects either.
    #[test]
    fn orient_every_input_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        o in any_orientation(),
    ) {
        let (h, w, c) = img_a.dim();
        let expected = if o.transposes() { (w, h, c) } else { (h, w, c) };

        let out_a = orient(img_a.view(), o)?;
        let out_b = orient(img_b.view(), o)?;
        prop_assert_eq!(out_a.dim(), expected);
        prop_assert_eq!(out_b.dim(), expected);
        prop_assert!(out_a.iter().all(|v| v.is_finite()), "orient invented a non-finite value");

        assert_layout_agnostic(&img_a, |v| orient(v, o))?;
        assert_deterministic_f32(|| orient(img_a.view(), o))?;
    }
}

// ── Geometry: resize ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6 on valid targets.
    #[test]
    fn resize_valid_target_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        out_w in 1u32..=48u32, out_h in 1u32..=48u32,
        filter in prop_oneof![
            Just(ResizeFilter::Area), Just(ResizeFilter::Bilinear), Just(ResizeFilter::CatmullRom)
        ],
    ) {
        let (_, _, c) = img_a.dim();
        let params = ResizeParams::new(out_w, out_h, filter);

        let out_a = resize(img_a.view(), &params)?;
        let out_b = resize(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), (out_h as usize, out_w as usize, c));
        prop_assert!(out_a.iter().all(|v| v.is_finite()), "resize produced a non-finite value from finite input");
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| resize(v, &params))?;
        assert_deterministic_f32(|| resize(img_a.view(), &params))?;
    }

    /// P2 + P3b: a zero target dimension is rejected regardless of pixel
    /// content and names the target.
    #[test]
    fn resize_zero_target_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        zero_is_width in any::<bool>(),
        other in 1u32..=48u32,
        filter in prop_oneof![
            Just(ResizeFilter::Area), Just(ResizeFilter::Bilinear), Just(ResizeFilter::CatmullRom)
        ],
    ) {
        let params = if zero_is_width {
            ResizeParams::new(0, other, filter)
        } else {
            ResizeParams::new(other, 0, filter)
        };
        let ra = catch_call(|| resize(img_a.view(), &params));
        let rb = catch_call(|| resize(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "resize target")?;
        let msg_b = expect_rejected_message(rb, "resize target")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P4: a target whose f32 footprint clears the 8 GiB single-allocation
    /// limit is refused as `Allocation`, and refused fast — before the
    /// allocation is attempted, per `alloc::zeros3`'s check-then-allocate
    /// contract (`src/alloc.rs`, `docs/ffi.md` "Allocation is bounded").
    /// `resize` is the kernel named in the task as the clearest example of
    /// a caller-controlled output shape: `width`/`height` come straight
    /// from `ResizeParams`, independent of the input's own size.
    #[test]
    fn resize_oversized_target_is_refused_before_allocating(
        width in 60_000u32..=200_000u32,
        height in 60_000u32..=200_000u32,
        c in prop_oneof![Just(1usize), Just(3usize)],
    ) {
        // The smallest combination here, 60_000 x 60_000 x 1 x 4 bytes, is
        // already ~14.4 GB — comfortably past the 8 GiB limit.
        prop_assert!(u64::from(width) * u64::from(height) * c as u64 * 4 > MAX_ALLOCATION_BYTES);

        let img = Array3::<f32>::zeros((2, 2, c));
        let params = ResizeParams::new(width, height, ResizeFilter::Area);

        let start = Instant::now();
        let result = resize(img.view(), &params);
        let elapsed = start.elapsed();

        prop_assert!(
            matches!(result, Err(PhaiosError::Allocation(_))),
            "a {width}x{height}x{c} target must be refused as Allocation, got {result:?}"
        );
        // The bound is deliberately coarse. Materialising an 8 GiB-plus
        // buffer takes many seconds on any machine, so two seconds still
        // separates "refused by the guard" from "allocated, then failed",
        // while a tight bound flakes on a loaded CI runner — 100 ms did,
        // once in three local runs under contention. The `Err(Allocation)`
        // check above is the assertion; this is the tell.
        prop_assert!(
            elapsed.as_secs() < 2,
            "rejection took {elapsed:?}; the guard must fire before allocating, not after"
        );
    }
}

// ── Geometry: straighten ─────────────────────────────────────────────────────

/// As [`dim`], weighted toward small, but starting at 2: `straighten`'s
/// inscribed rectangle is never empty for `h, w >= 2` across the whole
/// ±45° range (verified against the kernel's own max-area construction:
/// at exactly 45°, the tightest case, a 2 px short side still yields a
/// 1 px inscribed dimension) — `h` or `w` of 1 is the separate,
/// shape-dependent "leaves no whole pixel inscribed" rejection, already
/// exercised by `straighten`'s own `#[cfg(test)]` module.
fn dim_at_least_2() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => 2usize..=6usize,
        2 => 7usize..=20usize,
        1 => 21usize..=48usize,
    ]
}

prop_compose! {
    fn valid_straighten_case()(h in dim_at_least_2(), w in dim_at_least_2(), c in any_channels())
                              (degrees in -45.0f32..=45.0f32,
                               seed_a in any::<u64>(),
                               seed_b in any::<u64>(),
                               h in Just(h), w in Just(w), c in Just(c))
                              -> (Array3<f32>, Array3<f32>, StraightenParams) {
        (image(h, w, c, seed_a), image(h, w, c, seed_b), StraightenParams::new(degrees))
    }
}

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6.
    #[test]
    fn straighten_valid_angle_is_accepted((img_a, img_b, params) in valid_straighten_case()) {
        let (h, w, c) = img_a.dim();
        let out_a = straighten(img_a.view(), &params)?;
        let out_b = straighten(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim().2, c);
        prop_assert!(out_a.dim().0 >= 1 && out_a.dim().0 <= h);
        prop_assert!(out_a.dim().1 >= 1 && out_a.dim().1 <= w);
        prop_assert_eq!(out_a.dim(), out_b.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()), "straighten produced a non-finite value");

        assert_layout_agnostic(&img_a, |v| straighten(v, &params))?;
        assert_deterministic_f32(|| straighten(img_a.view(), &params))?;
    }

    /// P2 + P3b: an angle outside ±45° (or non-finite) is rejected
    /// regardless of pixel content, and the message names `degrees` —
    /// true of every rejection branch in `straighten_geometry`, including
    /// the shape-dependent "leaves no whole pixel inscribed" one, which
    /// also names `degrees` in its message.
    #[test]
    fn straighten_invalid_angle_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        degrees in prop_oneof![
            2 => non_finite_f32(),
            1 => 45.000_2f32..1.0e6f32,
            1 => -1.0e6f32..-45.000_2f32,
        ],
    ) {
        let params = StraightenParams::new(degrees);
        let ra = catch_call(|| straighten(img_a.view(), &params));
        let rb = catch_call(|| straighten(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "degrees")?;
        let msg_b = expect_rejected_message(rb, "degrees")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── exposure ──────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6. `exposure` has no shape constraint at all.
    #[test]
    fn exposure_valid_stops_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        stops in -50.0f32..50.0f32,
    ) {
        let out_a = exposure(img_a.view(), stops)?;
        let out_b = exposure(img_b.view(), stops)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| exposure(v, stops))?;
        assert_deterministic_f32(|| exposure(img_a.view(), stops))?;
    }

    /// P2 + P3b: non-finite `stops` is rejected regardless of pixel
    /// content and names `stops`.
    #[test]
    fn exposure_non_finite_stops_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        stops in non_finite_f32(),
    ) {
        let ra = catch_call(|| exposure(img_a.view(), stops));
        let rb = catch_call(|| exposure(img_b.view(), stops));
        let msg_a = expect_rejected_message(ra, "stops")?;
        let msg_b = expect_rejected_message(rb, "stops")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── B&W: luminance_bw, channel_mixer_bw, color_filter_bw ────────────────────
//
// All three share `validate_rgb`: the only requirement is `shape[2] == 3`.
// `channel_mixer_bw`'s weights are not validated at all (confirmed by
// reading `src/bw.rs`: `channel_mixer_bw` calls only `validate_rgb`), so its
// P2 property covers shape alone, not a "bad weight" domain that does not
// exist. `luminance_bw` and `color_filter_bw` take closed enums
// (`LuminanceStandard`, `ColorFilter`) with no invalid discriminant
// reachable from safe Rust, so likewise no param-level P2 for them.

fn any_luminance_standard() -> impl Strategy<Value = LuminanceStandard> {
    prop_oneof![
        Just(LuminanceStandard::Bt601),
        Just(LuminanceStandard::Bt709),
        Just(LuminanceStandard::Bt2020),
    ]
}

fn any_color_filter() -> impl Strategy<Value = ColorFilter> {
    prop_oneof![
        Just(ColorFilter::NoFilter),
        Just(ColorFilter::Yellow8K2),
        Just(ColorFilter::Orange21),
        Just(ColorFilter::Red25A),
        Just(ColorFilter::Green11X1),
        Just(ColorFilter::Blue47C5),
    ]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn luminance_bw_valid_rgb_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(3),
        standard in any_luminance_standard(),
    ) {
        let out_a = luminance_bw(img_a.view(), standard)?;
        let out_b = luminance_bw(img_b.view(), standard)?;
        prop_assert_eq!(out_a.dim(), (img_a.dim().0, img_a.dim().1, 1));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| luminance_bw(v, standard))?;
        assert_deterministic_f32(|| luminance_bw(img_a.view(), standard))?;
    }

    #[test]
    fn luminance_bw_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(3),
        standard in any_luminance_standard(),
    ) {
        let ra = catch_call(|| luminance_bw(img_a.view(), standard));
        let rb = catch_call(|| luminance_bw(img_b.view(), standard));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    #[test]
    fn channel_mixer_bw_valid_rgb_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(3),
        weights in prop::array::uniform3(-20.0f32..20.0f32),
    ) {
        let out_a = channel_mixer_bw(img_a.view(), weights)?;
        let out_b = channel_mixer_bw(img_b.view(), weights)?;
        prop_assert_eq!(out_a.dim(), (img_a.dim().0, img_a.dim().1, 1));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| channel_mixer_bw(v, weights))?;
        assert_deterministic_f32(|| channel_mixer_bw(img_a.view(), weights))?;
    }

    #[test]
    fn channel_mixer_bw_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(3),
        weights in prop::array::uniform3(-20.0f32..20.0f32),
    ) {
        let ra = catch_call(|| channel_mixer_bw(img_a.view(), weights));
        let rb = catch_call(|| channel_mixer_bw(img_b.view(), weights));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    #[test]
    fn color_filter_bw_valid_rgb_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(3),
        filter in any_color_filter(),
        standard in any_luminance_standard(),
    ) {
        let out_a = color_filter_bw(img_a.view(), filter, standard)?;
        let out_b = color_filter_bw(img_b.view(), filter, standard)?;
        prop_assert_eq!(out_a.dim(), (img_a.dim().0, img_a.dim().1, 1));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| color_filter_bw(v, filter, standard))?;
        assert_deterministic_f32(|| color_filter_bw(img_a.view(), filter, standard))?;
    }

    #[test]
    fn color_filter_bw_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(3),
        filter in any_color_filter(),
        standard in any_luminance_standard(),
    ) {
        let ra = catch_call(|| color_filter_bw(img_a.view(), filter, standard));
        let rb = catch_call(|| color_filter_bw(img_b.view(), filter, standard));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }
}

// ── B&W: hsl_bw ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn hsl_bw_valid_params_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(3),
        hue_weights in prop::array::uniform8(-5.0f32..5.0f32),
        standard in any_luminance_standard(),
        sigma_deg in 0.001f32..360.0f32,
    ) {
        let params = HslWeightedParams::new(hue_weights, standard, sigma_deg);
        let out_a = hsl_bw(img_a.view(), &params)?;
        let out_b = hsl_bw(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), (img_a.dim().0, img_a.dim().1, 1));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| hsl_bw(v, &params))?;
        assert_deterministic_f32(|| hsl_bw(img_a.view(), &params))?;
    }

    #[test]
    fn hsl_bw_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(3),
        sigma_deg in 0.001f32..360.0f32,
    ) {
        let params = HslWeightedParams::new([0.0; 8], LuminanceStandard::Bt709, sigma_deg);
        let ra = catch_call(|| hsl_bw(img_a.view(), &params));
        let rb = catch_call(|| hsl_bw(img_b.view(), &params));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    /// P2 + P3b over `sigma_deg`: non-finite or non-positive.
    #[test]
    fn hsl_bw_bad_sigma_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(3),
        sigma_deg in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..=0.0f32],
    ) {
        let params = HslWeightedParams::new([0.0; 8], LuminanceStandard::Bt709, sigma_deg);
        let ra = catch_call(|| hsl_bw(img_a.view(), &params));
        let rb = catch_call(|| hsl_bw(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "sigma_deg")?;
        let msg_b = expect_rejected_message(rb, "sigma_deg")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over a single non-finite hue weight.
    #[test]
    fn hsl_bw_bad_hue_weight_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(3),
        bad_index in 0usize..8,
        bad in non_finite_f32(),
    ) {
        let mut weights = [0.0_f32; 8];
        weights[bad_index] = bad;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        let ra = catch_call(|| hsl_bw(img_a.view(), &params));
        let rb = catch_call(|| hsl_bw(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "hue weight")?;
        let msg_b = expect_rejected_message(rb, "hue weight")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── zone_system ───────────────────────────────────────────────────────────────

prop_compose! {
    /// A small, well-formed zone-offset map: 0-4 entries, indices always
    /// in the valid 0..=10 range, offsets bounded to a realistic-but-wide
    /// span (see the module doc's note on "valid" ranges: offsets are
    /// validated for finiteness only). The Gaussian blend can put close to
    /// 2.0 combined weight on a `zone_pos` sitting between up to four
    /// clustered zone indices, so the worst case here is
    /// `2.0.powf(4 * 20 * 2.0)` — nowhere close, but `±60` (an earlier
    /// draft) leaves only a ~2 order-of-magnitude margin to `f32::MAX`
    /// once several entries cluster; `±20` leaves a much wider one while
    /// still testing an order of magnitude past the "conventional ±3"
    /// the docs mention.
    fn valid_zone_offsets()(entries in prop::collection::vec((0i32..=10, -20.0f32..20.0f32), 0..5))
                          -> std::collections::HashMap<i32, f32> {
        entries.into_iter().collect()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn zone_system_valid_offsets_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(1),
        offsets in valid_zone_offsets(),
    ) {
        let params = ZoneParams::new(offsets);
        let out_a = zone_system(img_a.view(), &params)?;
        let out_b = zone_system(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| zone_system(v, &params))?;
        assert_deterministic_f32(|| zone_system(img_a.view(), &params))?;
    }

    #[test]
    fn zone_system_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(1),
        offsets in valid_zone_offsets(),
    ) {
        let params = ZoneParams::new(offsets);
        let ra = catch_call(|| zone_system(img_a.view(), &params));
        let rb = catch_call(|| zone_system(img_b.view(), &params));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    /// P2 + P3b: a zone index outside the eleven zones 0..=10.
    #[test]
    fn zone_system_bad_zone_index_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        bad_zone in prop_oneof![-1000i32..0, 11i32..1000],
        offset in -3.0f32..3.0f32,
    ) {
        let mut offsets = std::collections::HashMap::new();
        offsets.insert(bad_zone, offset);
        let params = ZoneParams::new(offsets);
        let ra = catch_call(|| zone_system(img_a.view(), &params));
        let rb = catch_call(|| zone_system(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "zone index")?;
        let msg_b = expect_rejected_message(rb, "zone index")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b: a non-finite offset on an otherwise-valid zone index.
    #[test]
    fn zone_system_non_finite_offset_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        zone in 0i32..=10,
        bad_offset in non_finite_f32(),
    ) {
        let mut offsets = std::collections::HashMap::new();
        offsets.insert(zone, bad_offset);
        let params = ZoneParams::new(offsets);
        let ra = catch_call(|| zone_system(img_a.view(), &params));
        let rb = catch_call(|| zone_system(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "offset for zone")?;
        let msg_b = expect_rejected_message(rb, "offset for zone")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── local_contrast ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// `radius` has no validated upper bound (any `u32` is legal — a
    /// radius larger than the image is harmless, windows clamp, per the
    /// kernel's own doc comment) but is bounded to `0..=8` here for
    /// runtime: the guided filter's cost scales with radius, and this
    /// file's point is the validation contract, not a radius stress test.
    #[test]
    fn local_contrast_valid_params_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(1),
        radius in 0u32..=8u32,
        eps in 0.0f32..1000.0f32,
        strength in -10.0f32..10.0f32,
    ) {
        let params = GuidedFilterParams::new(radius, eps);
        let out_a = local_contrast(img_a.view(), &params, strength)?;
        let out_b = local_contrast(img_b.view(), &params, strength)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| local_contrast(v, &params, strength))?;
        assert_deterministic_f32(|| local_contrast(img_a.view(), &params, strength))?;
    }

    #[test]
    fn local_contrast_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(1),
        radius in 0u32..=8u32,
        eps in 0.0f32..1000.0f32,
    ) {
        let params = GuidedFilterParams::new(radius, eps);
        let ra = catch_call(|| local_contrast(img_a.view(), &params, 1.0));
        let rb = catch_call(|| local_contrast(img_b.view(), &params, 1.0));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    /// P2 + P3b over `eps`: non-finite or negative.
    #[test]
    fn local_contrast_bad_eps_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        radius in 0u32..=8u32,
        eps in prop_oneof![2 => non_finite_f32(), 1 => -1000.0f32..0.0f32],
    ) {
        let params = GuidedFilterParams::new(radius, eps);
        let ra = catch_call(|| local_contrast(img_a.view(), &params, 1.0));
        let rb = catch_call(|| local_contrast(img_b.view(), &params, 1.0));
        let msg_a = expect_rejected_message(ra, "eps")?;
        let msg_b = expect_rejected_message(rb, "eps")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `strength`: non-finite.
    #[test]
    fn local_contrast_bad_strength_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        radius in 0u32..=8u32,
        strength in non_finite_f32(),
    ) {
        let params = GuidedFilterParams::new(radius, 0.01);
        let ra = catch_call(|| local_contrast(img_a.view(), &params, strength));
        let rb = catch_call(|| local_contrast(img_b.view(), &params, strength));
        let msg_a = expect_rejected_message(ra, "strength")?;
        let msg_b = expect_rejected_message(rb, "strength")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── film_grain ────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn film_grain_valid_params_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(1),
        intensity in 0.0f32..100.0f32,
        size_pixels in 0.01f32..64.0f32,
        seed in any::<u64>(),
    ) {
        let params = GrainParams::new(intensity, size_pixels, seed);
        let out_a = film_grain(img_a.view(), &params)?;
        let out_b = film_grain(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| film_grain(v, &params))?;
        assert_deterministic_f32(|| film_grain(img_a.view(), &params))?;
    }

    #[test]
    fn film_grain_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(1),
        intensity in 0.0f32..100.0f32,
    ) {
        let params = GrainParams::new(intensity, 2.0, 0);
        let ra = catch_call(|| film_grain(img_a.view(), &params));
        let rb = catch_call(|| film_grain(img_b.view(), &params));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    /// P2 + P3b over `intensity`: non-finite or negative.
    #[test]
    fn film_grain_bad_intensity_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        intensity in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = GrainParams::new(intensity, 2.0, 0);
        let ra = catch_call(|| film_grain(img_a.view(), &params));
        let rb = catch_call(|| film_grain(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "intensity")?;
        let msg_b = expect_rejected_message(rb, "intensity")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `size_pixels`: non-finite or non-positive.
    #[test]
    fn film_grain_bad_size_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        size_pixels in prop_oneof![2 => non_finite_f32(), 1 => -64.0f32..=0.0f32],
    ) {
        let params = GrainParams::new(0.3, size_pixels, 0);
        let ra = catch_call(|| film_grain(img_a.view(), &params));
        let rb = catch_call(|| film_grain(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "size_pixels")?;
        let msg_b = expect_rejected_message(rb, "size_pixels")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P6, the seed half: on a non-trivial (non-constant) image with
    /// non-zero intensity, two different seeds must give different grain.
    /// `film_grain`'s own `#[cfg(test)]` module already pins this on one
    /// fixed image; this sweeps random ones (still requiring at least 2x2
    /// so the grain has somewhere to vary).
    #[test]
    fn film_grain_different_seeds_differ(
        h in 2usize..=48usize, w in 2usize..=48usize,
        vals in prop::collection::vec(0.1f32..0.9f32, 4..(48 * 48)),
        seed_a in any::<u64>(), seed_b in any::<u64>(),
    ) {
        prop_assume!(seed_a != seed_b);
        let n = h * w;
        prop_assume!(vals.len() >= n);
        let img = mk_image(h, w, 1, vals[..n].to_vec());
        let a = film_grain(img.view(), &GrainParams::new(0.5, 2.0, seed_a))?;
        let b = film_grain(img.view(), &GrainParams::new(0.5, 2.0, seed_b))?;
        prop_assert_ne!(bits(&a), bits(&b), "two different seeds produced identical grain");
    }
}

// ── split_toning ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn split_toning_valid_params_is_accepted(
        (img_a, img_b) in fixed_c_image_pair(1),
        shadow in prop::array::uniform3(-0.5f32..0.5f32),
        highlight in prop::array::uniform3(-0.5f32..0.5f32),
        pivot in 0.0f32..=1.0f32,
        balance in -1.0f32..=1.0f32,
    ) {
        let params = SplitToningParams::new(shadow, highlight, pivot, balance);
        let out_a = split_toning(img_a.view(), &params)?;
        let out_b = split_toning(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), (img_a.dim().0, img_a.dim().1, 3));
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| split_toning(v, &params))?;
        assert_deterministic_f32(|| split_toning(img_a.view(), &params))?;
    }

    #[test]
    fn split_toning_wrong_channel_count_is_rejected(
        (img_a, img_b) in wrong_c_image_pair(1),
    ) {
        let params = SplitToningParams::default();
        let ra = catch_call(|| split_toning(img_a.view(), &params));
        let rb = catch_call(|| split_toning(img_b.view(), &params));
        assert_shape_rejected_pixel_independent(ra, rb)?;
    }

    /// P2 + P3b over `pivot`: non-finite or outside 0..=1.
    #[test]
    fn split_toning_bad_pivot_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        pivot in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = SplitToningParams::new([0.0; 3], [0.0; 3], pivot, 0.0);
        let ra = catch_call(|| split_toning(img_a.view(), &params));
        let rb = catch_call(|| split_toning(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "pivot")?;
        let msg_b = expect_rejected_message(rb, "pivot")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `balance`: non-finite or outside -1..=1.
    #[test]
    fn split_toning_bad_balance_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        balance in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..-1.000_1f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, balance);
        let ra = catch_call(|| split_toning(img_a.view(), &params));
        let rb = catch_call(|| split_toning(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "balance")?;
        let msg_b = expect_rejected_message(rb, "balance")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b: a non-finite component in either tint triple.
    #[test]
    fn split_toning_bad_tint_component_is_rejected(
        (img_a, img_b) in fixed_c_image_pair(1),
        shadow_not_highlight in any::<bool>(),
        idx in 0usize..3,
        bad in non_finite_f32(),
    ) {
        let mut shadow = [0.0_f32; 3];
        let mut highlight = [0.0_f32; 3];
        if shadow_not_highlight { shadow[idx] = bad; } else { highlight[idx] = bad; }
        let params = SplitToningParams::new(shadow, highlight, 0.5, 0.0);
        let expected = if shadow_not_highlight { "shadow_oklab" } else { "highlight_oklab" };
        let ra = catch_call(|| split_toning(img_a.view(), &params));
        let rb = catch_call(|| split_toning(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, expected)?;
        let msg_b = expect_rejected_message(rb, expected)?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── vignette ──────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn vignette_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        amount in -100.0f32..100.0f32,
        feather in 0.0f32..=1.0f32,
        roundness in 0.0f32..=1.0f32,
    ) {
        let params = VignetteParams::new(amount, feather, roundness);
        let out_a = vignette(img_a.view(), &params)?;
        let out_b = vignette(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| vignette(v, &params))?;
        assert_deterministic_f32(|| vignette(img_a.view(), &params))?;
    }

    /// P2 + P3b over `amount`: non-finite.
    #[test]
    fn vignette_bad_amount_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        amount in non_finite_f32(),
    ) {
        let params = VignetteParams::new(amount, 0.5, 0.0);
        let ra = catch_call(|| vignette(img_a.view(), &params));
        let rb = catch_call(|| vignette(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "amount")?;
        let msg_b = expect_rejected_message(rb, "amount")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `feather`: non-finite or outside 0..=1.
    #[test]
    fn vignette_bad_feather_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        feather in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = VignetteParams::new(0.5, feather, 0.0);
        let ra = catch_call(|| vignette(img_a.view(), &params));
        let rb = catch_call(|| vignette(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "feather")?;
        let msg_b = expect_rejected_message(rb, "feather")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `roundness`: non-finite or outside 0..=1.
    #[test]
    fn vignette_bad_roundness_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        roundness in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = VignetteParams::new(0.5, 0.5, roundness);
        let ra = catch_call(|| vignette(img_a.view(), &params));
        let rb = catch_call(|| vignette(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "roundness")?;
        let msg_b = expect_rejected_message(rb, "roundness")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── shadow_rolloff ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn shadow_rolloff_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        knee in 0.0f32..=1.0f32,
        strength in 0.0f32..=1.0f32,
    ) {
        let params = ShadowRolloffParams::new(knee, strength);
        let out_a = shadow_rolloff(img_a.view(), &params)?;
        let out_b = shadow_rolloff(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| shadow_rolloff(v, &params))?;
        assert_deterministic_f32(|| shadow_rolloff(img_a.view(), &params))?;
    }

    /// P2 + P3b over `knee`: non-finite or outside 0..=1.
    #[test]
    fn shadow_rolloff_bad_knee_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        knee in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = ShadowRolloffParams::new(knee, 0.5);
        let ra = catch_call(|| shadow_rolloff(img_a.view(), &params));
        let rb = catch_call(|| shadow_rolloff(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "knee")?;
        let msg_b = expect_rejected_message(rb, "knee")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `strength`: non-finite or outside 0..=1.
    #[test]
    fn shadow_rolloff_bad_strength_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        strength in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = ShadowRolloffParams::new(0.2, strength);
        let ra = catch_call(|| shadow_rolloff(img_a.view(), &params));
        let rb = catch_call(|| shadow_rolloff(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "strength")?;
        let msg_b = expect_rejected_message(rb, "strength")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── tone_curve ────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// `slope`, `offset` and `power` are validated for finiteness only
    /// (`power` additionally `> 0`), with no upper bound. Bounded here to
    /// avoid `t.powf(power)` overflowing to infinity on this file's own
    /// pixel range (see the module doc) — the validator's domain is wider
    /// than what is sampled, deliberately. Worst case in range: `v` at the
    /// 1e4 landmark, `slope = 5`, `offset = 5` gives `t <= 50005`; even at
    /// `power = 6`, `t.powf(power)` is ~4e28, comfortably under
    /// `f32::MAX` (~3.4e38) — found the hard way: an earlier draft using
    /// `-100.0..100.0` / `0.01..10.0` overflowed to infinity on a shrunk
    /// case (`slope = -96.5227, offset = -89.96682, power = 6.7404222`,
    /// `v = -10000.0`).
    #[test]
    fn tone_curve_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        slope in -5.0f32..5.0f32,
        offset in -5.0f32..5.0f32,
        power in 0.05f32..6.0f32,
    ) {
        let params = ToneCurveParams::new(slope, offset, power);
        let out_a = tone_curve(img_a.view(), &params)?;
        let out_b = tone_curve(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| tone_curve(v, &params))?;
        assert_deterministic_f32(|| tone_curve(img_a.view(), &params))?;
    }

    /// P2 + P3b over `slope`: non-finite.
    #[test]
    fn tone_curve_bad_slope_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        slope in non_finite_f32(),
    ) {
        let params = ToneCurveParams::new(slope, 0.0, 1.0);
        let ra = catch_call(|| tone_curve(img_a.view(), &params));
        let rb = catch_call(|| tone_curve(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "slope")?;
        let msg_b = expect_rejected_message(rb, "slope")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `offset`: non-finite.
    #[test]
    fn tone_curve_bad_offset_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        offset in non_finite_f32(),
    ) {
        let params = ToneCurveParams::new(1.0, offset, 1.0);
        let ra = catch_call(|| tone_curve(img_a.view(), &params));
        let rb = catch_call(|| tone_curve(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "offset")?;
        let msg_b = expect_rejected_message(rb, "offset")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `power`: non-finite or non-positive.
    #[test]
    fn tone_curve_bad_power_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        power in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..=0.0f32],
    ) {
        let params = ToneCurveParams::new(1.0, 0.0, power);
        let ra = catch_call(|| tone_curve(img_a.view(), &params));
        let rb = catch_call(|| tone_curve(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "power")?;
        let msg_b = expect_rejected_message(rb, "power")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── blur ──────────────────────────────────────────────────────────────────────

/// Lower than [`config`]'s default (256), for `blur_valid_sigma_is_accepted`
/// alone. `box_widths`'s search cost is roughly quadratic in sigma (its
/// own doc: "quadratic... starting near σ"), so even with the
/// small-weighted distribution below, running it at 256 cases cost ~20 s
/// on its own — the *rest* of this file's 69 properties combined cost
/// about 2 s at 256 cases. This many still walks every tier of the
/// distribution, including at or near `MAX_SIGMA`, on this file's fixed
/// seed; the exact boundary (`sigma == MAX_SIGMA` accepted,
/// `sigma > MAX_SIGMA` rejected) is additionally pinned by
/// `blur::tests::a_huge_sigma_is_refused_rather_than_searched_forever`.
fn blur_valid_config() -> ProptestConfig {
    ProptestConfig {
        cases: 24,
        ..config()
    }
}

proptest! {
    #![proptest_config(blur_valid_config())]

    /// Sigma is weighted toward the small end but reaches all the way to
    /// `MAX_SIGMA`, occasionally. The box *path* costs the same at any
    /// radius (the module doc: "costs the same at any radius") — but
    /// choosing the box *widths* for it is `box_widths`, a search whose
    /// own doc calls it "quadratic... starting near σ", and measured here
    /// at ~19 s for this property alone when sigma was sampled uniformly
    /// over the full range: large sigma is cheap to *blur with*, not
    /// cheap to *search for*, and this file calls it six times per case.
    #[test]
    fn blur_valid_sigma_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        sigma in prop_oneof![
            10 => 0.0f32..16.0f32,
            5 => 16.0f32..128.0f32,
            3 => 128.0f32..768.0f32,
            1 => 768.0f32..=blur_mod::MAX_SIGMA,
        ],
    ) {
        let params = BlurParams::new(sigma, BlurShape::Gaussian);
        let out_a = blur(img_a.view(), &params)?;
        let out_b = blur(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| blur(v, &params))?;
        assert_deterministic_f32(|| blur(img_a.view(), &params))?;
    }
}

proptest! {
    #![proptest_config(config())]

    /// P2 + P3b: sigma negative, non-finite, or above `MAX_SIGMA`. Cheap
    /// regardless of sigma's magnitude — `validate` rejects before
    /// `box_widths` ever runs — so this stays at the default case count.
    #[test]
    fn blur_bad_sigma_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        sigma in prop_oneof![
            2 => non_finite_f32(),
            1 => -1.0e6f32..0.0f32,
            1 => (blur_mod::MAX_SIGMA * 1.001)..1.0e9f32,
        ],
    ) {
        let params = BlurParams::new(sigma, BlurShape::Gaussian);
        let ra = catch_call(|| blur(img_a.view(), &params));
        let rb = catch_call(|| blur(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "sigma")?;
        let msg_b = expect_rejected_message(rb, "sigma")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── glow ──────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn glow_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        threshold in 0.0f32..100.0f32,
        sigma in 0.0f32..64.0f32,
        amount in 0.0f32..100.0f32,
    ) {
        let params = GlowParams::new(threshold, sigma, amount);
        let out_a = glow(img_a.view(), &params)?;
        let out_b = glow(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| glow(v, &params))?;
        assert_deterministic_f32(|| glow(img_a.view(), &params))?;
    }

    /// P2 + P3b over `threshold`: non-finite or negative.
    #[test]
    fn glow_bad_threshold_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        threshold in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = GlowParams::new(threshold, 4.0, 0.5);
        let ra = catch_call(|| glow(img_a.view(), &params));
        let rb = catch_call(|| glow(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "threshold")?;
        let msg_b = expect_rejected_message(rb, "threshold")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `amount`: non-finite or negative.
    #[test]
    fn glow_bad_amount_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        amount in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = GlowParams::new(0.5, 4.0, amount);
        let ra = catch_call(|| glow(img_a.view(), &params));
        let rb = catch_call(|| glow(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "amount")?;
        let msg_b = expect_rejected_message(rb, "amount")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b: `sigma` outside `blur`'s own domain — `glow` delegates to
    /// `blur::validate`, so the message is whatever that validator names.
    #[test]
    fn glow_bad_sigma_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        sigma in prop_oneof![2 => non_finite_f32(), 1 => -1.0e6f32..0.0f32],
    ) {
        let params = GlowParams::new(0.5, sigma, 0.5);
        let ra = catch_call(|| glow(img_a.view(), &params));
        let rb = catch_call(|| glow(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "sigma")?;
        let msg_b = expect_rejected_message(rb, "sigma")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── sharpen ──────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// P1 + P5 + P6.
    #[test]
    fn sharpen_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        amount in 0.0f32..100.0f32,
        sigma in 0.0f32..64.0f32,
        threshold in 0.0f32..100.0f32,
    ) {
        let params = SharpenParams::new(amount, sigma, threshold);
        let out_a = sharpen(img_a.view(), &params)?;
        let out_b = sharpen(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| sharpen(v, &params))?;
        assert_deterministic_f32(|| sharpen(img_a.view(), &params))?;
    }

    /// P2 + P3b over `amount`: non-finite or negative.
    #[test]
    fn sharpen_bad_amount_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        amount in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = SharpenParams::new(amount, 1.0, 0.0);
        let ra = catch_call(|| sharpen(img_a.view(), &params));
        let rb = catch_call(|| sharpen(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "amount")?;
        let msg_b = expect_rejected_message(rb, "amount")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `threshold`: non-finite or negative.
    #[test]
    fn sharpen_bad_threshold_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        threshold in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = SharpenParams::new(0.5, 1.0, threshold);
        let ra = catch_call(|| sharpen(img_a.view(), &params));
        let rb = catch_call(|| sharpen(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "threshold")?;
        let msg_b = expect_rejected_message(rb, "threshold")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b: `sigma` outside `blur`'s own domain — `sharpen` delegates
    /// to `blur::validate`, so the message is whatever that validator
    /// names (mirrors `glow_bad_sigma_is_rejected`).
    #[test]
    fn sharpen_bad_sigma_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        sigma in prop_oneof![2 => non_finite_f32(), 1 => -1.0e6f32..0.0f32],
    ) {
        let params = SharpenParams::new(0.5, sigma, 0.0);
        let ra = catch_call(|| sharpen(img_a.view(), &params));
        let rb = catch_call(|| sharpen(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "sigma")?;
        let msg_b = expect_rejected_message(rb, "sigma")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── hot_pixels ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// P1 + P5 + P6.
    #[test]
    fn hot_pixels_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        threshold in 0.0f32..100.0f32,
        relative in 0.0f32..100.0f32,
    ) {
        let params = HotPixelParams::new(threshold, relative);
        let out_a = hot_pixels(img_a.view(), &params)?;
        let out_b = hot_pixels(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| hot_pixels(v, &params))?;
        assert_deterministic_f32(|| hot_pixels(img_a.view(), &params))?;
    }

    /// P2 + P3b over `threshold`: non-finite or negative.
    #[test]
    fn hot_pixels_bad_threshold_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        threshold in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = HotPixelParams::new(threshold, 0.0);
        let ra = catch_call(|| hot_pixels(img_a.view(), &params));
        let rb = catch_call(|| hot_pixels(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "threshold")?;
        let msg_b = expect_rejected_message(rb, "threshold")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `relative`: non-finite or negative.
    #[test]
    fn hot_pixels_bad_relative_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        relative in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..0.0f32],
    ) {
        let params = HotPixelParams::new(0.05, relative);
        let ra = catch_call(|| hot_pixels(img_a.view(), &params));
        let rb = catch_call(|| hot_pixels(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "relative")?;
        let msg_b = expect_rejected_message(rb, "relative")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── highlight_rolloff ────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn highlight_rolloff_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        knee in 0.0f32..=1.0f32,
        white_point in 1.0f32..1.0e4f32,
    ) {
        let params = RolloffParams::new(knee, white_point);
        let out_a = highlight_rolloff(img_a.view(), &params)?;
        let out_b = highlight_rolloff(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| highlight_rolloff(v, &params))?;
        assert_deterministic_f32(|| highlight_rolloff(img_a.view(), &params))?;
    }

    /// P2 + P3b over `knee`: non-finite or outside 0..=1.
    #[test]
    fn highlight_rolloff_bad_knee_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        knee in prop_oneof![2 => non_finite_f32(), 1 => -10.0f32..0.0f32, 1 => 1.000_1f32..10.0f32],
    ) {
        let params = RolloffParams::new(knee, 2.0);
        let ra = catch_call(|| highlight_rolloff(img_a.view(), &params));
        let rb = catch_call(|| highlight_rolloff(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "knee")?;
        let msg_b = expect_rejected_message(rb, "knee")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over `white_point`: non-finite or below 1.0.
    #[test]
    fn highlight_rolloff_bad_white_point_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        white_point in prop_oneof![2 => non_finite_f32(), 1 => -100.0f32..1.0f32],
    ) {
        let params = RolloffParams::new(0.8, white_point);
        let ra = catch_call(|| highlight_rolloff(img_a.view(), &params));
        let rb = catch_call(|| highlight_rolloff(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "white_point")?;
        let msg_b = expect_rejected_message(rb, "white_point")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}

// ── encode_srgb ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// P1 + P3a + P5 + P6. `encode_srgb` takes no parameters and has no
    /// `validate*` function — "currently infallible" is the doc comment on
    /// the kernel itself — so P2 does not apply: there is no invalid input
    /// this kernel's contract defines.
    #[test]
    fn encode_srgb_every_input_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
    ) {
        let out_a = encode_srgb(img_a.view())?;
        let out_b = encode_srgb(img_b.view())?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, encode_srgb)?;
        assert_deterministic_f32(|| encode_srgb(img_a.view()))?;
    }
}

// ── quantize_u8, quantize_u16 ────────────────────────────────────────────────
//
// Neither has a `validate*` function — "infallible for every 3-D input" is
// the doc comment on both kernels — so P2 does not apply: `Dither` is a
// closed two-variant enum, `seed` is an unconstrained `u64`, and no shape
// is rejected either. P1, P5 and P6 (including the seed-difference half)
// are the properties that apply.

proptest! {
    #![proptest_config(config())]

    #[test]
    fn quantize_u8_every_input_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        dither in prop_oneof![Just(Dither::Off), Just(Dither::Tpdf)],
        seed in any::<u64>(),
    ) {
        let params = QuantizeParams::new(dither, seed);
        let out_a = quantize_u8(img_a.view(), &params)?;
        let out_b = quantize_u8(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert_eq!(out_b.dim(), img_b.dim());

        // P5, adapted for the u8 output type.
        let reversed = img_a.slice(s![..;-1, ..;-1, ..]);
        let compact = reversed.to_owned();
        let la = quantize_u8(reversed, &params)?;
        let lb = quantize_u8(compact.view(), &params)?;
        prop_assert_eq!(la, lb, "layout changed the quantised output");

        // P6.
        let d1 = quantize_u8(img_a.view(), &params)?;
        let d2 = quantize_u8(img_a.view(), &params)?;
        prop_assert_eq!(d1, d2, "same input, params and seed produced different bytes");
    }

    #[test]
    fn quantize_u16_every_input_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        dither in prop_oneof![Just(Dither::Off), Just(Dither::Tpdf)],
        seed in any::<u64>(),
    ) {
        let params = QuantizeParams::new(dither, seed);
        let out_a = quantize_u16(img_a.view(), &params)?;
        let out_b = quantize_u16(img_b.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        prop_assert_eq!(out_b.dim(), img_b.dim());

        let reversed = img_a.slice(s![..;-1, ..;-1, ..]);
        let compact = reversed.to_owned();
        let la = quantize_u16(reversed, &params)?;
        let lb = quantize_u16(compact.view(), &params)?;
        prop_assert_eq!(la, lb, "layout changed the quantised output");

        let d1 = quantize_u16(img_a.view(), &params)?;
        let d2 = quantize_u16(img_a.view(), &params)?;
        prop_assert_eq!(d1, d2, "same input, params and seed produced different bytes");
    }

    /// P6, the seed half: with dither on and a non-degenerate ramp, two
    /// different seeds must give different bytes somewhere.
    #[test]
    fn quantize_u8_different_seeds_differ(
        vals in prop::collection::vec(0.0f32..=1.0f32, 64..2000),
        seed_a in any::<u64>(), seed_b in any::<u64>(),
    ) {
        prop_assume!(seed_a != seed_b);
        let n = vals.len();
        let img = mk_image(1, n, 1, vals);
        let a = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, seed_a))?;
        let b = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, seed_b))?;
        prop_assert_ne!(a, b, "two different dither seeds produced identical bytes");
    }
}

// ── histogram ─────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    /// `bins` is kept small here for runtime (a valid call actually
    /// allocates and zeroes a `channels x bins` accumulator); the
    /// near-`MAX_BINS` and oversized-accumulator edges are exercised
    /// separately below, at a deliberately low weight.
    #[test]
    fn histogram_valid_params_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        bins in 2u32..=512u32,
        lo in -1.0e4f32..1.0e4f32,
        span in 1.0e-3f32..1.0e4f32,
    ) {
        let params = HistogramParams::new(bins, lo, lo + span);
        let ha = histogram(img_a.view(), &params)?;
        let hb = histogram(img_b.view(), &params)?;
        prop_assert_eq!(ha.bins, bins as usize);
        prop_assert_eq!(ha.channels, img_a.dim().2);
        prop_assert_eq!(ha.counts().dim(), (img_a.dim().2, bins as usize));
        for ch in 0..ha.channels {
            prop_assert_eq!(ha.total(ch), (img_a.dim().0 * img_a.dim().1) as u64);
        }
        for ch in 0..hb.channels {
            prop_assert_eq!(hb.total(ch), (img_b.dim().0 * img_b.dim().1) as u64);
        }

        // P5: same counts regardless of physical layout.
        let reversed = img_a.slice(s![..;-1, ..;-1, ..]);
        let compact = reversed.to_owned();
        let la = histogram(reversed, &params)?;
        let lb = histogram(compact.view(), &params)?;
        prop_assert_eq!(la.counts().clone(), lb.counts().clone());
        prop_assert_eq!(la.below().to_vec(), lb.below().to_vec());
        prop_assert_eq!(la.above().to_vec(), lb.above().to_vec());
        prop_assert_eq!(la.non_finite().to_vec(), lb.non_finite().to_vec());

        // P6: two calls agree.
        let d2 = histogram(img_a.view(), &params)?;
        prop_assert_eq!(ha.counts().clone(), d2.counts().clone());
        prop_assert_eq!(ha.below().to_vec(), d2.below().to_vec());
        prop_assert_eq!(ha.above().to_vec(), d2.above().to_vec());
    }

    /// P2 + P3b over `bins`: below 2, or above `MAX_BINS`.
    #[test]
    fn histogram_bad_bins_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        bins in prop_oneof![
            1 => 0u32..2u32,
            1 => (histogram_mod::MAX_BINS + 1)..(histogram_mod::MAX_BINS + 1_000_000),
        ],
    ) {
        let params = HistogramParams::new(bins, 0.0, 1.0);
        let ra = catch_call(|| histogram(img_a.view(), &params));
        let rb = catch_call(|| histogram(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "bins")?;
        let msg_b = expect_rejected_message(rb, "bins")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over the counted range: non-finite bounds, `max <= min`,
    /// or a span wider than `f32` can represent.
    #[test]
    fn histogram_bad_range_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        case in 0u8..4,
        a in non_finite_f32(),
    ) {
        let params = match case {
            0 => HistogramParams::new(256, a, 1.0),
            1 => HistogramParams::new(256, 0.0, a),
            2 => HistogramParams::new(256, 1.0, 0.0), // max <= min
            _ => HistogramParams::new(256, f32::MIN, f32::MAX), // span overflows
        };
        let ra = catch_call(|| histogram(img_a.view(), &params));
        let rb = catch_call(|| histogram(img_b.view(), &params));
        let msg_a = expect_rejected_message(ra, "range")?;
        let msg_b = expect_rejected_message(rb, "range")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2: the accumulator-size guard is a function of *shape and
    /// params together* (`channels x (bins + 3) x 8 bytes`), not either
    /// alone — both `bins` and `channels` are individually inside their
    /// own valid range here, but the product exceeds
    /// `MAX_ACCUMULATOR_BYTES` (256 MiB). This is the one histogram
    /// rejection that is `Parameter`, not `Shape`, even though it depends
    /// on the image's channel count — confirmed by reading
    /// `histogram::validate_shape` directly.
    #[test]
    fn histogram_oversized_accumulator_is_rejected(
        channels in 4_000usize..8_000usize,
        bins in 10_000u32..50_000u32,
    ) {
        prop_assert!(
            (channels as u64) * (u64::from(bins) + 3) * 8 > (256u64 << 20),
            "test bug: this combination must actually exceed the accumulator budget"
        );
        let img = Array3::<f32>::zeros((1, 1, channels));
        let params = HistogramParams::new(bins, 0.0, 1.0);
        let result = catch_call(|| histogram(img.view(), &params));
        let msg = expect_rejected_message(result, "accumulator")?;
        prop_assert!(!msg.is_empty());
    }
}

// ── apply_lut ─────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn apply_lut_valid_table_is_accepted(
        (img_a, img_b) in any_c_image_pair(),
        table in prop::collection::vec(-1.0e6f32..1.0e6f32, 2..64),
        lo in -1.0e4f32..1.0e4f32,
        span in 1.0e-3f32..1.0e4f32,
    ) {
        let lut = Array1::from_vec(table);
        let params = LutParams::new(lo, lo + span);
        let out_a = apply_lut(img_a.view(), lut.view(), &params)?;
        let out_b = apply_lut(img_b.view(), lut.view(), &params)?;
        prop_assert_eq!(out_a.dim(), img_a.dim());
        // The interpolation is a convex combination of two finite table
        // entries (`src/lut.rs`: written that way specifically so it
        // cannot overflow between finite knots, however far apart), so
        // finite table + finite domain really does mean finite output
        // here, with no magnitude caveat needed.
        prop_assert!(out_a.iter().all(|v| v.is_finite()));
        prop_assert!(out_b.iter().all(|v| v.is_finite()));

        assert_layout_agnostic(&img_a, |v| apply_lut(v, lut.view(), &params))?;
        assert_deterministic_f32(|| apply_lut(img_a.view(), lut.view(), &params))?;
    }

    /// P2 + P3b: fewer than 2 table entries.
    #[test]
    fn apply_lut_short_table_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        table in prop::collection::vec(-1.0f32..1.0f32, 0..2),
    ) {
        let lut = Array1::from_vec(table);
        let params = LutParams::default();
        let ra = catch_call(|| apply_lut(img_a.view(), lut.view(), &params));
        let rb = catch_call(|| apply_lut(img_b.view(), lut.view(), &params));
        let msg_a = expect_rejected_message(ra, "lut has")?;
        let msg_b = expect_rejected_message(rb, "lut has")?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b: a non-finite table entry — the message names the index.
    #[test]
    fn apply_lut_non_finite_entry_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        len in 2usize..16,
        bad_index in 0usize..16,
        bad in non_finite_f32(),
    ) {
        let idx = bad_index % len;
        let mut table = vec![0.0_f32; len];
        table[idx] = bad;
        let lut = Array1::from_vec(table);
        let params = LutParams::default();
        let ra = catch_call(|| apply_lut(img_a.view(), lut.view(), &params));
        let rb = catch_call(|| apply_lut(img_b.view(), lut.view(), &params));
        let msg_a = expect_rejected_message(ra, &format!("lut[{idx}]"))?;
        let msg_b = expect_rejected_message(rb, &format!("lut[{idx}]"))?;
        prop_assert_eq!(msg_a, msg_b);
    }

    /// P2 + P3b over the domain: non-finite bounds, `max <= min`, or an
    /// unrepresentable span.
    #[test]
    fn apply_lut_bad_domain_is_rejected(
        (img_a, img_b) in any_c_image_pair(),
        case in 0u8..4,
        a in non_finite_f32(),
    ) {
        let params = match case {
            0 => LutParams::new(a, 1.0),
            1 => LutParams::new(0.0, a),
            2 => LutParams::new(1.0, 0.0),
            _ => LutParams::new(f32::MIN, f32::MAX),
        };
        let lut = Array1::from_vec(vec![0.0_f32, 1.0]);
        let ra = catch_call(|| apply_lut(img_a.view(), lut.view(), &params));
        let rb = catch_call(|| apply_lut(img_b.view(), lut.view(), &params));
        let msg_a = expect_rejected_message(ra, "domain")?;
        let msg_b = expect_rejected_message(rb, "domain")?;
        prop_assert_eq!(msg_a, msg_b);
    }
}
