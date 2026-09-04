// SPDX-License-Identifier: GPL-3.0-or-later
//! CPU-vs-CUDA conformance tests (built with `--features cuda`).
//!
//! **The CPU implementation is the specification.** Every GPU kernel is
//! tested against the CPU function that already passes the crate's unit
//! and integration suites — never the other way round, and never against
//! committed golden images (those would turn a driver update into a red
//! run).
//!
//! Without a device the whole suite skips, so `cargo test --features
//! cuda` stays green on GPU-less machines. Set `PHAIOS_REQUIRE_GPU=1`
//! to turn a missing device into a failure — the local pre-merge gate,
//! so a broken driver cannot masquerade as a passing run.

#![cfg(feature = "cuda")]

use ndarray::{Array3, s};
use phaios_core::cuda;

/// Open device 0, or skip (or fail under `PHAIOS_REQUIRE_GPU`).
fn try_context() -> Option<cuda::Context> {
    match cuda::Context::new(0) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            if std::env::var_os("PHAIOS_REQUIRE_GPU").is_some() {
                panic!("PHAIOS_REQUIRE_GPU is set but no usable device: {e}");
            }
            eprintln!("skip: no usable CUDA device ({e})");
            None
        }
    }
}

/// Deterministic pseudo-random image, same generator as the benches.
fn pseudo_random_image(h: usize, w: usize, c: usize) -> Array3<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    Array3::from_shape_simple_fn((h, w, c), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0
    })
}

// ── exposure: the bit-exact kernel ───────────────────────────────────────────

/// GPU exposure must agree with CPU exposure to the last bit.
///
/// The kernel is one correctly-rounded IEEE-754 multiply, and the gain
/// is computed by the same host code — so there is no tolerance here,
/// and there must never be one. Failure means the *stack* is broken.
#[test]
fn exposure_is_bit_exact_across_shapes() {
    let Some(ctx) = try_context() else { return };

    // Ragged sizes on purpose: last-block bounds handling is where
    // launch-geometry bugs live. 1021 and 733 are prime.
    for (h, w, c) in [
        (64, 64, 1),
        (64, 64, 3),
        (1021, 733, 3),
        (1, 1, 1),
        (1, 4096, 1),
        (517, 1, 3),
    ] {
        let img = pseudo_random_image(h, w, c);
        for stops in [-3.0_f32, -0.75, 0.0, 0.5, 4.0] {
            let cpu = phaios_core::exposure::exposure(img.view(), stops).unwrap();
            let gpu = cuda::kernels::exposure(&ctx, img.view(), stops).unwrap();
            assert_eq!(
                cpu, gpu,
                "CPU and GPU disagree at ({h}, {w}, {c}), stops {stops}"
            );
        }
    }
}

/// Strided, Fortran-order and reversed views agree with their
/// contiguous copies — the same layout contract every CPU kernel obeys.
#[test]
fn exposure_is_layout_agnostic() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(64, 48, 3);

    let views = [
        img.slice(s![..;2, .., ..]),
        img.slice(s![.., ..;3, ..]),
        img.slice(s![..;-1, .., ..]),
    ];
    for view in views {
        let gpu = cuda::kernels::exposure(&ctx, view, 1.25).unwrap();
        let cpu = phaios_core::exposure::exposure(view, 1.25).unwrap();
        assert_eq!(cpu, gpu, "layout handling diverged");
        assert!(gpu.is_standard_layout(), "output must be C-contiguous");
    }
}

/// Eight runs, one byte pattern — within-backend determinism, the
/// assertion that must never be relaxed.
#[test]
fn exposure_is_deterministic_within_the_backend() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(256, 256, 3);
    let first = cuda::kernels::exposure(&ctx, img.view(), 0.7).unwrap();
    for run in 1..8 {
        let again = cuda::kernels::exposure(&ctx, img.view(), 0.7).unwrap();
        assert_eq!(first, again, "run {run} differed");
    }
}

/// Two independently created contexts on the same device agree.
#[test]
fn exposure_agrees_across_contexts() {
    let Some(ctx_a) = try_context() else { return };
    let ctx_b = cuda::Context::new(0).expect("second context on the same device");
    let img = pseudo_random_image(128, 128, 1);
    let a = cuda::kernels::exposure(&ctx_a, img.view(), -1.5).unwrap();
    let b = cuda::kernels::exposure(&ctx_b, img.view(), -1.5).unwrap();
    assert_eq!(a, b, "two contexts disagreed");
}

/// The GPU kernel rejects exactly what the CPU kernel rejects, with the
/// identical message — both call the same `validate`.
#[test]
fn exposure_validation_is_shared_with_the_cpu() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(4, 4, 1);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let cpu_err = phaios_core::exposure::exposure(img.view(), bad).unwrap_err();
        let gpu_err = cuda::kernels::exposure(&ctx, img.view(), bad).unwrap_err();
        assert_eq!(
            cpu_err.to_string(),
            gpu_err.to_string(),
            "error messages diverged for stops = {bad}"
        );
    }
}

/// Zero-size input: same contract as the CPU (empty array back).
#[test]
fn exposure_accepts_empty_input() {
    let Some(ctx) = try_context() else { return };
    let img = Array3::<f32>::zeros((0, 8, 1));
    let out = cuda::kernels::exposure(&ctx, img.view(), 1.0).unwrap();
    assert_eq!(out.dim(), (0, 8, 1));
}

// ── local_contrast: the reformulated kernel ──────────────────────────────────

/// Worst violation of the `|x − y| <= atol + rtol·|x|` bound, as a
/// multiple of the bound. <= 1.0 means every element passes. The same
/// two-term form numpy's `assert_allclose` uses: a pure relative metric
/// punishes outputs that legitimately cross zero (out = L + s·(L − q)
/// does), where a few-ULP absolute difference is a huge ratio.
///
/// Non-finite values are compared explicitly rather than arithmetically,
/// and the shapes are asserted rather than zipped. Both matter: this was
/// `.zip(...).fold(0.0, f32::max)`, and `f32::max` returns the *other*
/// operand when one is NaN, so every NaN violation was silently dropped.
/// A GPU kernel returning nothing but NaN scored 0.000 — a perfect
/// match — as did an empty output, or one truncated to a single element,
/// because `zip` stops at the shorter side. That is the oracle behind
/// every bounded assertion in this file.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32) -> f32 {
    assert_eq!(
        a.dim(),
        b.dim(),
        "worst_violation: shape mismatch, {:?} against {:?}",
        a.dim(),
        b.dim()
    );
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let v = if x.is_nan() || y.is_nan() {
            // Agreeing on NaN is agreement; disagreeing about whether a
            // value is NaN at all is total disagreement, not zero.
            if x.is_nan() && y.is_nan() {
                0.0
            } else {
                f32::INFINITY
            }
        } else if x.is_infinite() || y.is_infinite() {
            if x == y { 0.0 } else { f32::INFINITY }
        } else {
            (x - y).abs() / (atol + rtol * x.abs())
        };
        // Plain `>`, so a NaN could not sneak through here either.
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// The oracle needs its own test, because a lenient oracle silently
/// weakens every assertion built on it — which is exactly what happened.
#[test]
fn worst_violation_does_not_score_garbage_as_agreement() {
    let ok = Array3::from_shape_vec((1, 3, 1), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let nan = Array3::from_shape_vec((1, 3, 1), vec![f32::NAN; 3]).unwrap();
    let inf = Array3::from_shape_vec((1, 3, 1), vec![f32::INFINITY; 3]).unwrap();

    assert_eq!(worst_violation(&ok, &ok, 1e-5, 1e-7), 0.0);
    assert_eq!(
        worst_violation(&nan, &nan, 1e-5, 1e-7),
        0.0,
        "NaN agrees with NaN"
    );
    assert_eq!(
        worst_violation(&inf, &inf, 1e-5, 1e-7),
        0.0,
        "+inf agrees with +inf"
    );

    // Each of these scored 0.0 before.
    assert!(
        worst_violation(&ok, &nan, 1e-5, 1e-7).is_infinite(),
        "an all-NaN output must not read as agreement"
    );
    assert!(
        worst_violation(&nan, &ok, 1e-5, 1e-7).is_infinite(),
        "the NaN side being the reference must not read as agreement either"
    );
    assert!(
        worst_violation(&ok, &inf, 1e-5, 1e-7).is_infinite(),
        "a finite/infinite mismatch must not read as agreement"
    );

    // A real numerical difference still scores as before.
    let off = Array3::from_shape_vec((1, 3, 1), vec![1.0_f32, 2.0, 3.5]).unwrap();
    assert!(worst_violation(&ok, &off, 1e-5, 1e-7) > 1000.0);
}

/// GPU guided filter agrees with the CPU oracle within 1e-4 relative.
///
/// Not bit-exact by design: the CPU uses global f64 summed-area tables,
/// the GPU uses separable f32 box filters with Kahan compensation — a
/// documented algorithm reformulation (`docs/ffi.md` §6). The committed
/// bound is what makes a driver-update regression visible.
#[test]
fn local_contrast_agrees_with_cpu_oracle() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(517, 733, 1);

    for (radius, eps, strength) in [
        (8_u32, 0.01_f32, 0.5_f32),
        (3, 0.001, 1.0),
        (1, 0.1, 2.0),
        (64, 0.01, 0.7),
        (9_999, 0.1, 0.7), // window degenerates to the whole image
    ] {
        let params = phaios_core::local_contrast::GuidedFilterParams::new(radius, eps);
        let cpu =
            phaios_core::local_contrast::local_contrast(img.view(), &params, strength).unwrap();
        let gpu = cuda::kernels::local_contrast(&ctx, img.view(), &params, strength).unwrap();
        // Committed bound: rtol 1e-4, atol 1e-6 — the same numbers the
        // Python suite asserts through the FFI.
        let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
        assert!(
            v <= 1.0,
            "r={radius} eps={eps} s={strength}: worst element at {v:.2}x the (1e-4, 1e-6) bound"
        );
    }
}

/// The same agreement on input that spans a scene's dynamic range.
///
/// The guided filter's variance is `mean(L²) − mean(L)²`, a subtraction of
/// two nearly equal large numbers whose true difference can be many orders
/// smaller than either. The device computed it in f32 from
/// Kahan-compensated f32 partial sums, and Kahan does nothing for a
/// cancelling difference — it compensates a *sum*. On a uniformly bright
/// region, where the true variance is near zero and both terms are ~1e8,
/// the f32 result was noise, `a = var/(var+ε)` then read that noise as an
/// edge, and the filter smoothed where it should have sharpened.
///
/// Measured at 51.8× the committed bound before the L/L² path moved to
/// f64. Nothing caught it because every other case here draws from
/// `pseudo_random_image`, i.e. `[0, 1)`, where `mean(L²)` and `mean(L)²`
/// are both O(1) and the cancellation is harmless.
#[test]
fn local_contrast_agrees_within_bound_across_the_dynamic_range() {
    use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};

    let Some(ctx) = try_context() else { return };

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let mut img = ndarray::Array3::<f32>::from_elem((81, 97, 1), 1e-4);
        img.slice_mut(ndarray::s![30..34, 10..80, ..])
            .fill(highlight);

        // Small radii and a large ε are the worst case: the window sits
        // wholly inside the bright bar, so the true variance really is
        // near zero and the cancellation has nothing left to stand on.
        for radius in [1_u32, 2, 8, 32] {
            for eps in [1e-4_f32, 0.01, 0.5] {
                let params = GuidedFilterParams::new(radius, eps);
                let cpu = local_contrast(img.view(), &params, 0.5).unwrap();
                let gpu = cuda::kernels::local_contrast(&ctx, img.view(), &params, 0.5).unwrap();
                let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
                assert!(
                    v <= 1.0,
                    "r={radius} eps={eps} with a {highlight:e} highlight: \
                     {v:.4}x the (1e-4, 1e-6) bound"
                );
            }
        }
    }
}

/// radius = 0 makes the filter the identity — and because no windowed
/// arithmetic happens at all in that configuration, CPU and GPU agree
/// bit-for-bit, not merely closely.
#[test]
fn local_contrast_radius_zero_is_bit_exact_identity() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(64, 64, 1);
    let params = phaios_core::local_contrast::GuidedFilterParams::new(0, 0.0);
    let gpu = cuda::kernels::local_contrast(&ctx, img.view(), &params, 1.0).unwrap();
    assert_eq!(gpu, img, "radius 0 must be the exact identity");
}

/// The property tests the CPU kernel already answers to, re-asserted on
/// GPU output directly — a tolerance comparison would forgive a kernel
/// that is wrong the same way on both sides; these do not.
#[test]
fn local_contrast_gpu_output_satisfies_cpu_properties() {
    let Some(ctx) = try_context() else { return };
    let params = phaios_core::local_contrast::GuidedFilterParams::new(4, 0.01);

    // Constant image: detail is zero everywhere, output == input.
    let flat = Array3::from_elem((32, 32, 1), 0.3_f32);
    let out = cuda::kernels::local_contrast(&ctx, flat.view(), &params, 0.5).unwrap();
    for &v in out.iter() {
        assert!((v - 0.3).abs() < 1e-4, "constant image changed: {v}");
    }

    // Large constant: the f32 cancellation regime, and a worked example
    // of the documented f32-vs-f64 difference. One f32 ULP at 1e7 is
    // exactly 1.0, and the GPU's f32 means land within a couple of ULP
    // of the value — where the CPU's f64 tables hold it to < 1 ULP. The
    // GPU bound is therefore 4 ULP at this magnitude, not the CPU's
    // sub-ULP 1.0.
    let big = Array3::from_elem((128, 128, 1), 1.0e7_f32);
    let params8 = phaios_core::local_contrast::GuidedFilterParams::new(8, 0.01);
    let out = cuda::kernels::local_contrast(&ctx, big.view(), &params8, 1.0).unwrap();
    for &v in out.iter() {
        assert!(
            (v - 1.0e7).abs() <= 4.0,
            "1e7 constant: got {v} (> 4 ULP off)"
        );
    }
}

/// Eight runs, one byte pattern.
#[test]
fn local_contrast_is_deterministic_within_the_backend() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(256, 256, 1);
    let params = phaios_core::local_contrast::GuidedFilterParams::new(8, 0.01);
    let first = cuda::kernels::local_contrast(&ctx, img.view(), &params, 0.5).unwrap();
    for run in 1..8 {
        let again = cuda::kernels::local_contrast(&ctx, img.view(), &params, 0.5).unwrap();
        assert_eq!(first, again, "run {run} differed");
    }
}

/// Strided input agrees with its contiguous copy.
#[test]
fn local_contrast_is_layout_agnostic() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(64, 48, 1);
    let view = img.slice(s![..;2, ..;3, ..]);
    let params = phaios_core::local_contrast::GuidedFilterParams::new(3, 0.01);
    let gpu = cuda::kernels::local_contrast(&ctx, view, &params, 0.7).unwrap();
    let contiguous = view.to_owned();
    let gpu_c = cuda::kernels::local_contrast(&ctx, contiguous.view(), &params, 0.7).unwrap();
    assert_eq!(gpu, gpu_c, "strided and contiguous inputs diverged");
}

/// Validation parity with the CPU, message for message.
#[test]
fn local_contrast_validation_is_shared_with_the_cpu() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(4, 4, 1);
    let rgb = pseudo_random_image(4, 4, 3);

    let bad_eps = phaios_core::local_contrast::GuidedFilterParams::new(4, -0.01);
    let good = phaios_core::local_contrast::GuidedFilterParams::new(4, 0.01);

    let cases: [(
        &Array3<f32>,
        &phaios_core::local_contrast::GuidedFilterParams,
        f32,
    ); 3] = [
        (&rgb, &good, 0.5),           // wrong channel count
        (&img, &bad_eps, 0.5),        // negative eps
        (&img, &good, f32::INFINITY), // non-finite strength
    ];
    for (input, params, strength) in cases {
        let cpu_err = phaios_core::local_contrast::local_contrast(input.view(), params, strength)
            .unwrap_err();
        let gpu_err =
            cuda::kernels::local_contrast(&ctx, input.view(), params, strength).unwrap_err();
        assert_eq!(cpu_err.to_string(), gpu_err.to_string());
    }
}

// ── the element-wise kernels ─────────────────────────────────────────────────

/// vignette is bit-exact: every operation involved (mul, add, div,
/// sqrt, min/max) is correctly rounded on both sides and the PTX build
/// disables FMA contraction.
#[test]
fn vignette_is_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(517, 733, 3);
    for (amount, feather, roundness) in [
        (0.5_f32, 1.0_f32, 0.0_f32),
        (0.5, 0.2, 0.0),
        (0.5, 1.0, 1.0),
        (-0.5, 0.7, 0.3),
        (0.0, 0.5, 0.0), // identity fast path
        (4.0, 1.0, 0.0), // clamp-to-zero regime
        (0.5, 0.0, 0.5), // degenerate feather -> hard step
    ] {
        let params = phaios_core::vignette::VignetteParams::new(amount, feather, roundness);
        let cpu = phaios_core::vignette::vignette(img.view(), &params).unwrap();
        let gpu = cuda::kernels::vignette(&ctx, img.view(), &params).unwrap();
        assert_eq!(
            cpu, gpu,
            "vignette diverged at ({amount}, {feather}, {roundness})"
        );
    }
}

/// luminance_bw is bit-exact: a left-to-right dot product with no FMA.
#[test]
fn luminance_bw_is_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(521, 733, 3);
    for standard in [
        phaios_core::bw::LuminanceStandard::Bt601,
        phaios_core::bw::LuminanceStandard::Bt709,
        phaios_core::bw::LuminanceStandard::Bt2020,
    ] {
        let cpu = phaios_core::bw::luminance_bw(img.view(), standard).unwrap();
        let gpu = cuda::kernels::luminance_bw(&ctx, img.view(), standard).unwrap();
        assert_eq!(cpu, gpu, "luminance_bw diverged for {standard:?}");
        assert_eq!(gpu.dim().2, 1, "channel collapse");
    }
}

/// tone_curve: the identity and power == 1 paths are bit-exact; the
/// general path carries one powf and is bounded.
#[test]
fn tone_curve_exact_paths_and_powf_bound() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(517, 733, 1);

    for (slope, offset) in [(1.0_f32, 0.0_f32), (1.4, 0.05), (0.8, -0.02)] {
        let params = phaios_core::tone::ToneCurveParams::new(slope, offset, 1.0);
        let cpu = phaios_core::tone::tone_curve(img.view(), &params).unwrap();
        let gpu = cuda::kernels::tone_curve(&ctx, img.view(), &params).unwrap();
        assert_eq!(cpu, gpu, "power == 1 path diverged at ({slope}, {offset})");
    }

    for (slope, offset, power) in [
        (1.0_f32, 0.0_f32, 0.7_f32),
        (1.15, 0.01, 0.85),
        (1.3, -0.05, 1.7),
    ] {
        let params = phaios_core::tone::ToneCurveParams::new(slope, offset, power);
        let cpu = phaios_core::tone::tone_curve(img.view(), &params).unwrap();
        let gpu = cuda::kernels::tone_curve(&ctx, img.view(), &params).unwrap();
        let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
        assert!(
            v <= 1.0,
            "powf path at ({slope}, {offset}, {power}): {v:.2}x the (1e-5, 1e-7) bound"
        );
    }
}

/// encode_srgb: the linear branch is exact; the power branch carries
/// one powf and is bounded.
#[test]
fn encode_srgb_agrees_within_powf_bound() {
    let Some(ctx) = try_context() else { return };
    // Sweep through both branches, negatives included.
    let img =
        ndarray::Array3::from_shape_fn((1, 8192, 1), |(_, x, _)| (x as f32 / 8192.0) * 1.6 - 0.1);
    let cpu = phaios_core::encode::encode_srgb(img.view()).unwrap();
    let gpu = cuda::kernels::encode_srgb(&ctx, img.view()).unwrap();
    let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
    assert!(v <= 1.0, "encode_srgb: {v:.2}x the (1e-5, 1e-7) bound");
}

// ── the stage-D kernels ──────────────────────────────────────────────────────

/// The two other classic B&W conversions share luminance's dot-product
/// kernel and are bit-exact.
#[test]
fn mixer_and_filter_are_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3);
    // The last two leave the -2..+2 the doc calls conventional: every
    // weight used anywhere else in the crate is within |1.2|, so a clamp
    // introduced on one backend and not the other would show up only out
    // here.
    for weights in [
        [1.0_f32, 0.0, 0.0],
        [0.3, 0.59, 0.11],
        [-0.5, 1.2, 0.3],
        [3.0, -2.5, 0.4],
        [10.0, -10.0, 10.0],
    ] {
        let cpu = phaios_core::bw::channel_mixer_bw(img.view(), weights).unwrap();
        let gpu = cuda::kernels::channel_mixer_bw(&ctx, img.view(), weights).unwrap();
        assert_eq!(cpu, gpu, "channel_mixer diverged at {weights:?}");
    }
    for filter in [
        phaios_core::bw::ColorFilter::Yellow8K2,
        phaios_core::bw::ColorFilter::Red25A,
        phaios_core::bw::ColorFilter::Blue47C5,
    ] {
        let cpu = phaios_core::bw::color_filter_bw(img.view(), filter, Default::default()).unwrap();
        let gpu =
            cuda::kernels::color_filter_bw(&ctx, img.view(), filter, Default::default()).unwrap();
        assert_eq!(cpu, gpu, "color_filter diverged at {filter:?}");
    }
}

/// hsl_bw carries eight expf terms; bounded. The neutral-pixel property
/// (zero chroma → weights ignored) is exact and asserted separately.
#[test]
fn hsl_bw_agrees_within_expf_bound() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3);
    let params = phaios_core::bw::HslWeightedParams::new(
        [0.3, -0.2, 0.5, 0.1, 0.0, -0.6, 0.2, -0.1],
        Default::default(),
        30.0,
    );
    let cpu = phaios_core::bw::hsl_bw(img.view(), &params).unwrap();
    let gpu = cuda::kernels::hsl_bw(&ctx, img.view(), &params).unwrap();
    let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
    assert!(v <= 1.0, "hsl_bw: {v:.2}x the (1e-5, 1e-7) bound");

    // Neutral pixels bypass every transcendental: exact.
    let grey = Array3::from_elem((16, 16, 3), 0.5_f32);
    let cpu = phaios_core::bw::hsl_bw(grey.view(), &params).unwrap();
    let gpu = cuda::kernels::hsl_bw(&ctx, grey.view(), &params).unwrap();
    assert_eq!(cpu, gpu, "neutral pixels must be exact");
}

/// zone_system: log2f/expf/powf bound the general case; the empty-map
/// identity is bit-exact (device copy, as the CPU assigns through).
#[test]
fn zone_system_agrees_and_identity_is_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 1);

    let mut offsets = std::collections::HashMap::new();
    for (z, o) in [(0, -0.7_f32), (3, 0.9), (5, 0.6), (7, 0.45), (10, 0.55)] {
        offsets.insert(z, o);
    }
    let params = phaios_core::tone::ZoneParams::new(offsets);
    let cpu = phaios_core::tone::zone_system(img.view(), &params).unwrap();
    let gpu = cuda::kernels::zone_system(&ctx, img.view(), &params).unwrap();
    let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
    assert!(v <= 1.0, "zone_system: {v:.2}x the (1e-5, 1e-7) bound");

    let empty = phaios_core::tone::ZoneParams::default();
    let gpu = cuda::kernels::zone_system(&ctx, img.view(), &empty).unwrap();
    assert_eq!(gpu, img, "empty offsets must be the exact identity");
}

/// split_toning: cbrtf bounds the general case; shape restored 1 → 3.
#[test]
fn split_toning_agrees_within_cbrt_bound() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 1);
    let params = phaios_core::split_toning::SplitToningParams::new(
        [0.0, -0.02, -0.05],
        [0.0, 0.03, 0.04],
        0.5,
        0.2,
    );
    let cpu = phaios_core::split_toning::split_toning(img.view(), &params).unwrap();
    let gpu = cuda::kernels::split_toning(&ctx, img.view(), &params).unwrap();
    assert_eq!(gpu.dim().2, 3, "channel restoration");
    let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
    assert!(v <= 1.0, "split_toning: {v:.2}x the (1e-5, 1e-7) bound");
}

// ── film_grain: the integer-exactness kernel ─────────────────────────────────

/// The device-side splitmix64 pixel hash is bit-identical to the CPU's
/// over 2²⁰ coordinates. Integer arithmetic has no tolerance and gets
/// none: a single differing bit here means the grain field is a
/// different image, not a slightly different one.
#[test]
fn grain_hash_is_bit_exact_over_2_20_coordinates() {
    let Some(ctx) = try_context() else { return };
    let (h, w) = (1024_usize, 1024_usize); // 2^20 coordinates
    for seed in [0_u64, 20_260_815, u64::MAX] {
        let gpu = cuda::kernels::hash_grid(&ctx, seed, h, w).unwrap();
        for y in 0..h {
            for x in 0..w {
                let cpu = phaios_core::film_grain::pixel_hash(seed, x as u64, y as u64);
                assert_eq!(
                    gpu[y * w + x],
                    cpu,
                    "hash diverged at seed {seed}, ({x}, {y})"
                );
            }
        }
    }
}

/// film_grain agrees with the CPU oracle. The hash is exact; Box–Muller
/// and the box filters carry logf/sqrtf/cosf plus f32-vs-f64 window
/// sums, so agreement is bounded, far inside the plan's 1e-3.
#[test]
fn film_grain_agrees_with_cpu_oracle() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(517, 733, 1);
    for (intensity, size, seed) in [
        (0.3_f32, 2.0_f32, 12_345_u64),
        (0.15, 1.0, 1),
        (0.15, 4.0, 99),
        (1.0, 8.0, 7),
    ] {
        let params = phaios_core::film_grain::GrainParams::new(intensity, size, seed);
        let cpu = phaios_core::film_grain::film_grain(img.view(), &params).unwrap();
        let gpu = cuda::kernels::film_grain(&ctx, img.view(), &params).unwrap();
        let v = worst_violation(&cpu, &gpu, 1e-3, 1e-5);
        assert!(
            v <= 1.0,
            "grain at ({intensity}, {size}, {seed}): {v:.2}x the (1e-3, 1e-5) bound"
        );
    }
}

/// The grain properties that must hold exactly, on GPU output directly:
/// zero intensity is a bit-exact identity, and the envelope silences
/// grain completely at L = 0, 1 and above 1.
#[test]
fn film_grain_gpu_properties_hold_exactly() {
    let Some(ctx) = try_context() else { return };

    let img = pseudo_random_image(64, 64, 1);
    let identity = phaios_core::film_grain::GrainParams::new(0.0, 2.0, 42);
    let gpu = cuda::kernels::film_grain(&ctx, img.view(), &identity).unwrap();
    assert_eq!(gpu, img, "zero intensity must be the exact identity");

    let params = phaios_core::film_grain::GrainParams::new(1.0, 2.0, 5);
    for level in [0.0_f32, 1.0, 2.5] {
        let flat = Array3::from_elem((32, 32, 1), level);
        let gpu = cuda::kernels::film_grain(&ctx, flat.view(), &params).unwrap();
        assert_eq!(gpu, flat, "grain leaked at L = {level}");
    }

    // Determinism: 8 runs, one byte pattern.
    let p = phaios_core::film_grain::GrainParams::new(0.4, 2.0, 99);
    let first = cuda::kernels::film_grain(&ctx, img.view(), &p).unwrap();
    for run in 1..8 {
        assert_eq!(
            cuda::kernels::film_grain(&ctx, img.view(), &p).unwrap(),
            first,
            "run {run} differed"
        );
    }
}

// ── geometry: bit-exact by construction ──────────────────────────────────────

/// Crop and all eight orientations are pure index permutations and must
/// agree with the CPU to the bit, including on a resident chain where
/// geometry runs first (its pipeline position).
#[test]
fn geometry_is_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3);

    for params in [
        phaios_core::geometry::CropParams::new(0, 0, 389, 257),
        phaios_core::geometry::CropParams::new(10, 20, 300, 200),
        phaios_core::geometry::CropParams::new(388, 256, 1, 1),
        phaios_core::geometry::CropParams::new(5, 5, 0, 0),
    ] {
        let cpu = phaios_core::geometry::crop(img.view(), &params).unwrap();
        let gpu = cuda::kernels::crop(&ctx, img.view(), &params).unwrap();
        assert_eq!(cpu, gpu, "crop diverged at {params:?}");
    }

    for value in 1..=8_u16 {
        let o = phaios_core::geometry::Orientation::from_exif(value).unwrap();
        let cpu = phaios_core::geometry::orient(img.view(), o).unwrap();
        let gpu = cuda::kernels::orient(&ctx, img.view(), o).unwrap();
        assert_eq!(cpu, gpu, "orient diverged at {o:?}");
    }

    // Resident chain: orient → crop → luminance → vignette, one upload.
    let cp = phaios_core::geometry::CropParams::new(8, 16, 200, 150);
    let vg = phaios_core::vignette::VignetteParams::new(0.4, 0.8, 0.1);
    let o = phaios_core::geometry::Orientation::Rotate90;

    let c = phaios_core::geometry::orient(img.view(), o).unwrap();
    let c = phaios_core::geometry::crop(c.view(), &cp).unwrap();
    let c = phaios_core::bw::luminance_bw(c.view(), Default::default()).unwrap();
    let cpu = phaios_core::vignette::vignette(c.view(), &vg).unwrap();

    let d = ctx.upload(img.view()).unwrap();
    let d = cuda::kernels::orient_device(&d, o).unwrap();
    let d = cuda::kernels::crop_device(&d, &cp).unwrap();
    let d = cuda::kernels::luminance_bw_device(&d, Default::default()).unwrap();
    let d = cuda::kernels::vignette_device(&d, &vg).unwrap();
    let gpu = ctx.download(&d).unwrap();

    assert_eq!(cpu, gpu, "geometry-first resident chain diverged");
}

/// resize and straighten are polynomial resampling with host-computed
/// transcendentals, engineered to match the CPU tap-for-tap: bit-exact.
#[test]
fn resampling_is_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3);

    use phaios_core::geometry::{ResizeFilter, ResizeParams, StraightenParams};
    for filter in [
        ResizeFilter::Area,
        ResizeFilter::Bilinear,
        ResizeFilter::CatmullRom,
    ] {
        for (w, h) in [(389_u32, 257_u32), (200, 130), (777, 500), (97, 311)] {
            let params = ResizeParams::new(w, h, filter);
            let cpu = phaios_core::geometry::resize(img.view(), &params).unwrap();
            let gpu = cuda::kernels::resize(&ctx, img.view(), &params).unwrap();
            assert_eq!(cpu, gpu, "resize diverged: {filter:?} {w}x{h}");
        }
    }

    for degrees in [0.0_f32, 1.5, -7.3, 30.0, -45.0] {
        let params = StraightenParams::new(degrees);
        let cpu = phaios_core::geometry::straighten(img.view(), &params).unwrap();
        let gpu = cuda::kernels::straighten(&ctx, img.view(), &params).unwrap();
        assert_eq!(cpu, gpu, "straighten diverged at {degrees} deg");
    }
}

// ── resampling accuracy: an f64 oracle, because the backends agree by
//    construction and therefore cannot audit each other ──────────────────────

/// Unit roundoff for `f32`: 2⁻²⁴. `f32::EPSILON` is 2⁻²³, twice this.
const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// `geometry::filter_eval`, transcribed. It is `pub(crate)`, and an
/// integration test only sees the public API — but the oracle below has
/// to use the *shipped* weights, bit for bit, so this is a copy on
/// purpose rather than a reimplementation.
fn filter_eval_f32(filter: phaios_core::geometry::ResizeFilter, t: f32) -> f32 {
    use phaios_core::geometry::ResizeFilter;
    let t = t.abs();
    match filter {
        ResizeFilter::Area => {
            if t <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        ResizeFilter::Bilinear => (1.0 - t).max(0.0),
        ResizeFilter::CatmullRom => {
            if t <= 1.0 {
                ((1.5 * t - 2.5) * t) * t + 1.0
            } else if t < 2.0 {
                ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
            } else {
                0.0
            }
        }
    }
}

/// The taps `(source index, weight)` the shipped kernel visits for
/// output index `i`, in its order — centre, support and every weight
/// computed in f32 exactly as `resample_axis1` does.
///
/// Sharing the geometry with the kernel is the point. Re-deriving the
/// sample positions in f64 would move them by ~`in_len·2⁻²⁴` of a pixel,
/// and on an edge that steps from 1e-4 to 1e8 that shift changes the
/// true answer by far more than any accumulator error — it would measure
/// the wrong thing.
fn resize_taps(
    filter: phaios_core::geometry::ResizeFilter,
    scale: f32,
    i: usize,
) -> Vec<(i64, f32)> {
    use phaios_core::geometry::ResizeFilter;
    let centre = (i as f32 + 0.5) * scale - 0.5;
    let denom = scale.max(1.0);
    let base = match filter {
        ResizeFilter::Area => 0.5_f32,
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::CatmullRom => 2.0,
    };
    let support = base * scale.max(1.0);
    let area_minify = filter == ResizeFilter::Area && scale > 1.0;
    let k0 = (centre - support).floor() as i64;
    let k1 = (centre + support).ceil() as i64;

    let mut taps = Vec::new();
    for k in k0..=k1 {
        let w = if area_minify {
            let lo = (k as f32 - 0.5).max(centre - scale * 0.5);
            let hi = (k as f32 + 0.5).min(centre + scale * 0.5);
            (hi - lo).max(0.0)
        } else {
            filter_eval_f32(filter, (k as f32 - centre) / denom)
        };
        if w != 0.0 {
            taps.push((k, w));
        }
    }
    taps
}

/// One pass of the separable resample along axis 1, accumulated in f64.
///
/// Carries a second plane, `mass`, holding Σ|wᵢ·xᵢ| / |Σwᵢ| — the
/// quantity the textbook forward-error bound for a floating-point dot
/// product is stated against. Propagating it through both passes gives
/// the composed operator's condition, which is what makes an assertion
/// on a cancelling output pixel meaningful at all.
///
/// Returns `(values, mass, worst tap count)`.
fn resample_axis1_f64(
    val: &ndarray::Array2<f64>,
    mass: &ndarray::Array2<f64>,
    scale: f32,
    filter: phaios_core::geometry::ResizeFilter,
    out_len: usize,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>, usize) {
    let (rows, in_len) = val.dim();
    let mut out_val = ndarray::Array2::<f64>::zeros((rows, out_len));
    let mut out_mass = ndarray::Array2::<f64>::zeros((rows, out_len));
    let mut worst_taps = 0_usize;

    for i in 0..out_len {
        let taps = resize_taps(filter, scale, i);
        worst_taps = worst_taps.max(taps.len());
        let wsum: f64 = taps.iter().map(|&(_, w)| f64::from(w)).sum();
        if wsum == 0.0 {
            continue; // the kernel writes 0.0 here; zeros already are.
        }
        for r in 0..rows {
            let mut acc = 0.0_f64;
            let mut m = 0.0_f64;
            for &(k, w) in &taps {
                let kc = k.clamp(0, in_len as i64 - 1) as usize;
                acc += f64::from(w) * val[[r, kc]];
                m += f64::from(w).abs() * mass[[r, kc]];
            }
            out_val[[r, i]] = acc / wsum;
            out_mass[[r, i]] = m / wsum.abs();
        }
    }
    (out_val, out_mass, worst_taps)
}

/// `geometry::resize` recomputed in f64: horizontal then vertical, the
/// shipped pass order and the shipped taps.
///
/// Returns `(values, mass, n)` where `n` is the number of rounding steps
/// the f32 kernel takes on the worst output pixel, so `n · u · mass`
/// is its forward-error bound (u = 2⁻²⁴).
/// The resample computed *entirely* in f64 — scale, tap geometry, weights
/// and accumulation — as a reference independent of the shipped kernel.
///
/// [`resize_f64_oracle`] deliberately reuses the kernel's f32 tap
/// geometry, so it isolates accumulation error and nothing else. This one
/// shares no arithmetic with the kernel at all, so what it measures is the
/// total distance from the mathematically intended answer, weight
/// evaluation included.
fn filter_eval_f64(filter: phaios_core::geometry::ResizeFilter, t: f64) -> f64 {
    use phaios_core::geometry::ResizeFilter;
    let t = t.abs();
    match filter {
        ResizeFilter::Area => {
            if t <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        ResizeFilter::Bilinear => (1.0 - t).max(0.0),
        ResizeFilter::CatmullRom => {
            if t <= 1.0 {
                ((1.5 * t - 2.5) * t) * t + 1.0
            } else if t < 2.0 {
                ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
            } else {
                0.0
            }
        }
    }
}

fn resize_taps_f64(
    filter: phaios_core::geometry::ResizeFilter,
    scale: f64,
    i: usize,
) -> Vec<(i64, f64)> {
    use phaios_core::geometry::ResizeFilter;
    let centre = (i as f64 + 0.5) * scale - 0.5;
    let denom = scale.max(1.0);
    let base = match filter {
        ResizeFilter::Area => 0.5_f64,
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::CatmullRom => 2.0,
    };
    let support = base * scale.max(1.0);
    let area_minify = filter == ResizeFilter::Area && scale > 1.0;
    let k0 = (centre - support).floor() as i64;
    let k1 = (centre + support).ceil() as i64;

    let mut taps = Vec::new();
    for k in k0..=k1 {
        let w = if area_minify {
            let lo = (k as f64 - 0.5).max(centre - scale * 0.5);
            let hi = (k as f64 + 0.5).min(centre + scale * 0.5);
            (hi - lo).max(0.0)
        } else {
            filter_eval_f64(filter, (k as f64 - centre) / denom)
        };
        if w != 0.0 {
            taps.push((k, w));
        }
    }
    taps
}

fn resample_axis1_all_f64(
    val: &ndarray::Array2<f64>,
    scale: f64,
    filter: phaios_core::geometry::ResizeFilter,
    out_len: usize,
) -> ndarray::Array2<f64> {
    let (rows, in_len) = val.dim();
    let mut out = ndarray::Array2::<f64>::zeros((rows, out_len));
    for i in 0..out_len {
        let taps = resize_taps_f64(filter, scale, i);
        let wsum: f64 = taps.iter().map(|&(_, w)| w).sum();
        if wsum == 0.0 {
            continue;
        }
        for r in 0..rows {
            let mut acc = 0.0_f64;
            for &(k, w) in &taps {
                let kc = k.clamp(0, in_len as i64 - 1) as usize;
                acc += w * val[[r, kc]];
            }
            out[[r, i]] = acc / wsum;
        }
    }
    out
}

fn resize_reference_f64(
    img: &Array3<f32>,
    params: &phaios_core::geometry::ResizeParams,
) -> ndarray::Array2<f64> {
    let (in_h, in_w, _) = img.dim();
    let (out_w, out_h) = (params.width as usize, params.height as usize);
    let src =
        ndarray::Array2::<f64>::from_shape_fn((in_h, in_w), |(y, x)| f64::from(img[[y, x, 0]]));
    // The scale is f64 here too; the kernel computes it in f32.
    let mid = resample_axis1_all_f64(&src, in_w as f64 / out_w as f64, params.filter, out_w);
    let out_t = resample_axis1_all_f64(
        &mid.t().to_owned(),
        in_h as f64 / out_h as f64,
        params.filter,
        out_h,
    );
    out_t.t().to_owned()
}

fn resize_f64_oracle(
    img: &Array3<f32>,
    params: &phaios_core::geometry::ResizeParams,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>, f64) {
    let (in_h, in_w, _) = img.dim();
    let (out_w, out_h) = (params.width as usize, params.height as usize);

    let src_val =
        ndarray::Array2::<f64>::from_shape_fn((in_h, in_w), |(y, x)| f64::from(img[[y, x, 0]]));
    let src_mass = src_val.mapv(f64::abs);

    let scale_x = in_w as f32 / out_w as f32;
    let (mid_val, mid_mass, taps_x) =
        resample_axis1_f64(&src_val, &src_mass, scale_x, params.filter, out_w);

    // The vertical pass is the same routine on the transposed view, as
    // on both backends.
    let scale_y = in_h as f32 / out_h as f32;
    let (out_t, mass_t, taps_y) = resample_axis1_f64(
        &mid_val.t().to_owned(),
        &mid_mass.t().to_owned(),
        scale_y,
        params.filter,
        out_h,
    );

    // Per pass: m products each rounded once, m−1 additions, an m-term
    // weight sum, and the division — ≤ (2m+1)·u·mass. Two passes plus
    // the one f32 rounding of the intermediate plane.
    let n = (2 * (taps_x + taps_y) + 3) as f64;
    (out_t.t().to_owned(), mass_t.t().to_owned(), n)
}

/// `geometry::straighten` recomputed in f64, 4×4 Catmull-Rom in the
/// shipped j-then-i order, with the same host-computed f32 sin/cos.
///
/// Returns `(values, mass)`.
fn straighten_f64_oracle(
    img: &Array3<f32>,
    degrees: f32,
    out_h: usize,
    out_w: usize,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let (in_h, in_w, _) = img.dim();
    let r = f64::from(degrees).to_radians();
    let (sin_a, cos_a) = (r.sin() as f32, r.cos() as f32);

    let (cx_out, cy_out) = (out_w as f32 * 0.5, out_h as f32 * 0.5);
    let (cx_in, cy_in) = (in_w as f32 * 0.5, in_h as f32 * 0.5);

    let mut val = ndarray::Array2::<f64>::zeros((out_h, out_w));
    let mut mass = ndarray::Array2::<f64>::zeros((out_h, out_w));

    for oy in 0..out_h {
        for ox in 0..out_w {
            let dx = ox as f32 + 0.5 - cx_out;
            let dy = oy as f32 + 0.5 - cy_out;
            let sx = cos_a * dx + sin_a * dy + cx_in - 0.5;
            let sy = -sin_a * dx + cos_a * dy + cy_in - 0.5;

            let (fx, fy) = (sx.floor(), sy.floor());
            let (tx, ty) = (sx - fx, sy - fy);
            let (ix, iy) = (fx as i64, fy as i64);

            let weights = |t: f32| {
                use phaios_core::geometry::ResizeFilter::CatmullRom as Cr;
                [
                    filter_eval_f32(Cr, t + 1.0),
                    filter_eval_f32(Cr, t),
                    filter_eval_f32(Cr, 1.0 - t),
                    filter_eval_f32(Cr, 2.0 - t),
                ]
            };
            let wx = weights(tx);
            let wy = weights(ty);

            let mut acc = 0.0_f64;
            let mut m = 0.0_f64;
            for (j, &wyj) in wy.iter().enumerate() {
                let yj = (iy - 1 + j as i64).clamp(0, in_h as i64 - 1) as usize;
                let mut row_acc = 0.0_f64;
                let mut row_m = 0.0_f64;
                for (i, &wxi) in wx.iter().enumerate() {
                    let xi = (ix - 1 + i as i64).clamp(0, in_w as i64 - 1) as usize;
                    let s = f64::from(img[[yj, xi, 0]]);
                    row_acc += f64::from(wxi) * s;
                    row_m += f64::from(wxi).abs() * s.abs();
                }
                acc += f64::from(wyj) * row_acc;
                m += f64::from(wyj).abs() * row_m;
            }
            val[[oy, ox]] = acc;
            mass[[oy, ox]] = m;
        }
    }
    (val, mass)
}

/// A flat field with a brighter bar across it: linear scene-referred
/// data with a specular highlight, the same shape of input that exposed
/// the box blur and `local_contrast`.
fn field_with_bar(h: usize, w: usize, field: f32, bar: f32) -> Array3<f32> {
    let mut img = Array3::<f32>::from_elem((h, w, 1), field);
    let (y0, y1) = (h * 4 / 10, h * 4 / 10 + 4);
    let (x0, x1) = (w / 10, w - w / 10);
    img.slice_mut(s![y0..y1, x0..x1, ..]).fill(bar);
    img
}

/// The three content regimes the resampling oracles below sweep.
///
/// **dark field** is the input that broke `blur` and `local_contrast`:
/// a 12-decade step. One side of every window dominates so completely
/// that Catmull-Rom's negative lobes have nothing to cancel against, and
/// the operator stays well conditioned — measured worst Σ|wᵢxᵢ| / |out|
/// is 6.1e1 for `resize` and 9.3e1 for `straighten`. A step edge simply cannot make this filter singular,
/// which is worth recording because it is the opposite of what a
/// summation audit would guess.
///
/// **cancelling field** narrows the step to a ratio of 9, where the
/// cubic's undershoot is deepest relative to the signal. It stresses the
/// accumulator measurably harder than the 12-decade step (0.32 → 0.39 of
/// the bound for `straighten`) but is still well conditioned, for the
/// same reason: with taps `[B, B, F, F]` the bright pair's weights sum
/// to something in `[0, 1]`, never to the −1/8 that would zero the
/// output. Measured conditioning 4.4e0 (`resize`) and 5.6e0
/// (`straighten`) — lower than the 12-decade step, not higher.
///
/// **cancelling comb** is the geometry that *does* make it singular, and
/// it is why the assertion is stated against the tap mass rather than
/// against the output. One-pixel lines every third row put bright
/// samples on the two *outer* taps of a 4-tap window, whose weights sum
/// to exactly −0.125 at the half-pixel phase against the inner pair's
/// +1.125; a surround at 1/9 of the line level then cancels to zero. The
/// residue is only the f32 rounding of 1/9, so the output is ~1e-8 of a
/// tap mass of ~0.3·B. The period is 3 rather than a single pair so that
/// *some* window lands on the singular alignment whatever phase the
/// output grid happens to have — with one pair it depends on the parity
/// of the scale factor, and the 512 → 768 case missed it entirely.
///
/// Measured conditioning there: 3.355e8 on both kernels, at which the
/// relative error reaches 1500% (`resize`) and 1900% (`straighten`). No
/// relative bound could survive that pixel and none should be asserted;
/// `n·u·Σ|wᵢxᵢ|` does, with room to spare.
///
/// Fine periodic structure against a mid-grey surround — a picket fence,
/// a distant balustrade, a moiré-prone textile — is not a synthetic
/// input, and this is the aliasing-adjacent content a resize is most
/// often asked to handle.
fn resampling_content(h: usize, w: usize, highlight: f32) -> [(&'static str, Array3<f32>); 3] {
    let field = highlight / 9.0;

    // Full-width lines, so the horizontal pass sees a uniform row and
    // the cancellation happens once, cleanly, in the vertical pass —
    // which is the 1-D case the weights above were reasoned about.
    let mut comb = Array3::<f32>::from_elem((h, w, 1), field);
    for y in (0..h).step_by(3) {
        comb.slice_mut(s![y..y + 1, .., ..]).fill(highlight);
    }

    [
        ("dark field", field_with_bar(h, w, 1e-4, highlight)),
        ("cancelling field", field_with_bar(h, w, field, highlight)),
        ("cancelling comb", comb),
    ]
}

/// Is `resize`'s f32 accumulator accurate enough on high-dynamic-range
/// content? A cross-backend test cannot answer that.
///
/// `src/cuda/ptx/geometry.cu` calls these kernels "operation-for-operation
/// transcriptions", and `resampling_is_bit_exact` holds them to
/// `assert_eq!`. Both backends therefore accumulate the weighted sum in
/// **f32**, deliberately, and an HDR conformance sweep of the kind
/// `blur` and `local_contrast` now have would pass here while proving
/// nothing: there is no divergence to find, by design. The open question
/// is not whether the two agree but whether what they agree on is right.
///
/// So the reference is an independent oracle — the same taps, the same
/// f32 weights, an f64 accumulator — and the assertion is the textbook
/// forward-error bound for a floating-point dot product, `n·u·Σ|wᵢxᵢ|`
/// with u = 2⁻²⁴ (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., SIAM 2002, §3.1). Stating it against the tap
/// mass rather than against the output value is the whole point: at a
/// Catmull-Rom output pixel where the negative lobes cancel a 1e8 tap
/// down to near zero, no *relative* bound can hold and none should be
/// asserted.
///
/// MEASURED (RTX 5070 Ti, highlights 1e0…1e8, the two backends
/// identical throughout): worst **0.2358x** the `n·u·mass` bound, at
/// Catmull-Rom 512 → 768 on the dark field. In absolute terms the worst
/// departure is 2.759e-7 of the scene peak — about 1/55 of a 16-bit
/// quantisation step, so it cannot reach a shipped file. The answer to
/// the question is therefore *yes, f32 accumulation is accurate enough
/// here*, and this test exists to keep it that way.
///
/// The worst **relative** error over the same sweep is 1500%, at a
/// `cancelling comb` pixel conditioned 3.4e8 where the true value is
/// ~1e-8 of the tap mass. Both numbers describe the same well-behaved
/// kernel; only one of them could have been asserted.
#[test]
fn resize_f32_accumulation_stays_within_its_error_bound() {
    use phaios_core::geometry::{ResizeFilter, ResizeParams, resize};

    let Some(ctx) = try_context() else { return };

    let mut worst_ratio = 0.0_f64;
    let mut worst_full_scale = 0.0_f64;
    let mut worst_relative = 0.0_f64;
    let mut worst_where = String::new();
    // Worst Σ|wᵢxᵢ| / |output| reached, per content regime — the number
    // that decides whether a relative bound was ever an option.
    let mut conditioning: [(&'static str, f64); 3] = [("", 0.0); 3];

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        for (r, (regime, img)) in resampling_content(512, 512, highlight)
            .into_iter()
            .enumerate()
        {
            let peak = f64::from(highlight);

            for (w, h, filter) in [
                // Large downscales: at 512 → 8 the Area filter sums
                // 4096 source samples per output pixel across the two
                // passes, at 512 → 64 it sums 64. That is where a
                // running f32 total has the most to lose.
                (8_u32, 8_u32, ResizeFilter::Area),
                (64, 64, ResizeFilter::Area),
                (8, 8, ResizeFilter::CatmullRom),
                (64, 64, ResizeFilter::CatmullRom),
                (64, 64, ResizeFilter::Bilinear),
                // And an upscale, the only geometry here whose 4-tap
                // window spans four *adjacent* source pixels — so it is
                // the only one that still resolves the one-pixel comb
                // when the negative lobes reach it.
                (768, 768, ResizeFilter::CatmullRom),
            ] {
                let params = ResizeParams::new(w, h, filter);
                let cpu = resize(img.view(), &params).unwrap();
                let gpu = cuda::kernels::resize(&ctx, img.view(), &params).unwrap();
                // The premise of this test, restated where it is used: the
                // two backends have nothing to tell us apart from each other.
                assert_eq!(cpu, gpu, "{filter:?} {w}x{h}: the backends diverged");

                let (oracle, mass, n) = resize_f64_oracle(&img, &params);
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let err = (f64::from(cpu[[y, x, 0]]) - oracle[[y, x]]).abs();
                        let bound = n * F32_UNIT_ROUNDOFF * mass[[y, x]];
                        let ratio = if bound > 0.0 {
                            err / bound
                        } else {
                            assert_eq!(err, 0.0, "zero mass with a non-zero error");
                            0.0
                        };
                        if ratio > worst_ratio {
                            worst_ratio = ratio;
                            worst_where = format!("{filter:?} {w}x{h}, {regime} at {highlight:e}");
                        }
                        worst_full_scale = worst_full_scale.max(err / peak);
                        let truth = oracle[[y, x]].abs();
                        if truth > 0.0 {
                            worst_relative = worst_relative.max(err / truth);
                            conditioning[r] = (regime, conditioning[r].1.max(mass[[y, x]] / truth));
                        }
                        assert!(
                            ratio <= 1.0,
                            "resize {filter:?} {w}x{h}, {regime} at {highlight:e}, \
                             pixel ({y}, {x}): f32 accumulation is {ratio:.3}x \
                             its {n}·u·mass error bound (err {err:e}, mass {:e})",
                            mass[[y, x]]
                        );
                    }
                }
            }
        }
    }

    eprintln!(
        "resize oracle: worst {worst_ratio:.4}x the n·u·mass bound ({worst_where}); \
         worst absolute error {worst_full_scale:.3e} of the scene peak; \
         worst relative error {worst_relative:.3e}"
    );
    for (regime, cond) in conditioning {
        eprintln!("resize oracle: {regime} reached a conditioning of {cond:.3e}");
    }
}

/// The same question for `straighten`, whose 16 Catmull-Rom taps are
/// where the negative lobes cancel hardest.
///
/// Only 16 taps, so the summation itself is short — but the two outer
/// lobes carry weight −0.5·t(1−t)² and the bar is 12 orders above the
/// field, so an output pixel a pixel outside the bar is a difference of
/// two ~1e8 quantities. That is not a defect: it is what a cubic does,
/// and the f64 oracle says so too. What is worth pinning is that the
/// f32 accumulator adds nothing beyond the rounding the arithmetic
/// forces.
///
/// MEASURED (RTX 5070 Ti, highlights 1e0…1e8, the two backends
/// identical throughout): worst **0.3899x** the `12·u·mass` bound, at
/// −45° on the `cancelling field`. Worst absolute departure 2.831e-7 of
/// the scene peak, ~1/54 of a 16-bit quantisation step. Worst relative
/// error 1900%, at a `cancelling comb` pixel conditioned 3.4e8.
///
/// So: accurate enough, with the 16 f32 taps costing about a third of
/// the rounding the arithmetic already allows them.
#[test]
fn straighten_f32_accumulation_stays_within_its_error_bound() {
    use phaios_core::geometry::{StraightenParams, straighten};

    // Products rounded once each and summed 4-wide, the row total
    // rounded, then the same again across the four rows: ≤ 12 rounding
    // steps on the worst pixel.
    const N_ROUNDINGS: f64 = 12.0;

    let Some(ctx) = try_context() else { return };

    let mut worst_ratio = 0.0_f64;
    let mut worst_full_scale = 0.0_f64;
    let mut worst_relative = 0.0_f64;
    let mut worst_where = String::new();
    // Worst Σ|wᵢxᵢ| / |output| reached, per content regime — the number
    // that decides whether a relative bound was ever an option.
    let mut conditioning: [(&'static str, f64); 3] = [("", 0.0); 3];

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        for (r, (regime, img)) in resampling_content(257, 389, highlight)
            .into_iter()
            .enumerate()
        {
            let peak = f64::from(highlight);

            for degrees in [0.35_f32, 1.5, -7.3, 30.0, -45.0] {
                let params = StraightenParams::new(degrees);
                let cpu = straighten(img.view(), &params).unwrap();
                let gpu = cuda::kernels::straighten(&ctx, img.view(), &params).unwrap();
                assert_eq!(cpu, gpu, "{degrees} deg: the backends diverged");

                let (out_h, out_w, _) = cpu.dim();
                let (oracle, mass) = straighten_f64_oracle(&img, degrees, out_h, out_w);
                for y in 0..out_h {
                    for x in 0..out_w {
                        let err = (f64::from(cpu[[y, x, 0]]) - oracle[[y, x]]).abs();
                        let bound = N_ROUNDINGS * F32_UNIT_ROUNDOFF * mass[[y, x]];
                        let ratio = if bound > 0.0 {
                            err / bound
                        } else {
                            assert_eq!(err, 0.0, "zero mass with a non-zero error");
                            0.0
                        };
                        if ratio > worst_ratio {
                            worst_ratio = ratio;
                            worst_where = format!("{degrees} deg, {regime} at {highlight:e}");
                        }
                        worst_full_scale = worst_full_scale.max(err / peak);
                        let truth = oracle[[y, x]].abs();
                        if truth > 0.0 {
                            worst_relative = worst_relative.max(err / truth);
                            conditioning[r] = (regime, conditioning[r].1.max(mass[[y, x]] / truth));
                        }
                        assert!(
                            ratio <= 1.0,
                            "straighten {degrees} deg, {regime} at {highlight:e}, \
                             pixel ({y}, {x}): f32 accumulation is {ratio:.3}x \
                             its 12·u·mass error bound (err {err:e}, mass {:e})",
                            mass[[y, x]]
                        );
                    }
                }
            }
        }
    }

    eprintln!(
        "straighten oracle: worst {worst_ratio:.4}x the 12·u·mass bound ({worst_where}); \
         worst absolute error {worst_full_scale:.3e} of the scene peak; \
         worst relative error {worst_relative:.3e}"
    );
    for (regime, cond) in conditioning {
        eprintln!("straighten oracle: {regime} reached a conditioning of {cond:.3e}");
    }
}

// ── the resident pipeline ────────────────────────────────────────────────────

/// The full nine-stage v0.2 pipeline, resident end to end — one upload,
/// one download — against the identical CPU chain. Every kernel the
/// crate ships now runs on the backend.
#[test]
fn full_pipeline_resident_matches_cpu() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(512, 768, 3);

    let hsl = phaios_core::bw::HslWeightedParams::new(
        [0.0, 0.0, 0.4, 0.2, 0.0, -0.5, 0.0, 0.0],
        Default::default(),
        30.0,
    );
    let zones = phaios_core::tone::ZoneParams::new([(3, -0.3_f32), (7, 0.4)].into_iter().collect());
    let gf = phaios_core::local_contrast::GuidedFilterParams::new(8, 0.01);
    let grain = phaios_core::film_grain::GrainParams::new(0.12, 1.5, 20_260_815);
    let toning = phaios_core::split_toning::SplitToningParams::new(
        [0.0, -0.02, -0.04],
        [0.0, 0.03, 0.03],
        0.5,
        0.0,
    );
    let vg = phaios_core::vignette::VignetteParams::new(0.35, 0.8, 0.1);
    let tc = phaios_core::tone::ToneCurveParams::new(1.1, 0.0, 0.9);

    // CPU chain.
    let c = phaios_core::exposure::exposure(img.view(), 0.5).unwrap();
    let c = phaios_core::bw::hsl_bw(c.view(), &hsl).unwrap();
    let c = phaios_core::tone::zone_system(c.view(), &zones).unwrap();
    let c = phaios_core::local_contrast::local_contrast(c.view(), &gf, 0.4).unwrap();
    let c = phaios_core::film_grain::film_grain(c.view(), &grain).unwrap();
    let c = phaios_core::split_toning::split_toning(c.view(), &toning).unwrap();
    let c = phaios_core::vignette::vignette(c.view(), &vg).unwrap();
    let c = phaios_core::tone::tone_curve(c.view(), &tc).unwrap();
    let cpu = phaios_core::encode::encode_srgb(c.view()).unwrap();

    // GPU chain, resident throughout.
    let d = ctx.upload(img.view()).unwrap();
    let d = cuda::kernels::exposure_device(&d, 0.5).unwrap();
    let d = cuda::kernels::hsl_bw_device(&d, &hsl).unwrap();
    let d = cuda::kernels::zone_system_device(&d, &zones).unwrap();
    let d = cuda::kernels::local_contrast_device(&d, &gf, 0.4).unwrap();
    let d = cuda::kernels::film_grain_device(&d, &grain).unwrap();
    let d = cuda::kernels::split_toning_device(&d, &toning).unwrap();
    let d = cuda::kernels::vignette_device(&d, &vg).unwrap();
    let d = cuda::kernels::tone_curve_device(&d, &tc).unwrap();
    let d = cuda::kernels::encode_srgb_device(&d).unwrap();
    let gpu = ctx.download(&d).unwrap();

    assert_eq!(gpu.dim(), cpu.dim());
    let v = worst_violation(&cpu, &gpu, 1e-3, 1e-5);
    assert!(v <= 1.0, "full pipeline: {v:.2}x the (1e-3, 1e-5) bound");
}

/// One upload, five device-resident stages, one download — against the
/// same five stages on the CPU. This is the chaining property Stage C
/// exists to prove; the bound is local_contrast's (loosest in chain).
#[test]
fn resident_chain_matches_cpu_chain() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(512, 768, 3);
    let gf = phaios_core::local_contrast::GuidedFilterParams::new(8, 0.01);
    let tc = phaios_core::tone::ToneCurveParams::new(1.15, 0.01, 0.85);
    let vg = phaios_core::vignette::VignetteParams::new(0.35, 0.8, 0.1);

    // CPU chain.
    let c = phaios_core::bw::luminance_bw(img.view(), Default::default()).unwrap();
    let c = phaios_core::local_contrast::local_contrast(c.view(), &gf, 0.4).unwrap();
    let c = phaios_core::tone::tone_curve(c.view(), &tc).unwrap();
    let c = phaios_core::vignette::vignette(c.view(), &vg).unwrap();
    let cpu = phaios_core::encode::encode_srgb(c.view()).unwrap();

    // GPU chain: one upload, one download.
    let d = ctx.upload(img.view()).unwrap();
    let d = cuda::kernels::luminance_bw_device(&d, Default::default()).unwrap();
    let d = cuda::kernels::local_contrast_device(&d, &gf, 0.4).unwrap();
    let d = cuda::kernels::tone_curve_device(&d, &tc).unwrap();
    let d = cuda::kernels::vignette_device(&d, &vg).unwrap();
    let d = cuda::kernels::encode_srgb_device(&d).unwrap();
    let gpu = ctx.download(&d).unwrap();

    assert_eq!(gpu.dim(), cpu.dim());
    let v = worst_violation(&cpu, &gpu, 2e-4, 1e-6);
    assert!(v <= 1.0, "resident chain: {v:.2}x the (2e-4, 1e-6) bound");
}

/// Upload → download round trip is bit-exact, for every layout.
#[test]
fn upload_download_round_trips_exactly() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(64, 48, 3);
    for view in [
        img.view(),
        img.slice(s![..;2, ..;3, ..]),
        img.slice(s![..;-1, .., ..]),
    ] {
        let device = ctx.upload(view).unwrap();
        let back = ctx.download(&device).unwrap();
        assert_eq!(back, view.to_owned(), "round trip corrupted data");
        assert!(back.is_standard_layout());
    }
}

// ── enumeration and context behaviour ────────────────────────────────────────

/// Enumeration never panics, and every reported-supported device can
/// actually be opened.
#[test]
fn devices_enumeration_is_consistent() {
    for d in cuda::devices() {
        if d.supported {
            let ctx = cuda::Context::new(d.ordinal)
                .unwrap_or_else(|e| panic!("device {} reported supported but: {e}", d.ordinal));
            assert!(
                ctx.fingerprint().starts_with("cuda/"),
                "fingerprint format changed: {}",
                ctx.fingerprint()
            );
        }
    }
}

/// An out-of-range ordinal is a clean Backend error, not a panic.
#[test]
fn bad_ordinal_is_a_backend_error() {
    let err = cuda::Context::new(9_999).unwrap_err();
    assert!(
        matches!(err, phaios_core::error::PhaiosError::Backend(_)),
        "expected Backend error, got: {err:?}"
    );
}

/// Non-finite pixels must not split the backends *for element-wise
/// kernels*, whose arithmetic is per-pixel and so cannot spread a
/// poisoned sample. `hsl_bw` is the delicate one: an all-infinite pixel
/// makes `delta = inf - inf = NaN`, which fails every comparison, so the
/// CPU guard written as a negative test fell *through* into the hue
/// branch and returned 0.0 where CUDA returned inf.
///
/// Neighbourhood kernels are explicitly out of scope — see
/// `local_contrast_non_finite_is_documented_as_divergent` below and
/// docs/ffi.md §1. Found by an empirical CPU/GPU sweep over degenerate
/// inputs, not by this suite, which is why the sweep's cases live here.
#[test]
fn non_finite_pixels_agree_across_backends() {
    let Some(ctx) = try_context() else { return };
    let params = phaios_core::bw::HslWeightedParams::new(
        [0.2; 8],
        phaios_core::bw::LuminanceStandard::Bt709,
        30.0,
    );

    for pixel in [
        [f32::INFINITY, f32::INFINITY, f32::INFINITY],
        [f32::INFINITY, 1.0, 1.0],
        [f32::NAN, f32::NAN, f32::NAN],
        [f32::NAN, 0.5, 0.25],
        [f32::NEG_INFINITY, 0.5, 0.25],
        [f32::INFINITY, f32::NEG_INFINITY, 0.0],
    ] {
        let img = ndarray::array![[[pixel[0], pixel[1], pixel[2]]]];
        let cpu = phaios_core::bw::hsl_bw(img.view(), &params).unwrap();
        let gpu = cuda::kernels::hsl_bw(&ctx, img.view(), &params).unwrap();
        let (c, g) = (cpu[[0, 0, 0]], gpu[[0, 0, 0]]);
        // Strict bit comparison. `is_nan() && is_nan()` would accept any
        // NaN as equal to any other, and a NaN sign or payload flip is
        // exactly the divergence class the negated comparisons in these
        // kernels exist to prevent — the shadow_rolloff review found the
        // looser form hiding a real one.
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "hsl_bw diverges on {pixel:?}: cpu={c}, gpu={g}"
        );
    }
}

/// Downloading an image through a *different* Context on the same device
/// must return the same bytes as downloading through its own. An audit
/// claim held that `Context::download` synchronises `self.stream` rather
/// than the stream that produced the image, making this racy; both
/// Contexts take `CudaContext::default_stream()` for the same device, so
/// the streams coincide and the download is ordered. This test exists to
/// keep that true — if `Context` ever allocates its own stream, it fails.
#[test]
fn cross_context_download_is_ordered() {
    let Some(a) = try_context() else { return };
    let Some(b) = try_context() else { return };

    // Large and expensive enough that the kernel is still in flight when
    // the download is issued, if the streams were ever independent.
    let img = pseudo_random_image(1024, 1024, 1);
    let device = a.upload(img.view()).unwrap();
    let params = phaios_core::local_contrast::GuidedFilterParams::new(16, 0.01);
    let processed = cuda::kernels::local_contrast_device(&device, &params, 0.8).unwrap();

    let via_b = b.download(&processed).unwrap();
    let via_a = a.download(&processed).unwrap();
    assert_eq!(via_a, via_b, "cross-context download raced the producer");
}

/// `local_contrast` is the one kernel whose backends genuinely disagree
/// on non-finite input, and docs/ffi.md §1 says so in those terms. This
/// test pins the *shape* of that disagreement rather than papering over
/// it: the CPU's global summed-area tables spread one non-finite sample
/// across the whole image, the GPU's separable box passes confine it to a
/// neighbourhood, and every finite GPU pixel remains bit-correct.
///
/// If a future change makes these agree, this test fails and the doc
/// paragraph should be rewritten — that would be good news, not a
/// regression.
#[test]
fn local_contrast_non_finite_is_documented_as_divergent() {
    let Some(ctx) = try_context() else { return };

    let (h, w, r) = (96_usize, 96_usize, 4_u32);
    let mut img = pseudo_random_image(h, w, 1);
    img[[h / 2, w / 2, 0]] = f32::INFINITY;
    let params = phaios_core::local_contrast::GuidedFilterParams::new(r, 0.01);

    let cpu = phaios_core::local_contrast::local_contrast(img.view(), &params, 1.0).unwrap();
    let gpu = cuda::kernels::local_contrast_device(&ctx.upload(img.view()).unwrap(), &params, 1.0)
        .and_then(|d| ctx.download(&d))
        .unwrap();

    let cpu_bad = cpu.iter().filter(|v| !v.is_finite()).count();
    let gpu_bad = gpu.iter().filter(|v| !v.is_finite()).count();

    // The GPU's damage is bounded by the two box passes: (4r+1)^2.
    let bound = ((4 * r + 1) * (4 * r + 1)) as usize;
    assert!(
        gpu_bad <= bound,
        "GPU contamination should stay within {bound} pixels, saw {gpu_bad}"
    );
    assert!(
        cpu_bad > gpu_bad,
        "the documented divergence is that the CPU spreads further \
         (cpu {cpu_bad}, gpu {gpu_bad})"
    );

    // Where the GPU is finite and the CPU is too, they must still agree.
    let mut compared = 0_usize;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        if c.is_finite() && g.is_finite() {
            compared += 1;
            assert!(
                // The suite's committed bound (docs/ffi.md section 6).
                (c - g).abs() <= 1e-4 * c.abs() + 1e-6,
                "finite pixels must still agree: cpu={c}, gpu={g}"
            );
        }
    }
    assert!(
        compared > 0,
        "nothing was comparable — the test proved nothing"
    );
}

/// `highlight_rolloff` is bit-exact: the shoulder uses only add,
/// subtract, multiply, divide and sqrt, every one of which IEEE-754-2008
/// §5.4.1 requires to be correctly rounded, and `-fmad=false` stops the
/// compiler contracting any of them into a fused multiply-add. There is
/// no transcendental, so there is no libm to disagree with. No tolerance.
#[test]
fn highlight_rolloff_is_bit_exact() {
    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3);

    for (knee, white) in [
        (1.0_f32, 1.0_f32), // the default: hard clip
        (0.8, 2.0),
        (0.5, 4.0),
        (0.0, 1.0), // knee at black
        (0.6, 1.4), // a = W + k - 2 = 0: the degenerate linear solve
        (0.95, 16.0),
        (0.0, 64.0), // the whole range in the shoulder
    ] {
        let params = phaios_core::highlight_rolloff::RolloffParams::new(knee, white);
        let cpu = phaios_core::highlight_rolloff::highlight_rolloff(img.view(), &params).unwrap();
        let gpu = cuda::kernels::highlight_rolloff(&ctx, img.view(), &params).unwrap();
        assert_eq!(cpu, gpu, "highlight_rolloff knee={knee} white={white}");
    }
}

/// Values above the white point and non-finite samples must agree too —
/// the branches the pseudo-random image (values in [0,1)) never reaches.
#[test]
fn highlight_rolloff_edge_values_agree() {
    let Some(ctx) = try_context() else { return };
    let img = ndarray::array![[
        [-1.0_f32, 0.0, 0.5],
        [1.0, 2.0, 1e30],
        [f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    ]];
    let params = phaios_core::highlight_rolloff::RolloffParams::new(0.7, 3.0);
    let cpu = phaios_core::highlight_rolloff::highlight_rolloff(img.view(), &params).unwrap();
    let gpu = cuda::kernels::highlight_rolloff(&ctx, img.view(), &params).unwrap();
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        // Strict, for the reason given on `hsl_bw_non_finite`: the
        // NaN-tolerant form cannot see a payload or sign divergence.
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "edge value diverges: cpu={c}, gpu={g}"
        );
    }
}

/// Quantisation is bit-exact, dithered or not. The dither is exact
/// 64-bit integer arithmetic (splitmix64, already asserted over 2²⁰
/// coordinates by the grain suite) and the rounding is `floor(v + 0.5)`
/// — an exact operation composed with a correctly-rounded one, rather
/// than a library rounding routine that could differ between host and
/// device. No tolerance.
#[test]
fn quantize_is_bit_exact() {
    use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};

    let Some(ctx) = try_context() else { return };
    // Values spanning and exceeding [0, 1] so the clamp branches run too.
    let img = pseudo_random_image(257, 389, 3).mapv(|v| v * 1.4 - 0.2);

    for (dither, seed) in [
        (Dither::Off, 0_u64),
        (Dither::Off, 12345), // seed must be ignored here
        (Dither::Tpdf, 20260825),
        (Dither::Tpdf, 1),
        (Dither::Tpdf, u64::MAX),
    ] {
        let params = QuantizeParams::new(dither, seed);
        assert_eq!(
            quantize_u8(img.view(), &params).unwrap(),
            cuda::kernels::quantize_u8(&ctx, img.view(), &params).unwrap(),
            "quantize_u8 {dither:?} seed={seed}"
        );
        assert_eq!(
            quantize_u16(img.view(), &params).unwrap(),
            cuda::kernels::quantize_u16(&ctx, img.view(), &params).unwrap(),
            "quantize_u16 {dither:?} seed={seed}"
        );
    }
}

/// The dither key mixes the channel index, so the flat-index arithmetic
/// on the device has to recover (x, y, channel) exactly as the CPU's
/// `Zip::indexed` reports it. A single-channel image cannot catch a
/// mistake there; a three-channel one at a non-square size can.
#[test]
fn quantize_channel_keying_agrees() {
    use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8};

    let Some(ctx) = try_context() else { return };
    for (h, w, c) in [(1, 1, 1), (1, 7, 3), (7, 1, 3), (5, 3, 1), (13, 17, 3)] {
        let img = pseudo_random_image(h, w, c);
        let params = QuantizeParams::new(Dither::Tpdf, 4242);
        assert_eq!(
            quantize_u8(img.view(), &params).unwrap(),
            cuda::kernels::quantize_u8(&ctx, img.view(), &params).unwrap(),
            "shape ({h}, {w}, {c})"
        );
    }
}

/// Non-finite and out-of-range samples must land on the same codes.
#[test]
fn quantize_edge_values_agree() {
    use phaios_core::quantize::{Dither, QuantizeParams, quantize_u16};

    let Some(ctx) = try_context() else { return };
    let img = ndarray::array![[
        [-1.0_f32, 0.0, 1.0],
        [1.5, 1e30, -1e30],
        [f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    ]];
    for dither in [Dither::Off, Dither::Tpdf] {
        let params = QuantizeParams::new(dither, 8);
        assert_eq!(
            quantize_u16(img.view(), &params).unwrap(),
            cuda::kernels::quantize_u16(&ctx, img.view(), &params).unwrap(),
            "edge values, {dither:?}"
        );
    }
}

/// The histogram is bit-identical across backends, not merely close.
/// The only arithmetic on pixel values is the bin assignment — subtract,
/// divide, multiply, truncate, all exact or correctly rounded — and
/// everything after it is integer counting, whose total cannot depend on
/// the order in which the device's atomics complete.
///
/// Both device paths are exercised: bins small enough to privatise in
/// shared memory, and the global-atomic fallback above that.
#[test]
fn histogram_is_bit_identical() {
    use phaios_core::histogram::{HistogramParams, histogram};

    let Some(ctx) = try_context() else { return };
    // Values reaching outside [0, 1] so below/above are non-zero too.
    let img = pseudo_random_image(257, 389, 3).mapv(|v| v * 1.4 - 0.2);

    for bins in [2_u32, 3, 256, 1024, 12_288, 65_536] {
        let p = HistogramParams::new(bins, 0.0, 1.0);
        let cpu = histogram(img.view(), &p).unwrap();
        let gpu = cuda::kernels::histogram(&ctx, img.view(), &p).unwrap();
        assert_eq!(cpu.counts(), gpu.counts(), "counts differ at bins={bins}");
        assert_eq!(cpu.below(), gpu.below(), "below differs at bins={bins}");
        assert_eq!(cpu.above(), gpu.above(), "above differs at bins={bins}");
        assert_eq!(
            cpu.non_finite(),
            gpu.non_finite(),
            "non_finite differs at bins={bins}"
        );
        assert_eq!(cpu.total(0), gpu.total(0), "totals differ at bins={bins}");
    }
}

/// Non-finite and out-of-range samples must be classified identically.
#[test]
fn histogram_edge_values_agree() {
    use phaios_core::histogram::{HistogramParams, histogram};

    let Some(ctx) = try_context() else { return };
    let img = ndarray::array![[
        [-1.0_f32, 0.0, 1.0],
        [1.5, f32::INFINITY, f32::NEG_INFINITY],
        [f32::NAN, 0.5, 2.0],
    ]];
    let p = HistogramParams::default();
    let cpu = histogram(img.view(), &p).unwrap();
    let gpu = cuda::kernels::histogram(&ctx, img.view(), &p).unwrap();
    assert_eq!(cpu.counts(), gpu.counts());
    assert_eq!(cpu.below(), gpu.below());
    assert_eq!(cpu.above(), gpu.above());
    assert_eq!(cpu.non_finite(), gpu.non_finite());
}

/// `apply_lut` is bit-exact: subtract, divide, multiply, truncate and one
/// linear interpolation, with `-fmad=false` stopping the interpolation's
/// multiply-add from being contracted.
#[test]
fn apply_lut_is_bit_exact() {
    use phaios_core::lut::{LutParams, apply_lut};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(257, 389, 3).mapv(|v| v * 1.4 - 0.2);

    for n in [2_usize, 3, 17, 256, 4096, 65_536] {
        // A curved table, so interpolation actually does something.
        let lut = ndarray::Array1::from_shape_fn(n, |i| (i as f32 / (n - 1) as f32).powf(0.7));
        for params in [LutParams::default(), LutParams::new(-0.5, 2.0)] {
            assert_eq!(
                apply_lut(img.view(), lut.view(), &params).unwrap(),
                cuda::kernels::apply_lut(&ctx, img.view(), lut.view(), &params).unwrap(),
                "lut n={n} domain=({}, {})",
                params.min,
                params.max
            );
        }
    }
}

/// A non-monotone table — solarisation — and NaN propagation, neither of
/// which the pseudo-random image reaches on its own.
#[test]
fn apply_lut_edge_cases_agree() {
    use phaios_core::lut::{LutParams, apply_lut};

    let Some(ctx) = try_context() else { return };
    let img = ndarray::array![[
        [-1.0_f32, 0.0, 0.5],
        [1.0, 2.0, f32::INFINITY],
        [f32::NEG_INFINITY, f32::NAN, 0.25],
    ]];
    let solarise = ndarray::Array1::from_shape_fn(256, |i| {
        let t = i as f32 / 255.0;
        if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 }
    });
    // A negative NaN, which is what ordinary f32 arithmetic produces on
    // x86 — the CPU used to substitute a canonical positive NaN here and
    // silently disagree with the device.
    let img = {
        let mut img = img;
        img[[0, 2, 1]] = f32::from_bits(0xFFC0_0000);
        img
    };
    let cpu = apply_lut(img.view(), solarise.view(), &LutParams::default()).unwrap();
    let gpu =
        cuda::kernels::apply_lut(&ctx, img.view(), solarise.view(), &LutParams::default()).unwrap();
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        // Bit-for-bit including the NaN payload: `is_nan() && is_nan()`
        // would let a sign flip through.
        assert_eq!(c.to_bits(), g.to_bits(), "diverges: cpu={c}, gpu={g}");
    }

    // A reversed (negative-stride) table must reach the same answer on
    // both backends, having panicked on both before the review.
    let reversed = solarise.slice(ndarray::s![..;-1]);
    assert!(reversed.as_slice().is_none());
    let rcpu = apply_lut(img.view(), reversed, &LutParams::default()).unwrap();
    let rgpu = cuda::kernels::apply_lut(&ctx, img.view(), reversed, &LutParams::default()).unwrap();
    // Bitwise, because the image carries NaN and `NaN == NaN` is false.
    for (c, g) in rcpu.iter().zip(rgpu.iter()) {
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "reversed table diverges: {c} vs {g}"
        );
    }
}

/// The two backends must refuse an oversized request identically — the
/// CPU used to abort the process where the GPU returned an error.
#[test]
fn oversized_histogram_requests_are_refused_by_both() {
    use phaios_core::histogram::{HistogramParams, histogram};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(4, 4, 3);
    for params in [
        HistogramParams::new(500_000_000, 0.0, 1.0),
        HistogramParams::new(u32::MAX, 0.0, 1.0),
    ] {
        let cpu = histogram(img.view(), &params).unwrap_err();
        let gpu = cuda::kernels::histogram(&ctx, img.view(), &params).unwrap_err();
        assert_eq!(
            cpu.to_string(),
            gpu.to_string(),
            "both backends must reject with the same message"
        );
    }
}

/// `shadow_rolloff` is bit-exact: a cubic in Horner form built from
/// multiply, add, subtract and one divide, every one correctly rounded,
/// with `-fmad=false` stopping the compiler contracting the polynomial's
/// multiply-adds. No tolerance.
#[test]
fn shadow_rolloff_is_bit_exact() {
    use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};

    let Some(ctx) = try_context() else { return };
    // Reaching below zero and above the knee so every branch is taken.
    let img = pseudo_random_image(257, 389, 3).mapv(|v| v * 1.3 - 0.15);

    for (knee, strength) in [
        (0.2_f32, 0.0_f32), // the identity fast path
        (0.2, 0.25),
        (0.2, 0.5),
        (0.2, 1.0), // zero slope at black
        (1.0, 0.7), // the whole range is toe
        (0.05, 1.0),
        (0.0, 1.0), // empty region: also the identity
    ] {
        let params = ShadowRolloffParams::new(knee, strength);
        assert_eq!(
            shadow_rolloff(img.view(), &params).unwrap(),
            cuda::kernels::shadow_rolloff(&ctx, img.view(), &params).unwrap(),
            "shadow_rolloff knee={knee} strength={strength}"
        );
    }
}

/// Non-finite and negative samples must agree too — the branches the
/// pseudo-random image alone does not reach.
///
/// Two things this test had to learn. It sweeps `strength`, because the
/// only broken case was `strength == 1.0`, where the continuation slope
/// is exactly zero and `0.0 · −∞` is NaN. And it compares **bit
/// patterns**, not `is_nan() && is_nan()` — the looser form accepted any
/// NaN as equal to any other and so could not see the divergence at all,
/// which is exactly the class the kernel's negated comparisons exist to
/// prevent.
#[test]
fn shadow_rolloff_edge_values_agree() {
    use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};

    let Some(ctx) = try_context() else { return };
    let img = ndarray::array![[
        [-1.0_f32, -0.001, 0.0],
        [1e-30, 0.1, 0.2],
        [f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
    ]];
    for step in 0..=10 {
        let strength = step as f32 / 10.0;
        let params = ShadowRolloffParams::new(0.2, strength);
        let cpu = shadow_rolloff(img.view(), &params).unwrap();
        let gpu = cuda::kernels::shadow_rolloff(&ctx, img.view(), &params).unwrap();
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "strength={strength}: cpu={c} and gpu={g} differ bitwise"
            );
        }
    }
}

/// The device entry point must validate. Deleting the `validate` call in
/// `shadow_rolloff_device` left every conformance test green, yet that is
/// the function the Python GPU binding calls — so an out-of-domain
/// parameter would have produced pixels instead of an error.
#[test]
fn shadow_rolloff_device_rejects_what_the_cpu_rejects() {
    use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(4, 4, 1);
    for (knee, strength) in [
        (1.5_f32, 0.5_f32),
        (-0.1, 0.5),
        (0.2, 1.5),
        (0.2, -0.1),
        (f32::NAN, 0.5),
        (0.2, f32::INFINITY),
    ] {
        let params = ShadowRolloffParams::new(knee, strength);
        let cpu = shadow_rolloff(img.view(), &params).expect_err("the CPU kernel must reject this");
        let device = ctx.upload(img.view()).unwrap();
        let gpu = cuda::kernels::shadow_rolloff_device(&device, &params)
            .expect_err("the device kernel must reject it too");
        assert_eq!(
            cpu.to_string(),
            gpu.to_string(),
            "knee={knee} strength={strength}: backends must refuse identically"
        );
    }
}

/// Both backends must refuse the same input, with the same message.
///
/// Compared by rendered string rather than by variant: the message is
/// what reaches a Python caller, and the whole reason validation is
/// extracted into shared `validate*` helpers (CLAUDE.md §2) is that the
/// two backends should be indistinguishable at that boundary.
fn rejects_identically<T, U>(
    label: &str,
    cpu: Result<T, phaios_core::error::PhaiosError>,
    gpu: Result<U, phaios_core::error::PhaiosError>,
) {
    let Err(cpu) = cpu else {
        panic!("{label}: the CPU kernel accepted an input it is supposed to refuse");
    };
    let Err(gpu) = gpu else {
        panic!("{label}: the device kernel accepted what the CPU refused");
    };
    assert_eq!(
        cpu.to_string(),
        gpu.to_string(),
        "{label}: the backends must refuse identically"
    );
}

/// Every fallible `_device` entry point refuses what the CPU refuses.
///
/// The `_device` forms are what the `phaios_core.gpu` submodule calls,
/// and each carries its own guard. The per-call offload wrappers
/// validate *separately*, so a test routed through those would not
/// notice a `validate` deleted from the device form — every case below
/// therefore uploads first and calls the device entry point directly.
///
/// Written because that guard was unpinned everywhere but one kernel:
/// the `validate` call could be deleted from every device entry point
/// except `shadow_rolloff` with the whole suite green, after which
/// `blur_device(sigma = NaN)` returns `Ok` full of numbers, and the
/// B&W entry points read a (H, W, 1) upload as if it held three
/// channels — a 3x out-of-bounds device read.
#[test]
fn every_fallible_device_entry_point_rejects_what_the_cpu_rejects() {
    use phaios_core::blur::{BlurParams, BlurShape};
    use phaios_core::bw::{ColorFilter, HslWeightedParams, LuminanceStandard};
    use phaios_core::denoise::DenoiseParams;
    use phaios_core::film_grain::GrainParams;
    use phaios_core::geometry::{CropParams, ResizeFilter, ResizeParams};
    use phaios_core::glow::GlowParams;
    use phaios_core::highlight_rolloff::RolloffParams;
    use phaios_core::histogram::HistogramParams;
    use phaios_core::hot_pixels::HotPixelParams;
    use phaios_core::local_contrast::GuidedFilterParams;
    use phaios_core::lut::LutParams;
    use phaios_core::shadow_rolloff::ShadowRolloffParams;
    use phaios_core::sharpen::SharpenParams;
    use phaios_core::split_toning::SplitToningParams;
    use phaios_core::tone::{ToneCurveParams, ZoneParams};
    use phaios_core::vignette::VignetteParams;
    use std::collections::HashMap;

    let Some(ctx) = try_context() else { return };

    let rgb = pseudo_random_image(4, 4, 3);
    let luma = pseudo_random_image(4, 4, 1);
    let d_rgb = ctx.upload(rgb.view()).unwrap();
    let d_luma = ctx.upload(luma.view()).unwrap();

    // ── parameter guards ────────────────────────────────────────────

    let p = BlurParams::new(f32::NAN, BlurShape::Gaussian);
    rejects_identically(
        "blur sigma=NaN",
        phaios_core::blur::blur(rgb.view(), &p),
        cuda::kernels::blur_device(&d_rgb, &p),
    );

    let p = GlowParams::new(0.8, 2.0, -1.0);
    rejects_identically(
        "glow amount=-1",
        phaios_core::glow::glow(rgb.view(), &p),
        cuda::kernels::glow_device(&d_rgb, &p),
    );

    let p = SharpenParams::new(0.5, f32::NAN, 0.0);
    rejects_identically(
        "sharpen sigma=NaN",
        phaios_core::sharpen::sharpen(rgb.view(), &p),
        cuda::kernels::sharpen_device(&d_rgb, &p),
    );

    let p = SharpenParams::new(-1.0, 2.0, 0.0);
    rejects_identically(
        "sharpen amount=-1",
        phaios_core::sharpen::sharpen(rgb.view(), &p),
        cuda::kernels::sharpen_device(&d_rgb, &p),
    );

    let p = HotPixelParams::new(-1.0, 0.0);
    rejects_identically(
        "hot_pixels threshold=-1",
        phaios_core::hot_pixels::hot_pixels(rgb.view(), &p),
        cuda::kernels::hot_pixels_device(&d_rgb, &p),
    );

    let p = HotPixelParams::new(0.05, f32::NAN);
    rejects_identically(
        "hot_pixels relative=NaN",
        phaios_core::hot_pixels::hot_pixels(rgb.view(), &p),
        cuda::kernels::hot_pixels_device(&d_rgb, &p),
    );

    let p = DenoiseParams::new(2, -1.0, 0.5, LuminanceStandard::Bt709);
    rejects_identically(
        "denoise noise_sigma=-1",
        phaios_core::denoise::denoise(rgb.view(), &p),
        cuda::kernels::denoise_device(&d_rgb, &p),
    );

    let p = DenoiseParams::new(2, 0.02, 1.5, LuminanceStandard::Bt709);
    rejects_identically(
        "denoise amount=1.5",
        phaios_core::denoise::denoise(rgb.view(), &p),
        cuda::kernels::denoise_device(&d_rgb, &p),
    );

    let p = DenoiseParams::new(
        phaios_core::denoise::MAX_RADIUS + 1,
        0.02,
        0.5,
        LuminanceStandard::Bt709,
    );
    rejects_identically(
        "denoise radius=33",
        phaios_core::denoise::denoise(rgb.view(), &p),
        cuda::kernels::denoise_device(&d_rgb, &p),
    );

    rejects_identically(
        "exposure stops=inf",
        phaios_core::exposure::exposure(rgb.view(), f32::INFINITY),
        cuda::kernels::exposure_device(&d_rgb, f32::INFINITY),
    );

    let p = ToneCurveParams::new(1.0, 0.0, 0.0);
    rejects_identically(
        "tone_curve power=0",
        phaios_core::tone::tone_curve(rgb.view(), &p),
        cuda::kernels::tone_curve_device(&d_rgb, &p),
    );

    let p = RolloffParams::new(1.5, 2.0);
    rejects_identically(
        "highlight_rolloff knee=1.5",
        phaios_core::highlight_rolloff::highlight_rolloff(rgb.view(), &p),
        cuda::kernels::highlight_rolloff_device(&d_rgb, &p),
    );

    let p = ShadowRolloffParams::new(1.5, 0.5);
    rejects_identically(
        "shadow_rolloff knee=1.5",
        phaios_core::shadow_rolloff::shadow_rolloff(rgb.view(), &p),
        cuda::kernels::shadow_rolloff_device(&d_rgb, &p),
    );

    let p = VignetteParams::new(0.4, 1.5, 0.5);
    rejects_identically(
        "vignette feather=1.5",
        phaios_core::vignette::vignette(rgb.view(), &p),
        cuda::kernels::vignette_device(&d_rgb, &p),
    );

    let p = GrainParams::new(-1.0, 2.0, 1);
    rejects_identically(
        "film_grain intensity=-1",
        phaios_core::film_grain::film_grain(luma.view(), &p),
        cuda::kernels::film_grain_device(&d_luma, &p),
    );

    let p = GuidedFilterParams::new(2, -1.0);
    rejects_identically(
        "local_contrast eps=-1",
        phaios_core::local_contrast::local_contrast(luma.view(), &p, 0.5),
        cuda::kernels::local_contrast_device(&d_luma, &p, 0.5),
    );

    let p = SplitToningParams::new([0.0; 3], [0.0; 3], 1.5, 0.0);
    rejects_identically(
        "split_toning pivot=1.5",
        phaios_core::split_toning::split_toning(luma.view(), &p),
        cuda::kernels::split_toning_device(&d_luma, &p),
    );

    let mut offsets = HashMap::new();
    offsets.insert(11_i32, 1.0_f32);
    let p = ZoneParams::new(offsets);
    rejects_identically(
        "zone_system zone=11",
        phaios_core::tone::zone_system(luma.view(), &p),
        cuda::kernels::zone_system_device(&d_luma, &p),
    );

    let p = HistogramParams::new(1, 0.0, 1.0);
    rejects_identically(
        "histogram bins=1",
        phaios_core::histogram::histogram(rgb.view(), &p),
        cuda::kernels::histogram_device(&d_rgb, &p),
    );

    let short = ndarray::Array1::<f32>::zeros(1);
    let p = LutParams::new(0.0, 1.0);
    rejects_identically(
        "apply_lut lut.len()=1",
        phaios_core::lut::apply_lut(rgb.view(), short.view(), &p),
        cuda::kernels::apply_lut_device(&d_rgb, short.view(), &p),
    );

    let p = CropParams::new(0, 0, 100, 100);
    rejects_identically(
        "crop exceeds the frame",
        phaios_core::geometry::crop(rgb.view(), &p),
        cuda::kernels::crop_device(&d_rgb, &p),
    );

    let p = ResizeParams::new(0, 4, ResizeFilter::Area);
    rejects_identically(
        "resize width=0",
        phaios_core::geometry::resize(rgb.view(), &p),
        cuda::kernels::resize_device(&d_rgb, &p),
    );

    let p = HslWeightedParams::new([0.0; 8], LuminanceStandard::Bt709, 0.0);
    rejects_identically(
        "hsl_bw sigma_deg=0",
        phaios_core::bw::hsl_bw(rgb.view(), &p),
        cuda::kernels::hsl_bw_device(&d_rgb, &p),
    );

    // ── shape guards ────────────────────────────────────────────────
    //
    // The B&W entry points duplicate the channel check inline in the
    // .cu-facing Rust rather than calling `bw::validate_rgb`, so the
    // two messages agree only by both being written out by hand. That
    // is exactly what needs pinning: without it, a (H, W, 1) upload is
    // read as if it held three channels.

    rejects_identically(
        "luminance_bw on (H, W, 1)",
        phaios_core::bw::luminance_bw(luma.view(), LuminanceStandard::Bt709),
        cuda::kernels::luminance_bw_device(&d_luma, LuminanceStandard::Bt709),
    );

    rejects_identically(
        "channel_mixer_bw on (H, W, 1)",
        phaios_core::bw::channel_mixer_bw(luma.view(), [0.3, 0.6, 0.1]),
        cuda::kernels::channel_mixer_bw_device(&d_luma, [0.3, 0.6, 0.1]),
    );

    rejects_identically(
        "color_filter_bw on (H, W, 1)",
        phaios_core::bw::color_filter_bw(
            luma.view(),
            ColorFilter::Red25A,
            LuminanceStandard::Bt709,
        ),
        cuda::kernels::color_filter_bw_device(
            &d_luma,
            ColorFilter::Red25A,
            LuminanceStandard::Bt709,
        ),
    );

    let p = HslWeightedParams::new([0.0; 8], LuminanceStandard::Bt709, 30.0);
    rejects_identically(
        "hsl_bw on (H, W, 1)",
        phaios_core::bw::hsl_bw(luma.view(), &p),
        cuda::kernels::hsl_bw_device(&d_luma, &p),
    );

    // The luminance-only kernels, given RGB.

    let p = GrainParams::new(0.2, 2.0, 1);
    rejects_identically(
        "film_grain on (H, W, 3)",
        phaios_core::film_grain::film_grain(rgb.view(), &p),
        cuda::kernels::film_grain_device(&d_rgb, &p),
    );

    let p = GuidedFilterParams::new(2, 0.01);
    rejects_identically(
        "local_contrast on (H, W, 3)",
        phaios_core::local_contrast::local_contrast(rgb.view(), &p, 0.5),
        cuda::kernels::local_contrast_device(&d_rgb, &p, 0.5),
    );

    let p = SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, 0.0);
    rejects_identically(
        "split_toning on (H, W, 3)",
        phaios_core::split_toning::split_toning(rgb.view(), &p),
        cuda::kernels::split_toning_device(&d_rgb, &p),
    );

    let mut offsets = HashMap::new();
    offsets.insert(5_i32, 1.0_f32);
    let p = ZoneParams::new(offsets);
    rejects_identically(
        "zone_system on (H, W, 3)",
        phaios_core::tone::zone_system(rgb.view(), &p),
        cuda::kernels::zone_system_device(&d_rgb, &p),
    );
}

/// Gaussian blur, across the crossover in both directions.
///
/// The committed bound is (1e-5, 1e-7) — the crate's tolerance for a
/// one-transcendental kernel — rather than `local_contrast`'s looser
/// (1e-4, 1e-6), because the measured worst case is 0.038x of the looser
/// one and a bound that is never approached cannot catch a regression.
///
/// Both paths are exercised: direct convolution below σ = 4 (where the
/// device only has to sum the same weights in a different order) and
/// three box passes at or above it (where it also swaps f64 accumulation
/// for Kahan-compensated f32).
#[test]
fn blur_agrees_within_bound_on_both_paths() {
    use phaios_core::blur::{BlurParams, BlurShape, blur};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(129, 173, 3);

    for sigma in [0.5_f32, 0.8, 1.5, 2.5, 3.9, 4.0, 6.0, 12.0, 30.0] {
        let params = BlurParams::new(sigma, BlurShape::Gaussian);
        let cpu = blur(img.view(), &params).unwrap();
        let gpu = cuda::kernels::blur(&ctx, img.view(), &params).unwrap();
        let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
        assert!(
            v <= 1.0,
            "blur sigma={sigma}: {v:.2}x the (1e-5, 1e-7) bound"
        );
    }
}

/// The same agreement, on input that actually spans a scene's dynamic
/// range.
///
/// `pseudo_random_image` draws from `[0, 1)`, so nothing above ever put a
/// bright sample and a dark one in the same sliding window — and that is
/// precisely what the box path could not survive. With an f32 accumulator
/// holding ~1e8, the ~1e-4 samples entering behind it were annihilated on
/// contact, and when the bright sample left the window their contribution
/// was gone: 164% relative error against an exact oracle at a 1e4
/// highlight, and 2.8e7 times this bound at 1e8. Kahan compensation does
/// not help, because its error bound scales with Σ|xᵢ|, which the bright
/// sample dominates.
///
/// A dark field with a specular bar is not an exotic input for this
/// crate. It is linear scene-referred data with a highlight in it.
#[test]
fn blur_agrees_within_bound_across_the_dynamic_range() {
    use phaios_core::blur::{BlurParams, BlurShape, blur};

    let Some(ctx) = try_context() else { return };

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let mut img = ndarray::Array3::<f32>::from_elem((97, 131, 1), 1e-4);
        img.slice_mut(ndarray::s![40..44, 20..110, ..])
            .fill(highlight);

        // Both paths: below the crossover a direct convolution, at or above
        // it the sliding window that was wrong.
        for sigma in [3.0_f32, 12.0, 30.0] {
            let params = BlurParams::new(sigma, BlurShape::Gaussian);
            let cpu = blur(img.view(), &params).unwrap();
            let gpu = cuda::kernels::blur(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
            assert!(
                v <= 1.0,
                "blur sigma={sigma} with a {highlight:e} highlight: \
                 {v:.4}x the (1e-5, 1e-7) bound"
            );
        }
    }
}

/// The identity path must be bit-exact on both backends: σ = 0 is a copy,
/// not a filter, and a copy has no rounding to disagree about.
#[test]
fn blur_identity_is_bit_exact() {
    use phaios_core::blur::{BlurParams, BlurShape, blur};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(64, 96, 3);
    let params = BlurParams::new(0.0, BlurShape::Gaussian);
    assert_eq!(
        blur(img.view(), &params).unwrap(),
        cuda::kernels::blur(&ctx, img.view(), &params).unwrap()
    );
    // And it really is the input, not merely equal across backends.
    assert_eq!(blur(img.view(), &params).unwrap(), img);
}

/// Degenerate shapes and non-finite samples must agree too — the single
/// row and single column cases are where a separable filter's axis
/// handling goes wrong.
#[test]
fn blur_degenerate_shapes_agree() {
    use phaios_core::blur::{BlurParams, BlurShape, blur};

    let Some(ctx) = try_context() else { return };
    for (h, w, c) in [(1, 1, 1), (1, 33, 3), (33, 1, 3), (2, 2, 1), (17, 5, 3)] {
        let img = pseudo_random_image(h, w, c);
        for sigma in [1.5_f32, 6.0] {
            let params = BlurParams::new(sigma, BlurShape::Gaussian);
            let cpu = blur(img.view(), &params).unwrap();
            let gpu = cuda::kernels::blur(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
            assert!(
                v <= 1.0,
                "blur {h}x{w}x{c} sigma={sigma}: {v:.2}x the bound"
            );
        }
    }
}

/// Glow across its three parameter regimes. The bound is inherited from
/// the blur between the two element-wise halves, so it is the blur's
/// (1e-5, 1e-7) rather than anything looser.
#[test]
fn glow_agrees_within_bound() {
    use phaios_core::glow::{GlowParams, glow};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(97, 131, 3).mapv(|v| v * 2.0);

    for (threshold, sigma, amount) in [
        (0.0_f32, 8.0_f32, 0.0_f32), // the identity fast path
        (0.8, 8.0, 0.35),            // halation
        (0.5, 20.0, 0.4),            // diffusion
        (0.0, 400.0, 0.06),          // veiling glare, sigma beyond the frame
        (3.0, 4.0, 0.5),             // threshold above everything present
    ] {
        let params = GlowParams::new(threshold, sigma, amount);
        let cpu = glow(img.view(), &params).unwrap();
        let gpu = cuda::kernels::glow(&ctx, img.view(), &params).unwrap();
        let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
        assert!(
            v <= 1.0,
            "glow threshold={threshold} sigma={sigma} amount={amount}: {v:.2}x the bound"
        );
    }
}

/// The same agreement, on input that actually spans a scene's dynamic
/// range.
///
/// `glow` is the crate's third kernel with an accumulator over many
/// samples — it *is* a blur, wrapped in two element-wise halves — and
/// when the audit found the box path wrong by 2.8e7x and
/// `local_contrast` by 51.8x, this one had never been shown anything
/// above 2.0. `glow_agrees_within_bound` draws from
/// `pseudo_random_image`, so no sliding window here has ever held a
/// bright sample and a dark one at the same time.
///
/// Two things make this more than the blur's sweep relabelled. The
/// threshold subtraction happens *before* the blur, so `threshold = 0`
/// (veiling glare, where every part of the scene scatters) is the only
/// setting that leaves the 1e-4 field in the accumulator at all — a
/// threshold above the field zeroes it exactly, and the sweep covers
/// both so the two cannot be confused. And the final `in + amount·spread`
/// puts the blur's error over an output that includes the *unblurred*
/// input, so the ratio a violation is scored against is not the blur's.
///
/// Measured worst case across the sweep: 0.0510x the (1e-5, 1e-7)
/// bound, at threshold 0, σ = 12, amount 0.35 and a 1e8 highlight —
/// the box path with the 1e-4 field still in the accumulator, which is
/// exactly the combination the paragraph above predicts — and since
/// that is the sweep's maximum, the thresholded settings scored no
/// higher, as they must.
#[test]
fn glow_agrees_within_bound_across_the_dynamic_range() {
    use phaios_core::glow::{GlowParams, glow};

    let Some(ctx) = try_context() else { return };
    let mut worst = 0.0_f32;
    let mut worst_where = String::new();

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let mut img = ndarray::Array3::<f32>::from_elem((97, 131, 1), 1e-4);
        img.slice_mut(ndarray::s![40..44, 20..110, ..])
            .fill(highlight);

        // σ = 3 is the direct convolution; 12, 30 and 400 are the box
        // path — the one that could not survive a bright sample leaving
        // a running f32 total.
        for (threshold, sigma, amount) in [
            (0.0_f32, 3.0_f32, 0.35_f32), // halation, direct path
            (0.0, 12.0, 0.35),            // halation, box path
            (0.0, 30.0, 0.4),             // diffusion
            (0.0, 400.0, 0.06),           // veiling glare, σ beyond the frame
            (0.5, 12.0, 0.4),             // threshold above the field: the safe case
            (0.5, 400.0, 0.06),
        ] {
            let params = GlowParams::new(threshold, sigma, amount);
            let cpu = glow(img.view(), &params).unwrap();
            let gpu = cuda::kernels::glow(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
            if v > worst {
                worst = v;
                worst_where = format!("t={threshold} sigma={sigma} a={amount} hl={highlight:e}");
            }
            assert!(
                v <= 1.0,
                "glow threshold={threshold} sigma={sigma} amount={amount} \
                 with a {highlight:e} highlight: {v:.4}x the (1e-5, 1e-7) bound"
            );
        }
    }

    eprintln!("glow HDR sweep: worst {worst:.4}x the bound ({worst_where})");
}

/// A NaN sample stays where it is, on both backends.
///
/// `glow::scatter_weight` tests `excess > 0.0` rather than taking a
/// `max` so that a NaN fails the test and contributes 0 to the blur.
/// That reasoning is written out *twice* — once in Rust and once in
/// `src/cuda/ptx/glow.cu` — so either copy can rot alone, and no test
/// on either backend ever passed glow a non-finite sample.
///
/// It matters because the blur is separable: a NaN that gets in becomes
/// a NaN row after the horizontal pass and a NaN frame after the
/// vertical one, so one dead sensor pixel would take the whole image
/// with it.
///
/// `docs/ffi.md` §1 leaves non-finite samples unspecified in general.
/// This is the deliberate exception, and the exception is the thing
/// worth pinning.
#[test]
fn glow_contains_a_nan_sample_on_both_backends() {
    use phaios_core::glow::{GlowParams, glow};

    let Some(ctx) = try_context() else { return };

    let mut img = Array3::<f32>::from_elem((9, 9, 1), 0.5);
    img[[4, 4, 0]] = f32::NAN;
    let params = GlowParams::new(0.2, 2.0, 1.0);

    let cpu = glow(img.view(), &params).unwrap();
    let gpu = cuda::kernels::glow(&ctx, img.view(), &params).unwrap();

    for (backend, out) in [("cpu", &cpu), ("gpu", &gpu)] {
        let spread = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            spread, 1,
            "{backend}: the NaN spread to {spread} of 81 outputs"
        );
        assert!(
            out[[4, 4, 0]].is_nan(),
            "{backend}: the poisoned sample itself must stay NaN"
        );
    }
}

/// `amount = 0` is a copy on both backends, so it must be bit-exact.
#[test]
fn glow_identity_is_bit_exact() {
    use phaios_core::glow::{GlowParams, glow};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(48, 64, 3);
    let params = GlowParams::new(0.5, 8.0, 0.0);
    assert_eq!(
        glow(img.view(), &params).unwrap(),
        cuda::kernels::glow(&ctx, img.view(), &params).unwrap()
    );
    assert_eq!(glow(img.view(), &params).unwrap(), img);
}

// ── sharpen: one pointwise kernel after the shared device blur ──────────────

/// Sharpen across amount/sigma/threshold. The bound is inherited from
/// the blur underneath the pointwise gate-and-combine kernel, so it is
/// the blur's (1e-5, 1e-7) rather than anything looser — the pointwise
/// half is bit-exact (no transcendentals, `-fmad=false`; see
/// `src/cuda/ptx/sharpen.cu`).
#[test]
fn sharpen_agrees_within_bound() {
    use phaios_core::sharpen::{SharpenParams, sharpen};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(97, 131, 3).mapv(|v| v * 2.0);

    let mut worst = 0.0_f32;
    let mut worst_where = String::new();
    for (amount, sigma, threshold) in [
        (0.0_f32, 8.0_f32, 0.0_f32), // the identity fast path
        (0.35, 1.5, 0.0),            // direct blur path, gate off (threshold=0)
        (0.5, 8.0, 0.0),             // box blur path, gate off
        (0.4, 1.2, 0.05),            // direct path, gate engaged
        (0.3, 10.0, 0.2),            // box path, gate engaged
    ] {
        let params = SharpenParams::new(amount, sigma, threshold);
        let cpu = sharpen(img.view(), &params).unwrap();
        let gpu = cuda::kernels::sharpen(&ctx, img.view(), &params).unwrap();
        let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
        if v > worst {
            worst = v;
            worst_where = format!("amount={amount} sigma={sigma} threshold={threshold}");
        }
        assert!(
            v <= 1.0,
            "sharpen amount={amount} sigma={sigma} threshold={threshold}: {v:.2}x the bound"
        );
    }
    eprintln!("sharpen unit-range sweep: worst {worst:.4}x the bound ({worst_where})");
}

/// The same agreement, on input that actually spans a scene's dynamic
/// range — `blur`'s and `glow`'s own reason (`docs/ffi.md` §6): the box
/// path's f64-vs-Kahan-f32 divergence only shows up when a bright sample
/// and a dark one share a sliding window, which `pseudo_random_image`
/// never produces.
///
/// `detail = img - blurred` inherits the blur's own error at the
/// highlight edge, so this is the same stress `blur_agrees_within_bound
/// _across_the_dynamic_range` and `glow_agrees_within_bound_across_the
/// _dynamic_range` apply, one stage later.
#[test]
fn sharpen_agrees_within_bound_across_the_dynamic_range() {
    use phaios_core::sharpen::{SharpenParams, sharpen};

    let Some(ctx) = try_context() else { return };
    let mut worst = 0.0_f32;
    let mut worst_where = String::new();

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let mut img = ndarray::Array3::<f32>::from_elem((97, 131, 1), 1e-4);
        img.slice_mut(ndarray::s![40..44, 20..110, ..])
            .fill(highlight);

        // sigma 3 is the direct convolution (below BOX_CROSSOVER_SIGMA =
        // 6); 12 and 30 are the box path — the one that could not
        // survive a bright sample leaving a running f32 total.
        for (amount, sigma, threshold) in [
            (0.35_f32, 3.0_f32, 0.0_f32),
            (0.35, 12.0, 0.0),
            (0.4, 30.0, 0.0),
            (0.4, 12.0, 0.5),
            (0.4, 30.0, 0.5),
        ] {
            let params = SharpenParams::new(amount, sigma, threshold);
            let cpu = sharpen(img.view(), &params).unwrap();
            let gpu = cuda::kernels::sharpen(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
            if v > worst {
                worst = v;
                worst_where =
                    format!("amount={amount} sigma={sigma} threshold={threshold} hl={highlight:e}");
            }
            assert!(
                v <= 1.0,
                "sharpen amount={amount} sigma={sigma} threshold={threshold} \
                 with a {highlight:e} highlight: {v:.4}x the (1e-5, 1e-7) bound"
            );
        }
    }

    eprintln!("sharpen HDR sweep: worst {worst:.4}x the bound ({worst_where})");
}

/// `amount = 0` and `sigma = 0` are each a copy on both backends, so
/// both fast paths must be bit-exact.
#[test]
fn sharpen_identity_is_bit_exact() {
    use phaios_core::sharpen::{SharpenParams, sharpen};

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(48, 64, 3);

    let amount_zero = SharpenParams::new(0.0, 3.0, 0.1);
    assert_eq!(
        sharpen(img.view(), &amount_zero).unwrap(),
        cuda::kernels::sharpen(&ctx, img.view(), &amount_zero).unwrap()
    );
    assert_eq!(sharpen(img.view(), &amount_zero).unwrap(), img);

    let sigma_zero = SharpenParams::new(0.5, 0.0, 0.1);
    assert_eq!(
        sharpen(img.view(), &sigma_zero).unwrap(),
        cuda::kernels::sharpen(&ctx, img.view(), &sigma_zero).unwrap()
    );
    assert_eq!(sharpen(img.view(), &sigma_zero).unwrap(), img);
}

/// Degenerate shapes must agree too — the single row and single column
/// cases are where a separable filter's axis handling goes wrong, and
/// `sharpen` inherits that risk entirely from the blur beneath it.
#[test]
fn sharpen_degenerate_shapes_agree() {
    use phaios_core::sharpen::{SharpenParams, sharpen};

    let Some(ctx) = try_context() else { return };
    for (h, w, c) in [(1, 1, 1), (1, 33, 3), (33, 1, 3), (2, 2, 1), (17, 5, 3)] {
        let img = pseudo_random_image(h, w, c);
        for (amount, sigma, threshold) in [(0.5_f32, 1.5_f32, 0.0_f32), (0.4, 6.0, 0.1)] {
            let params = SharpenParams::new(amount, sigma, threshold);
            let cpu = sharpen(img.view(), &params).unwrap();
            let gpu = cuda::kernels::sharpen(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-5, 1e-7);
            assert!(
                v <= 1.0,
                "sharpen {h}x{w}x{c} amount={amount} sigma={sigma} threshold={threshold}: \
                 {v:.2}x the bound"
            );
        }
    }
}

// ── hot_pixels: comparison-only, no arithmetic in the median ────────────────

/// Assert every element of `cpu` and `gpu` carries the identical bit
/// pattern -- stricter than `assert_eq!` on the arrays directly, which
/// treats NaN as unequal to itself even when the bits match (see
/// `non_finite_pixels_agree_across_backends` above). `hot_pixels` is
/// committed to full bit-exactness (`docs/ffi.md` §6), not a tolerance,
/// so every test in this section uses this rather than `worst_violation`.
fn assert_bit_exact(cpu: &Array3<f32>, gpu: &Array3<f32>, label: &str) {
    assert_eq!(cpu.dim(), gpu.dim(), "{label}: shape mismatch");
    for ((idx, c), g) in cpu.indexed_iter().zip(gpu.iter()) {
        assert_eq!(
            c.to_bits(),
            g.to_bits(),
            "{label}: diverged at {idx:?}: cpu={c}, gpu={g}"
        );
    }
}

/// Pseudo-random image across several `(threshold, relative)` pairs,
/// including `0.0/0.0` -- the unconditional median, where every pixel
/// takes the replace branch. No tolerance: the median step is
/// IEEE-754-2008 `minNum`/`maxNum` comparisons only (`f32::min`/
/// `f32::max` on the CPU, `fminf`/`fmaxf` on the device), and the one
/// arithmetic step (`threshold + relative * m.abs()`, then
/// `(p - m).abs() > limit`) is ordinary correctly-rounded `f32`
/// arithmetic with `-fmad=false` on both sides -- see
/// `src/hot_pixels.rs`'s module documentation and
/// `src/cuda/ptx/hot_pixels.cu`.
#[test]
fn hot_pixels_is_bit_exact() {
    use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

    let Some(ctx) = try_context() else { return };
    for c in [1, 3] {
        let img = pseudo_random_image(257, 389, c);
        for (threshold, relative) in [
            (0.0_f32, 0.0_f32), // unconditional median
            (0.05, 0.0),
            (0.0, 0.05),
            (0.05, 0.02),
            (1.0, 1.0), // above the image's own range: the identity in practice
        ] {
            let params = HotPixelParams::new(threshold, relative);
            let cpu = hot_pixels(img.view(), &params).unwrap();
            let gpu = cuda::kernels::hot_pixels(&ctx, img.view(), &params).unwrap();
            assert_bit_exact(
                &cpu,
                &gpu,
                &format!("C={c} threshold={threshold} relative={relative}"),
            );
        }
    }
}

/// The same agreement on linear scene-referred data spanning a real
/// dynamic range -- `field_with_bar`, the input that broke `blur` and
/// `local_contrast` (see those kernels' own HDR sweeps elsewhere in this
/// file). `hot_pixels` carries no accumulation of any kind, so this is
/// not expected to find anything the unit-range sweep above didn't --
/// asserted anyway because the plan calls for checking, not assuming
/// ("checked, not assumed", `src/cuda/ptx/local_contrast.cu`).
#[test]
fn hot_pixels_agrees_within_the_dynamic_range() {
    use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

    let Some(ctx) = try_context() else { return };
    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let img = field_with_bar(64, 96, 1e-4, highlight);
        for (threshold, relative) in [(0.0_f32, 0.0_f32), (0.05, 0.02)] {
            let params = HotPixelParams::new(threshold, relative);
            let cpu = hot_pixels(img.view(), &params).unwrap();
            let gpu = cuda::kernels::hot_pixels(&ctx, img.view(), &params).unwrap();
            assert_bit_exact(
                &cpu,
                &gpu,
                &format!("highlight={highlight:e} threshold={threshold} relative={relative}"),
            );
        }
    }
}

/// A smooth ramp (not a flat field, per CONTRIBUTING.md's warning about
/// constant-image tests) with one bright and one dark planted outlier,
/// stressing the replace-vs-keep decision boundary the way the
/// pseudo-random sweep above cannot: neighbouring samples there are
/// unrelated by construction, so a tight threshold makes nearly every
/// pixel look like an outlier. On a ramp, only the two planted pixels
/// take the replace branch (mirrors the CPU-only ramp tests added in
/// `tests/kernels.rs` step 1, generalised here to a CPU/GPU comparison).
#[test]
fn hot_pixels_ramp_with_planted_outliers_is_bit_exact() {
    use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

    let Some(ctx) = try_context() else { return };
    let (h, w) = (9, 15);
    let mut img = Array3::from_shape_fn((h, w, 1), |(_, x, _)| 0.1_f32 + 0.02 * x as f32);
    img[[4, 7, 0]] = 50.0; // bright outlier
    img[[4, 10, 0]] = -50.0; // dark outlier

    for (threshold, relative) in [(0.0_f32, 0.0_f32), (0.1, 0.05)] {
        let params = HotPixelParams::new(threshold, relative);
        let cpu = hot_pixels(img.view(), &params).unwrap();
        let gpu = cuda::kernels::hot_pixels(&ctx, img.view(), &params).unwrap();
        assert_bit_exact(
            &cpu,
            &gpu,
            &format!("ramp threshold={threshold} relative={relative}"),
        );
    }
}

/// Degenerate shapes must agree too, at both `C = 1` and `C = 3`: `1x1`
/// has no distinct neighbour at all (every clamped offset reads the
/// single pixel back), and `1xN`/`Nx1` collapse one axis of the border
/// clamp entirely.
#[test]
fn hot_pixels_degenerate_shapes_agree() {
    use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

    let Some(ctx) = try_context() else { return };
    for (h, w) in [(1, 1), (1, 33), (33, 1), (2, 2)] {
        for c in [1, 3] {
            let img = pseudo_random_image(h, w, c);
            for (threshold, relative) in [(0.0_f32, 0.0_f32), (0.05, 0.02)] {
                let params = HotPixelParams::new(threshold, relative);
                let cpu = hot_pixels(img.view(), &params).unwrap();
                let gpu = cuda::kernels::hot_pixels(&ctx, img.view(), &params).unwrap();
                assert_bit_exact(
                    &cpu,
                    &gpu,
                    &format!("{h}x{w}x{c} threshold={threshold} relative={relative}"),
                );
            }
        }
    }
}

/// An image containing one NaN pixel and one +Inf pixel, far enough
/// apart that their 3x3 neighbourhoods do not overlap, run through both
/// backends. `docs/ffi.md` §1 leaves non-finite input unspecified in
/// general, but `hot_pixels` is a documented exception the plan predicts
/// (`.cache/scratch/denoise/PLAN.md`, "Determinism"): the median step is
/// IEEE-754-2008 `minNum`/`maxNum` comparisons only, which both
/// `f32::min`/`f32::max` and `fminf`/`fmaxf` implement identically, and
/// the kept-vs-replaced branch is decided by an ordinary `>` comparison
/// that is `false` whenever a NaN reaches it on either backend -- so a
/// single non-finite sample cannot make the two backends disagree here.
/// If this ever fails, the doc paragraph is wrong and should be
/// corrected -- not this test loosened.
#[test]
fn hot_pixels_agrees_bit_for_bit_on_nan_and_inf_input() {
    use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

    let Some(ctx) = try_context() else { return };
    let mut img = Array3::<f32>::from_elem((9, 9, 1), 0.5_f32);
    img[[2, 2, 0]] = f32::NAN;
    img[[6, 6, 0]] = f32::INFINITY;

    for (threshold, relative) in [(0.0_f32, 0.0_f32), (0.1, 0.05)] {
        let params = HotPixelParams::new(threshold, relative);
        let cpu = hot_pixels(img.view(), &params).unwrap();
        let gpu = cuda::kernels::hot_pixels(&ctx, img.view(), &params).unwrap();
        assert_bit_exact(
            &cpu,
            &gpu,
            &format!("nan/inf threshold={threshold} relative={relative}"),
        );
    }
}

// ── denoise: local_contrast's own bound, confirmed for the new cross- ──────
// ── guided kernels too ──────────────────────────────────────────────────────

/// `denoise`'s self-guided path at `C == 1` is a direct call to
/// `local_contrast_device` with `strength = -amount` -- the same
/// exact-negation argument `src/denoise.rs`'s module documentation gives
/// for the CPU kernel (negating one multiply operand is an exact sign
/// flip) applies bit for bit on the device too, since both device entry
/// points round the identical sequence of operations. `amount` is kept
/// away from `0.0` throughout so every case here actually reaches that
/// shared computation rather than `denoise_device`'s own identity fast
/// path.
#[test]
fn denoise_device_c1_is_bit_exact_with_local_contrast_device() {
    use phaios_core::bw::LuminanceStandard;
    use phaios_core::denoise::DenoiseParams;
    use phaios_core::local_contrast::GuidedFilterParams;

    let Some(ctx) = try_context() else { return };
    let img = pseudo_random_image(97, 131, 1);
    let d_img = ctx.upload(img.view()).unwrap();

    for (radius, noise_sigma, amount) in [
        (0_u32, 0.0_f32, 1.0_f32),
        (4, 0.02, 0.6),
        (8, 0.1, 1.0),
        (phaios_core::denoise::MAX_RADIUS, 0.0, 0.35),
        (2, 0.5, 0.01),
    ] {
        let denoise_params =
            DenoiseParams::new(radius, noise_sigma, amount, LuminanceStandard::Bt709);
        let via_denoise = cuda::kernels::denoise_device(&d_img, &denoise_params).unwrap();
        let via_denoise = ctx.download(&via_denoise).unwrap();

        let eps = noise_sigma * noise_sigma;
        let gf = GuidedFilterParams::new(radius, eps);
        let via_local_contrast =
            cuda::kernels::local_contrast_device(&d_img, &gf, -amount).unwrap();
        let via_local_contrast = ctx.download(&via_local_contrast).unwrap();

        assert_eq!(
            via_denoise, via_local_contrast,
            "radius={radius} noise_sigma={noise_sigma} amount={amount}: \
             denoise_device(C=1) diverged from local_contrast_device(strength=-amount)"
        );
    }
}

/// GPU `denoise` agrees with the CPU oracle within the guided filter's
/// own bound (rtol 1e-4, atol 1e-6, `docs/ffi.md` §6) -- confirmed
/// empirically for both channel counts `denoise` dispatches specially:
/// `C == 1` (self-guided, inherits the bound automatically since it is
/// the very same device kernel) and `C == 3` (cross-guided, three new
/// kernels adding one new cancelling subtraction, `cov(I, p_c)`, the
/// self-guided path never exercises). Prints the worst ratio to the
/// bound with `--nocapture`.
///
/// `radius = 0` is deliberately paired with a *nonzero* `noise_sigma`
/// here (`0.05`, not `0.0`): at `radius = 0` **and** `eps` exactly
/// `0.0`, `C == 3`'s `a_c = cov / (var_i + eps)` divides two
/// independently-SAT-rounded near-zero residuals by each other on the
/// CPU side (`var_i`/`cov` are each mathematically exactly `0.0` at a
/// single-pixel window, but the SAT's four-corner query does not
/// recover that *exactly*, unlike a direct single-term sum) --
/// measured at up to 222069x the bound on this test's own image, a
/// genuine CPU-side instability, investigated and confirmed unrelated
/// to the device kernels (see
/// `denoise_cross_guided_diverges_from_the_cpu_at_extreme_highlights`
/// for the same investigation at the other pathological corner, extreme
/// highlights). The exact `radius = 0` identity itself is separately
/// pinned, bit-exact against the input directly rather than against
/// this CPU oracle, by `denoise_degenerate_shapes_and_identity_agree`.
#[test]
fn denoise_agrees_with_cpu_oracle() {
    use phaios_core::bw::LuminanceStandard;
    use phaios_core::denoise::{DenoiseParams, denoise};

    let Some(ctx) = try_context() else { return };

    for &c in &[1_usize, 3] {
        let img = pseudo_random_image(89, 113, c);
        let mut worst = 0.0_f32;
        let mut worst_where = String::new();
        for (radius, noise_sigma, amount) in [
            (0_u32, 0.05_f32, 1.0_f32),
            (2, 0.02, 0.6),
            (4, 0.05, 1.0),
            (8, 0.01, 0.4),
            (32, 0.1, 0.8),
        ] {
            let params = DenoiseParams::new(radius, noise_sigma, amount, LuminanceStandard::Bt709);
            let cpu = denoise(img.view(), &params).unwrap();
            let gpu = cuda::kernels::denoise(&ctx, img.view(), &params).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
            if v > worst {
                worst = v;
                worst_where =
                    format!("C={c} radius={radius} noise_sigma={noise_sigma} amount={amount}");
            }
            assert!(
                v <= 1.0,
                "C={c} radius={radius} noise_sigma={noise_sigma} amount={amount}: \
                 {v:.4}x the (1e-4, 1e-6) bound"
            );
        }
        eprintln!("denoise unit-range sweep (C={c}): worst {worst:.4}x the bound ({worst_where})");
    }
}

/// The same agreement on input that spans a scene's dynamic range --
/// `field_with_bar`, the input that broke `local_contrast` before its
/// L/L² fix (51.8× the bound, `src/cuda/ptx/local_contrast.cu`), for
/// both `C == 1` and a uniform bar shared by all three channels at
/// `C == 3`. Small radius with a large `noise_sigma` (hence a large
/// `eps`) is the worst case, mirroring
/// `local_contrast_agrees_within_bound_across_the_dynamic_range`
/// exactly -- the three `noise_sigma` values are chosen so
/// `noise_sigma²` lands on that test's own `eps` sweep, `{1e-4, 0.01,
/// 0.5}`.
///
/// Both `C == 1` and `C == 3` are asserted across the full required
/// highlight range, `{1e0, 1e2, 1e4, 1e6, 1e8}`. `C == 1` inherits
/// `local_contrast_device`'s own already-proven behaviour there
/// directly. `C == 3` held only across `{1e0, 1e2, 1e4}` before the CPU
/// kernel summed its window statistics directly instead of through a
/// global summed-area table (`.cache/scratch/denoise/PLAN.md`,
/// "Decision 2"): the missing residual at `{1e6, 1e8}` -- `cov(I, p_c)`
/// differencing two separately-accumulated global tables -- no longer
/// exists once neither side of that subtraction is global, so the full
/// range now holds here too.
///
/// See `denoise_cross_guided_agrees_within_bound_when_only_some_channels
/// _carry_the_bar` below for the harder `C == 3` case this sweep cannot
/// reach: here the bar is identical in every channel, so `cov(I, p_c)`
/// degenerates to `var(I)` exactly as the self-guided path's `cov(I, I)`
/// does, and never exercises a genuine covariance between two different
/// signals.
#[test]
fn denoise_agrees_within_bound_across_the_dynamic_range() {
    use phaios_core::bw::LuminanceStandard;
    use phaios_core::denoise::{DenoiseParams, denoise};

    let Some(ctx) = try_context() else { return };
    let (h, w) = (64, 96);

    let mut worst_c1 = 0.0_f32;
    let mut worst_c1_where = String::new();
    let mut worst_c3 = 0.0_f32;
    let mut worst_c3_where = String::new();

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let img_c1 = field_with_bar(h, w, 1e-4, highlight);
        let bar = field_with_bar(h, w, 1e-4, highlight);
        let mut img_c3 = Array3::<f32>::zeros((h, w, 3));
        for ch in 0..3 {
            img_c3.slice_mut(s![.., .., ch..ch + 1]).assign(&bar);
        }

        for radius in [1_u32, 2, 8, 32] {
            // sqrt(1e-4), sqrt(0.01), ~sqrt(0.5): the same eps sweep, roughly
            // local_contrast's own HDR test uses.
            for noise_sigma in [0.01_f32, 0.1, 0.7] {
                let params = DenoiseParams::new(radius, noise_sigma, 0.5, LuminanceStandard::Bt709);

                let cpu = denoise(img_c1.view(), &params).unwrap();
                let gpu = cuda::kernels::denoise(&ctx, img_c1.view(), &params).unwrap();
                let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
                if v > worst_c1 {
                    worst_c1 = v;
                    worst_c1_where =
                        format!("radius={radius} noise_sigma={noise_sigma} hl={highlight:e}");
                }
                assert!(
                    v <= 1.0,
                    "C=1 radius={radius} noise_sigma={noise_sigma} with a {highlight:e} \
                     highlight: {v:.4}x the (1e-4, 1e-6) bound"
                );

                let cpu = denoise(img_c3.view(), &params).unwrap();
                let gpu = cuda::kernels::denoise(&ctx, img_c3.view(), &params).unwrap();
                let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
                if v > worst_c3 {
                    worst_c3 = v;
                    worst_c3_where =
                        format!("radius={radius} noise_sigma={noise_sigma} hl={highlight:e}");
                }
                assert!(
                    v <= 1.0,
                    "C=3 radius={radius} noise_sigma={noise_sigma} with a {highlight:e} \
                     highlight: {v:.4}x the (1e-4, 1e-6) bound"
                );
            }
        }
    }

    eprintln!("denoise HDR sweep C=1: worst {worst_c1:.4}x the bound ({worst_c1_where})");
    eprintln!(
        "denoise HDR sweep C=3 (uniform bar): worst {worst_c3:.4}x the bound ({worst_c3_where})"
    );
}

/// The `C == 3` HDR case the sweep above cannot reach: the bright bar
/// present in only some channels (R and B; G stays dark everywhere), so
/// `cov(I, p_c)` is a genuine covariance between two *different*
/// bright/dark signals rather than the guide against itself. Generalises
/// `local_contrast.cu`'s own HDR lesson (`var(I)` needs f64 because nothing
/// cancels it) to `coeff_ab_cross`'s covariance term, which the
/// self-guided path and the uniform-bar sweep above cannot exercise at
/// all -- there, "cov" is always "var" of one signal.
///
/// Asserted across the full required highlight range,
/// `{1e0, 1e2, 1e4, 1e6, 1e8}`: this is the exact construction (bar in
/// only some channels) that diverged from the CPU oracle at `{1e6, 1e8}`
/// before the CPU kernel summed its window statistics directly instead
/// of through a global summed-area table
/// (`.cache/scratch/denoise/PLAN.md`, "Decision 2") -- with neither side
/// of `cov(I, p_c)`'s subtraction global any more, the missing residual
/// at extreme highlights no longer exists, and the device (unchanged
/// throughout) agrees with the fixed CPU here just as it always did at
/// the lower highlights.
#[test]
fn denoise_cross_guided_agrees_within_bound_when_only_some_channels_carry_the_bar() {
    use phaios_core::bw::LuminanceStandard;
    use phaios_core::denoise::{DenoiseParams, denoise};

    let Some(ctx) = try_context() else { return };
    let (h, w) = (64, 96);

    let mut worst = 0.0_f32;
    let mut worst_where = String::new();

    for highlight in [1.0_f32, 1e2, 1e4, 1e6, 1e8] {
        let dark = Array3::<f32>::from_elem((h, w, 1), 1e-4_f32);
        let bright = field_with_bar(h, w, 1e-4, highlight);
        let mut img = Array3::<f32>::zeros((h, w, 3));
        img.slice_mut(s![.., .., 0..1]).assign(&bright); // R: bright bar
        img.slice_mut(s![.., .., 1..2]).assign(&dark); // G: stays dark
        img.slice_mut(s![.., .., 2..3]).assign(&bright); // B: bright bar

        for radius in [1_u32, 2, 8, 32] {
            for noise_sigma in [0.01_f32, 0.1, 0.7] {
                let params = DenoiseParams::new(radius, noise_sigma, 0.5, LuminanceStandard::Bt709);
                let cpu = denoise(img.view(), &params).unwrap();
                let gpu = cuda::kernels::denoise(&ctx, img.view(), &params).unwrap();
                let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
                if v > worst {
                    worst = v;
                    worst_where =
                        format!("radius={radius} noise_sigma={noise_sigma} hl={highlight:e}");
                }
                assert!(
                    v <= 1.0,
                    "partial-channel bar radius={radius} noise_sigma={noise_sigma} with a \
                     {highlight:e} highlight: {v:.4}x the (1e-4, 1e-6) bound"
                );
            }
        }
    }

    eprintln!(
        "denoise HDR sweep C=3 (partial-channel bar): worst {worst:.4}x the bound ({worst_where})"
    );
}

/// Degenerate shapes and the two identity cases, across every channel
/// count the dispatch cares about: `C == 1` (the direct
/// `local_contrast_device` alias), `C == 3` (cross-guided), and two
/// channel counts that take neither special path (`C == 2`, `C == 4`,
/// routed through the host-round-trip branch documented in
/// `src/cuda/kernels/denoise.rs`).
///
/// `amount = 0` is the device-copy fast path, bit-exact by construction.
/// `radius = 0` collapses every window to one pixel: `var`/`cov` become
/// an exact `x - x = 0.0` on both backends for any input (worked out by
/// hand in `.cache/scratch/denoise/PROGRESS.md`: the same value cast or
/// multiplied twice via the same deterministic operation always produces
/// identical bits, so the cancelling subtraction is exactly zero, not
/// merely close to it), so it is *also* an exact identity -- against the
/// original input directly, not merely "close to the CPU" -- for any
/// `amount`, mirroring `local_contrast_radius_zero_is_bit_exact_identity`.
#[test]
fn denoise_degenerate_shapes_and_identity_agree() {
    use phaios_core::bw::LuminanceStandard;
    use phaios_core::denoise::{DenoiseParams, denoise};

    let Some(ctx) = try_context() else { return };

    for (h, w) in [(1, 1), (1, 33), (33, 1), (9, 13)] {
        for c in [1_usize, 2, 3, 4] {
            let img = pseudo_random_image(h, w, c);

            let zero_amount = DenoiseParams::new(4, 0.1, 0.0, LuminanceStandard::Bt709);
            let gpu = cuda::kernels::denoise(&ctx, img.view(), &zero_amount).unwrap();
            assert_eq!(gpu, img, "{h}x{w}x{c}: amount=0 must be the exact identity");

            let zero_radius = DenoiseParams::new(0, 0.1, 0.7, LuminanceStandard::Bt709);
            let gpu = cuda::kernels::denoise(&ctx, img.view(), &zero_radius).unwrap();
            assert_eq!(gpu, img, "{h}x{w}x{c}: radius=0 must be the exact identity");

            // A real (radius > 0, amount > 0) case still has to agree
            // with the CPU within the guided-filter bound on these same
            // tiny/degenerate shapes -- window clamping at a
            // single-row or single-column extent is exactly the kind of
            // edge case a box-sum implementation gets wrong, and C=2/4
            // additionally exercise the host-round-trip branch.
            let real = DenoiseParams::new(3, 0.05, 0.6, LuminanceStandard::Bt709);
            let cpu = denoise(img.view(), &real).unwrap();
            let gpu = cuda::kernels::denoise(&ctx, img.view(), &real).unwrap();
            let v = worst_violation(&cpu, &gpu, 1e-4, 1e-6);
            assert!(v <= 1.0, "{h}x{w}x{c}: {v:.4}x the (1e-4, 1e-6) bound");
        }
    }
}

/// `blur` and `glow` diverge on non-finite input **below the box
/// crossover**, and `docs/ffi.md` §1 says so. This pins the *shape* of
/// that divergence rather than papering over it: on the direct path the
/// device's Kahan compensation computes ∞ − ∞ and yields NaN where the
/// host's uncompensated f64 sum keeps ±∞. σ here is deliberately 1.5.
///
/// At or above the crossover the two agree — both accumulate the box
/// passes in f64, and a sliding window turns ∞ into NaN on either side.
///
/// If a future change makes these agree, this test fails and the doc
/// paragraph should be rewritten — that would be good news.
#[test]
fn blur_non_finite_is_documented_as_divergent() {
    use phaios_core::blur::{BlurParams, BlurShape, blur};

    let Some(ctx) = try_context() else { return };
    for fill in [f32::INFINITY, f32::NEG_INFINITY] {
        let img = ndarray::Array3::<f32>::from_elem((5, 7, 1), fill);
        let params = BlurParams::new(1.5, BlurShape::Gaussian);
        let cpu = blur(img.view(), &params).unwrap();
        let gpu = cuda::kernels::blur(&ctx, img.view(), &params).unwrap();

        assert!(
            cpu[[2, 3, 0]].is_infinite(),
            "the host keeps the infinity: {}",
            cpu[[2, 3, 0]]
        );
        assert!(
            gpu[[2, 3, 0]].is_nan(),
            "the device's compensation turns it into NaN: {}",
            gpu[[2, 3, 0]]
        );
    }

    // Finite input, by contrast, agrees inside the committed bound —
    // this is a non-finite-only divergence, not a broken kernel.
    let finite = pseudo_random_image(5, 7, 1);
    let params = BlurParams::new(1.5, BlurShape::Gaussian);
    let v = worst_violation(
        &blur(finite.view(), &params).unwrap(),
        &cuda::kernels::blur(&ctx, finite.view(), &params).unwrap(),
        1e-5,
        1e-7,
    );
    assert!(v <= 1.0, "finite input must still agree: {v:.2}x the bound");
}

/// How far the shipped f32 resample sits from the mathematically intended
/// answer, measured against a reference that shares no arithmetic with it.
///
/// `resize_f32_accumulation_stays_within_its_error_bound` deliberately
/// reuses the kernel's own f32 tap geometry, so it isolates accumulation
/// error and can hold it to a derived `n·u·mass` bound. That is the right
/// question for "is the running total wide enough", and the wrong one for
/// "is the answer right": an error in the *weights* is invisible to an
/// oracle that computes the weights the same way.
///
/// This one shares nothing — scale, tap positions, support, weights and
/// accumulation are all f64 — so it measures the total distance from the
/// intended result. The bound is empirical rather than derived: a forward
/// bound covering weight evaluation as well as summation would be far
/// looser than what the kernel achieves and would assert almost nothing.
///
/// It is the *weaker* detector of the two, which is worth stating plainly
/// because the opposite is the natural assumption. Measured by mutating
/// `filter_eval`: perturbing the Keys `a` coefficient to 1.5001 fails
/// both, but adding 1e-6 to the constant term fails only the accumulation
/// test. That test re-implements the weights in f32 and compares against a
/// tight derived bound, so a weight that disagrees with the intended
/// formula shows up immediately; here the same error hides inside the
/// 9.44e-6 of honest f32 weight error this bound must tolerate. (The
/// constant term also largely cancels in the division by the weight sum,
/// which is why it is small to begin with.)
///
/// What it adds instead is the answer to a question the other cannot
/// reach: how far the shipped result sits from the true one. Both oracles
/// could agree while both were wrong, since the f32 one mirrors the
/// kernel's formula by hand; only a reference computed differently
/// throughout can say the f32 pipeline is accurate rather than merely
/// self-consistent.
///
/// Measured at HEAD: worst 9.44e-6 of the scene peak, on a Catmull-Rom
/// upscale of the cancelling comb, whose conditioning reaches 3.4e8. The
/// two other regimes and every downscale stay far below it. The bound is
/// set at roughly twice that, which still catches a wrong filter (error
/// ~1e-1), a dropped or misplaced tap (~1e-2), or any material loss of
/// accumulator width, while leaving room for a different rounding
/// elsewhere. Nothing here is transcendental, so the figure is
/// reproducible on any IEEE-754 host.
#[test]
fn resize_agrees_with_a_fully_independent_f64_reference() {
    use phaios_core::geometry::{ResizeFilter, ResizeParams, resize};

    let mut worst_rel = 0.0_f64;
    let mut worst_where = String::new();
    let mut worst_tapset = 0_usize;

    for highlight in [1.0_f32, 1e4, 1e8] {
        for (regime, img) in resampling_content(256, 256, highlight) {
            for (w, h, filter) in [
                (8_u32, 8_u32, ResizeFilter::Area),
                (64, 64, ResizeFilter::Area),
                (64, 64, ResizeFilter::CatmullRom),
                (64, 64, ResizeFilter::Bilinear),
                (384, 384, ResizeFilter::CatmullRom),
            ] {
                let params = ResizeParams::new(w, h, filter);
                let got = resize(img.view(), &params).unwrap();
                let want = resize_reference_f64(&img, &params);

                // Does the f32 geometry even select the same taps as f64?
                let scale = img.dim().1 as f64 / w as f64;
                for i in [0_usize, (w / 3) as usize, (w - 1) as usize] {
                    let a = resize_taps(filter, img.dim().1 as f32 / w as f32, i).len();
                    let b = resize_taps_f64(filter, scale, i).len();
                    worst_tapset = worst_tapset.max(a.abs_diff(b));
                }

                let peak = f64::from(highlight);
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let err = (f64::from(got[[y, x, 0]]) - want[[y, x]]).abs();
                        let rel = err / peak;
                        if rel > worst_rel {
                            worst_rel = rel;
                            worst_where = format!("{filter:?} {w}x{h}, {regime} at {highlight:e}");
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "independent f64 reference: worst {worst_rel:.3e} of the scene peak ({worst_where}); \
         largest tap-count difference {worst_tapset}"
    );
    // The f32 geometry must select the same taps as the f64 geometry.
    // A difference here would mean the two are not merely different in
    // precision but structurally different at the window edges, and the
    // error figure below would be comparing two different resamples.
    assert_eq!(
        worst_tapset, 0,
        "f32 and f64 tap geometry disagree on the number of taps"
    );
    assert!(
        worst_rel < 2.0e-5,
        "resize sits {worst_rel:.3e} from the f64 reference ({worst_where}); \
         HEAD measures 9.44e-6"
    );
}
