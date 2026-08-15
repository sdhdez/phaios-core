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
