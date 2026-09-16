// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 44 — Unsharp masking with a soft threshold, on the CUDA backend.
//!
//! The GPU twin of `43_sharpen.rs`. Same chart, same three renderings,
//! same two printed tables — run on the device, so the two can be put
//! side by side. Read example 43 first: it makes the argument that
//! `threshold` gates by the *derived* detail signal `|img − blur(img)|`,
//! not the raw pixel value, and this example does not repeat it. It asks
//! whether the card reproduces it.
//!
//! On the device `sharpen` is **not** composed the way `glow` is: `glow`
//! brackets its blur with a weight kernel before and an add kernel
//! after, but `sharpen`'s blur runs on `img` directly, so the
//! subtract/gate/combine work all happens after it and fits one
//! pointwise launch. The blur itself is the crate's one device
//! Gaussian, [`blur_device`], so there is a single blur implementation
//! on the card and this kernel cannot drift from it — which is also why
//! its agreement bound is the blur's, the same class `glow` holds.
//!
//! Everything runs **device-resident**: the chart is uploaded once,
//! `luminance_bw_device` prepares it, and each rendering then runs its
//! own `sharpen_device` call without an intermediate crossing the bus —
//! only the finished images come back.
//!
//! Three renderings are written, matching example 43 one for one:
//!
//! 1. `44_none` — the reference.
//! 2. `44_plain` — `threshold = 0`, every pixel's detail amplified.
//! 3. `44_thresholded` — same `amount`/`sigma`, small detail gated out.
//!
//! What to look for in the output:
//!
//! - **The agreement column.** `docs/ffi.md` §6 does *not* promise
//!   `sharpen` bit-exact. It inherits the blur's committed (rtol 1e-5,
//!   atol 1e-7) — the pointwise subtract/gate/combine kernel underneath
//!   is free of transcendentals and built with `-fmad=false`, so the
//!   blur is the only inexact part of it, exactly as for `glow`. Each
//!   row prints the worst element as a multiple of the bound; 1.000 is
//!   the limit.
//! - **The step-edge table**, computed on the device: the same
//!   measurable steepening example 43 prints on the CPU, plus a
//!   threshold high enough to shut the gate, reproduced here bit for
//!   bit against the CPU's algebraic identity.
//! - **The threshold table.** A ripple's contribution to the output
//!   fades away as the threshold rises past its `|detail|`; a step's
//!   does not, because its `|detail|` is far larger. The two probes are
//!   48 and 40 pixels wide — small enough to exercise the separable
//!   filter on a degenerate shape, where axis handling goes wrong if it
//!   is going to.
//!
//! The renderings are passed through `encode_srgb` **on the host**
//! before they are written (`shared::write_ppm_grey_display`), exactly
//! as example 43 does.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, as example 15 does.
//!
//! [`blur_device`]: phaios_core::cuda::kernels::blur_device

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, Context, DeviceImage, kernels as k};
use phaios_core::sharpen::{SharpenParams, sharpen};

/// The committed cross-backend bound, from `docs/ffi.md` §6.
///
/// `sharpen` inherits the blur's, exactly as `glow` does, so a single
/// constant covers every comparison in this file.
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-7;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: <= 1.0 means every element passes.
///
/// The same two-term form `tests/cuda_conformance.rs` uses, including its
/// NaN handling — a metric built on `f32::max` over a `zip` scores a
/// wholly-NaN or truncated output as a perfect match, which is how the
/// conformance oracle was wrong before it was fixed.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch");
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let v = if x.is_nan() || y.is_nan() {
            if x.is_nan() && y.is_nan() {
                0.0
            } else {
                f32::INFINITY
            }
        } else if x.is_infinite() || y.is_infinite() {
            if x == y { 0.0 } else { f32::INFINITY }
        } else {
            (x - y).abs() / (ATOL + RTOL * x.abs())
        };
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Largest absolute difference, element for element.
///
/// Reduced with a plain `>` rather than `f32::max`, for the reason given
/// above: `f32::max` returns the *other* operand when one side is NaN,
/// so a fold over it reports a NaN-filled output as a perfect match.
fn max_abs_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    worst
}

/// Download a device image, write it as `44_<name>.ppm`, and report how
/// it compares with the CPU rendering of the same thing.
fn emit(ctx: &Context, name: &str, gpu: &DeviceImage, cpu: &Array3<f32>) {
    let host = ctx.download(gpu).unwrap();
    let path = format!("examples/output/44_{name}.ppm");
    shared::write_ppm_grey_display(Path::new(&path), host.view(), shared::WIDTH, shared::HEIGHT);
    let differing = cpu
        .iter()
        .zip(host.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let share = format!("{differing}/{}", cpu.len());
    println!(
        "{name:>12} {:>13.3e} {:>10.4} {share:>19}",
        max_abs_diff(cpu, &host),
        worst_violation(cpu, &host)
    );
}

fn main() {
    let devices = cuda::devices();
    if devices.is_empty() {
        println!("no CUDA device available, skipping");
        return;
    }
    for d in &devices {
        println!(
            "device {}: {} (cc {}.{}, supported: {})",
            d.ordinal, d.name, d.compute_capability.0, d.compute_capability.1, d.supported
        );
    }
    let Ok(ctx) = cuda::Context::new(0) else {
        println!("device present but unusable, skipping");
        return;
    };
    println!("fingerprint: {}\n", ctx.fingerprint());

    std::fs::create_dir_all("examples/output").unwrap();

    // ── The same input as example 43, prepared on both backends ──────
    // `luminance_bw` is documented bit-exact (§6), so both chains
    // provably start from the same bits.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // The one upload.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let d_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();
    drop(uploaded);
    println!(
        "input to sharpen (luminance_bw): CPU == GPU bit-for-bit: {}",
        bw == ctx.download(&d_bw).unwrap()
    );

    // ── The three renderings ──────────────────────────────────────────
    println!("\nthree renderings, device-resident — sharpen on the card:");
    println!(
        "{:>12} {:>13} {:>10} {:>19}",
        "rendering", "max |delta|", "x bound", "elements differing"
    );

    // 1. Reference: unsharpened.
    emit(&ctx, "none", &d_bw, &bw);

    // 2. Plain unsharp mask — no threshold.
    let plain_params = SharpenParams::new(1.5, 2.0, 0.0);
    let cpu_plain = sharpen(bw.view(), &plain_params).unwrap();
    let d_plain = k::sharpen_device(&d_bw, &plain_params).unwrap();
    emit(&ctx, "plain", &d_plain, &cpu_plain);

    // 3. Thresholded — same amount/sigma, small detail gated out.
    let gated_params = SharpenParams::new(1.5, 2.0, 0.03);
    let cpu_thresholded = sharpen(bw.view(), &gated_params).unwrap();
    let d_thresholded = k::sharpen_device(&d_bw, &gated_params).unwrap();
    emit(&ctx, "thresholded", &d_thresholded, &cpu_thresholded);

    println!("wrote examples/output/44_*.ppm (the GPU renderings)");
    println!(
        "  the bound is the committed (rtol {RTOL:e}, atol {ATOL:e}) from docs/ffi.md\n  \
         section 6, expressed as a multiple: 1.0000 is the limit. sharpen inherits the\n  \
         blur's bound — the pointwise subtract/gate/combine kernel underneath it is\n  \
         free of transcendentals and built with -fmad=false, so it is exact on its own."
    );

    // ── A step edge gets steeper, and a threshold can shut that off ──
    println!("\na step edge gets steeper at threshold=0, unchanged once threshold covers it:");
    let n = 48_usize;
    let mut step = Array3::<f32>::zeros((1, n, 1));
    for x in (n / 2)..n {
        step[[0, x, 0]] = 0.6;
    }
    let sigma = 3.0_f32;
    let half = 6_usize;
    let slope = |img: &Array3<f32>| img[[0, n / 2 + half, 0]] - img[[0, n / 2 - half, 0]];
    let d_step = ctx.upload(step.view()).unwrap();
    println!(
        "{:>28} {:>10} {:>10}",
        "threshold", "slope (gpu)", "x bound"
    );
    for threshold in [0.0_f32, 0.004, 0.008, 0.012, 5.0] {
        let params = SharpenParams::new(2.0, sigma, threshold);
        let cpu = sharpen(step.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::sharpen_device(&d_step, &params).unwrap())
            .unwrap();
        println!(
            "{threshold:>28.3} {:>10.4} {:>10.4}",
            slope(&gpu),
            worst_violation(&cpu, &gpu)
        );
    }
    println!(
        "  slope falls back toward the input's 0.6000 as threshold rises past the\n  \
         |detail| this window sees, and threshold 5.0 exceeds every |detail| the step\n  \
         produces anywhere, so both backends hit the identity's algebraic zero and\n  \
         agree exactly, not merely within bound."
    );

    // ── What the threshold buys: gating by DETAIL, not by pixel value ─
    println!("\nthreshold gates by |detail| at each position, not by the pixel's own value:");
    println!(
        "a flat ripple (small |detail|) beside a step (large |detail|), sigma 1.5, amount 2.0, on the device:"
    );
    // A flat buffer (10..20) separates the ripple from the step by more
    // than the blur's reach, so each region's |detail| reflects only
    // its own content, not the other's.
    let mut mixed = Array3::<f32>::from_elem((1, 40, 1), 0.2_f32);
    for x in 0..10 {
        mixed[[0, x, 0]] += if x % 2 == 0 { 0.01 } else { -0.01 };
    }
    for x in 20..40 {
        mixed[[0, x, 0]] = 1.4;
    }
    let d_mixed = ctx.upload(mixed.view()).unwrap();
    for threshold in [0.0_f32, 0.05, 0.2] {
        let params = SharpenParams::new(2.0, 1.5, threshold);
        let cpu = sharpen(mixed.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::sharpen_device(&d_mixed, &params).unwrap())
            .unwrap();
        let ripple_amp = (2..8)
            .map(|x| (gpu[[0, x, 0]] - mixed[[0, x, 0]]).abs())
            .fold(0.0_f32, f32::max);
        let step_amp = (17..24)
            .map(|x| (gpu[[0, x, 0]] - mixed[[0, x, 0]]).abs())
            .fold(0.0_f32, f32::max);
        println!(
            "  threshold {threshold:>4.2}: max |added| in the ripple {ripple_amp:.4}, at the step {step_amp:.4}   ({:.4}x bound)",
            worst_violation(&cpu, &gpu)
        );
    }
    println!(
        "  (as threshold rises the ripple's contribution disappears while the step's does not)"
    );

    println!(
        "\ndiff against the CPU twin: examples/output/43_<name>.ppm — same three names.\n  \
         Run `cargo run --example 43_sharpen` first, then e.g.\n  \
         `cmp examples/output/43_thresholded.ppm examples/output/44_thresholded.ppm`.\n  \
         On this chart all three pairs come out byte-identical, which is a measurement\n  \
         and not a promise: sharpen is not bit-exact in f32, and the column above shows\n  \
         a few pixels differing by one or two ULP — small enough that none of them\n  \
         crossed an 8-bit rounding boundary here. On other input, or another card, a\n  \
         few codes may differ by one. The f32 table is the real verdict."
    );
}
