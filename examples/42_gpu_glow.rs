// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 42 — Halation, diffusion and veiling glare, on the CUDA backend.
//!
//! The GPU twin of `21_glow.rs`. Same chart, same four renderings, same
//! two printed tables — run on the device, so the two can be put side by
//! side. Read example 21 first: it makes the argument that the three
//! named effects are one operation at different parameters *and
//! different pipeline positions*,
//!
//! ```text
//! out = in + amount · blur(max(in − threshold, 0), σ)
//! ```
//!
//! and this example does not repeat it. It asks whether the card
//! reproduces it.
//!
//! On the device `glow` is composed rather than fused: the threshold
//! weight and the add are element-wise kernels, and the spread between
//! them is the crate's one device Gaussian, [`blur_device`]. So there is
//! a single blur implementation on the card and this kernel cannot drift
//! from it — which is also why its agreement bound is the blur's.
//!
//! Everything runs **device-resident**: the chart is uploaded once,
//! `luminance_bw_device` → `exposure_device` prepare it, each rendering
//! then runs its `glow_device` and `tone_curve_device` calls without an
//! intermediate crossing the bus, and only the finished image comes
//! back. Note that image 3 reuses image 1's device buffer as its input —
//! diffusion happens *at the print*, so it is applied to the already
//! toned result — which on the host would have meant another round trip.
//!
//! Four renderings are written, matching example 21 one for one:
//!
//! 1. `42_none` — the reference.
//! 2. `42_halation` — high threshold, moderate σ, applied **early**
//!    (right after exposure, before the tone stages).
//! 3. `42_diffusion` — mid threshold, large σ, applied **late**.
//! 4. `42_glare` — no threshold, frame-spanning σ, applied **earliest**.
//!
//! What to look for in the output:
//!
//! - **The agreement column.** `docs/ffi.md` §6 does *not* promise
//!   `glow` bit-exact. It inherits the blur's committed (rtol 1e-5, atol
//!   1e-7), and the `tone_curve` used for the grade here has
//!   `power = 0.9`, which takes the `powf` path and carries the same
//!   one-transcendental bound. So **none** of these four renderings is
//!   bit-exact, including `42_none`, which is a tone curve and nothing
//!   else. Each row prints the worst element as a multiple of the bound;
//!   1.000 is the limit.
//! - **The veiling-glare table**, computed on the device. The same black
//!   corner pixel, in two frames that differ only elsewhere, ends up at
//!   two different values. No per-pixel transfer can do that, which is
//!   example 21's whole argument — and it survives the port, because the
//!   device blur is a real spatial filter and not a lookup.
//! - **The threshold table.** A higher threshold scatters less. The
//!   probe is a `(1, 5, 1)` image — five pixels — which also exercises
//!   the separable filter on a degenerate shape, where axis handling
//!   goes wrong if it is going to.
//!
//! The renderings are passed through `encode_srgb` **on the host**
//! before they are written (`shared::write_ppm_grey_display`), exactly
//! as example 21 does.
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
use phaios_core::exposure::exposure;
use phaios_core::glow::{GlowParams, glow};
use phaios_core::tone::{ToneCurveParams, tone_curve};

/// The committed cross-backend bound, from `docs/ffi.md` §6.
///
/// `glow` inherits the blur's, and `tone_curve` with `power != 1` holds
/// the same one-transcendental pair, so a single constant covers every
/// comparison in this file.
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

/// Download a device image, write it as `42_<name>.ppm`, and report how
/// it compares with the CPU rendering of the same thing.
fn emit(ctx: &Context, name: &str, gpu: &DeviceImage, cpu: &Array3<f32>) {
    let host = ctx.download(gpu).unwrap();
    let path = format!("examples/output/42_{name}.ppm");
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

    // ── The same input as example 21, prepared on both backends ──────
    // `luminance_bw` and `exposure` are documented bit-exact (§6), so
    // both chains provably start from the same bits.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let pushed = exposure(bw.view(), 2.0).unwrap();
    let grade = ToneCurveParams::new(1.15, 0.0, 0.9);

    // The one upload.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let d_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();
    let d_pushed = k::exposure_device(&d_bw, 2.0).unwrap();
    drop(d_bw);
    drop(uploaded);
    println!(
        "input to the glow (luminance_bw then +2 EV): CPU == GPU bit-for-bit: {}",
        pushed == ctx.download(&d_pushed).unwrap()
    );

    // ── The four renderings, in their pipeline positions ─────────────
    println!("\nfour renderings, device-resident — glow and tone_curve on the card:");
    println!(
        "{:>12} {:>13} {:>10} {:>19}",
        "rendering", "max |delta|", "x bound", "elements differing"
    );

    // 1. Reference: exposure then tone, nothing scattered.
    let cpu_plain = tone_curve(pushed.view(), &grade).unwrap();
    let d_plain = k::tone_curve_device(&d_pushed, &grade).unwrap();
    emit(&ctx, "none", &d_plain, &cpu_plain);

    // 2. Halation — in the emulsion, so BEFORE the tone stages.
    let halation = GlowParams::new(0.8, 8.0, 0.35);
    let cpu_halation = tone_curve(glow(pushed.view(), &halation).unwrap().view(), &grade).unwrap();
    let d_halation =
        k::tone_curve_device(&k::glow_device(&d_pushed, &halation).unwrap(), &grade).unwrap();
    emit(&ctx, "halation", &d_halation, &cpu_halation);

    // 3. Diffusion — at the print, so AFTER them. Note the input: the
    //    already-toned image, which is still on the device.
    let diffusion = GlowParams::new(0.5, 20.0, 0.4);
    let cpu_diffusion = glow(cpu_plain.view(), &diffusion).unwrap();
    let d_diffusion = k::glow_device(&d_plain, &diffusion).unwrap();
    emit(&ctx, "diffusion", &d_diffusion, &cpu_diffusion);

    // 4. Veiling glare — in the lens, so earliest of all.
    let glare = GlowParams::new(0.0, 400.0, 0.06);
    let cpu_glare = tone_curve(glow(pushed.view(), &glare).unwrap().view(), &grade).unwrap();
    let d_glare =
        k::tone_curve_device(&k::glow_device(&d_pushed, &glare).unwrap(), &grade).unwrap();
    emit(&ctx, "glare", &d_glare, &cpu_glare);

    println!("wrote examples/output/42_*.ppm (the GPU renderings)");
    println!(
        "  the bound is the committed (rtol {RTOL:e}, atol {ATOL:e}) from docs/ffi.md\n  \
         section 6, expressed as a multiple: 1.0000 is the limit. None of the four is\n  \
         bit-exact — glow inherits the blur's bound, and this grade's tone_curve has\n  \
         power = 0.9, which calls powf."
    );

    // ── Why a tone curve cannot be veiling glare ─────────────────────
    // The same black pixel, in two frames that differ only elsewhere.
    println!("\nveiling glare: the black point lifts by an amount set by the WHOLE frame");
    println!(
        "{:>18} {:>14} {:>16} {:>10}",
        "scene", "corner before", "corner after", "x bound"
    );
    let n = 64_usize;
    let lens = GlowParams::new(0.0, 200.0, 0.5);
    for (label, peak) in [("mostly dark", 0.2_f32), ("bright subject", 4.0)] {
        let mut scene = Array3::<f32>::zeros((n, n, 1));
        for y in 0..8 {
            for x in 0..8 {
                scene[[y, x, 0]] = peak;
            }
        }
        let cpu = glow(scene.view(), &lens).unwrap();
        let d_scene = ctx.upload(scene.view()).unwrap();
        let gpu = ctx
            .download(&k::glow_device(&d_scene, &lens).unwrap())
            .unwrap();
        println!(
            "{label:>18} {:>14.5} {:>16.5} {:>10.4}",
            scene[[n - 1, n - 1, 0]],
            gpu[[n - 1, n - 1, 0]],
            worst_violation(&cpu, &gpu)
        );
    }
    println!(
        "  the corner pixel is 0.0 in both frames, and ends up different —\n  \
         no per-pixel transfer can do that, which is the whole argument"
    );

    // ── And what the threshold buys ──────────────────────────────────
    println!("\nthreshold selects what scatters (sigma 6, amount 0.5), on the device:");
    let probe = Array3::<f32>::from_shape_fn((1, 5, 1), |(_, x, _)| x as f32 * 0.5);
    let d_probe = ctx.upload(probe.view()).unwrap();
    for threshold in [0.0_f32, 0.5, 1.0, 2.0] {
        let params = GlowParams::new(threshold, 6.0, 0.5);
        let cpu = glow(probe.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::glow_device(&d_probe, &params).unwrap())
            .unwrap();
        let deltas: Vec<String> = (0..5)
            .map(|i| format!("{:+.4}", gpu[[0, i, 0]] - probe[[0, i, 0]]))
            .collect();
        println!(
            "  threshold {threshold:>4.1}: {}   ({:.4}x bound)",
            deltas.join("  "),
            worst_violation(&cpu, &gpu)
        );
    }
    println!("  (inputs 0.0 0.5 1.0 1.5 2.0 — a higher threshold scatters less)");

    println!(
        "\ndiff against the CPU twin: examples/output/21_<name>.ppm — same four names.\n  \
         Run `cargo run --example 21_glow` first, then e.g.\n  \
         `cmp examples/output/21_halation.ppm examples/output/42_halation.ppm`.\n  \
         On this chart all four pairs come out byte-identical, which is a measurement\n  \
         and not a promise: none of these renderings is bit-exact in f32, and the\n  \
         column above shows a few thousand pixels differing by one or two ULP — small\n  \
         enough that none of them crossed an 8-bit rounding boundary here. On other\n  \
         input, or another card, a few codes may differ by one. The f32 table is the\n  \
         real verdict."
    );
}
