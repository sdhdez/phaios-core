// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 31 — Parametric tone curve (ASC CDL), on the CUDA backend.
//!
//! The GPU twin of `examples/09_tone_curve.rs`: the same chart, the same
//! five grades, the same zone-anchor table at the end. The pair exists
//! to be run one after the other and diffed.
//!
//! Demonstrates `tone_curve_device`, the slope/offset/power primary
//! correction, isolating one control at a time as example 09 does:
//!
//! 1. Reference — the identity `(1.0, 0.0, 1.0)`.
//! 2. Slope 1.4 — gain. Scales everything about black.
//! 3. Offset +0.05 — lift. Moves black itself, which is what makes the
//!    shadows go milky.
//! 4. Power 0.7 — gamma. Opens the midtones while pinning 0 and 1.
//! 5. A combined grade: slight lift, gentle gain, midtone bend.
//!
//! What to look for:
//!
//! - **The pictures are example 09's pictures.** Image 2 pivots on
//!   black, so it costs highlights first while the darkest patches
//!   barely move; image 4 lifts the same midtones and leaves the white
//!   patch where it was. Image 3 is the one to study for what not to do
//!   by accident: the black patch (#24) is no longer black, and no later
//!   stage can put it back.
//! - **This one kernel spans both of §6's agreement classes, which is
//!   why it is the most instructive twin in the set.** Rows 1–3 have
//!   `power == 1`: the device skips `powf` as the CPU does, leaving a
//!   multiply, an add and a `max`. Row 1 does not even launch a kernel:
//!   it is the identity fast path, a device-to-device copy. Rows 2 and 3
//!   do, and `build.rs` compiles the kernel with fast math, but the `.cu`
//!   pins `slope * x + offset` to the CPU's two roundings (`__fmul_rn`,
//!   `__fadd_rn`), which fast math never fuses into an FMA. `docs/ffi.md`
//!   §6 therefore promises all three bit-exact unless an input or
//!   `slope * x + offset` is subnormal — flushed to zero then, and the
//!   chart has none — so they are checked with `==` and nothing else. Rows 4 and 5
//!   raise a `powf`, which IEEE-754 does not standardise and fast math
//!   computes from approximate `log2` and `exp2`, so §6 commits
//!   rtol 1e-5 / atol 1e-7 instead and those rows print the worst
//!   element as a multiple of that bound. A single kernel, two promises,
//!   decided by a parameter — which is exactly why the promise column is
//!   read from §6 rather than guessed from the kernel name.
//! - **The zone-anchor table.** The combined grade is evaluated at the
//!   eleven Zone System anchors on both backends and printed side by
//!   side in ULP, so any disagreement is visible as a count of
//!   representable steps at known photographic stops rather than as an
//!   abstract tolerance. On the card this was written against the
//!   anchors differ by at most two ULP, about two hundredths of §6's
//!   bound. Those counts are a measurement, not a promise; §6's bound is
//!   the promise.
//! - **The 8-bit verdict.** The last column counts how many of the
//!   98 304 codes in the written PPM differ after the same `encode_srgb`
//!   that `write_ppm_grey_display` applies — the disagreement that
//!   survives into a file.
//!
//! Shape of the run: one upload, the luminance conversion and all five
//! grades on the card, one download per image written. The eleven-pixel
//! anchor probe at the end is a second, deliberate upload; it is a
//! measurement, not part of the picture.
//!
//! Reference: American Society of Cinematographers Technology Committee,
//! "ASC Color Decision List (ASC CDL) Transfer Functions and Interchange
//! Syntax", version 1.2 (2009), §2.1.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::tone::{ToneCurveParams, tone_curve};

/// The committed cross-backend bound of `docs/ffi.md` §6's
/// one-transcendental class, which `tone_curve` belongs to at
/// `power != 1`: a single `powf`, compiled with fast math. Copied from
/// the document rather than invented here. It applies only to the rows
/// with `power != 1`; the rest are promised exact and get no tolerance.
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-7;

/// Middle grey, 18% reflectance — Zone V of the Adams/Archer scale.
const MIDDLE_GREY: f32 = 0.18;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: `<= 1.0` means every element passes.
///
/// The same form `tests/cuda_conformance.rs` uses, including its care
/// with non-finite values. A metric built on `f32::max` over a `zip`
/// scores a wholly-NaN or truncated output as a perfect match, because
/// `f32::max` returns the *other* operand when one side is NaN — which
/// is how the conformance oracle was wrong before it was fixed.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32) -> f32 {
    if a.dim() != b.dim() {
        return f32::INFINITY;
    }
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
            (x - y).abs() / (atol + rtol * x.abs())
        };
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Largest absolute difference, element for element. Plain `>` rather
/// than `f32::max`, for the reason given above.
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

/// Distance in representable f32 steps between two finite, like-signed
/// values. Meaningful here because the curve clamps its input at zero,
/// so every value in play is non-negative.
fn ulps_apart(a: f32, b: f32) -> i64 {
    i64::from(a.to_bits()) - i64::from(b.to_bits())
}

/// The 8-bit code `shared::write_ppm` emits for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// How many 8-bit codes differ between the two PPMs that would be
/// written, i.e. after the same CPU `encode_srgb` both sides go through.
fn codes_differing(cpu: &Array3<f32>, gpu: &Array3<f32>) -> usize {
    let a = phaios_core::encode::encode_srgb(cpu.view()).expect("encode_srgb is infallible here");
    let b = phaios_core::encode::encode_srgb(gpu.view()).expect("encode_srgb is infallible here");
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| code(**x) != code(**y))
        .count()
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

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    // The CPU chain, as example 09 computes it. It is the specification
    // (`docs/ffi.md` §6); the device output is measured against it, never
    // the other way round.
    let cpu_bw = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // One upload; the conversion and all five grades run on the card.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let device_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, ToneCurveParams); 5] = [
        ("reference", ToneCurveParams::new(1.0, 0.0, 1.0)),
        ("slope_gain", ToneCurveParams::new(1.4, 0.0, 1.0)),
        ("offset_lift", ToneCurveParams::new(1.0, 0.05, 1.0)),
        ("power_gamma", ToneCurveParams::new(1.0, 0.0, 0.7)),
        ("combined", ToneCurveParams::new(1.15, 0.01, 0.85)),
    ];

    println!(
        "{:<12} {:>5} {:<10} {:>9} {:>12} {:>9} {:>18}",
        "grade", "power", "promise", "CPU == GPU", "max |delta|", "bound", "8-bit codes differ"
    );
    let mut all_within = true;
    for (name, params) in configs {
        let cpu = tone_curve(cpu_bw.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::tone_curve_device(&device_bw, &params).unwrap())
            .unwrap();

        // §6 decides the class, and for this kernel the deciding fact is
        // the exponent: `power == 1` skips powf on both backends and the
        // affine step is pinned to the CPU's roundings, so it is promised
        // exact; everything else is bounded.
        let exact = params.power == 1.0;
        let (promise, bound) = if exact {
            all_within &= cpu == gpu;
            ("bit-exact", "-".to_string())
        } else {
            let v = worst_violation(&cpu, &gpu, RTOL, ATOL);
            all_within &= v <= 1.0;
            ("1e-5/1e-7", format!("{v:.4}"))
        };
        println!(
            "{name:<12} {:>5} {promise:<10} {:>9} {:>12.3e} {bound:>9} {:>18}",
            params.power,
            cpu == gpu,
            max_abs_diff(&cpu, &gpu),
            format!("{}/{}", codes_differing(&cpu, &gpu), cpu.len()),
        );

        let path = format!("examples/output/31_gpu_curve_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
    }
    println!(
        "  bound is the committed one-transcendental pair (rtol 1e-5, atol 1e-7)\n  \
         from docs/ffi.md §6, as a multiple: 1.0000 is the limit. The three power == 1\n  \
         rows are promised bit-exact there, so they are checked with == and given\n  \
         no tolerance at all: the kernel is compiled with fast math, but it pins\n  \
         slope * x + offset to the CPU's two roundings. reference is the identity\n  \
         fast path, a device copy"
    );
    println!("every grade within its docs/ffi.md §6 promise: {all_within}");

    // The shape of the combined curve at the eleven zone anchors, as
    // example 09 prints it — with the device column beside it, which is
    // the whole reason for the twin. Eleven pixels in one image: one
    // upload, one launch, one download.
    let combined = ToneCurveParams::new(1.15, 0.01, 0.85);
    let anchors = Array3::from_shape_fn((1, 11, 1), |(_, z, _)| {
        MIDDLE_GREY * 2.0_f32.powi(z as i32 - 5)
    });
    let cpu_anchors = tone_curve(anchors.view(), &combined).unwrap();
    let gpu_anchors = ctx
        .download(&k::tone_curve_device(&ctx.upload(anchors.view()).unwrap(), &combined).unwrap())
        .unwrap();

    println!("\ncombined grade, sampled at the zone anchors:");
    println!(
        "  {:<8} {:>10} {:>12} {:>12} {:>6}",
        "zone", "linear", "CPU", "GPU", "ULP"
    );
    for zone in 0..=10_usize {
        let linear = anchors[[0, zone, 0]];
        let c = cpu_anchors[[0, zone, 0]];
        let g = gpu_anchors[[0, zone, 0]];
        println!(
            "  {zone:>4}     {linear:>10.5} {c:>12.7} {g:>12.7} {:>6}",
            ulps_apart(c, g)
        );
    }

    println!("\nwrote examples/output/31_gpu_curve_*.ppm");
    println!(
        "diff against the CPU twin: examples/output/09_curve_*.ppm \
         (cargo run --example 09_tone_curve)"
    );
}
