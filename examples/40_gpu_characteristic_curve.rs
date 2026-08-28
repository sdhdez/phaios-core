// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 40 — The characteristic curve, on the CUDA backend.
//!
//! The GPU twin of `19_characteristic_curve.rs`. Same chart, same five
//! gradings, same printed tables — run on the device instead of the
//! host, so the two can be put side by side. Read example 19 first: it
//! explains *why* a film tone scale has a toe, a straight section and a
//! shoulder. This one only asks whether the card agrees.
//!
//! The three stages compose exactly as they do on the CPU:
//!
//! ```text
//! shadow_rolloff  →  tone_curve  →  highlight_rolloff
//!      toe            straight          shoulder
//! ```
//!
//! and all three run **device-resident**: the chart is uploaded once,
//! `luminance_bw_device` → `exposure_device` prepare it, each grading
//! then runs its three `*_device` kernels without the intermediates
//! crossing the bus, and only the finished rendering is downloaded.
//! That is the API the GPU backend exists for; the per-call offload form
//! (example 15) would pay four PCIe round trips per grading to compute
//! the same thing.
//!
//! Five renderings are written, matching example 19 one for one:
//!
//! 1. `40_linear` — none of the three. A straight line, clipped at white.
//! 2. `40_toe_only` — the toe alone.
//! 3. `40_shoulder_only` — the shoulder alone.
//! 4. `40_characteristic` — all three: the full curve.
//! 5. `40_hard_toe` — the toe at full strength, where the deepest
//!    shadows lose their separation entirely.
//!
//! What to look for in the output:
//!
//! - **The agreement column, which should read `identical` on every
//!   row.** `docs/ffi.md` §6 promises this composition bit-exact across
//!   backends, and unusually the promise covers the whole chain rather
//!   than its ends: `shadow_rolloff` is a cubic in Horner form,
//!   `highlight_rolloff` a quadratic solve whose `sqrt` IEEE-754-2008
//!   §5.4.1 requires to be correctly rounded, and every `tone_curve`
//!   used here has `power == 1`, which takes the path that never calls
//!   `powf`. Nothing in these five gradings is transcendental, so the
//!   comparison is `==` over the whole array with no tolerance — and if
//!   a row ever prints `DIFFERS`, the GPU kernel is wrong, not the
//!   bound.
//! - **The slope table**, which is example 19's argument in numbers and
//!   is computed here on the device: low slope at both ends, highest in
//!   the midtones. The probe images are `(1, 2, 1)` — two pixels — which
//!   is also a cheap check that the launch geometry survives a
//!   degenerate shape.
//! - **The toe separation table.** Two samples 0.02 apart just above
//!   black, closing up as toe strength rises.
//!
//! The renderings are passed through `encode_srgb` **on the host**
//! before they are written (`shared::write_ppm_grey_display`), exactly
//! as example 19 does, so nothing in the file format can hide a
//! difference the f32 comparison would have caught.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::{Array3, ArrayView3};
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, DeviceImage, kernels as k};
use phaios_core::exposure::exposure;
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};
use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};
use phaios_core::tone::{ToneCurveParams, tone_curve};

/// The three stages in pipeline order, on the host.
///
/// Copied from `19_characteristic_curve.rs` unchanged, on purpose: it is
/// the specification this example checks the device against (CLAUDE.md
/// §2), so it must not be paraphrased.
fn characteristic_cpu(
    img: ArrayView3<f32>,
    toe: &ShadowRolloffParams,
    straight: &ToneCurveParams,
    shoulder: &RolloffParams,
) -> Array3<f32> {
    let a = shadow_rolloff(img, toe).unwrap();
    let b = tone_curve(a.view(), straight).unwrap();
    highlight_rolloff(b.view(), shoulder).unwrap()
}

/// The same three stages, device-resident.
///
/// `a` is rebound at each stage, so the previous buffer is freed as soon
/// as the next exists and the device working set stays at two images.
fn characteristic_gpu(
    img: &DeviceImage,
    toe: &ShadowRolloffParams,
    straight: &ToneCurveParams,
    shoulder: &RolloffParams,
) -> DeviceImage {
    let a = k::shadow_rolloff_device(img, toe).unwrap();
    let b = k::tone_curve_device(&a, straight).unwrap();
    k::highlight_rolloff_device(&b, shoulder).unwrap()
}

/// Number of elements whose bit patterns differ.
///
/// Stricter than `==`: it separates `+0.0` from `-0.0` and reports two
/// NaNs in the same place as agreement, neither of which `PartialEq`
/// does. Printed next to the `==` verdict so a row cannot claim
/// bit-exactness on a technicality.
fn differing_bits(a: &Array3<f32>, b: &Array3<f32>) -> usize {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
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

    // ── The same input as example 19, prepared on both backends ──────
    // `luminance_bw` and `exposure` are themselves documented bit-exact
    // (§6), so the two chains provably start from the same bits — which
    // the line below states rather than assumes.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let pushed = exposure(bw.view(), 2.0).unwrap();

    // The one upload. Everything below runs from this buffer.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let d_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();
    let d_pushed = k::exposure_device(&d_bw, 2.0).unwrap();
    drop(d_bw);
    drop(uploaded);

    let prepared = ctx.download(&d_pushed).unwrap();
    println!(
        "input to the curve (luminance_bw then +2 EV): CPU == GPU bit-for-bit: {} \
         ({} of {} elements differ)",
        pushed == prepared,
        differing_bits(&pushed, &prepared),
        pushed.len()
    );

    // ── The five gradings of example 19, unchanged ───────────────────
    let no_toe = ShadowRolloffParams::default();
    let toe = ShadowRolloffParams::new(0.18, 0.8);
    let hard_toe = ShadowRolloffParams::new(0.30, 1.0);
    let flat = ToneCurveParams::new(1.0, 0.0, 1.0);
    let contrast = ToneCurveParams::new(1.2, 0.0, 1.0);
    let no_shoulder = RolloffParams::default();
    let shoulder = RolloffParams::new(0.7, 3.0);

    let configs: [(&str, ShadowRolloffParams, ToneCurveParams, RolloffParams); 5] = [
        ("linear", no_toe.clone(), flat.clone(), no_shoulder.clone()),
        ("toe_only", toe.clone(), flat.clone(), no_shoulder.clone()),
        (
            "shoulder_only",
            no_toe.clone(),
            flat.clone(),
            shoulder.clone(),
        ),
        (
            "characteristic",
            toe.clone(),
            contrast.clone(),
            shoulder.clone(),
        ),
        ("hard_toe", hard_toe.clone(), contrast, shoulder.clone()),
    ];

    println!("\nfive gradings, three device kernels each, one download apiece:");
    println!(
        "{:>16} {:>12} {:>19} {:>13}",
        "grading", "CPU == GPU", "elements differing", "max |delta|"
    );
    let mut all_exact = true;
    for (name, t, c, sh) in configs.iter() {
        let cpu = characteristic_cpu(pushed.view(), t, c, sh);
        let gpu = ctx
            .download(&characteristic_gpu(&d_pushed, t, c, sh))
            .unwrap();

        let equal = cpu == gpu;
        all_exact &= equal;
        let differing = differing_bits(&cpu, &gpu);
        // Reduced with `>` rather than `f32::max`, which returns the
        // other operand when one side is NaN and would score a
        // NaN-filled output as a perfect match.
        let mut worst = 0.0_f32;
        for (x, y) in cpu.iter().zip(gpu.iter()) {
            let d = (x - y).abs();
            if d > worst || d.is_nan() {
                worst = d;
            }
        }

        let path = format!("examples/output/40_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);

        let share = format!("{differing}/{}", cpu.len());
        println!("{name:>16} {equal:>12} {share:>19} {worst:>13.3e}");
    }
    println!("wrote examples/output/40_*.ppm (the GPU renderings)");
    println!(
        "  docs/ffi.md section 6 promises shadow_rolloff, highlight_rolloff and\n  \
         tone_curve at power == 1 bit-exact across backends, so this is `==`\n  \
         with no tolerance: {}",
        if all_exact {
            "every grading matched"
        } else {
            "A GRADING DID NOT MATCH — the GPU kernel is wrong"
        }
    );

    // ── The curve shape, in numbers, measured on the device ──────────
    // Local slope d(out)/d(in) at four scene values, spanning shadow to
    // highlight. The characteristic shape is low, high, high, low.
    let probes = [0.01_f32, 0.05, 0.40, 2.00];
    let step = 1e-4_f32;

    println!("\nlocal slope d(out)/d(in), computed on the device:");
    println!(
        "{:>16} {:>10} {:>10} {:>10} {:>10}  {:>10}",
        "", "in 0.01", "in 0.05", "in 0.40", "in 2.00", "vs CPU"
    );
    for (name, t, c, sh) in configs.iter() {
        let mut cells = String::new();
        let mut matched = true;
        for p in probes {
            let pair = Array3::from_shape_vec((1, 2, 1), vec![p, p + step]).unwrap();
            let d_pair = ctx.upload(pair.view()).unwrap();
            let gpu = ctx
                .download(&characteristic_gpu(&d_pair, t, c, sh))
                .unwrap();
            matched &= gpu == characteristic_cpu(pair.view(), t, c, sh);
            let slope = (gpu[[0, 1, 0]] - gpu[[0, 0, 0]]) / step;
            cells.push_str(&format!("{slope:>10.3}"));
        }
        let verdict = if matched { "identical" } else { "DIFFERS" };
        println!("{name:>16}{cells}  {verdict:>10}");
    }

    println!("\nthe toe compresses shadow separation — two samples 0.02 apart just above black:");
    for strength in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let pair = Array3::from_shape_vec((1, 2, 1), vec![0.0_f32, 0.02]).unwrap();
        let params = ShadowRolloffParams::new(0.2, strength);
        let d_pair = ctx.upload(pair.view()).unwrap();
        let gpu = ctx
            .download(&k::shadow_rolloff_device(&d_pair, &params).unwrap())
            .unwrap();
        let cpu = shadow_rolloff(pair.view(), &params).unwrap();
        let sep = gpu[[0, 1, 0]] - gpu[[0, 0, 0]];
        println!(
            "  strength {strength:.2}: {sep:.5}  ({:.0}% of the input separation, {})",
            sep / 0.02 * 100.0,
            if gpu == cpu {
                "identical to the CPU"
            } else {
                "DIFFERS from the CPU"
            }
        );
    }

    println!(
        "\ndiff against the CPU twin: examples/output/19_<name>.ppm — same five names.\n  \
         Run `cargo run --example 19_characteristic_curve` first, then e.g.\n  \
         `cmp examples/output/19_characteristic.ppm examples/output/40_characteristic.ppm`;\n  \
         because the f32 arrays agree to the bit and both are encoded by the same\n  \
         host `encode_srgb`, the PPM files are expected to be byte-identical."
    );
}
