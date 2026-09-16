// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 28 — Local contrast via the guided filter, on the CUDA backend.
//!
//! The GPU twin of `examples/05_local_contrast.rs`: the same chart, the
//! same reference image, the same two strength levels. The pair exists
//! to be run one after the other and diffed.
//!
//! Demonstrates `local_contrast_device` at example 05's settings:
//! 1. Unprocessed luminance (reference).
//! 2. `radius=8, eps=0.01, strength=0.5` — moderate enhancement.
//! 3. `radius=16, eps=0.01, strength=1.0` — aggressive enhancement.
//!
//! What to look for:
//!
//! - **The pictures are example 05's pictures.** Patch boundaries
//!   sharpen and grow a halo at the higher strength; flat patch
//!   interiors are untouched, because the guided filter is the identity
//!   there and the residual `L − guided` is zero.
//! - **The agreement table, which is why the twin exists.** This is the
//!   kernel whose two backends differ most in formulation: the CPU
//!   builds four global f64 summed-area tables, the device runs
//!   separable box filters with Kahan-compensated f32 accumulation
//!   (`docs/ffi.md` §6). §6 therefore commits a bound of
//!   rtol 1e-4 / atol 1e-6 across backends and **not** bit-exactness,
//!   so the table prints the worst element as a multiple of that bound
//!   rather than claiming an equality this kernel is not promised.
//! - **The reference row is a control.** `luminance_bw` is one of the
//!   kernels §6 does promise exactly, so both backends enter the filter
//!   from a bit-identical image. Whatever the other two rows show
//!   therefore belongs to the guided filter alone, not to anything
//!   upstream of it.
//! - **The 8-bit verdict.** Agreement in f32 is not the question a
//!   photographer asks. The last column counts how many of the 98 304
//!   codes in the written PPM differ after the same `encode_srgb` that
//!   `write_ppm_grey_display` applies — the disagreement that survives
//!   into a file.
//!
//! Shape of the run: one upload, the luminance conversion and both
//! filters on the card, one download per image written. Nothing crosses
//! the bus between the conversion and the filter. The per-call offload
//! form would pay a PCIe round trip per stage for identical arithmetic;
//! example 22 measures what that costs.
//!
//! Reference: Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image
//! Filtering", *ECCV 2010*, LNCS 6311, pp. 1–14; extended as IEEE
//! *TPAMI* 35(6), 2013, pp. 1397–1409.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::local_contrast::GuidedFilterParams;

/// The committed cross-backend bound for `local_contrast` (`docs/ffi.md`
/// §6). Copied from the document rather than invented here, so a driver
/// that regresses accuracy shows up as a number above 1.0 instead of
/// being quietly absorbed.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-6;

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

/// Largest absolute difference, element for element.
///
/// Reduced with a plain `>` rather than `f32::max`, for the reason given
/// above.
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

    // The CPU chain, computed exactly as example 05 computes it. It is
    // the specification (`docs/ffi.md` §6), so it is what the device output
    // is measured against, never the other way round.
    let cpu_bw = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // The device chain: one upload, then the luminance conversion and
    // both filters without a further bus crossing.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let device_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();

    let moderate = GuidedFilterParams::new(8, 0.01);
    let aggressive = GuidedFilterParams::new(16, 0.01);

    // (file suffix, promise, CPU result, GPU result). The reference case
    // runs no filter at all: it is the luminance image itself.
    let runs: [(&str, &str, Array3<f32>, Array3<f32>); 3] = [
        (
            "reference",
            "bit-exact",
            cpu_bw.clone(),
            ctx.download(&device_bw).unwrap(),
        ),
        (
            "moderate",
            "1e-4/1e-6",
            phaios_core::local_contrast::local_contrast(cpu_bw.view(), &moderate, 0.5).unwrap(),
            ctx.download(&k::local_contrast_device(&device_bw, &moderate, 0.5).unwrap())
                .unwrap(),
        ),
        (
            "aggressive",
            "1e-4/1e-6",
            phaios_core::local_contrast::local_contrast(cpu_bw.view(), &aggressive, 1.0).unwrap(),
            ctx.download(&k::local_contrast_device(&device_bw, &aggressive, 1.0).unwrap())
                .unwrap(),
        ),
    ];

    println!(
        "{:<11} {:<10} {:>9} {:>12} {:>9} {:>18}",
        "variant", "promise", "CPU == GPU", "max |delta|", "bound", "8-bit codes differ"
    );
    for (name, promise, cpu, gpu) in &runs {
        let violation = worst_violation(cpu, gpu, RTOL, ATOL);
        let bound = if *promise == "bit-exact" {
            "-".to_string()
        } else {
            format!("{violation:.4}")
        };
        println!(
            "{name:<11} {promise:<10} {:>9} {:>12.3e} {bound:>9} {:>18}",
            cpu == gpu,
            max_abs_diff(cpu, gpu),
            format!("{}/{}", codes_differing(cpu, gpu), cpu.len()),
        );

        let path = format!("examples/output/28_gpu_contrast_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
    }
    println!(
        "  bound is the committed local_contrast pair (rtol 1e-4, atol 1e-6) from\n  \
         docs/ffi.md §6, as a multiple: 1.0000 is the limit. Bit-exactness is not\n  \
         promised for this kernel and is not claimed here."
    );

    println!("\nwrote examples/output/28_gpu_contrast_{{reference,moderate,aggressive}}.ppm");
    println!(
        "diff against the CPU twin: examples/output/05_contrast_{{reference,moderate,aggressive}}.ppm \
         (cargo run --example 05_local_contrast)"
    );
}
