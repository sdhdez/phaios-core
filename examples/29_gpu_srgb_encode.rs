// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 29 — sRGB transfer encoding, on the CUDA backend.
//!
//! The GPU twin of `examples/06_srgb_encode.rs`: the same chart, the
//! same two images, the same point about what the transfer is for. The
//! pair exists to be run one after the other and diffed.
//!
//! Demonstrates `encode_srgb_device` by writing:
//! 1. Linear output of the luminance kernel (too dark — no transfer).
//! 2. sRGB-encoded output (correct display brightness).
//!
//! Both are written with the raw `write_ppm_grey`, not the `*_display`
//! helper — exactly as example 06 does, and for the same reason: the
//! helper would apply the transfer itself, and image 1 is the picture of
//! what happens when it is skipped.
//!
//! What to look for:
//!
//! - **The pictures are example 06's pictures.** Image 1 is far too dark
//!   in the shadows: linear 0.18 is middle grey but reads as 18% of
//!   peak, where the eye expects roughly 46%. In image 2 the neutral
//!   ramp (row 4) runs from near-black to near-white in a perceptually
//!   even progression.
//! - **The two agreement rows say different things, and the difference
//!   is the lesson.** `luminance_bw` is a dot product with no
//!   transcendental in it, so `docs/ffi.md` §6 promises it bit-exact
//!   across backends and the row is checked with `==`. `encode_srgb` is
//!   `1.055·x^(1/2.4) − 0.055`, and IEEE-754 standardises no
//!   transcendental: neither the host libm nor the device's is required
//!   to round `powf` correctly, so §6 commits a bound of
//!   rtol 1e-5 / atol 1e-7 instead. That row prints the worst element as
//!   a multiple of the bound. Claiming bit-exactness for it would be
//!   claiming something the crate does not promise and cannot enforce
//!   across a driver update.
//! - **The neutral-patch table, and the count beneath it.** The chart
//!   holds only 24 distinct luminance values, 4096 pixels each, so the
//!   entire disagreement between the two backends is a property of 24
//!   numbers. Row 4's six neutrals are printed side by side in ULP —
//!   the units the difference is actually measured in — and the line
//!   under the table says how many of all 24 differ, and by how much.
//!   Every value on this chart is above the 0.0031308 threshold and so
//!   takes the `powf` branch; the linear segment below it is a bare
//!   multiply and cannot differ at all.
//! - **A degenerate input does not prove exactness.** 24 distinct
//!   arguments is a very small sample of `powf`, and the two libms may
//!   well agree on most or all of them. A row reading `0 ULP` is a
//!   measurement on this chart, on this card, under this driver — not
//!   a promise. §6's bound is the promise.
//! - **The 8-bit verdict.** A handful of ULP in f32 almost never crosses
//!   a rounding boundary, so the count of differing codes in the written
//!   file is the number that matters to a photographer, and it is much
//!   smaller than the f32 count above it.
//!
//! Shape of the run: one upload, the conversion and the encode on the
//! card, one download per image written.
//!
//! Reference: IEC 61966-2-1:1999, "Multimedia systems and equipment —
//! Colour measurement and management — Part 2-1: Colour management —
//! Default RGB colour space — sRGB."
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};

/// The committed cross-backend bound for a kernel whose transcendental
/// content is a single `powf` (`docs/ffi.md` §6). Copied from the
/// document rather than invented here.
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-7;

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
/// values. Meaningful here because every value in play is non-negative.
fn ulps_apart(a: f32, b: f32) -> i64 {
    i64::from(a.to_bits()) - i64::from(b.to_bits())
}

/// The 8-bit code `shared::write_ppm` emits for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// One row of the agreement table: how the two backends compare on one
/// image, and what §6 promises about it.
fn report(name: &str, promise: &str, exact: bool, cpu: &Array3<f32>, gpu: &Array3<f32>) {
    let differing = cpu
        .iter()
        .zip(gpu.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let codes = cpu
        .iter()
        .zip(gpu.iter())
        .filter(|(x, y)| code(**x) != code(**y))
        .count();
    let bound = if exact {
        "-".to_string()
    } else {
        format!("{:.4}", worst_violation(cpu, gpu, RTOL, ATOL))
    };
    println!(
        "{name:<10} {promise:<10} {:>9} {:>12.3e} {bound:>9} {:>15} {:>15}",
        cpu == gpu,
        max_abs_diff(cpu, gpu),
        format!("{differing}/{}", cpu.len()),
        format!("{codes}/{}", cpu.len()),
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

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    // The CPU chain, computed exactly as example 06 computes it. It is
    // the specification (`docs/ffi.md` §6); the device output is measured
    // against it, never the other way round.
    let cpu_linear = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let cpu_encoded = phaios_core::encode::encode_srgb(cpu_linear.view()).unwrap();

    // The device chain: one upload, conversion and encode on the card.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let device_linear = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();
    let device_encoded = k::encode_srgb_device(&device_linear).unwrap();
    let gpu_linear = ctx.download(&device_linear).unwrap();
    let gpu_encoded = ctx.download(&device_encoded).unwrap();

    println!(
        "{:<10} {:<10} {:>9} {:>12} {:>9} {:>15} {:>15}",
        "image", "promise", "CPU == GPU", "max |delta|", "bound", "f32 differing", "codes differ"
    );
    report("linear", "bit-exact", true, &cpu_linear, &gpu_linear);
    report("srgb", "1e-5/1e-7", false, &cpu_encoded, &gpu_encoded);
    println!(
        "  bound is the committed one-powf pair (rtol 1e-5, atol 1e-7) from\n  \
         docs/ffi.md §6, as a multiple: 1.0000 is the limit. §6 promises\n  \
         luminance_bw bit-exact and encode_srgb only bounded, so only the first\n  \
         row is checked with =="
    );

    // The neutral patches (row 4 of the chart), where the transfer does
    // its most visible work. Read from the images themselves rather than
    // from a second upload, so the numbers are the ones in the files.
    println!("\nneutral patches, centre pixel — the sRGB transfer on both backends:");
    println!(
        "  {:<14} {:>9} {:>11} {:>11} {:>6} {:>10}",
        "patch", "linear", "CPU encoded", "GPU encoded", "ULP", "8-bit code"
    );
    let y = 3 * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
    for col in 0..shared::GRID_COLS {
        let x = col * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
        let linear = gpu_linear[[y, x, 0]];
        let c = cpu_encoded[[y, x, 0]];
        let g = gpu_encoded[[y, x, 0]];
        let codes = if code(c) == code(g) {
            format!("{}", code(c))
        } else {
            format!("{} vs {}", code(c), code(g))
        };
        println!(
            "  #{:<13} {linear:>9.5} {c:>11.7} {g:>11.7} {:>6} {codes:>10}",
            19 + col,
            ulps_apart(c, g),
        );
    }

    // ... and the same question over all 24, which is the whole of the
    // chart's information: 4096 pixels carry each of these values, so
    // the f32 count in the table above is 4096 times this one.
    let mut patches_differing = 0_usize;
    let mut patches_recoded = 0_usize;
    let mut worst_ulp = 0_i64;
    for i in 0..shared::PATCHES.len() {
        let y = (i / shared::GRID_COLS) * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
        let x = (i % shared::GRID_COLS) * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
        let (c, g) = (cpu_encoded[[y, x, 0]], gpu_encoded[[y, x, 0]]);
        let d = ulps_apart(c, g);
        if d != 0 {
            patches_differing += 1;
            worst_ulp = worst_ulp.max(d.abs());
        }
        if code(c) != code(g) {
            patches_recoded += 1;
        }
    }
    println!(
        "  of the chart's {} distinct luminance values, {patches_differing} encode to a \
         different f32 on the\n  two backends (worst {worst_ulp} ULP), and {patches_recoded} \
         to a different 8-bit code",
        shared::PATCHES.len()
    );

    // 1 — Linear luminance (no transfer encoding — will look too dark).
    shared::write_ppm_grey(
        Path::new("examples/output/29_gpu_encode_linear.ppm"),
        gpu_linear.as_slice().expect("kernel output is contiguous"),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — sRGB-encoded (correct display brightness). Already
    // display-referred, so it is written as it is.
    shared::write_ppm_grey(
        Path::new("examples/output/29_gpu_encode_srgb.ppm"),
        gpu_encoded.as_slice().expect("kernel output is contiguous"),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("\nwrote examples/output/29_gpu_encode_{{linear,srgb}}.ppm");
    println!(
        "diff against the CPU twin: examples/output/06_encode_{{linear,srgb}}.ppm \
         (cargo run --example 06_srgb_encode)"
    );
}
