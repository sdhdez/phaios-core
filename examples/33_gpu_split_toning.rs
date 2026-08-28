// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 33 — Split-toning in OKLab, on the CUDA backend.
//!
//! The GPU twin of `examples/11_split_toning.rs`: the same chart, the
//! same five configurations — reference, sepia, selenium, cross-process
//! and the shifted balance — computed on the device. One
//! [`Context::upload`], `luminance_bw_device` then `split_toning_device`
//! with the monochrome intermediate staying on the card, one download
//! per rendered frame.
//!
//! What to look for in the output:
//!
//! - **The agreement table, and why it is a bound rather than `==`.**
//!   OKLab is a cube-root round trip: linear sRGB → LMS → `cbrt` →
//!   Lab, and back. IEEE-754 standardises `+ − × ÷ √` and requires
//!   correct rounding for those; it standardises **no** transcendental,
//!   and `cbrtf` is one. Two CPU machines with different libms already
//!   disagree in the low bits here — the device is one more libm, not a
//!   new category of problem. `docs/ffi.md` §6 commits this kernel to
//!   (rtol 1e-5, atol 1e-7), and the *bound* column expresses the worst
//!   element as a multiple of it: at or below 1.0000 passes. Claiming
//!   bit-exactness for this kernel would be claiming something §6 does
//!   not promise and the arithmetic cannot deliver.
//! - **The two right-hand columns, which are not the same question.**
//!   How many f32 values differ by so much as one ULP is a fact about
//!   the arithmetic; how many 8-bit codes differ after `encode_srgb` is
//!   the fact a photographer can see. The second number is far smaller,
//!   because a fraction of a ULP almost never crosses a rounding
//!   boundary — so whether these PPMs diff clean against example 11's is
//!   measured here rather than asserted. The f32 counts arrive in
//!   multiples of 4096 because the chart is flat: 24 distinct patch
//!   values over 4096 pixels each, so a value that differs differs
//!   everywhere it appears.
//! - **The untinted round trip.** Configuration 1 has no chroma in
//!   either slot, so monochrome in must give neutral out. The line below
//!   measures the device's departure from the luminance it was given. If
//!   that number were large, every tint in images 2–5 would be sitting
//!   on top of a colour cast rather than on grey.
//!
//! What the five configurations *mean* photographically — where the
//! crossover sits in the neutral ramp, and why OKLab keeps the patches'
//! relative brightness when tinting in linear sRGB would not — is
//! example 11's subject and is not repeated here.
//!
//! Reference for the colour space: Björn Ottosson, "A perceptual color
//! space for image processing" (2020),
//! <https://bottosson.github.io/posts/oklab/> — the title spelt as
//! published.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), by the same host-side writer example 11
//! uses, so the two sets of files are comparable code for code.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::split_toning::SplitToningParams;

/// The committed bound for a one-`cbrtf` kernel (`docs/ffi.md` §6).
const RTOL: f32 = 1e-5;
/// Absolute term of the same bound.
const ATOL: f32 = 1e-7;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: at or below 1.0 means every element passes.
///
/// Non-finite values are compared explicitly rather than arithmetically,
/// and the reduction uses a plain `>` rather than `f32::max`, for the
/// reason recorded in `tests/cuda_conformance.rs`: `f32::max` returns
/// the *other* operand when one side is NaN, so a fold over it scores an
/// all-NaN output as a perfect match. That is how the conformance oracle
/// was wrong before it was fixed.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
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
            (x - y).abs() / (ATOL + RTOL * x.abs())
        };
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Largest absolute difference, element for element. Reduced with `>`
/// so a NaN cannot read as agreement, for the reason above.
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

/// The 8-bit code `shared::write_ppm` would emit for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
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

    // One upload; the B&W conversion runs on the card, so the toning
    // kernel's input is produced where it is consumed.
    let dev_rgb = ctx.upload(rgb.view()).unwrap();
    let dev_bw = k::luminance_bw_device(&dev_rgb, LuminanceStandard::Bt709).unwrap();

    // The CPU side of the comparison, from the same chart. The CPU
    // implementation is the specification (CLAUDE.md §2).
    let bw = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, SplitToningParams); 5] = [
        (
            "reference",
            SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, 0.0),
        ),
        (
            "sepia",
            SplitToningParams::new([0.0, 0.03, 0.05], [0.0, 0.03, 0.05], 0.5, 0.0),
        ),
        (
            "selenium",
            SplitToningParams::new([0.0, 0.01, -0.05], [0.0, 0.0, 0.0], 0.5, 0.0),
        ),
        (
            "cross_process",
            SplitToningParams::new([0.0, -0.02, -0.05], [0.0, 0.03, 0.04], 0.5, 0.0),
        ),
        (
            "balance_high",
            SplitToningParams::new([0.0, -0.02, -0.05], [0.0, 0.03, 0.04], 0.5, 0.6),
        ),
    ];

    println!("split_toning against the CPU kernel on the same chart.");
    println!("docs/ffi.md §6 bounds this one at (rtol 1e-5, atol 1e-7) — a cbrt round trip");
    println!("is not bit-exact across libms — so `bound` is the worst element as a multiple");
    println!("of that pair: 1.0000 is the limit, not the target.");
    println!(
        "  {:<14} {:>12} {:>9} {:>17} {:>17}",
        "config", "max |delta|", "bound", "f32 differing", "8-bit differing"
    );

    let mut worst_overall = 0.0_f32;
    let mut neutral_cast = 0.0_f32;
    for (name, params) in configs {
        let gpu = ctx
            .download(&k::split_toning_device(&dev_bw, &params).unwrap())
            .unwrap();
        let cpu = phaios_core::split_toning::split_toning(bw.view(), &params).unwrap();

        let abs = max_abs_diff(&cpu, &gpu);
        let viol = worst_violation(&cpu, &gpu);
        // Plain `>` rather than `f32::max`, for the same reason the
        // helper above avoids it.
        if viol > worst_overall {
            worst_overall = viol;
        }
        let bits = cpu
            .iter()
            .zip(gpu.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        // The codes are compared after the same encode the writer
        // applies, which is what makes this the number that reaches a
        // file rather than a number about the arithmetic.
        let cpu_enc = phaios_core::encode::encode_srgb(cpu.view()).unwrap();
        let gpu_enc = phaios_core::encode::encode_srgb(gpu.view()).unwrap();
        let codes = cpu_enc
            .iter()
            .zip(gpu_enc.iter())
            .filter(|(x, y)| code(**x) != code(**y))
            .count();

        if name == "reference" {
            // Monochrome in, neutral out: the departure of the device's
            // three channels from the luminance they were built from.
            for y in 0..shared::HEIGHT {
                for x in 0..shared::WIDTH {
                    for c in 0..3 {
                        let d = (gpu[[y, x, c]] - bw[[y, x, 0]]).abs();
                        if d > neutral_cast || d.is_nan() {
                            neutral_cast = d;
                        }
                    }
                }
            }
        }

        let path = format!("examples/output/33_gpu_split_toning_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);

        let n = cpu.len();
        println!(
            "  {name:<14} {abs:>12.3e} {viol:>9.4} {:>17} {:>17}",
            format!("{bits}/{n}"),
            format!("{codes}/{n}")
        );
    }
    println!(
        "worst element across all five: {worst_overall:.4} of the committed bound (pass: {})",
        worst_overall <= 1.0
    );
    println!(
        "untinted round trip on the device (luminance -> OKLab -> linear sRGB): \
         max error {neutral_cast:.3e}"
    );

    println!("wrote examples/output/33_gpu_split_toning_*.ppm");
    println!(
        "diff against the CPU twin, examples/output/11_toning_*.ppm \
         (`cargo run --example 11_split_toning`); the 8-bit column above says what to expect"
    );
}
