// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 27 — Adams/Archer Zone System tone curve, on the CUDA backend.
//!
//! The GPU twin of `examples/04_zone_system.rs`. Same chart, same three
//! configurations, same three output images:
//!
//! 1. No-op (empty params) — output == input.
//! 2. "Pull" Zone V: offset Zone V by −1 stop (darker mid-tones).
//! 3. "Push" Zone VII by +1 stop, "deepen" Zone III by −0.5 stops — a
//!    classic landscape / darkroom interpretation curve.
//!
//! What to look for in the output:
//!
//! - **The photography, unchanged from example 04.** Compare the grey
//!   ramp in row 4 across configurations 1 and 3. The Gaussian blending
//!   (σ = 0.8 zones) means a Zone V offset also moves Zones IV and VI:
//!   the influence is gradual, not a sharp step, and the patches
//!   nearest the targeted zone move most.
//! - **The agreement lines — and this is the one kernel of the four
//!   B&W-stage twins that is *not* promised bit-exact.** The zone
//!   position is `5 + log2(L / 0.18)` and each zone's weight is a
//!   Gaussian, so the kernel evaluates `log2f` and `expf`. IEEE-754
//!   standardises `+ − × ÷ √` and requires those to be correctly
//!   rounded; it standardises **no** transcendental. Host libm and
//!   device libm are therefore allowed to differ in the last bits, and
//!   `docs/ffi.md` §6 commits a bound instead of exactness: rtol 1e-5,
//!   atol 1e-7. The lines below print the worst element as a multiple
//!   of that bound, where 1.000 is the limit — copied from §6 rather
//!   than chosen here, so a driver update that regresses accuracy shows
//!   up as a number above 1 instead of being quietly absorbed.
//! - **The identity path is still exact.** With no offsets the kernel
//!   does no zone arithmetic at all — the CPU assigns its input
//!   through and the device issues a device-to-device copy — so
//!   configuration 1 is checked with `==` and no tolerance.
//! - **What survives to 8 bits.** A few ULP in f32 is not the question
//!   a photographer asks. Each line therefore also counts how many of
//!   the written 8-bit codes differ from the CPU twin's, after the same
//!   `encode_srgb` and rounding `shared::write_ppm` applies. That is
//!   the number that decides whether `cmp` on the two PPM files is
//!   silent.
//!
//! The whole thing runs from a **single** upload. `luminance_bw_device`
//! produces the mono image on the card and the three `zone_system_device`
//! calls read it there — the (H, W, 1) intermediate never crosses the
//! bus. Its own agreement is checked first and separately: `luminance_bw`
//! is bit-exact per §6, so the three zone lines below are attributable
//! to `zone_system` alone rather than to something inherited upstream.
//!
//! Reference: Ansel Adams, *The Negative*, Little, Brown (1948), ch. 5;
//! modernised in Davis, *Beyond the Zone System*, Focal Press (1999),
//! from which the σ = 0.8 zone blending comes.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda;
use phaios_core::encode::encode_srgb;
use phaios_core::tone::{ZoneParams, zone_system};

/// `docs/ffi.md` §6's committed cross-backend bound for a kernel whose
/// transcendental content is a single `powf` / `expf` / `log2f`.
///
/// Copied from §6, not invented here: the point of printing a multiple
/// of a *committed* bound is that the number means the same thing on
/// every card that runs this example.
const RTOL: f32 = 1e-5;
/// Absolute term of the same bound; see [`RTOL`].
const ATOL: f32 = 1e-7;
/// How the bound is written in §6 and in example 23's table.
const BOUND_LABEL: &str = "1e-5/1e-7";

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: `<= 1.0` means every element passes.
///
/// The same two-term form `tests/cuda_conformance.rs` and example 23
/// use, including their care with non-finite values. Comparing NaN
/// arithmetically, or reducing with `f32::max`, scores a wholly-NaN or
/// truncated output as a perfect match — which is how the conformance
/// oracle was wrong before it was fixed.
fn worst_violation(cpu: &Array3<f32>, gpu: &Array3<f32>) -> f32 {
    if cpu.dim() != gpu.dim() {
        return f32::INFINITY;
    }
    let mut worst = 0.0_f32;
    for (x, y) in cpu.iter().zip(gpu.iter()) {
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
/// above: `f32::max` returns the *other* operand when one side is NaN.
fn max_abs_diff(cpu: &Array3<f32>, gpu: &Array3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in cpu.iter().zip(gpu.iter()) {
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

/// How many written 8-bit codes differ, once both arrays have been
/// through the terminal `encode_srgb` the PPM writers apply.
///
/// This is the end-to-end question — whether `cmp` on the two files is
/// silent — rather than the f32 one.
fn codes_differing(cpu: &Array3<f32>, gpu: &Array3<f32>) -> usize {
    let cpu_srgb = encode_srgb(cpu.view()).unwrap();
    let gpu_srgb = encode_srgb(gpu.view()).unwrap();
    cpu_srgb
        .iter()
        .zip(gpu_srgb.iter())
        .filter(|(a, b)| code(**a) != code(**b))
        .count()
}

/// Print one bounded configuration's agreement: the worst absolute
/// difference, that difference as a multiple of §6's committed bound,
/// and what reaches the file.
fn report_bounded(label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) {
    let violation = worst_violation(cpu, gpu);
    let differing = codes_differing(cpu, gpu);
    let verdict = if violation <= 1.0 { "within" } else { "OVER" };
    println!(
        "{label:<26} max |delta| {:.3e}   {violation:>7.4}x the {BOUND_LABEL} bound ({verdict})   \
         8-bit codes differing: {differing} of {}",
        max_abs_diff(cpu, gpu),
        cpu.len()
    );
}

/// Print an exact configuration's agreement: `==`, no tolerance.
///
/// Used for the identity path, which §6 and example 23 both require to
/// be bit-exact — it is a copy on the device and an assignment on the
/// host, so there is no arithmetic to disagree about.
fn report_exact(label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) {
    if cpu == gpu {
        println!(
            "{label:<26} CPU == GPU bit-for-bit: true  ({} elements, no tolerance)",
            cpu.len()
        );
    } else {
        let differing = cpu
            .iter()
            .zip(gpu.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "{label:<26} CPU == GPU bit-for-bit: FALSE  ({differing} of {} elements differ, \
             worst |delta| {:.3e}) — exactness is required here, so this is a bug",
            cpu.len(),
            max_abs_diff(cpu, gpu)
        );
    }
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

    // One upload. The mono intermediate is produced on the card and
    // stays there; only the three finished images come back.
    let dev = ctx.upload(rgb.view()).unwrap();
    let dev_mono = cuda::kernels::luminance_bw_device(&dev, LuminanceStandard::Bt709).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // Checked first, so the zone lines below cannot be blamed on it.
    report_exact(
        "input (luminance_bw)",
        &bw,
        &ctx.download(&dev_mono).unwrap(),
    );
    println!();

    let write = |name: &str, img: &Array3<f32>| {
        let path = format!("examples/output/27_gpu_zone_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), img.view(), shared::WIDTH, shared::HEIGHT);
    };

    // 1 — No-op reference. The identity path: a device copy, no zone
    //     arithmetic, hence exact rather than bounded.
    let params_noop = ZoneParams::default();
    let gpu_ref = ctx
        .download(&cuda::kernels::zone_system_device(&dev_mono, &params_noop).unwrap())
        .unwrap();
    let cpu_ref = zone_system(bw.view(), &params_noop).unwrap();
    report_exact("1. no-op reference", &cpu_ref, &gpu_ref);
    write("reference", &gpu_ref);

    // 2 — Pull Zone V (mid-tones) down by 1 stop
    let mut offsets_pull = HashMap::new();
    offsets_pull.insert(5_i32, -1.0_f32);
    let params_pull = ZoneParams::new(offsets_pull);
    let gpu_pull = ctx
        .download(&cuda::kernels::zone_system_device(&dev_mono, &params_pull).unwrap())
        .unwrap();
    let cpu_pull = zone_system(bw.view(), &params_pull).unwrap();
    report_bounded("2. pull Zone V by -1", &cpu_pull, &gpu_pull);
    write("pull_v", &gpu_pull);

    // 3 — Push Zone VII (upper mid-tones/highlights) up, deepen Zone III
    let mut offsets_push = HashMap::new();
    offsets_push.insert(7_i32, 1.0_f32);
    offsets_push.insert(3_i32, -0.5_f32);
    let params_push = ZoneParams::new(offsets_push);
    let gpu_push = ctx
        .download(&cuda::kernels::zone_system_device(&dev_mono, &params_push).unwrap())
        .unwrap();
    let cpu_push = zone_system(bw.view(), &params_push).unwrap();
    report_bounded("3. push VII, deepen III", &cpu_push, &gpu_push);
    write("push_vii", &gpu_push);

    println!("\nwrote examples/output/27_gpu_zone_{{reference,pull_v,push_vii}}.ppm");
    println!(
        "diff against the CPU twin (run example 04 first), for each of the three:\n  \
         cmp examples/output/27_gpu_zone_<name>.ppm examples/output/04_zone_<name>.ppm\n  \
         a silent cmp means the 8-bit code count above was 0; a bounded kernel does not\n  \
         promise that, so read the count rather than assuming it."
    );
}
