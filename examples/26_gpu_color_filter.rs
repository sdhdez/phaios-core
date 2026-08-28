// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 26 — Colour filter B&W simulation, on the CUDA backend.
//!
//! The GPU twin of `examples/03_color_filter.rs`. Same chart, same six
//! Wratten-style presets (None, Yellow #8 K2, Orange #21, Red #25 A,
//! Green #11 X1, Blue #47 C5), one PPM per preset.
//!
//! What to look for in the output:
//!
//! - **The photography, unchanged from example 03.** Red #25 A drives
//!   the blue-sky patch (#03) very dark and lightens the reds (#09,
//!   #15) — the classic dramatic-sky effect. Green #11 X1 brightens the
//!   foliage patches (#04, #11, #14) and darkens reds and blues, the
//!   natural landscape look. Blue #47 C5 inverts what Red does: sky
//!   bright, reds dark, as used for atmospheric haze.
//! - **The agreement lines.** The kernel multiplies RGB by the filter's
//!   three transmission factors and then takes the luminance dot
//!   product — six multiplies and two adds per pixel, and not a single
//!   transcendental — so `docs/ffi.md` §6 promises it *bit-exact*
//!   between CPU and CUDA. Each of the six lines is `==` over the whole
//!   array, with no tolerance. The PTX is compiled with `-fmad=false`,
//!   so the device cannot contract the filter multiply and the
//!   luminance add into one fused operation that the CPU never
//!   performs.
//! - **All six PPM pairs are byte-identical** to example 03's: equal
//!   f32 arrays, and the same host-side sRGB encode and 8-bit
//!   quantisation on both sides.
//!
//! All six presets run from a **single** upload — the chart crosses the
//! bus once, six `*_device` kernels read the same device buffer, and
//! each result is pulled back once. The filter factors themselves are
//! kernel arguments, not buffers, so changing preset costs no transfer
//! whatsoever.
//!
//! Reference: Kodak Wratten Gelatin Filters datasheet, Publication
//! B3-203 (5th ed.) — the spectral transmission curves the presets
//! approximate; luminance weights from ITU-R BT.709-6 (2015), Part 2,
//! item 3.2.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{ColorFilter, LuminanceStandard, color_filter_bw};
use phaios_core::cuda;

/// Print the CPU/GPU agreement for a kernel `docs/ffi.md` §6 promises
/// bit-exact, as one line.
///
/// The headline comparison is `==` over the whole array, because that
/// is what §6 promises and a tolerance here would quietly absorb a
/// regression. The failure branch exists only to answer "how badly?":
/// it counts differing *bit patterns*, which is stricter than `==`
/// (it separates `-0.0` from `+0.0`), and reduces the worst magnitude
/// with a plain `>` rather than `f32::max`, since `f32::max` returns
/// the other operand when one side is NaN and would score an all-NaN
/// output as a perfect match.
///
/// Returns whether the two agreed, so the caller can report a verdict
/// over all six presets rather than leaving the reader to scan.
fn report_bit_exact(label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) -> bool {
    if cpu.dim() != gpu.dim() {
        println!(
            "{label}: shape mismatch, {:?} against {:?}",
            cpu.dim(),
            gpu.dim()
        );
        return false;
    }
    if cpu == gpu {
        println!(
            "{label:<22} CPU == GPU bit-for-bit: true  ({} elements, no tolerance)",
            cpu.len()
        );
        return true;
    }
    let differing = cpu
        .iter()
        .zip(gpu.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let mut worst = 0.0_f32;
    for (a, b) in cpu.iter().zip(gpu.iter()) {
        let d = (a - b).abs();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    println!(
        "{label:<22} CPU == GPU bit-for-bit: FALSE  ({differing} of {} elements differ, \
         worst |delta| {worst:.3e}) — §6 promises exactness here, so this is a bug",
        cpu.len()
    );
    false
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

    // One upload; every preset reads this same device buffer.
    let dev = ctx.upload(rgb.view()).unwrap();
    let standard = LuminanceStandard::Bt709;

    // The file-name stems match example 03's exactly, so the two
    // directories can be compared preset by preset.
    let presets: &[(&str, &str, ColorFilter)] = &[
        ("none", "no filter", ColorFilter::NoFilter),
        ("yellow", "Yellow #8 K2", ColorFilter::Yellow8K2),
        ("orange", "Orange #21", ColorFilter::Orange21),
        ("red", "Red #25 A", ColorFilter::Red25A),
        ("green", "Green #11 X1", ColorFilter::Green11X1),
        ("blue", "Blue #47 C5", ColorFilter::Blue47C5),
    ];

    let mut all_exact = true;
    for (name, label, filter) in presets {
        let gpu = ctx
            .download(&cuda::kernels::color_filter_bw_device(&dev, *filter, standard).unwrap())
            .unwrap();
        let cpu = color_filter_bw(rgb.view(), *filter, standard).unwrap();
        all_exact &= report_bit_exact(label, &cpu, &gpu);

        let out = format!("examples/output/26_gpu_filter_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&out), gpu.view(), shared::WIDTH, shared::HEIGHT);
        println!("  wrote {out}");
    }

    println!("\nall six presets bit-exact: {all_exact}  (one upload, six kernels, six downloads)");
    println!(
        "diff against the CPU twin (run example 03 first), for each of the six:\n  \
         cmp examples/output/26_gpu_filter_<name>.ppm examples/output/03_filter_<name>.ppm"
    );
}
