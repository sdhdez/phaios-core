// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 25 — Channel mixer B&W conversion, on the CUDA backend.
//!
//! The GPU twin of `examples/02_channel_mixer.rs`. Same chart, same
//! three mixes, same three output images:
//!
//! 1. Standard BT.709 weights (0.2126, 0.7152, 0.0722) as a reference —
//!    taken, as in example 02, through `luminance_bw_device` so the
//!    weights are the exact BT.709 constants rather than a rounded
//!    triple typed in by hand.
//! 2. Red-boosted weights (1.0, 0.0, 0.0) — infrared-like: reds bright,
//!    greens and blues very dark.
//! 3. Custom weights (0.0, 0.5, 0.5) — equal green+blue, no red.
//!
//! What to look for in the output:
//!
//! - **The photography, unchanged from example 02.** Image 2 inverts
//!   the tonal relationship between the red (#15) and green (#14)
//!   patches relative to image 1. Negative weights are valid and invert
//!   tones further; the self-test in example 23 exercises
//!   `[-0.2, 1.4, -0.2]` for exactly that reason.
//! - **The agreement lines.** `channel_mixer_bw` is `wR·R + wG·G +
//!   wB·B` — three multiplies and two adds per pixel, no
//!   transcendental — so `docs/ffi.md` §6 promises it *bit-exact*
//!   between CPU and CUDA, and each line below is `==` over the whole
//!   array with no tolerance. Arbitrary weights change nothing about
//!   that: the PTX is compiled with `-fmad=false`, so the device does
//!   not contract `a*b + c` into an FMA that the CPU never performs,
//!   whatever the coefficients are.
//! - **All three PPM pairs are byte-identical** to example 02's, the
//!   two backends' f32 arrays being equal and the sRGB encode and 8-bit
//!   quantisation happening on the host in both cases.
//!
//! All three mixes run from a **single** upload: the chart crosses the
//! bus once and the three `*_device` kernels read the same device
//! buffer, one download per output image. Three uploads of the same
//! array would be three copies of one thing.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, channel_mixer_bw, luminance_bw};
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
fn report_bit_exact(label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) {
    if cpu.dim() != gpu.dim() {
        println!(
            "{label}: shape mismatch, {:?} against {:?}",
            cpu.dim(),
            gpu.dim()
        );
        return;
    }
    if cpu == gpu {
        println!(
            "{label:<28} CPU == GPU bit-for-bit: true  ({} elements, no tolerance)",
            cpu.len()
        );
        return;
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
        "{label:<28} CPU == GPU bit-for-bit: FALSE  ({differing} of {} elements differ, \
         worst |delta| {worst:.3e}) — §6 promises exactness here, so this is a bug",
        cpu.len()
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

    // One upload for all three mixes.
    let dev = ctx.upload(rgb.view()).unwrap();

    let write = |name: &str, img: &Array3<f32>| {
        let path = format!("examples/output/25_gpu_mixer_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), img.view(), shared::WIDTH, shared::HEIGHT);
    };

    // 1 — BT.709 reference (via luminance_bw for exact BT.709 weights)
    let gpu_bt709 = ctx
        .download(&cuda::kernels::luminance_bw_device(&dev, LuminanceStandard::Bt709).unwrap())
        .unwrap();
    let cpu_bt709 = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    report_bit_exact("BT.709 reference", &cpu_bt709, &gpu_bt709);
    write("bt709", &gpu_bt709);

    // 2 — Red channel only (infrared-like)
    let weights_red = [1.0_f32, 0.0, 0.0];
    let gpu_red = ctx
        .download(&cuda::kernels::channel_mixer_bw_device(&dev, weights_red).unwrap())
        .unwrap();
    let cpu_red = channel_mixer_bw(rgb.view(), weights_red).unwrap();
    report_bit_exact("red only (1.0, 0.0, 0.0)", &cpu_red, &gpu_red);
    write("red_only", &gpu_red);

    // 3 — Equal green+blue, no red
    let weights_gb = [0.0_f32, 0.5, 0.5];
    let gpu_gb = ctx
        .download(&cuda::kernels::channel_mixer_bw_device(&dev, weights_gb).unwrap())
        .unwrap();
    let cpu_gb = channel_mixer_bw(rgb.view(), weights_gb).unwrap();
    report_bit_exact("green+blue (0.0, 0.5, 0.5)", &cpu_gb, &gpu_gb);
    write("green_blue", &gpu_gb);

    println!("wrote examples/output/25_gpu_mixer_{{bt709,red_only,green_blue}}.ppm");
    println!(
        "diff against the CPU twin (run example 02 first), for each of the three:\n  \
         cmp examples/output/25_gpu_mixer_<name>.ppm examples/output/02_mixer_<name>.ppm"
    );
}
