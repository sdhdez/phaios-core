// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 24 — Standard luminance B&W conversion, on the CUDA backend.
//!
//! The GPU twin of `examples/01_luminance.rs`: the same synthetic
//! Macbeth chart, the same default BT.709 weights, the same single
//! output image. Only the processor differs, and the whole point of the
//! example is that this changes nothing about the pixels.
//!
//! What to look for in the output:
//!
//! - **The photography, unchanged from example 01.** The neutral
//!   patches (row 4) form a smooth grey ramp, and the yellow-green
//!   patch (#11) renders brighter than the red one (#15) although both
//!   are vivid — BT.709 weights green at 0.71 against red's 0.21.
//! - **The agreement line.** `luminance_bw` is one dot product per
//!   pixel: three multiplies and two adds, no transcendental. So
//!   `docs/ffi.md` §6 lists it among the kernels promised *bit-exact*
//!   between CPU and CUDA, and the check below is `==` over the whole
//!   array with no tolerance at all. This holds because the PTX is
//!   compiled with `-fmad=false`: Rust does not contract `a*b + c` into
//!   a fused multiply-add, and the device is told not to either, so
//!   every remaining operation is correctly rounded on both sides.
//! - **The two PPM files are byte-identical.** Both backends yield the
//!   same f32 array, and both files are written by the same host-side
//!   `encode_srgb` and 8-bit quantisation, so `cmp` over the pair is
//!   the end-to-end form of the same claim.
//!
//! The kernel runs through the device-resident entry point
//! [`phaios_core::cuda::kernels::luminance_bw_device`] — one upload,
//! one kernel, one download. That is the API the backend exists for;
//! example 22 shows why the shape matters once there is more than one
//! stage to run.
//!
//! Reference: ITU-R BT.709-6 (2015), Part 2, item 3.2 — the luminance
//! coefficients (0.2126, 0.7152, 0.0722).
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
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
            "{label}: CPU == GPU bit-for-bit: true  ({} elements, no tolerance)",
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
        "{label}: CPU == GPU bit-for-bit: FALSE  ({differing} of {} elements differ, \
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

    // The same chart every other example uses.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    // One upload, one kernel, one download.
    let dev = ctx.upload(rgb.view()).unwrap();
    let result = cuda::kernels::luminance_bw_device(&dev, LuminanceStandard::Bt709).unwrap();
    let gpu = ctx.download(&result).unwrap();

    let cpu = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    report_bit_exact("BT.709 luminance", &cpu, &gpu);

    // The GPU result is what gets written: a twin that wrote the CPU
    // array would be checking nothing.
    shared::write_ppm_grey_display(
        Path::new("examples/output/24_gpu_luminance_bt709.ppm"),
        gpu.view(),
        shared::WIDTH,
        shared::HEIGHT,
    );
    println!("wrote examples/output/24_gpu_luminance_bt709.ppm");
    println!(
        "diff against the CPU twin (run example 01 first):\n  \
         cmp examples/output/24_gpu_luminance_bt709.ppm examples/output/01_luminance_bt709.ppm"
    );
}
