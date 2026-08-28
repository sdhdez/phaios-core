// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 36 — Resampling: resize and straighten, on the CUDA backend.
//!
//! The GPU twin of `examples/14_resample.rs`. Same synthetic Macbeth
//! chart, same three resize filters, same two straighten angles — run
//! through the device-resident entry points (`resize_device`,
//! `straighten_device`) rather than the CPU ones, so that the two sets
//! of PPMs can be compared file by file.
//!
//! The chart is uploaded **once** and all five kernels read that same
//! device buffer; only the finished images come back. That is what the
//! resident API is for: a geometry stage in a real pipeline is followed
//! by a dozen more, and none of them should pay a PCIe crossing.
//!
//! `docs/ffi.md` §6 lists `resize` and `straighten` among the kernels
//! promised **bit-exact** across backends. The filters are polynomial —
//! multiplies and adds, each correctly rounded by IEEE-754 — the PTX is
//! compiled with `-fmad=false` so neither side contracts a multiply-add,
//! and straighten's sin/cos (the one transcendental in sight) is
//! computed once on the host and handed to the device, so it cannot
//! differ. The table below therefore compares with `==` and no
//! tolerance. The CPU implementation is the specification: a row that
//! is not `true` means the GPU kernel is wrong.
//!
//! What to look for:
//! - The `CPU == GPU` column: five `true`s, and a `max |delta|` column
//!   of exact zeros beside them. The two are not the same claim —
//!   `==` also separates `+0.0` from `-0.0` and rejects NaN — which is
//!   why the verdict is the equality and the magnitude is only context.
//! - `36_resize_area_down.ppm` (½ size) keeps every patch's tone: area
//!   averaging preserves the mean. `36_resize_catmull_up.ppm` (2×)
//!   keeps the patch borders crisp without ringing overshoot on the
//!   flat chart.
//! - The straighten outputs are smaller than the input — the largest
//!   inscribed rectangle — and their patch edges stay smooth
//!   (Catmull-Rom), not staircased. The printed dimensions are the same
//!   numbers example 14 prints, because the inscribed-rectangle
//!   geometry is host arithmetic shared by both backends.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred, and
//! that encode happens on the host exactly as in example 14 — the files
//! are meant to be byte-comparable, so nothing in the writing path may
//! differ either.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::{Array3, ArrayView3};
use phaios_core::cuda::{self, kernels as k};
use phaios_core::geometry::{ResizeFilter, ResizeParams, StraightenParams, resize, straighten};

/// Largest absolute difference between two equally-shaped images.
///
/// Reduced with a plain `>` and an explicit NaN test rather than
/// `f32::max`: `f32::max` returns the *other* operand when one side is
/// NaN, so a fold over it scores a NaN-filled output as a perfect
/// match. Example 22 records the same trap in the conformance oracle.
fn max_abs_diff(a: ArrayView3<f32>, b: ArrayView3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    worst
}

/// Print one row of the agreement table; return whether it was exact.
fn report(label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) -> bool {
    assert_eq!(
        cpu.dim(),
        gpu.dim(),
        "shape mismatch: {:?} vs {:?}",
        cpu.dim(),
        gpu.dim()
    );
    let (h, w, _) = gpu.dim();
    let size = format!("{w}x{h}");
    let exact = cpu == gpu;
    println!(
        "  {label:<20} {size:>11} {exact:>12} {:>13.3e}",
        max_abs_diff(cpu.view(), gpu.view())
    );
    exact
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
    let (w, h) = (shared::WIDTH as u32, shared::HEIGHT as u32);

    // One upload. Every kernel below resamples this same device buffer.
    let device = ctx.upload(rgb.view()).unwrap();

    let configs: [(&str, ResizeParams); 3] = [
        (
            "area_down",
            ResizeParams::new(w / 2, h / 2, ResizeFilter::Area),
        ),
        (
            "bilinear_down",
            ResizeParams::new(w / 2, h / 2, ResizeFilter::Bilinear),
        ),
        (
            "catmull_up",
            ResizeParams::new(w * 2, h * 2, ResizeFilter::CatmullRom),
        ),
    ];

    println!("device-resident resample vs the CPU specification:");
    println!(
        "  {:<20} {:>11} {:>12} {:>13}",
        "output", "size", "CPU == GPU", "max |delta|"
    );

    let mut all_exact = true;
    for (name, params) in configs.iter() {
        let gpu = ctx
            .download(&k::resize_device(&device, params).unwrap())
            .unwrap();
        let cpu = resize(rgb.view(), params).unwrap();
        let (oh, ow, _) = gpu.dim();
        let path = format!("examples/output/36_resize_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), gpu.view(), ow, oh);
        all_exact &= report(&format!("resize_{name}"), &cpu, &gpu);
    }

    for degrees in [3.0_f32, -12.0] {
        let params = StraightenParams::new(degrees);
        let gpu = ctx
            .download(&k::straighten_device(&device, &params).unwrap())
            .unwrap();
        let cpu = straighten(rgb.view(), &params).unwrap();
        let (oh, ow, _) = gpu.dim();
        let path = format!("examples/output/36_straighten_{degrees}.ppm");
        shared::write_ppm_display(Path::new(&path), gpu.view(), ow, oh);
        all_exact &= report(&format!("straighten_{degrees}"), &cpu, &gpu);
    }
    println!("wrote examples/output/36_{{resize_*,straighten_*}}.ppm");

    // The inscribed-rectangle geometry, printed as example 14 prints it.
    // Both backends take these dimensions from the same host function,
    // so a disagreement here would be a shape bug, not a rounding one.
    println!("\ninscribed crop after straightening (host geometry, shared by both backends):");
    for degrees in [3.0_f32, -12.0] {
        let out = straighten(rgb.view(), &StraightenParams::new(degrees)).unwrap();
        let (oh, ow, _) = out.dim();
        println!(
            "  straighten {degrees:>6}: {}x{} -> {ow}x{oh}",
            shared::WIDTH,
            shared::HEIGHT
        );
    }

    println!(
        "\nall five outputs bit-exact against the CPU (docs/ffi.md §6 promises this): {all_exact}"
    );
    println!(
        "diff against example 14: for f in resize_area_down resize_bilinear_down resize_catmull_up straighten_3 straighten_-12; do cmp examples/output/36_$f.ppm examples/output/14_$f.ppm; done"
    );
}
