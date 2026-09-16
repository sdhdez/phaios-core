// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 46 — Hot-pixel removal, on the CUDA backend.
//!
//! The GPU twin of `45_hot_pixels.rs`. Same corrupted chart, same two
//! cleaning passes, run on the device so the two can be put side by
//! side. Read example 45 first: it makes the argument that the
//! two-term criterion removes huge planted defects while leaving a
//! genuine fine line and a modest highlight texture bump alone, and
//! this example does not repeat it. It asks whether the card reproduces
//! it, and by how much it can disagree.
//!
//! Answer: not at all. `hot_pixels` is comparisons and selection only —
//! `median9`'s 19-comparator sorting network uses
//! `f32::min`/`f32::max` (IEEE-754-2008 `minNum`/`maxNum`) on both
//! backends, and the one arithmetic step (`|p - m|` against
//! `threshold + relative * |m|`) is correctly rounded under
//! `-fmad=false` — `docs/ffi.md` §6's bit-exact list. So this example
//! does not print a ratio to a bound the way the bounded-class GPU
//! examples do: it counts elements where the two backends' bit patterns
//! differ at all, and requires that count to be exactly zero.
//!
//! Everything runs device-resident: the corrupted chart is uploaded
//! once and each cleaning pass runs its own `hot_pixels_device` call
//! without an intermediate crossing the bus.
//!
//! Diff against the CPU twin: `examples/output/45_<name>.ppm` — the
//! same three names (`corrupted`, `cleaned`, `cleaned_relative`).
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, kernels as k};
use phaios_core::hot_pixels::HotPixelParams;

// Identical to examples/45_hot_pixels.rs -- kept in sync so the two
// PPMs are directly comparable.
const BLACK_SPIKE: (usize, usize) = (210, 340);
const GREY_DIP: (usize, usize) = (210, 90);
const LINE_COLS: (usize, usize) = (205, 250);
const LINE_ROW: usize = 220;
const WHITE_TEXTURE: (usize, usize) = (210, 15);
const WHITE_SPIKE: (usize, usize) = (210, 45);

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
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let mut corrupted = bw.clone();
    corrupted[[BLACK_SPIKE.0, BLACK_SPIKE.1, 0]] += 1.0;
    corrupted[[GREY_DIP.0, GREY_DIP.1, 0]] -= 0.5;
    for x in LINE_COLS.0..LINE_COLS.1 {
        corrupted[[LINE_ROW, x, 0]] += 0.02;
    }
    corrupted[[WHITE_TEXTURE.0, WHITE_TEXTURE.1, 0]] += 0.05;
    corrupted[[WHITE_SPIKE.0, WHITE_SPIKE.1, 0]] += 0.6;

    let write = |name: &str, img: ndarray::ArrayView3<f32>| {
        let path = format!("examples/output/46_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
    };
    write("corrupted", corrupted.view());

    let d_corrupted = ctx.upload(corrupted.view()).unwrap();

    println!("bit-exact class (docs/ffi.md §6): every differing sample must be 0.");
    println!("{:>18} {:>20}", "rendering", "differing samples");

    let emit = |name: &str, params: &HotPixelParams| {
        let cpu = phaios_core::hot_pixels::hot_pixels(corrupted.view(), params).unwrap();
        let gpu_dev = k::hot_pixels_device(&d_corrupted, params).unwrap();
        let gpu = ctx.download(&gpu_dev).unwrap();
        write(name, gpu.view());
        let differing = cpu
            .iter()
            .zip(gpu.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("{name:>18} {differing:>20}");
        assert_eq!(
            differing, 0,
            "{name}: hot_pixels is bit-exact; CPU and GPU must match exactly"
        );
    };

    emit("cleaned", &HotPixelParams::new(0.15, 0.0));
    emit("cleaned_relative", &HotPixelParams::new(0.03, 0.15));

    println!("\nwrote examples/output/46_*.ppm (the GPU renderings)");
    println!(
        "\ndiff against the CPU twin: examples/output/45_<name>.ppm — same three names.\n  \
         Run `cargo run --example 45_hot_pixels` first, then e.g.\n  \
         `cmp examples/output/45_cleaned.ppm examples/output/46_cleaned.ppm`."
    );
}
