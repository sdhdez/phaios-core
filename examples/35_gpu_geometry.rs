// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 35 — Exact geometry: orientation and crop, on the CUDA backend.
//!
//! The GPU twin of `examples/13_geometry.rs`: the same chart, all eight
//! Exif transforms through `orient_device`, and the same order property
//! — vignetting a crop is not the same image as cropping a vignette —
//! built from `crop_device` and `vignette_device`. One
//! [`Context::upload`] of the chart; every transform reads that one
//! device buffer, and only the finished frames come back.
//!
//! What to look for in the output:
//!
//! - **The agreement column, which is `==` for every row.** `crop` and
//!   `orient` are pure index permutations: they move floats, they do not
//!   compute with them, so there is nothing for two backends to round
//!   differently. `docs/ffi.md` §6 lists both as bit-exact, and
//!   `vignette` with them, so the whole of this example is checked
//!   against the CPU with `==` and not a tolerance. This is also the
//!   cheapest place for a launch-geometry bug to show itself: an
//!   off-by-one in the index arithmetic of a transposing kernel is not a
//!   small numerical difference but a visibly wrong image, and the four
//!   transposed transforms (5–8) exercise a different index path from
//!   the four that keep the frame's aspect.
//! - **The shape column.** Four transforms keep 6×4 patches and four are
//!   transposed to 4×6. The device is required to agree with the CPU on
//!   the output *shape* as well as its contents; a shape mismatch is
//!   reported here rather than being hidden by a comparison that zips
//!   two arrays and stops at the shorter one.
//! - **The two order-property images.** `35_gpu_geometry_crop_then_vignette.ppm`
//!   darkens toward the corners of the *crop*;
//!   `35_gpu_geometry_vignette_then_crop.ppm` keeps the off-centre falloff
//!   of the original frame. Geometry runs first in the pipeline
//!   (CLAUDE.md §3) precisely so that the first behaviour is what a
//!   sidecar reproduces — and the property survives the port, because
//!   residency does not reorder anything.
//!
//! Patch #01 (dark skin, top-left) tracks the corner through every
//! transform, exactly as in example 13.
//!
//! Reference for the orientation encoding, as in `src/geometry.rs`:
//! JEITA CP-3451C / CIPA DC-008-2012 (Exif 2.3), tag 0x0112, values
//! 1..=8 — which are the enum discriminants.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), by the same host-side writer example 13
//! uses, so the two sets of files are comparable byte for byte.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::geometry::{CropParams, Orientation, crop, orient};
use phaios_core::vignette::{VignetteParams, vignette};

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

    // One upload. All ten frames below are computed from this buffer.
    let dev = ctx.upload(rgb.view()).unwrap();

    println!("orient: the eight Exif transforms — docs/ffi.md §6 promises bit-exactness for");
    println!("pure index permutations, so this is `==` over the whole array:");
    println!(
        "  {:<5} {:<16} {:>12} {:>12}",
        "exif", "transform", "shape", "CPU == GPU"
    );

    let mut all_exact = true;
    for value in 1..=8_u16 {
        let o = Orientation::from_exif(value).unwrap();
        let gpu = ctx.download(&k::orient_device(&dev, o).unwrap()).unwrap();
        let cpu = orient(rgb.view(), o).unwrap();

        // Shapes first: comparing contents without them would let a
        // truncated output read as agreement.
        let (h, w, _) = gpu.dim();
        let same_shape = cpu.dim() == gpu.dim();
        let exact = same_shape && cpu == gpu;
        all_exact = all_exact && exact;

        let path = format!("examples/output/35_gpu_geometry_orient_{value}.ppm");
        shared::write_ppm_display(Path::new(&path), gpu.view(), w, h);
        println!(
            "  {value:<5} {:<16} {:>12} {exact:>12}",
            format!("{o:?}"),
            format!("{w}x{h}")
        );
    }

    // The order property, made visible — the same two configurations
    // example 13 renders, with both kernels device-resident so the
    // intermediate of each pair never crosses the bus.
    let cp = CropParams::new(0, 0, shared::WIDTH as u32 / 2, shared::HEIGHT as u32);
    let vg = VignetteParams::new(0.7, 1.0, 0.0);

    let gpu_crop_then_vignette = ctx
        .download(&k::vignette_device(&k::crop_device(&dev, &cp).unwrap(), &vg).unwrap())
        .unwrap();
    let gpu_vignette_then_crop = ctx
        .download(&k::crop_device(&k::vignette_device(&dev, &vg).unwrap(), &cp).unwrap())
        .unwrap();

    let cpu_crop_then_vignette = vignette(crop(rgb.view(), &cp).unwrap().view(), &vg).unwrap();
    let cpu_vignette_then_crop = crop(vignette(rgb.view(), &vg).unwrap().view(), &cp).unwrap();

    let (h, w, _) = gpu_crop_then_vignette.dim();
    let write = |name: &str, img: &Array3<f32>| {
        let path = format!("examples/output/35_gpu_geometry_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), img.view(), w, h);
    };
    write("crop_then_vignette", &gpu_crop_then_vignette);
    write("vignette_then_crop", &gpu_vignette_then_crop);

    let order_a = cpu_crop_then_vignette == gpu_crop_then_vignette;
    let order_b = cpu_vignette_then_crop == gpu_vignette_then_crop;
    all_exact = all_exact && order_a && order_b;

    println!("order property, both compositions device-resident:");
    println!("  crop then vignette   CPU == GPU bit-for-bit: {order_a}");
    println!("  vignette then crop   CPU == GPU bit-for-bit: {order_b}");
    // The two orders must also differ from each other, or the images
    // above would agree with the CPU while demonstrating nothing.
    println!(
        "  the two orders differ from each other on the device: {}",
        gpu_crop_then_vignette != gpu_vignette_then_crop
    );
    println!("all ten frames bit-exact: {all_exact}");

    println!(
        "wrote examples/output/35_gpu_geometry_{{orient_1..8,crop_then_vignette,vignette_then_crop}}.ppm"
    );
    println!(
        "diff against the CPU twin, examples/output/13_geo_*.ppm \
         (`cargo run --example 13_geometry`): byte-identical, every kernel here being bit-exact"
    );
}
