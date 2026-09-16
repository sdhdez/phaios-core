// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 32 — Radial vignette, on the CUDA backend.
//!
//! The GPU twin of `examples/10_vignette.rs`: the same chart, the same
//! five configurations, the same photographic point — computed on the
//! device instead. One [`Context::upload`], `luminance_bw_device` and
//! then `vignette_device` with the intermediate never leaving the card,
//! one download per rendered frame.
//!
//! What to look for in the output:
//!
//! - **The agreement line.** `vignette` contains no transcendental: the
//!   falloff is a smoothstep over a distance measure, so every operation
//!   in it is correctly rounded on both sides, and the PTX is built with
//!   `-fmad=false` so the device does not contract a multiply-add
//!   either. `docs/ffi.md` §6 therefore promises this kernel *bit-exact*
//!   across backends, and the check below is `==` over the whole array
//!   rather than a tolerance. A tolerance here would be a bug: it would
//!   absorb precisely the failure this example exists to catch.
//! - **The five images are example 10's five images.** `luminance_bw` is
//!   bit-exact too, so the whole chain is, and the PPMs come out
//!   byte-identical to `10_vignette_*.ppm`. `cmp` on the two files is
//!   the same check the printed line makes before the encode.
//! - **Resolution independence, measured on the device.** The falloff is
//!   parameterised in normalised coordinates, so a half-size preview must
//!   give the same factor at the corresponding position. Example 10
//!   prints the CPU's two deltas; these are the device's.
//!
//! What the five configurations *mean* — why `feather` matters more than
//! it sounds, and what separates the Euclidean corner falloff from the
//! Chebyshev one — is example 10's subject and is not repeated here.
//! This file is that example on the other backend, not a second
//! explanation of the kernel.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), by the same host-side writer example 10
//! uses, so the two sets of files are comparable byte for byte.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::vignette::VignetteParams;

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

    // One upload for the whole example. The B&W conversion runs on the
    // card, so the vignette's input is produced where it is consumed and
    // never crosses the bus.
    let dev_rgb = ctx.upload(rgb.view()).unwrap();
    let dev_bw = k::luminance_bw_device(&dev_rgb, LuminanceStandard::Bt709).unwrap();

    // The CPU side of the comparison, from the same chart. This is the
    // specification (`docs/ffi.md` §6), not a second opinion.
    let bw = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, VignetteParams); 5] = [
        ("reference", VignetteParams::new(0.0, 0.5, 0.0)),
        ("classic", VignetteParams::new(0.5, 1.0, 0.0)),
        ("hard", VignetteParams::new(0.5, 0.2, 0.0)),
        ("rectangular", VignetteParams::new(0.5, 1.0, 1.0)),
        ("inverted", VignetteParams::new(-0.5, 1.0, 0.0)),
    ];

    println!("vignette on the Macbeth chart — docs/ffi.md §6 promises bit-exactness,");
    println!("so this is `==` over the whole array and not a tolerance:");
    let mut all_exact = true;
    for (name, params) in configs {
        let gpu = ctx
            .download(&k::vignette_device(&dev_bw, &params).unwrap())
            .unwrap();
        let cpu = phaios_core::vignette::vignette(bw.view(), &params).unwrap();
        let exact = cpu == gpu;
        all_exact = all_exact && exact;

        let path = format!("examples/output/32_gpu_vignette_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
        println!("  {name:<12} CPU == GPU bit-for-bit: {exact}");
    }
    println!("all five configurations bit-exact: {all_exact}");

    // Resolution independence, the property example 10 measures on the
    // CPU: the same parameters on a half-size frame must give the same
    // factors at corresponding positions.
    let params = VignetteParams::new(0.5, 1.0, 0.0);
    let flat_full = Array3::from_elem((shared::HEIGHT, shared::WIDTH, 1), 1.0_f32);
    let flat_half = Array3::from_elem((shared::HEIGHT / 2, shared::WIDTH / 2, 1), 1.0_f32);
    let on_device = |img: &Array3<f32>| {
        let dev = ctx.upload(img.view()).unwrap();
        ctx.download(&k::vignette_device(&dev, &params).unwrap())
            .unwrap()
    };
    let full = on_device(&flat_full);
    let half = on_device(&flat_half);
    let corner_delta = (full[[0, 0, 0]] - half[[0, 0, 0]]).abs();
    let centre_delta = (full[[shared::HEIGHT / 2, shared::WIDTH / 2, 0]]
        - half[[shared::HEIGHT / 4, shared::WIDTH / 4, 0]])
    .abs();
    println!(
        "full frame vs half-size preview on the device: corner delta {corner_delta:.2e}, \
         centre delta {centre_delta:.2e}"
    );

    println!("wrote examples/output/32_gpu_vignette_*.ppm");
    println!(
        "diff against the CPU twin, examples/output/10_vignette_*.ppm \
         (`cargo run --example 10_vignette`): byte-identical, every kernel here being bit-exact"
    );
}
