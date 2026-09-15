// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 48 — Guided-filter noise reduction, on the CUDA backend.
//!
//! The GPU twin of `47_denoise.rs`. Same noisy chart, same synthetic
//! RGB probe, run on the device so the two can be put side by side.
//! Read example 47 first: it makes the case that cross-guided denoise
//! quiets flat-patch noise while keeping real patch boundaries, and that
//! a channel's smoothing rate is driven by its covariance with the
//! shared luminance guide rather than its own variance. This example
//! does not repeat that argument; it asks whether the card reproduces
//! it, and how closely.
//!
//! `denoise` is **not** bit-exact (`docs/ffi.md` §6): it inherits
//! `local_contrast`'s own GUIDED_FILTER bound (rtol 1e-4, atol 1e-6) —
//! at `C == 1` because it is, bit for bit, the same device kernel
//! ([`local_contrast_device`] with `strength = -amount`); at `C == 3`
//! because the three new cross-guided kernels add one new cancelling
//! subtraction, `cov(I, p_c)`, that the self-guided path never
//! exercises. Each row below prints the worst element as a multiple of
//! that bound; 1.000 is the limit.
//!
//! Everything runs device-resident: the chart is uploaded once and
//! `denoise_device` runs on it directly, no intermediate crossing the
//! bus.
//!
//! Diff against the CPU twin: `examples/output/47_<name>.ppm` — the
//! same three names (`none`, `noisy`, `denoised`).
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, as example 15 does.
//!
//! [`local_contrast_device`]: phaios_core::cuda::kernels::local_contrast_device

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::denoise::{DenoiseParams, denoise};
use phaios_core::film_grain::splitmix64;

/// The committed cross-backend bound for `denoise` (docs/ffi.md §6):
/// `local_contrast`'s own GUIDED_FILTER class.
const RTOL: f32 = 1e-4;
const ATOL: f32 = 1e-6;

const RADIUS: u32 = 4;
const NOISE_SIGMA: f32 = 0.04;

/// Worst violation of `|x - y| <= atol + rtol*|x|`, as a multiple of the
/// bound. Mirrors `tests/cuda_conformance.rs`'s own helper, including its
/// NaN/infinity handling.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim(), "shape mismatch");
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
    let none = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let write = |name: &str, img: ndarray::ArrayView3<f32>| {
        let path = format!("examples/output/48_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
    };
    write("none", none.view());

    // Identical seed and formula to examples/47_denoise.rs.
    let mut state = 0xD1B5_4A32_D192_ED03_u64;
    let noisy = Array3::from_shape_fn(none.raw_dim(), |(_, _, _)| {
        state = splitmix64(state);
        let bits24 = (state >> 40) as i32;
        (bits24 - 0x0080_0000) as f32 / 8_388_608.0 * 0.05
    }) + &none;
    write("noisy", noisy.view());

    let d_noisy = ctx.upload(noisy.view()).unwrap();
    let params = DenoiseParams::new(RADIUS, NOISE_SIGMA, 1.0, LuminanceStandard::Bt709);
    let cpu_denoised = denoise(noisy.view(), &params).unwrap();
    let d_denoised = k::denoise_device(&d_noisy, &params).unwrap();
    let gpu_denoised = ctx.download(&d_denoised).unwrap();
    write("denoised", gpu_denoised.view());

    println!("three renderings, device-resident -- denoise (C=3, cross-guided) on the card:");
    println!("{:>12} {:>10}", "rendering", "x bound");
    println!(
        "{:>12} {:>10.4}",
        "denoised",
        worst_violation(&cpu_denoised, &gpu_denoised)
    );
    println!(
        "  the bound is the committed (rtol {RTOL:e}, atol {ATOL:e}) from docs/ffi.md\n  \
         section 6, expressed as a multiple: 1.0000 is the limit. denoise's C=3 path\n  \
         inherits local_contrast's own class, with one new cancelling subtraction,\n  \
         cov(I, p_c), the self-guided path never exercises."
    );

    // ── The same synthetic RGB probe as example 47, on the device ─────
    println!(
        "\nsynthetic RGB probe, on the device: does blue's own noise care where the guide's edge is?"
    );
    const PROBE_RADIUS: u32 = 4;
    let (ph, pw) = (24_usize, 48_usize);
    let mid = pw / 2;
    let probe = Array3::from_shape_fn((ph, pw, 3), |(y, x, c)| match c {
        0 | 1 => {
            if x < mid {
                0.2_f32
            } else {
                0.8_f32
            }
        }
        _ => {
            0.5 + if (x + y) % 2 == 0 {
                0.05_f32
            } else {
                -0.05_f32
            }
        }
    });
    let probe_params = DenoiseParams::new(PROBE_RADIUS, 0.1, 1.0, LuminanceStandard::Bt709);
    let cpu_probe = denoise(probe.view(), &probe_params).unwrap();
    let d_probe = ctx.upload(probe.view()).unwrap();
    let gpu_probe = ctx
        .download(&k::denoise_device(&d_probe, &probe_params).unwrap())
        .unwrap();
    println!(
        "  probe agreement: {:.4}x the bound",
        worst_violation(&cpu_probe, &gpu_probe)
    );
    let reach = 2 * PROBE_RADIUS as usize;
    let checker_p2p = |img: &Array3<f32>, col: usize| -> f32 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for y in reach..(ph - reach) {
            let v = img[[y, col, 2]];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi - lo
    };
    println!(
        "  blue's checker amplitude (gpu), near the edge: {:.4}, far from it: {:.4}",
        checker_p2p(&gpu_probe, mid),
        checker_p2p(&gpu_probe, reach + 1)
    );

    println!(
        "\ndiff against the CPU twin: examples/output/47_<name>.ppm — same three names.\n  \
         Run `cargo run --example 47_denoise` first, then e.g.\n  \
         `cmp examples/output/47_denoised.ppm examples/output/48_denoised.ppm`."
    );
}
