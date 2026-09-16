// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 37 — Highlight roll-off: the clip-versus-shoulder decision,
//! on the CUDA backend.
//!
//! The GPU twin of `examples/16_highlight_rolloff.rs`. Same chart, same
//! +2 EV push, the same five settings, the same two printed tables —
//! computed by `highlight_rolloff_device` instead of the CPU kernel, so
//! the ten output files can be compared five pairs at a time.
//!
//! The whole run to the left of the shoulder is device-resident too:
//! one `Context::upload`, then `luminance_bw_device` and
//! `exposure_device`, and only the five finished frames come back.
//! Both of those upstream kernels are themselves bit-exact (one dot
//! product, one multiply), which is what lets the rows below be read as
//! a statement about `highlight_rolloff` alone rather than about the
//! chain that fed it — the first printed line checks exactly that.
//!
//! `docs/ffi.md` §6 promises `highlight_rolloff` is **bit-exact**
//! across backends: the shoulder is a quadratic solve, and IEEE-754-2008
//! §5.4.1 requires `sqrt` to be correctly rounded just as it does the
//! four arithmetic operations, so a curve built from those five alone
//! carries across unchanged. The comparison is therefore `==`, with no
//! tolerance, and a row that is not `true` means the GPU kernel is
//! wrong — the CPU implementation is the specification.
//!
//! The five settings, as in example 16:
//!
//! 1. `hard_clip` — the default `(1.0, 1.0)`, identical to `np.clip`.
//! 2. `white_2` — knee 0.7, white point 2.0. One stop recovered.
//! 3. `white_4` — knee 0.7, white point 4.0. Two stops recovered.
//! 4. `white_8` — knee 0.7, white point 8.0. Three stops, visibly flatter
//!    near white: recovering range costs highlight contrast, always.
//! 5. `low_knee` — knee 0.3, white point 4.0. The shoulder starts in the
//!    midtones, which is usually too much.
//!
//! What to look for:
//! - The `CPU == GPU` column: five `true`s. That is the point of the
//!   example for anyone beta-testing the backend on their own card.
//! - In image 1 the top row of light patches is one flat white area with
//!   a hard border. In images 2–4 that border is gone and the patches
//!   separate again. That difference is the entire point of the kernel.
//! - The survival table: one distinct output code from eight distinct
//!   highlights under the hard clip, and four, six or eight under a
//!   shoulder according to how much range it reaches for. The counts
//!   are printed for both backends side by side, and they match —
//!   counting distinct *bit patterns* is the strictest reading of
//!   "agree" there is, since two results one ULP apart count as two.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred. That
//! encode runs on the host exactly as in example 16: the files are meant
//! to be byte-comparable, so nothing in the writing path may differ.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::{Array3, ArrayView3};
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, kernels as k};
use phaios_core::exposure::exposure;
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};

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

/// How many distinct bit patterns survive in an image.
///
/// Distinct *bits*, not distinct values: two outputs that differ by one
/// ULP are two codes here, which is the strict reading the survival
/// table wants.
fn distinct_bits(img: &Array3<f32>) -> usize {
    let mut bits: Vec<u32> = img.iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
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

    // CPU reference, exactly as example 16 builds it: B&W, then +2 EV so
    // the light patches genuinely exceed 1.0 and there is something for
    // the shoulder to recover.
    let cpu_bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let cpu_pushed = exposure(cpu_bw.view(), 2.0).unwrap();

    // The same two stages on the device: one upload, nothing downloaded
    // between them.
    let device = ctx.upload(rgb.view()).unwrap();
    let gpu_bw = k::luminance_bw_device(&device, LuminanceStandard::Bt709).unwrap();
    let gpu_pushed = k::exposure_device(&gpu_bw, 2.0).unwrap();

    // Check the shoulder's input before checking the shoulder, so the
    // rows below are about one kernel and not about its feed.
    let pushed_host = ctx.download(&gpu_pushed).unwrap();
    println!(
        "input to the shoulder (luminance_bw -> +2 EV, both bit-exact): CPU == GPU: {}",
        cpu_pushed == pushed_host
    );

    let configs: [(&str, RolloffParams); 5] = [
        ("hard_clip", RolloffParams::default()),
        ("white_2", RolloffParams::new(0.7, 2.0)),
        ("white_4", RolloffParams::new(0.7, 4.0)),
        ("white_8", RolloffParams::new(0.7, 8.0)),
        ("low_knee", RolloffParams::new(0.3, 4.0)),
    ];

    println!("\ndevice-resident highlight_rolloff vs the CPU specification:");
    println!(
        "  {:<10} {:>6} {:>7} {:>12} {:>13}",
        "setting", "knee", "white", "CPU == GPU", "max |delta|"
    );
    let mut all_exact = true;
    for (name, params) in configs.iter() {
        let gpu = ctx
            .download(&k::highlight_rolloff_device(&gpu_pushed, params).unwrap())
            .unwrap();
        let cpu = highlight_rolloff(cpu_pushed.view(), params).unwrap();
        let path = format!("examples/output/37_rolloff_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
        let exact = cpu == gpu;
        all_exact &= exact;
        println!(
            "  {name:<10} {:>6.2} {:>7.1} {exact:>12} {:>13.3e}",
            params.knee,
            params.white_point,
            max_abs_diff(cpu.view(), gpu.view())
        );
    }
    println!("wrote examples/output/37_rolloff_*.ppm");

    // How much highlight detail each setting actually keeps. Eight scene
    // values from nominal white to three stops over; count how many
    // remain distinguishable after the stage — on the device, and on the
    // CPU beside it, because the two counts agreeing is a stronger claim
    // than either count alone.
    let probes: Vec<f32> = (0..8).map(|i| 2.0_f32.powf(i as f32 * 3.0 / 7.0)).collect();
    let probe_img = Array3::from_shape_vec((1, probes.len(), 1), probes).unwrap();
    let probe_dev = ctx.upload(probe_img.view()).unwrap();
    println!("\ndistinct output codes from 8 distinct highlights (1.0 .. 8.0):");
    for (name, params) in configs.iter() {
        let gpu = ctx
            .download(&k::highlight_rolloff_device(&probe_dev, params).unwrap())
            .unwrap();
        let cpu = highlight_rolloff(probe_img.view(), params).unwrap();
        all_exact &= cpu == gpu;
        println!(
            "  {name:>10}: {} of 8 survive on the GPU, {} on the CPU  (knee {:.2}, white {:.1})",
            distinct_bits(&gpu),
            distinct_bits(&cpu),
            params.knee,
            params.white_point
        );
    }

    // The curve itself, at the zone anchors, for the two-stop setting.
    // Thirteen anchors in one (1, 13, 1) image rather than thirteen
    // one-pixel launches: the numbers are the same, and a kernel that
    // indexes by absolute element could not tell the difference anyway.
    let params = RolloffParams::new(0.7, 4.0);
    let anchors = Array3::from_shape_fn((1, 13, 1), |(_, z, _)| {
        0.18_f32 * 2.0_f32.powi(z as i32 - 5)
    });
    let anchor_dev = ctx.upload(anchors.view()).unwrap();
    let gpu_anchors = ctx
        .download(&k::highlight_rolloff_device(&anchor_dev, &params).unwrap())
        .unwrap();
    let cpu_anchors = highlight_rolloff(anchors.view(), &params).unwrap();
    all_exact &= cpu_anchors == gpu_anchors;

    println!("\nknee 0.70, white 4.0 — sampled at the zone anchors, computed on the device:");
    for zone in 0..=12_usize {
        let linear = anchors[[0, zone, 0]];
        let out = gpu_anchors[[0, zone, 0]];
        let mark = if linear <= params.knee {
            ""
        } else {
            "  <- shoulder"
        };
        println!("  zone {zone:>2}: {linear:>9.4} -> {out:>8.5}{mark}");
    }
    println!(
        "  all 13 anchors equal to the CPU bit-for-bit: {}",
        cpu_anchors == gpu_anchors
    );

    println!("\nevery comparison above bit-exact (docs/ffi.md §6 promises this): {all_exact}");
    println!(
        "diff against example 16: for f in hard_clip white_2 white_4 white_8 low_knee; do cmp examples/output/37_rolloff_$f.ppm examples/output/16_rolloff_$f.ppm; done"
    );
}
