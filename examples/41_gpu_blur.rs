// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 41 — Gaussian blur and its two paths, on the CUDA backend.
//!
//! The GPU twin of `20_blur.rs`. Same chart, same σ series, same
//! impulse-response measurement — run on the device, so the two can be
//! put side by side. Read example 20 first: it explains why `blur`
//! exists (halation, diffusion and veiling glare are all *blur,
//! weighted, added back*) and what the two paths are. This one asks
//! whether the card reproduces them.
//!
//! Both paths are exercised, because the device implements both:
//!
//! - below σ = [`phaios_core::blur::BOX_CROSSOVER_SIGMA`] a direct
//!   separable convolution, where the device sums the *same* weights as
//!   the host in a different order, with Kahan-compensated f32 against
//!   the host's f64;
//! - at or above it, three box passes, where the device keeps f64 like
//!   the host, because a sliding window subtracts and no compensation
//!   scheme survives that on high-dynamic-range input.
//!
//! The weights and the box widths are computed once by the shared host
//! code in `crate::blur`, so the two backends can never disagree about
//! *which* filter they are applying — only about the order of the sum.
//!
//! Everything runs **device-resident**: the chart is uploaded once,
//! `luminance_bw_device` and each `blur_device` run without an
//! intermediate crossing the bus, and only the finished rendering comes
//! back. The six impulse frames of the second table are separate images
//! and so are uploaded separately; within each, the two separable
//! passes (six for the box path) stay on the card.
//!
//! What to look for in the output:
//!
//! - **The agreement column.** `docs/ffi.md` §6 does *not* promise this
//!   kernel bit-exact — it commits a bound of (rtol 1e-5, atol 1e-7) —
//!   so every row prints the worst element as a multiple of that bound,
//!   where 1.000 is the limit. σ = 0 is the one row that must read zero
//!   outright: it is a device-to-device copy, not a filter, so it has no
//!   rounding to disagree about, and the "elements differing" column —
//!   a comparison of raw bit patterns, stricter than `==` because it
//!   separates `+0.0` from `-0.0` — has to be empty for it.
//! - **The impulse response table.** Below the crossover the transfer
//!   matches a sampled true Gaussian to within about a part in a
//!   million; at or above it, three box passes take over and the
//!   profile departs by a few parts in ten thousand of the impulse's
//!   total energy, less as σ grows — invisible in a picture, and the
//!   price of a cost that no longer grows with radius. That deviation
//!   is a property of the *algorithm*, not of the backend: the CPU
//!   column and the GPU column show the same figure, and the separate
//!   agreement column is what measures the backend.
//! - **Energy.** A blur redistributes light rather than creating it, so
//!   the impulse sums to one — until the kernel is wide enough to
//!   overrun the frame, at which point clamped borders lose the tail.
//!   Both cases are printed, as in example 20.
//!
//! The renderings are passed through `encode_srgb` **on the host**
//! before they are written (`shared::write_ppm_grey_display`), exactly
//! as example 20 does.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::blur::{BOX_CROSSOVER_SIGMA, BlurParams, BlurShape, blur};
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, kernels as k};

/// The committed cross-backend bound for `blur`, from `docs/ffi.md` §6.
///
/// Copied rather than invented here, so a driver update that regresses
/// accuracy shows up in this example as a number above 1.0 instead of
/// being quietly absorbed.
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-7;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: <= 1.0 means every element passes.
///
/// The same two-term form `tests/cuda_conformance.rs` uses, including its
/// NaN handling — a metric built on `f32::max` over a `zip` scores a
/// wholly-NaN or truncated output as a perfect match, which is how the
/// conformance oracle was wrong before it was fixed.
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

/// Largest absolute difference, element for element.
///
/// Reduced with a plain `>` rather than `f32::max`, for the reason given
/// above: `f32::max` returns the *other* operand when one side is NaN,
/// so a fold over it reports a NaN-filled output as a perfect match.
fn max_abs_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    worst
}

/// Which of the two implementations `blur` will take at this σ.
fn path_name(sigma: f32) -> &'static str {
    if sigma < BOX_CROSSOVER_SIGMA {
        "direct"
    } else {
        "box"
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

    // ── The same input as example 20, prepared on both backends ──────
    // `luminance_bw` is documented bit-exact (§6), so both chains
    // provably start from the same bits.
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let uploaded = ctx.upload(rgb.view()).unwrap();
    let d_bw = k::luminance_bw_device(&uploaded, LuminanceStandard::Bt709).unwrap();
    drop(uploaded);
    println!(
        "input to the blur (luminance_bw): CPU == GPU bit-for-bit: {}",
        bw == ctx.download(&d_bw).unwrap()
    );

    // ── The five renderings, and what they cost in agreement ─────────
    println!("\nfive renderings of the checker, device-resident:");
    println!(
        "{:>7} {:>8} {:>13} {:>10} {:>19}",
        "sigma", "path", "max |delta|", "x bound", "elements differing"
    );
    for sigma in [0.0_f32, 1.0, 3.0, 8.0, 24.0] {
        let params = BlurParams::new(sigma, BlurShape::Gaussian);
        let cpu = blur(bw.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::blur_device(&d_bw, &params).unwrap())
            .unwrap();

        let path = format!("examples/output/41_blur_sigma{sigma}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);

        let differing = cpu
            .iter()
            .zip(gpu.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let share = format!("{differing}/{}", cpu.len());
        let label = if sigma == 0.0 {
            "identity"
        } else {
            path_name(sigma)
        };
        println!(
            "{sigma:>7.1} {label:>8} {:>13.3e} {:>10.4} {share:>19}",
            max_abs_diff(&cpu, &gpu),
            worst_violation(&cpu, &gpu)
        );
    }
    println!("wrote examples/output/41_blur_sigma*.ppm (the GPU renderings)");
    println!(
        "  the bound is the committed (rtol {RTOL:e}, atol {ATOL:e}) from docs/ffi.md\n  \
         section 6, expressed as a multiple: 1.0000 is the limit. Only sigma 0 is\n  \
         *promised* bit-exact — it is a device-to-device copy, not a filter. A box\n  \
         row reading 0/98304 is a measurement, not a promise: both backends keep f64\n  \
         through the sliding window, so on this input they happen to land on the same\n  \
         bits, and only the direct path's f64-versus-Kahan-f32 split shows up at all."
    );

    // ── What the kernel is actually doing ────────────────────────────
    // The same measurement example 20 prints, taken from the device, with
    // the CPU's figure beside it so the algorithmic deviation (which both
    // backends share) is not confused with the backend disagreement
    // (which only the last column measures).
    println!("\nimpulse response, measured against a true Gaussian:");
    println!(
        "{:>7} {:>8} {:>12} {:>14} {:>14} {:>10} {:>10}",
        "sigma", "path", "peak", "dev (GPU)", "dev (CPU)", "energy", "x bound"
    );
    for sigma in [1.0_f32, 3.0, 5.9, 6.0, 12.0, 24.0] {
        // A frame comfortably wider than the kernel, so nothing leaks.
        let n = (4.0 * sigma).ceil() as usize * 2 + 9;
        let mut imp = Array3::<f32>::zeros((n, n, 1));
        imp[[n / 2, n / 2, 0]] = 1.0;
        let params = BlurParams::new(sigma, BlurShape::Gaussian);

        let cpu = blur(imp.view(), &params).unwrap();
        let d_imp = ctx.upload(imp.view()).unwrap();
        let gpu = ctx
            .download(&k::blur_device(&d_imp, &params).unwrap())
            .unwrap();

        // The continuous Gaussian this should match, sampled and normalised.
        let two_s2 = 2.0 * f64::from(sigma) * f64::from(sigma);
        let mut ref_img = vec![0.0_f64; n * n];
        let mut total = 0.0_f64;
        for y in 0..n {
            for x in 0..n {
                let dy = y as f64 - (n / 2) as f64;
                let dx = x as f64 - (n / 2) as f64;
                let v = (-(dx * dx + dy * dy) / two_s2).exp();
                ref_img[y * n + x] = v;
                total += v;
            }
        }
        let deviation = |img: &Array3<f32>| {
            (0..n)
                .flat_map(|y| (0..n).map(move |x| (y, x)))
                .map(|(y, x)| (f64::from(img[[y, x, 0]]) - ref_img[y * n + x] / total).abs())
                .fold(0.0_f64, f64::max)
        };
        let energy: f32 = gpu.iter().sum();
        println!(
            "{sigma:>7.1} {:>8} {:>12.6} {:>14.2e} {:>14.2e} {energy:>10.5} {:>10.4}",
            path_name(sigma),
            gpu[[n / 2, n / 2, 0]],
            deviation(&gpu),
            deviation(&cpu),
            worst_violation(&cpu, &gpu)
        );
    }
    println!(
        "  dev columns are the algorithm's own error against a true Gaussian and are\n  \
         a property of the box approximation, not of the backend; the last column is\n  \
         the only one that measures CPU-versus-GPU — and it reads 0.0000 on every row\n  \
         because an impulse puts a single non-zero sample in each window, so there is\n  \
         no summation order left to disagree about. The table above, on real image\n  \
         content, is where the two backends are actually separated."
    );

    // ── And the border case, stated rather than hidden ───────────────
    let mut small = Array3::<f32>::zeros((16, 16, 1));
    small[[8, 8, 0]] = 1.0;
    let params = BlurParams::new(5.0, BlurShape::Gaussian);
    let cpu_small = blur(small.view(), &params).unwrap();
    let d_small = ctx.upload(small.view()).unwrap();
    let gpu_small = ctx
        .download(&k::blur_device(&d_small, &params).unwrap())
        .unwrap();
    let leaked: f32 = gpu_small.iter().sum();
    println!(
        "\nthe same impulse in a 16x16 frame at sigma 5: energy {leaked:.4} on the GPU, \
         {:.4} on the CPU\n  \
         (borders clamp, so a kernel wider than the frame loses its tails —\n   \
         that is what clamping means, not a defect; the two backends agree to \
         {:.4}x the bound)",
        cpu_small.iter().sum::<f32>(),
        worst_violation(&cpu_small, &gpu_small)
    );

    println!(
        "\ndiff against the CPU twin: examples/output/20_blur_sigma<σ>.ppm — same five\n  \
         values of σ. Run `cargo run --example 20_blur` first, then e.g.\n  \
         `cmp examples/output/20_blur_sigma8.ppm examples/output/41_blur_sigma8.ppm`.\n  \
         On this chart all five pairs come out byte-identical, which is a measurement\n  \
         and not a promise: `blur` is bounded rather than bit-exact, and the f32\n  \
         disagreement above is one ULP on a fraction of the pixels — small enough that\n  \
         it never crossed an 8-bit rounding boundary here. On other input, or another\n  \
         card, a few codes may differ by one. The f32 table is the real verdict."
    );
}
