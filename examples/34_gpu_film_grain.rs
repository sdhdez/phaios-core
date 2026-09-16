// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 34 — Procedural film grain, on the CUDA backend.
//!
//! The GPU twin of `examples/12_film_grain.rs`: the same chart, the same
//! five configurations — reference, fine, coarse, heavy, and heavy with
//! a different seed — computed on the device. One [`Context::upload`],
//! `luminance_bw_device` then `film_grain_device` with the monochrome
//! intermediate staying on the card, one download per rendered frame.
//!
//! Grain is the kernel where a GPU port is most easily wrong in a way
//! that still *looks* like grain, so the verdict here is split into
//! parts rather than printed as one number:
//!
//! - **The integer hash, which gets no tolerance at all.** Each pixel's
//!   noise is `splitmix64(seed ^ splitmix64(mix(x, y)))` — no sequential
//!   generator, no per-tile state, so no pixel depends on another's and
//!   the field is the same at any thread count or launch geometry. It is
//!   *integer* arithmetic, so `docs/ffi.md` §6 requires the device to
//!   reproduce it bit-for-bit, and the check below compares 2²⁰
//!   coordinates against the CPU's `pixel_hash` with `==`. A single
//!   differing bit here would not be a slightly different image; it
//!   would be a different grain field entirely.
//! - **The Box–Muller half, which is bounded.** Turning those bits into
//!   a normal deviate costs a `logf`, a `sqrtf` and a `cosf`, and
//!   IEEE-754 standardises none of the three. §6 commits that half to
//!   (rtol 1e-3, atol 1e-5) — the loosest bound in the crate, because a
//!   transcendental evaluated near `ln(0)` is where two libms diverge
//!   most — and the *bound* column expresses the worst element as a
//!   multiple of it: 1.0000 is the limit.
//! - **Zero intensity, which is an identity.** Identities are exact on
//!   both backends or the fast path is wrong, so that row is `==` again.
//! - **The 8-bit column, which is the question a photographer asks.**
//!   Bounded is not the same as visible: the last column counts the
//!   codes that differ after `encode_srgb`, and it says whether a byte
//!   diff against example 12's files should come out clean.
//!
//! Also worth looking at:
//!
//! - **The determinism line.** The same parameters rendered twice on the
//!   device must give the same bytes. This is the property the whole
//!   position-keyed scheme exists to buy (`docs/ffi.md` §6), and it is
//!   demonstrated rather than asserted in prose.
//! - **The envelope table.** Grain is scaled by `4·t·(1−t)`, so it
//!   vanishes at both ends of the scale and peaks at mid-grey — which is
//!   why it reads as film rather than as sensor noise. The two σ columns
//!   are the CPU's and the device's on flat patches; they should track
//!   each other to the printed precision at every level.
//!
//! What the five configurations *mean* — why images 2 and 3 carry the
//! same amount of grain at different coarseness, and what the analytic
//! normalisation is for — is example 12's subject and is not repeated
//! here.
//!
//! Reference for the hash: Steele, Lea and Flood, *Fast splittable
//! pseudorandom number generators*, OOPSLA (2014); the constants are
//! Vigna's public-domain `splitmix64.c`.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), by the same host-side writer example 12
//! uses, so the two sets of files are comparable code for code.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, kernels as k};
use phaios_core::film_grain::{GrainParams, pixel_hash};

/// The committed bound for `film_grain`'s Box–Muller half
/// (`docs/ffi.md` §6).
const RTOL: f32 = 1e-3;
/// Absolute term of the same bound.
const ATOL: f32 = 1e-5;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: at or below 1.0 means every element passes.
///
/// Non-finite values are compared explicitly rather than arithmetically,
/// and the reduction uses a plain `>` rather than `f32::max`, for the
/// reason recorded in `tests/cuda_conformance.rs`: `f32::max` returns
/// the *other* operand when one side is NaN, so a fold over it scored an
/// all-NaN device output as a perfect match.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    if a.dim() != b.dim() {
        return f32::INFINITY;
    }
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

/// Largest absolute difference, element for element. Reduced with `>`
/// so a NaN cannot read as agreement, for the reason above.
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

/// The 8-bit code `shared::write_ppm` would emit for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Standard deviation of an image, in f64 so the accumulation order
/// cannot move the printed digits (`docs/ffi.md` §6, ordered reductions).
fn sigma(img: &Array3<f32>) -> f64 {
    let n = img.len() as f64;
    let mean = img.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let var = img
        .iter()
        .map(|&v| (f64::from(v) - mean) * (f64::from(v) - mean))
        .sum::<f64>()
        / n;
    var.sqrt()
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

    // One upload; the B&W conversion runs on the card, so the grain
    // kernel's input is produced where it is consumed.
    let dev_rgb = ctx.upload(rgb.view()).unwrap();
    let dev_bw = k::luminance_bw_device(&dev_rgb, LuminanceStandard::Bt709).unwrap();

    // The CPU side of the comparison. The CPU implementation is the
    // specification (`docs/ffi.md` §6), never the other way round.
    let bw = phaios_core::bw::luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, GrainParams); 5] = [
        ("reference", GrainParams::new(0.0, 1.0, 20_260_815)),
        ("fine", GrainParams::new(0.15, 1.0, 20_260_815)),
        ("coarse", GrainParams::new(0.15, 4.0, 20_260_815)),
        ("heavy", GrainParams::new(0.4, 2.0, 20_260_815)),
        ("heavy_other_seed", GrainParams::new(0.4, 2.0, 1_234_567)),
    ];

    println!("film_grain against the CPU kernel on the same chart.");
    println!("docs/ffi.md §6: the identity path is required exact; everything else carries");
    println!("Box-Muller's logf/sqrtf/cosf and is bounded at (rtol 1e-3, atol 1e-5), shown");
    println!("as a multiple of that bound.");
    println!(
        "  {:<18} {:>12} {:>9} {:>9} {:>17}",
        "config", "max |delta|", "bound", "class", "8-bit differing"
    );

    let mut worst_overall = 0.0_f32;
    let mut identity_exact = true;
    for (name, params) in configs {
        let gpu = ctx
            .download(&k::film_grain_device(&dev_bw, &params).unwrap())
            .unwrap();
        let cpu = phaios_core::film_grain::film_grain(bw.view(), &params).unwrap();

        let abs = max_abs_diff(&cpu, &gpu);
        // Zero intensity is the identity fast path on both backends, so
        // it is checked with `==` and reported as such.
        let (class, viol) = if params.intensity == 0.0 {
            identity_exact = cpu == gpu;
            ("exact", if identity_exact { 0.0 } else { f32::INFINITY })
        } else {
            let v = worst_violation(&cpu, &gpu);
            if v > worst_overall {
                worst_overall = v;
            }
            ("bounded", v)
        };

        // The codes are compared after the same encode the writer
        // applies, so this column is the number that reaches a file
        // rather than a number about the arithmetic.
        let cpu_enc = phaios_core::encode::encode_srgb(cpu.view()).unwrap();
        let gpu_enc = phaios_core::encode::encode_srgb(gpu.view()).unwrap();
        let codes = cpu_enc
            .iter()
            .zip(gpu_enc.iter())
            .filter(|(x, y)| code(**x) != code(**y))
            .count();

        let path = format!("examples/output/34_gpu_film_grain_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
        println!(
            "  {name:<18} {abs:>12.3e} {viol:>9.4} {class:>9} {:>17}",
            format!("{codes}/{}", cpu.len())
        );
    }
    println!(
        "identity (intensity 0) bit-exact: {identity_exact}; worst bounded case \
         {worst_overall:.4} of the committed bound (pass: {})",
        worst_overall <= 1.0
    );

    // ── The integer hash, over 2^20 coordinates ──────────────────────
    //
    // `hash_grid` is the conformance oracle's view of the device-side
    // `pixel_hash` — not part of the image pipeline, and the only way to
    // see the integer half on its own. The grid is 1024 x 1024 to match
    // the coordinate count `tests/cuda_conformance.rs` asserts over.
    let (hh, hw) = (1024_usize, 1024_usize);
    let mut hash_exact = true;
    let mut first_divergence = None;
    for seed in [20_260_815_u64, 0, u64::MAX] {
        let gpu = k::hash_grid(&ctx, seed, hh, hw).unwrap();
        for y in 0..hh {
            for x in 0..hw {
                if gpu[y * hw + x] != pixel_hash(seed, x as u64, y as u64) {
                    hash_exact = false;
                    first_divergence.get_or_insert((seed, x, y));
                }
            }
        }
    }
    println!(
        "splitmix64 pixel hash, 3 seeds x 2^20 coordinates, CPU == GPU bit-for-bit: {hash_exact}"
    );
    if let Some((seed, x, y)) = first_divergence {
        println!("  first divergence at seed {seed}, ({x}, {y})");
    }

    // ── Determinism on the device ────────────────────────────────────
    let params = GrainParams::new(0.4, 2.0, 20_260_815);
    let first = ctx
        .download(&k::film_grain_device(&dev_bw, &params).unwrap())
        .unwrap();
    let second = ctx
        .download(&k::film_grain_device(&dev_bw, &params).unwrap())
        .unwrap();
    println!(
        "same seed renders identically on the device: {}",
        first == second
    );

    // ── The envelope, measured rather than described ─────────────────
    println!("grain spread by luminance (intensity 0.4, size 2):");
    println!("  {:<8} {:>10} {:>10}", "L", "sigma CPU", "sigma GPU");
    for level in [0.0_f32, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
        let patch = Array3::from_elem((64, 64, 1), level);
        let dev_patch = ctx.upload(patch.view()).unwrap();
        let gpu = ctx
            .download(&k::film_grain_device(&dev_patch, &params).unwrap())
            .unwrap();
        let cpu = phaios_core::film_grain::film_grain(patch.view(), &params).unwrap();
        let (sc, sg) = (sigma(&cpu), sigma(&gpu));
        let bar = "#".repeat((sg * 200.0).round() as usize);
        println!("  {level:<8} {sc:>10.4} {sg:>10.4}  {bar}");
    }

    println!("wrote examples/output/34_gpu_film_grain_*.ppm");
    println!(
        "diff against the CPU twin, examples/output/12_grain_*.ppm \
         (`cargo run --example 12_film_grain`); the 8-bit column above says what to expect"
    );
}
