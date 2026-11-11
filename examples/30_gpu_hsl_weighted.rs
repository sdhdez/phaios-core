// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 30 — HSL-weighted B&W conversion, on the CUDA backend.
//!
//! The GPU twin of `examples/08_hsl_weighted.rs`: the same chart, the
//! same four configurations, the same photographic point. The pair
//! exists to be run one after the other and diffed.
//!
//! Demonstrates `hsl_bw_device`, which scales each pixel's luminance by
//! a Gaussian blend of eight per-hue-band weights, modulated by how
//! saturated the pixel is. Example 08's four configurations:
//!
//! 1. All weights zero — identical to `luminance_bw` (the reference).
//! 2. Blue −0.8 — the classic darkened sky, without touching anything
//!    else.
//! 3. Yellow +0.8, green +0.4 — foliage separation.
//! 4. Red −0.9 — skin and reds pushed down, a look no Wratten filter
//!    produces because a real filter cannot subtract.
//!
//! What to look for:
//!
//! - **The pictures are example 08's pictures.** The neutral patches
//!   (row 4) are identical in all four images — zero chroma means zero
//!   modulation, whatever the weights say. That is a property of each
//!   backend's own output, checked on the device below; between the
//!   backends the neutrals are bounded like every other pixel, for the
//!   reason given two points down. In image 2 the blue-sky patch
//!   (#03) darkens sharply while the bluish-green patch (#06) barely
//!   moves, about 70° from the blue band centre with a 30° sigma. In
//!   image 4 the red patch (#15) goes nearly black while the orange
//!   patch (#07) only dims: the bands overlap, they do not partition.
//!   That last property is printed from the device output below rather
//!   than left to the eye.
//! - **Why the promise is a bound.** The device kernel mirrors the CPU
//!   one line for line — hexagonal hue, chroma ratio, circular band
//!   distance, fixed-order eight-term Gaussian sum — but `build.rs`
//!   compiles it with fast math. The GPU side's `expf` is the hardware's
//!   approximate `ex2`, its divisions are approximate, and its
//!   multiply-adds are contracted into FMAs; the CPU, which is the
//!   specification, calls the platform libm (glibc on Linux) and rounds
//!   every step separately. `docs/ffi.md` §6 therefore commits
//!   rtol 1e-5 / atol 1e-7 rather than bit-exactness, and the table
//!   prints the worst element as a multiple of that bound.
//! - **The reference row is bounded too, and so are the neutrals.**
//!   With every weight zero each of the eight terms is `0.0 · expf(…)`,
//!   exactly zero on both backends whatever `expf` returns, so the hue
//!   modulation `1 + multiplier · chroma` is exactly 1 and the kernel
//!   reduces to the luminance dot product `wr·r + wg·g + wb·b`.
//!   `luminance_bw` computes that same product and *is* bit-exact,
//!   because it is compiled strict; here fast math contracts it into
//!   FMAs, so the reference row can differ from the CPU by about a
//!   rounding step. A neutral pixel has zero chroma, so whatever the
//!   weights it reduces to the same product and is bounded as well — it
//!   may compare equal on a given input, but that is not a promise.
//! - **The chart is still a degenerate input for `expf`.** 24 flat
//!   patches means 24 distinct hues, so the other three rows evaluate
//!   `expf` at a handful of arguments, and their small multiples say
//!   little about a photograph. Example 29 makes the same point about
//!   `encode_srgb`'s `powf`. A photograph is not 24 colours.
//! - **So the example runs a second, harder input.** The same xorshift
//!   image the conformance suite and `benches/gpu.rs` use — 98 304
//!   pixels, essentially all distinct — goes through the same four
//!   configurations, and that table is where `expf` is exercised across
//!   the hue circle. Its numbers are comparable with example 23's
//!   `hsl_bw` row, which uses the same generator. If the first table
//!   were the whole example, the twin would be measuring the bound on
//!   24 hues and presenting it as the kernel's.
//! - **The 8-bit verdict.** The last column counts how many of the
//!   98 304 codes in the written PPM differ, after the same
//!   `encode_srgb` that `write_ppm_grey_display` applies — the
//!   disagreement that survives into a file.
//!
//! Shape of the run: one upload, four `hsl_bw_device` calls branching
//! from the same device image, one download per image written.
//!
//! Reference: the hue/chroma geometry is the standard hexagonal
//! projection given in Joblove & Greenberg, "Color spaces for computer
//! graphics", *SIGGRAPH '78*, pp. 20–25; `docs/architecture.md` carries
//! the derivation of the band weighting.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{HslWeightedParams, LuminanceStandard};
use phaios_core::cuda::{self, kernels as k};

/// The committed cross-backend bound for the one-transcendental class
/// (`docs/ffi.md` §6), which takes `hsl_bw`: eight `expf` per pixel, on
/// the device compiled with fast math. Copied from the document rather
/// than invented here.
const RTOL: f32 = 1e-5;
const ATOL: f32 = 1e-7;

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: `<= 1.0` means every element passes.
///
/// The same form `tests/cuda_conformance.rs` uses, including its care
/// with non-finite values. A metric built on `f32::max` over a `zip`
/// scores a wholly-NaN or truncated output as a perfect match, because
/// `f32::max` returns the *other* operand when one side is NaN — which
/// is how the conformance oracle was wrong before it was fixed.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32) -> f32 {
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
            (x - y).abs() / (atol + rtol * x.abs())
        };
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Largest absolute difference, element for element. Plain `>` rather
/// than `f32::max`, for the reason given above.
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

/// The 8-bit code `shared::write_ppm` emits for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// How many 8-bit codes differ between the two PPMs that would be
/// written, i.e. after the same CPU `encode_srgb` both sides go through.
fn codes_differing(cpu: &Array3<f32>, gpu: &Array3<f32>) -> usize {
    let a = phaios_core::encode::encode_srgb(cpu.view()).expect("encode_srgb is infallible here");
    let b = phaios_core::encode::encode_srgb(gpu.view()).expect("encode_srgb is infallible here");
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| code(**x) != code(**y))
        .count()
}

/// Deterministic pseudo-random image — the xorshift that
/// `tests/cuda_conformance.rs`, `benches/gpu.rs` and example 23 all use,
/// so a number measured here is comparable with theirs. No RNG
/// dependency, and no seed hidden in a thread-local.
fn pseudo_random_image(h: usize, w: usize, c: usize) -> Array3<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    Array3::from_shape_simple_fn((h, w, c), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0
    })
}

/// Value at the centre of chart patch `index` (1-based, as the patches
/// are numbered in `shared::PATCHES` and in example 08's commentary).
fn patch_centre(img: &Array3<f32>, index: usize) -> f32 {
    let i = index - 1;
    let y = (i / shared::GRID_COLS) * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
    let x = (i % shared::GRID_COLS) * shared::PATCH_SIZE + shared::PATCH_SIZE / 2;
    img[[y, x, 0]]
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

    // One upload; all four configurations branch from this image.
    let uploaded = ctx.upload(rgb.view()).unwrap();

    // Band order: red, orange, yellow, green, aqua, blue, purple, magenta.
    let configs: [(&str, [f32; 8]); 4] = [
        ("reference", [0.0; 8]),
        ("blue_down", [0.0, 0.0, 0.0, 0.0, 0.0, -0.8, 0.0, 0.0]),
        ("foliage", [0.0, 0.0, 0.8, 0.4, 0.0, 0.0, 0.0, 0.0]),
        ("red_down", [-0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];

    println!(
        "{:<11} {:<10} {:>9} {:>12} {:>9} {:>18}",
        "config", "promise", "CPU == GPU", "max |delta|", "bound", "8-bit codes differ"
    );
    // The worst row of both tables, against the one bound they share:
    // the verdict printed at the end.
    let mut worst_row = 0.0_f32;
    let mut cpu_reference: Option<Array3<f32>> = None;
    let mut gpu_outputs: Vec<Array3<f32>> = Vec::with_capacity(configs.len());
    for (name, weights) in configs {
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);

        // The CPU kernel is the specification (`docs/ffi.md` §6); the device
        // output is measured against it, never the other way round.
        let cpu = phaios_core::bw::hsl_bw(rgb.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::hsl_bw_device(&uploaded, &params).unwrap())
            .unwrap();
        let violation = worst_violation(&cpu, &gpu, RTOL, ATOL);
        if violation > worst_row {
            worst_row = violation;
        }

        println!(
            "{name:<11} {:<10} {:>9} {:>12.3e} {:>9.4} {:>18}",
            "1e-5/1e-7",
            cpu == gpu,
            max_abs_diff(&cpu, &gpu),
            violation,
            format!("{}/{}", codes_differing(&cpu, &gpu), cpu.len()),
        );

        let path = format!("examples/output/30_gpu_hsl_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), gpu.view(), shared::WIDTH, shared::HEIGHT);
        cpu_reference.get_or_insert(cpu);
        gpu_outputs.push(gpu);
    }
    println!(
        "  bound is the committed one-transcendental pair (rtol 1e-5, atol 1e-7)\n  \
         from docs/ffi.md §6, as a multiple: 1.0000 is the limit. §6 promises this\n  \
         kernel bounded, not bit-exact, the reference row included: no expf term\n  \
         survives zero weights, but fast math contracts the luminance dot product\n  \
         into FMAs. The chart gives the other three only 24 distinct hues to\n  \
         disagree over; the xorshift table below is the one that exercises expf."
    );

    // The properties example 08 asks the reader to look for, read from
    // the device output rather than left to the eye: the neutrals are
    // checked, the band overlap is printed.
    let reference = &gpu_outputs[0];
    let neutrals_held = gpu_outputs.iter().all(|out| {
        (19..=24).all(|p| patch_centre(out, p).to_bits() == patch_centre(reference, p).to_bits())
    });
    println!(
        "\ndevice output, the properties example 08 is about:\n  \
         neutral patches (#19-#24) bit-identical across the four device configs: {neutrals_held}"
    );

    // The same neutrals against the CPU. Zero chroma removes the hue
    // terms on both sides, but the dot product left over is contracted
    // on the device, so this is a bounded comparison, not an exact one.
    let cpu_reference = cpu_reference.expect("configs is not empty");
    let neutrals =
        |img: &Array3<f32>| Array3::from_shape_fn((1, 6, 1), |(_, i, _)| patch_centre(img, 19 + i));
    let (cpu_neutrals, gpu_neutrals) = (neutrals(&cpu_reference), neutrals(reference));
    let neutrals_differing = cpu_neutrals
        .iter()
        .zip(gpu_neutrals.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    println!(
        "  the same neutrals against the CPU: {neutrals_differing} of 6 differ, {:.4}x the bound \
         (bounded, not promised exact)",
        worst_violation(&cpu_neutrals, &gpu_neutrals, RTOL, ATOL)
    );
    println!(
        "  blue_down:  sky #03 {:.5} -> {:.5}, bluish green #06 {:.5} -> {:.5} (70 deg away)",
        patch_centre(reference, 3),
        patch_centre(&gpu_outputs[1], 3),
        patch_centre(reference, 6),
        patch_centre(&gpu_outputs[1], 6),
    );
    println!(
        "  red_down:   red #15 {:.5} -> {:.5}, orange #07 {:.5} -> {:.5} (bands overlap)",
        patch_centre(reference, 15),
        patch_centre(&gpu_outputs[3], 15),
        patch_centre(reference, 7),
        patch_centre(&gpu_outputs[3], 7),
    );

    // The chart is 24 flat patches, so it puts 24 arguments through
    // `expf` and settles nothing about the other four billion. The same
    // four grades are therefore run again over the xorshift image, where
    // nearly every pixel is a distinct hue. No PPM is written from it:
    // it is a measurement, not a picture.
    let noise = pseudo_random_image(shared::HEIGHT, shared::WIDTH, 3);
    let noise_device = ctx.upload(noise.view()).unwrap();
    println!(
        "\nthe same four grades over the xorshift image ({} pixels, essentially all distinct\n\
         hues) — the input the conformance suite and benches/gpu.rs use:",
        shared::HEIGHT * shared::WIDTH
    );
    println!(
        "{:<11} {:<10} {:>9} {:>12} {:>9} {:>18}",
        "config", "promise", "CPU == GPU", "max |delta|", "bound", "f32 differing"
    );
    for (name, weights) in configs {
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        let cpu = phaios_core::bw::hsl_bw(noise.view(), &params).unwrap();
        let gpu = ctx
            .download(&k::hsl_bw_device(&noise_device, &params).unwrap())
            .unwrap();
        let differing = cpu
            .iter()
            .zip(gpu.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let violation = worst_violation(&cpu, &gpu, RTOL, ATOL);
        if violation > worst_row {
            worst_row = violation;
        }
        println!(
            "{name:<11} {:<10} {:>9} {:>12.3e} {:>9.4} {:>18}",
            "1e-5/1e-7",
            cpu == gpu,
            max_abs_diff(&cpu, &gpu),
            violation,
            format!("{differing}/{}", cpu.len()),
        );
    }
    println!(
        "  verdict: all eight rows within the 1e-5/1e-7 bound: {} (worst {worst_row:.4}x)",
        worst_row <= 1.0
    );

    println!("\nwrote examples/output/30_gpu_hsl_{{reference,blue_down,foliage,red_down}}.ppm");
    println!(
        "diff against the CPU twin: examples/output/08_hsl_{{reference,blue_down,foliage,red_down}}.ppm \
         (cargo run --example 08_hsl_weighted)"
    );
}
