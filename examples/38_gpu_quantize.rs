// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 38 — Quantisation and dither, on the CUDA backend.
//!
//! The GPU twin of `examples/17_quantize.rs`. Same shallow gradient,
//! same colour checker, same seed, the same four files and the same
//! measurements — produced by `quantize_u8_device` and
//! `quantize_u16_device` instead of the CPU kernels.
//!
//! These are the only device kernels whose output is not an image.
//! Quantisation is terminal: the codes go into a file, not into another
//! kernel, so there is no resident form to chain — the `*_device`
//! entry points take a `DeviceImage` and return a host array, and that
//! return *is* the single download at the end of the pipeline.
//!
//! `docs/ffi.md` §6 promises `quantize_u8` and `quantize_u16` are
//! **bit-exact** across backends. The dither is exact 64-bit integer
//! arithmetic — the same splitmix64 the grain kernel uses, and integer
//! hashing has no rounding to disagree about — and the rounding is
//! `floor(v + 0.5)`, an exact operation composed with a correctly
//! rounded one rather than a library routine host and device could
//! implement differently. So the comparison is `==` on the integer
//! codes, with no tolerance anywhere.
//!
//! One honest complication, and it is worth understanding before
//! reading the table. The gradient is quantised straight from the
//! uploaded buffer, so that path is bit-exact end to end. The checker
//! is not: it runs the real terminal chain — `luminance_bw_device` →
//! `encode_srgb_device` → `quantize_u8_device`, one upload, one
//! download — and `encode_srgb` is **not** in §6's bit-exact list. It
//! evaluates `powf`, which IEEE-754 does not standardise, so it carries
//! a bound (rtol 1e-5, atol 1e-7) rather than a promise. The example
//! therefore reports the checker two ways: the full chain's code
//! disagreement as measured, and — feeding both backends the identical
//! display-referred buffer — the isolated `quantize_u8` verdict, which
//! is the one §6 actually promises.
//!
//! What to look for:
//! - `38_ramp_8bit_plain.ppm` and `38_ramp_8bit_dither.ppm` side by
//!   side are the entire argument for dither. The band in the first is
//!   a hard vertical edge; in the second there is no edge, only fine
//!   noise. `38_checker_8bit_*.ppm` should be almost indistinguishable
//!   from each other: dither is insurance, not a look.
//! - The `CPU == GPU` column on the ramp rows: `true`, on the dithered
//!   row as much as the plain one. A dither that agreed on average but
//!   not per pixel would be worthless for reproducible renders, which
//!   is why the position-keyed hash is integer arithmetic and not a
//!   generator.
//! - The mean shift from dithering, which must stay at zero codes:
//!   triangular dither is zero-mean and must not shift exposure. The
//!   device computes the same shift because it computes the same codes.
//!
//! Nothing here is re-encoded on the way out. The codes *are* the file,
//! so these PPMs are written byte for byte from the kernel's output —
//! the same reason example 17 carries its own writer instead of using
//! the shared `*_display` helpers.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::cuda::{self, kernels as k};
use phaios_core::encode::encode_srgb;
use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};

/// Write 8-bit codes straight out as a grey PPM. Unlike most examples
/// this one must not re-encode: the codes *are* the file.
fn write_codes(path: &Path, codes: &[u8], width: usize, height: usize) {
    let mut buf = format!("P6\n{width} {height}\n255\n").into_bytes();
    for &c in codes {
        buf.extend_from_slice(&[c, c, c]);
    }
    std::fs::write(path, buf).unwrap();
}

fn transitions(codes: &[u8]) -> usize {
    codes.windows(2).filter(|w| w[0] != w[1]).count()
}

fn mean(codes: &[u8]) -> f64 {
    codes.iter().map(|c| f64::from(*c)).sum::<f64>() / codes.len() as f64
}

/// How many codes differ between two equally-shaped code images, and by
/// how much at worst. Integer throughout: there is no tolerance to
/// choose, which is the whole appeal of comparing at this stage.
fn code_diff<T: Copy + PartialEq + Into<u32>>(a: &Array3<T>, b: &Array3<T>) -> (usize, u32) {
    let mut differing = 0_usize;
    let mut worst = 0_u32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            differing += 1;
            worst = worst.max(u32::abs_diff((*x).into(), (*y).into()));
        }
    }
    (differing, worst)
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

    // ── A gradient too shallow for 8 bits ────────────────────────────
    // 256 wide, 64 tall, spanning codes 100..101. Already display-
    // referred: quantisation is what happens *after* encode_srgb.
    let (w, h) = (256_usize, 64_usize);
    let ramp = Array3::<f32>::from_shape_fn((h, w, 1), |(_, x, _)| {
        (100.0 + x as f32 / (w as f32 - 1.0)) / 255.0
    });
    let plain_params = QuantizeParams::default();
    let dither_params = QuantizeParams::new(Dither::Tpdf, 20260825);

    // One upload; three quantisations read it, and each return value is
    // its own (small) download.
    let ramp_dev = ctx.upload(ramp.view()).unwrap();
    let plain = k::quantize_u8_device(&ramp_dev, &plain_params).unwrap();
    let dithered = k::quantize_u8_device(&ramp_dev, &dither_params).unwrap();
    let deep = k::quantize_u16_device(&ramp_dev, &plain_params).unwrap();

    let cpu_plain = quantize_u8(ramp.view(), &plain_params).unwrap();
    let cpu_dithered = quantize_u8(ramp.view(), &dither_params).unwrap();
    let cpu_deep = quantize_u16(ramp.view(), &plain_params).unwrap();

    let plain_codes: Vec<u8> = plain.iter().copied().collect();
    let dither_codes: Vec<u8> = dithered.iter().copied().collect();

    write_codes(
        Path::new("examples/output/38_ramp_8bit_plain.ppm"),
        &plain_codes,
        w,
        h,
    );
    write_codes(
        Path::new("examples/output/38_ramp_8bit_dither.ppm"),
        &dither_codes,
        w,
        h,
    );

    // ── Ordinary content, both ways, through the real terminal chain ─
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    let cpu_bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let cpu_display = encode_srgb(cpu_bw.view()).unwrap();

    let checker_dev = ctx.upload(rgb.view()).unwrap();
    let gpu_display = k::encode_srgb_device(
        &k::luminance_bw_device(&checker_dev, LuminanceStandard::Bt709).unwrap(),
    )
    .unwrap();
    // The same display-referred buffer the CPU quantises, put back on
    // the device: this is what isolates `quantize_u8` from the `powf` in
    // `encode_srgb` upstream of it.
    let shared_display_dev = ctx.upload(cpu_display.view()).unwrap();

    let mut checker_chain = Vec::new();
    let mut checker_isolated = Vec::new();
    for (name, params) in [("plain", &plain_params), ("dither", &dither_params)] {
        let codes = k::quantize_u8_device(&gpu_display, params).unwrap();
        let cpu_codes = quantize_u8(cpu_display.view(), params).unwrap();
        let isolated = k::quantize_u8_device(&shared_display_dev, params).unwrap();
        let flat: Vec<u8> = codes.iter().copied().collect();
        write_codes(
            Path::new(&format!("examples/output/38_checker_8bit_{name}.ppm")),
            &flat,
            shared::WIDTH,
            shared::HEIGHT,
        );
        checker_chain.push((name, code_diff(&cpu_codes, &codes)));
        checker_isolated.push((
            name,
            cpu_codes == isolated,
            code_diff(&cpu_codes, &isolated),
        ));
    }
    println!("wrote examples/output/38_*.ppm");

    // ── Agreement: the point of the example for a beta tester ────────
    println!("\ndevice-resident quantise vs the CPU specification, on integer codes:");
    println!(
        "  {:<32} {:>12} {:>13} {:>13}",
        "output", "CPU == GPU", "codes differ", "worst delta"
    );
    let mut rows: Vec<(String, bool, usize, u32)> = vec![
        {
            let (n, d) = code_diff(&cpu_plain, &plain);
            ("ramp_8bit_plain".to_owned(), cpu_plain == plain, n, d)
        },
        {
            let (n, d) = code_diff(&cpu_dithered, &dithered);
            (
                "ramp_8bit_dither".to_owned(),
                cpu_dithered == dithered,
                n,
                d,
            )
        },
        {
            let (n, d) = code_diff(&cpu_deep, &deep);
            ("ramp_16bit_plain".to_owned(), cpu_deep == deep, n, d)
        },
    ];
    for (name, equal, (n, d)) in &checker_isolated {
        rows.push((format!("checker_8bit_{name} (isolated)"), *equal, *n, *d));
    }
    let mut all_exact = true;
    for (label, equal, differing, worst) in &rows {
        all_exact &= *equal;
        println!("  {label:<32} {equal:>12} {differing:>13} {worst:>13}");
    }
    println!("  every row above is a kernel docs/ffi.md §6 promises bit-exact: {all_exact}");

    // The full chain is a weaker claim, and saying so is the point.
    println!(
        "\nthe written checker files come from the full device chain\n\
         (luminance_bw_device -> encode_srgb_device -> quantize_u8_device), whose\n\
         encode_srgb carries a bound rather than a promise — one powf, which\n\
         IEEE-754 does not standardise. Measured against the CPU chain:"
    );
    for (name, (differing, worst)) in &checker_chain {
        println!(
            "  checker_8bit_{name:<7} {differing} of {} codes differ, worst {worst} code(s)",
            shared::WIDTH * shared::HEIGHT
        );
    }

    // ── The measurements, as example 17 prints them ──────────────────
    let first_row = |c: &[u8]| c[..w].to_vec();
    let distinct = |c: &[u8]| {
        let mut v = c.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    };

    println!(
        "\na gradient spanning barely two 8-bit codes, 256 px wide (quantised on the device):"
    );
    println!(
        "  plain    {} distinct codes, {} transitions along the row, mean {:.3}",
        distinct(&plain_codes),
        transitions(&first_row(&plain_codes)),
        mean(&plain_codes)
    );
    println!(
        "  dithered {} distinct codes, {} transitions along the row, mean {:.3}",
        distinct(&dither_codes),
        transitions(&first_row(&dither_codes)),
        mean(&dither_codes)
    );
    println!(
        "  mean shift from dithering: {:+.4} codes (triangular dither is zero-mean)",
        mean(&dither_codes) - mean(&plain_codes)
    );

    // ── And why 16 bits rarely needs any of this ─────────────────────
    let mut uniq: Vec<u16> = deep.iter().copied().collect();
    uniq.sort_unstable();
    uniq.dedup();
    println!(
        "\nthe same gradient at 16 bits, undithered: {} distinct codes",
        uniq.len()
    );
    println!("  (8 bits cannot resolve it at all; 16 bits resolves it without help)");

    println!(
        "\ndiff against example 17: for f in ramp_8bit_plain ramp_8bit_dither checker_8bit_plain checker_8bit_dither; do cmp examples/output/38_$f.ppm examples/output/17_$f.ppm; done"
    );
}
