// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 17 — Quantisation and dither: where banding comes from.
//!
//! Demonstrates `quantize_u8` and `quantize_u16`, the terminal stage,
//! and the reason dither exists at all.
//!
//! The test image here is not the colour checker but a **shallow
//! gradient**: 256 pixels covering barely two 8-bit codes. That is the
//! worst case for quantisation and the one every smooth sky runs into.
//!
//! Four files are written:
//!
//! 1. `17_ramp_8bit_plain.ppm` — 8-bit, no dither. One hard step across
//!    the whole frame: a band.
//! 2. `17_ramp_8bit_dither.ppm` — 8-bit, TPDF dither. The step becomes a
//!    stippled transition that carries the gradient.
//! 3. `17_checker_8bit_plain.ppm` and `17_checker_8bit_dither.ppm` — the
//!    Macbeth checker both ways, to confirm dither costs nothing visible
//!    on ordinary content.
//!
//! What to look for:
//! - Files 1 and 2 side by side are the entire argument. The band in 1
//!   is a hard vertical edge; in 2 there is no edge, only fine noise.
//! - Files 3 and 4 should be almost indistinguishable. Dither is
//!   insurance, not a look.
//!
//! The printed numbers are the point as much as the images: how many
//! distinct codes each configuration produces, how many transitions
//! appear along the ramp, and — the check that matters — that the mean
//! is unchanged, because triangular dither is zero-mean and must not
//! shift exposure.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::encode::encode_srgb;
use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};

/// Write 8-bit codes straight out as a grey PPM. Unlike the other
/// examples this one must not re-encode: the codes *are* the file.
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

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    // ── A gradient too shallow for 8 bits ────────────────────────────
    // 256 wide, 64 tall, spanning codes 100..101. Already display-
    // referred: quantisation is what happens *after* encode_srgb.
    let (w, h) = (256_usize, 64_usize);
    let ramp = Array3::<f32>::from_shape_fn((h, w, 1), |(_, x, _)| {
        (100.0 + x as f32 / (w as f32 - 1.0)) / 255.0
    });

    let plain = quantize_u8(ramp.view(), &QuantizeParams::default()).unwrap();
    let dithered = quantize_u8(ramp.view(), &QuantizeParams::new(Dither::Tpdf, 20260825)).unwrap();

    let plain_codes: Vec<u8> = plain.iter().copied().collect();
    let dither_codes: Vec<u8> = dithered.iter().copied().collect();

    write_codes(
        Path::new("examples/output/17_ramp_8bit_plain.ppm"),
        &plain_codes,
        w,
        h,
    );
    write_codes(
        Path::new("examples/output/17_ramp_8bit_dither.ppm"),
        &dither_codes,
        w,
        h,
    );

    // ── Ordinary content, both ways ──────────────────────────────────
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let display = encode_srgb(bw.view()).unwrap();

    for (name, params) in [
        ("plain", QuantizeParams::default()),
        ("dither", QuantizeParams::new(Dither::Tpdf, 20260825)),
    ] {
        let codes = quantize_u8(display.view(), &params).unwrap();
        let flat: Vec<u8> = codes.iter().copied().collect();
        write_codes(
            Path::new(&format!("examples/output/17_checker_8bit_{name}.ppm")),
            &flat,
            shared::WIDTH,
            shared::HEIGHT,
        );
    }
    println!("wrote examples/output/17_*.ppm");

    // ── The measurements ─────────────────────────────────────────────
    let first_row = |c: &[u8]| c[..w].to_vec();
    let distinct = |c: &[u8]| {
        let mut v = c.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    };

    println!("\na gradient spanning barely two 8-bit codes, 256 px wide:");
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
    let deep = quantize_u16(ramp.view(), &QuantizeParams::default()).unwrap();
    let deep_codes: Vec<u16> = deep.iter().copied().collect();
    let mut uniq = deep_codes.clone();
    uniq.sort_unstable();
    uniq.dedup();
    println!(
        "\nthe same gradient at 16 bits, undithered: {} distinct codes",
        uniq.len()
    );
    println!("  (8 bits cannot resolve it at all; 16 bits resolves it without help)");
}
