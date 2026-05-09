// SPDX-License-Identifier: GPL-3.0-or-later
//! Shared utilities for phaios-core examples.
//!
//! Provides a synthetic Macbeth-style colour checker and a minimal
//! binary PPM writer. Neither function performs I/O except via the
//! explicit `path` argument to `write_ppm`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Patch count and grid layout for the synthetic Macbeth chart.
pub const PATCH_SIZE: usize = 64;
pub const GRID_COLS: usize = 6;
pub const GRID_ROWS: usize = 4;
pub const WIDTH: usize = PATCH_SIZE * GRID_COLS; // 384
pub const HEIGHT: usize = PATCH_SIZE * GRID_ROWS; // 256

/// Standard Macbeth ColorChecker patch values in scene-linear sRGB (D65).
///
/// 24 patches in row-major order (top-left to bottom-right). Values are
/// derived from the BabelColor average measurements
/// (http://www.babelcolor.com/colorchecker.htm) converted from sRGB to
/// linear by removing the sRGB transfer function (x/12.92 for the linear
/// segment, ((x+0.055)/1.055)^2.4 otherwise).
///
/// These are approximate; the exact values depend on the illuminant and
/// the measurement instrument. For the purposes of these examples they
/// are sufficient to produce visually meaningful output.
pub const PATCHES: [[f32; 3]; 24] = [
    // Row 1 — colour patches
    [0.400_f32, 0.225_f32, 0.155_f32], // 01 Dark skin
    [0.763_f32, 0.488_f32, 0.349_f32], // 02 Light skin
    [0.183_f32, 0.267_f32, 0.467_f32], // 03 Blue sky
    [0.149_f32, 0.200_f32, 0.086_f32], // 04 Foliage
    [0.341_f32, 0.345_f32, 0.625_f32], // 05 Blue flower
    [0.148_f32, 0.604_f32, 0.531_f32], // 06 Bluish green
    // Row 2 — colour patches
    [0.763_f32, 0.325_f32, 0.031_f32], // 07 Orange
    [0.102_f32, 0.145_f32, 0.510_f32], // 08 Purplish blue
    [0.631_f32, 0.165_f32, 0.165_f32], // 09 Moderate red
    [0.082_f32, 0.045_f32, 0.122_f32], // 10 Purple
    [0.416_f32, 0.620_f32, 0.059_f32], // 11 Yellow green
    [0.749_f32, 0.514_f32, 0.012_f32], // 12 Orange yellow
    // Row 3 — colour patches
    [0.027_f32, 0.063_f32, 0.416_f32], // 13 Blue
    [0.090_f32, 0.306_f32, 0.094_f32], // 14 Green
    [0.502_f32, 0.039_f32, 0.031_f32], // 15 Red
    [0.714_f32, 0.620_f32, 0.008_f32], // 16 Yellow
    [0.565_f32, 0.122_f32, 0.404_f32], // 17 Magenta
    [0.012_f32, 0.353_f32, 0.502_f32], // 18 Cyan
    // Row 4 — neutral patches
    [0.914_f32, 0.914_f32, 0.914_f32], // 19 White  (N9.5)
    [0.573_f32, 0.573_f32, 0.573_f32], // 20 Neutral 8
    [0.353_f32, 0.353_f32, 0.353_f32], // 21 Neutral 6.5
    [0.188_f32, 0.188_f32, 0.188_f32], // 22 Neutral 5   (≈18% grey)
    [0.086_f32, 0.086_f32, 0.086_f32], // 23 Neutral 3.5
    [0.031_f32, 0.031_f32, 0.031_f32], // 24 Black  (N2)
];

/// Build a linear sRGB f32 image of the synthetic Macbeth chart.
///
/// Returns a flat row-major buffer of length `HEIGHT * WIDTH * 3`.
/// Each group of three `f32` values is one pixel in RGB order.
#[must_use]
pub fn synthetic_macbeth() -> Vec<f32> {
    let mut buf = vec![0.0_f32; HEIGHT * WIDTH * 3];
    for (patch_idx, patch) in PATCHES.iter().enumerate() {
        let col = patch_idx % GRID_COLS;
        let row = patch_idx / GRID_COLS;
        let y0 = row * PATCH_SIZE;
        let x0 = col * PATCH_SIZE;
        for dy in 0..PATCH_SIZE {
            for dx in 0..PATCH_SIZE {
                let pixel = ((y0 + dy) * WIDTH + (x0 + dx)) * 3;
                buf[pixel] = patch[0];
                buf[pixel + 1] = patch[1];
                buf[pixel + 2] = patch[2];
            }
        }
    }
    buf
}

/// Write an 8-bit binary PPM file.
///
/// `pixels` is a flat row-major RGB buffer; values are clamped to [0,1]
/// and scaled to [0,255]. Panics if the file cannot be created.
pub fn write_ppm(path: &Path, pixels: &[f32], width: usize, height: usize) {
    let file = File::create(path).expect("cannot create PPM file");
    let mut w = BufWriter::new(file);
    write!(w, "P6\n{width} {height}\n255\n").expect("PPM header write failed");
    for &v in pixels {
        let byte = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        w.write_all(&[byte]).expect("PPM pixel write failed");
    }
}

/// Like `write_ppm` but for single-channel (luminance) data.
///
/// Each f32 is written as R=G=B (greyscale triplet) so the PPM viewer
/// renders a proper greyscale image.
pub fn write_ppm_grey(path: &Path, pixels: &[f32], width: usize, height: usize) {
    let rgb: Vec<f32> = pixels.iter().flat_map(|&v| [v, v, v]).collect();
    write_ppm(path, &rgb, width, height);
}
