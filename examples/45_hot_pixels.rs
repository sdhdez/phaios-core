// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 45 — Hot-pixel removal by conditional median.
//!
//! Demonstrates `hot_pixels`:
//!
//! ```text
//! m     = median9(3x3 window, index-clamped at the border)
//! limit = threshold + relative * |m|
//! out   = if |p - m| > limit { m } else { p }
//! ```
//!
//! A synthetic defect layer is planted on the B&W Macbeth chart in code
//! — five things, in four different patches, far enough apart that no
//! two 3x3 windows overlap:
//!
//! - a huge bright spike in the Black patch (a stuck-high photosite),
//! - a huge dark dip in the Neutral 8 patch (a stuck-low one),
//! - a one-pixel-wide, 45-pixel-long horizontal line in the Neutral 5
//!   patch, only 0.02 brighter than its background — genuine fine detail,
//!   not a defect,
//! - a modest 0.05 bump in the White patch — genuine highlight texture,
//! - a huge bright spike in the *same* White patch, 30 columns away —
//!   a real defect landing on a highlight rather than a midtone.
//!
//! Three files are written.
//!
//! What to look for:
//! - `45_corrupted` is the before picture: the chart with the defect
//!   layer planted, and the input both cleaning passes are given.
//! - `45_cleaned` (threshold 0.15, no relative term): both huge defects
//!   are gone, replaced by their neighbourhood's value; the fine line
//!   survives untouched, because its own deviation (0.02) never reaches
//!   the threshold.
//! - `45_cleaned_relative` (threshold 0.03, relative 0.15 — a threshold
//!   tight enough for ordinary midtone noise): the huge defects are
//!   still gone, including the one sitting in a highlight, but the
//!   White patch's 0.05 highlight texture now survives too, because
//!   `relative * |m|` widens the limit specifically where the local
//!   median is bright. The printed table below shows that the *same*
//!   tight threshold with `relative = 0.0` would have wrongly erased
//!   that texture as if it were a defect.
//!
//! The output is passed through `encode_srgb` before it is written (the
//! terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::hot_pixels::{HotPixelParams, hot_pixels};

// Planted-defect coordinates (row, col), one per patch, chosen so every
// 3x3 window is isolated from every other planted feature.
const BLACK_SPIKE: (usize, usize) = (210, 340); // Black patch (y0=192,x0=320)
const GREY_DIP: (usize, usize) = (210, 90); // Neutral 8 patch (y0=192,x0=64)
const LINE_ROW: usize = 220; // Neutral 5 patch (y0=192,x0=192)
const LINE_COLS: (usize, usize) = (205, 250); // half-open, inside the patch
const LINE_PROBE_COL: usize = 225; // well interior to the line segment
const WHITE_TEXTURE: (usize, usize) = (210, 15); // White patch (y0=192,x0=0)
const WHITE_SPIKE: (usize, usize) = (210, 45); // same patch, 30 cols away

fn write(name: &str, img: ndarray::ArrayView3<f32>) {
    let path = format!("examples/output/45_{name}.ppm");
    shared::write_ppm_grey_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // ── Plant the synthetic defect layer ──────────────────────────────
    let mut corrupted = bw.clone();
    let black_true = corrupted[[BLACK_SPIKE.0, BLACK_SPIKE.1, 0]];
    corrupted[[BLACK_SPIKE.0, BLACK_SPIKE.1, 0]] = black_true + 1.0;
    let grey_true = corrupted[[GREY_DIP.0, GREY_DIP.1, 0]];
    corrupted[[GREY_DIP.0, GREY_DIP.1, 0]] = grey_true - 0.5;
    for x in LINE_COLS.0..LINE_COLS.1 {
        corrupted[[LINE_ROW, x, 0]] += 0.02;
    }
    let white_texture_true = corrupted[[WHITE_TEXTURE.0, WHITE_TEXTURE.1, 0]];
    corrupted[[WHITE_TEXTURE.0, WHITE_TEXTURE.1, 0]] = white_texture_true + 0.05;
    let white_spike_true = corrupted[[WHITE_SPIKE.0, WHITE_SPIKE.1, 0]];
    corrupted[[WHITE_SPIKE.0, WHITE_SPIKE.1, 0]] = white_spike_true + 0.6;

    write("corrupted", corrupted.view());

    // ── Absolute-only cleaning: threshold wide enough for the defects, ─
    // ── narrow enough to leave the fine line and highlight bump alone ──
    let plain_params = HotPixelParams::new(0.15, 0.0);
    let cleaned = hot_pixels(corrupted.view(), &plain_params).unwrap();
    write("cleaned", cleaned.view());

    // ── Tight absolute threshold plus a relative term ─────────────────
    let relative_params = HotPixelParams::new(0.03, 0.15);
    let cleaned_relative = hot_pixels(corrupted.view(), &relative_params).unwrap();
    write("cleaned_relative", cleaned_relative.view());

    println!("wrote examples/output/45_*.ppm");

    // ── Outliers gone: both huge defects return to their background ──
    println!("\noutliers removed by the absolute-only cleaning (threshold=0.15):");
    println!(
        "{:>18} {:>10} {:>10} {:>10} {:>8}",
        "defect", "planted", "true bg", "cleaned", "restored"
    );
    for (label, (y, x), true_bg) in [
        ("black spike", BLACK_SPIKE, black_true),
        ("grey dip", GREY_DIP, grey_true),
        ("white spike", WHITE_SPIKE, white_spike_true),
    ] {
        let planted = corrupted[[y, x, 0]];
        let got = cleaned[[y, x, 0]];
        let restored = got.to_bits() == true_bg.to_bits();
        println!("{label:>18} {planted:>10.4} {true_bg:>10.4} {got:>10.4} {restored:>8}");
    }

    // ── A fine line is kept, bit-exact, by the same cleaning pass ─────
    println!("\na fine line (deviation 0.02, well under threshold=0.15) survives:");
    let line_before = corrupted[[LINE_ROW, LINE_PROBE_COL, 0]];
    let line_after = cleaned[[LINE_ROW, LINE_PROBE_COL, 0]];
    println!(
        "  before {line_before:.4}, after {line_after:.4}, bit-identical: {}",
        line_before.to_bits() == line_after.to_bits()
    );

    // ── The relative term protects highlight texture, without ─────────
    // ── protecting a genuine highlight defect ─────────────────────────
    println!("\nrelative term at a tight threshold=0.03 (appropriate for midtone noise):");
    println!(
        "{:>10} {:>10} {:>10} {:>10}",
        "relative", "limit", "texture", "spike"
    );
    let white_median = white_texture_true; // flat patch away from the spike
    for relative in [0.0_f32, 0.01, 0.02, 0.03, 0.05, 0.1] {
        let params = HotPixelParams::new(0.03, relative);
        let out = hot_pixels(corrupted.view(), &params).unwrap();
        let limit = 0.03 + relative * white_median.abs();
        let texture_kept = out[[WHITE_TEXTURE.0, WHITE_TEXTURE.1, 0]].to_bits()
            == corrupted[[WHITE_TEXTURE.0, WHITE_TEXTURE.1, 0]].to_bits();
        let spike_removed =
            out[[WHITE_SPIKE.0, WHITE_SPIKE.1, 0]].to_bits() == white_spike_true.to_bits();
        println!(
            "{relative:>10.2} {limit:>10.4} {:>10} {:>10}",
            if texture_kept { "kept" } else { "erased" },
            if spike_removed { "removed" } else { "kept!" }
        );
    }
    println!(
        "  (texture crosses from erased to kept between relative=0.02 and 0.03; the\n   \
         highlight spike stays removed throughout — relative widens the limit, it\n   \
         does not disable the kernel)"
    );
}
