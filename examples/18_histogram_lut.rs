// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 18 — The analysis pair: `histogram` and `apply_lut`.
//!
//! These two are shipped together because together they are a family.
//! `histogram` is the crate's first reduction — image to statistics
//! rather than image to image — and `apply_lut` is the general transfer
//! that turns any table into a tone curve. Neither is very interesting
//! alone; composed, they cover curves, equalisation, matching, film
//! emulation and solarisation without a kernel for each.
//!
//! The example works on a deliberately flat frame — a low-contrast
//! gradient confined to `[0.40, 0.60]`, the kind of thing a scan of a
//! flat negative produces — and demonstrates four things:
//!
//! 1. `18_original.ppm` — the flat input, and its histogram printed as
//!    a text bar chart. Everything is crammed into a fifth of the range.
//! 2. `18_equalised.ppm` — `histogram` → `equalisation_lut()` →
//!    `apply_lut`. That is the entire implementation of histogram
//!    equalisation: no third component, which is the argument for this
//!    factoring. (`equalisation_lut()` is `cdf()` with a leading zero,
//!    which aligns the CDF's bin-edge convention with how `apply_lut`
//!    spaces its table entries; feeding `cdf()` straight in biases the
//!    result by half a bin.)
//! 3. `18_solarised.ppm` — a deliberately **non-monotone** table. This
//!    is the one thing `tone_curve` and `zone_system` structurally
//!    cannot express, since both are monotone by construction.
//! 4. `18_film_curve.ppm` — a tabulated toe-and-shoulder characteristic
//!    curve, showing that "film emulation" needs no new kernel either.
//!
//! What to look for:
//! - The printed histograms before and after equalisation. The input
//!   occupies a narrow spike; the output spans the axis.
//! - The clipping tallies. Out-of-range samples are counted *separately*
//!   from the end bins, so a bright picture and a clipped one do not
//!   look alike — the flaw in most histogram displays.
//! - The solarised image reverses above its midpoint: highlights come
//!   back down towards black, which is the Sabattier effect.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::{Array1, Array3};
use phaios_core::histogram::{Histogram, HistogramParams, histogram};
use phaios_core::lut::{LutParams, apply_lut};

const W: usize = 384;
const H: usize = 256;

/// Print a histogram as a coarse text bar chart — 32 buckets, so it fits
/// in a terminal and the shape is what is visible rather than the noise.
fn print_histogram(label: &str, h: &Histogram) {
    let buckets = 32;
    let per = h.bins / buckets;
    let sums: Vec<u64> = (0..buckets)
        .map(|b| (0..per).map(|k| h.counts()[[0, b * per + k]]).sum())
        .collect();
    let peak = sums.iter().copied().max().unwrap_or(1).max(1);

    println!("\n{label}");
    for (b, &v) in sums.iter().enumerate() {
        let width = (v * 40 / peak) as usize;
        let lo = h.min + (h.max - h.min) * b as f32 / buckets as f32;
        println!("  {lo:>5.2} |{:<40}| {v}", "#".repeat(width));
    }
    println!(
        "  below {} · above {} · NaN {} · total {}",
        h.below()[0],
        h.above()[0],
        h.non_finite()[0],
        h.total(0)
    );
}

fn write(path: &str, img: &Array3<f32>) {
    let encoded = phaios_core::encode::encode_srgb(img.view()).unwrap();
    shared::write_ppm_grey(Path::new(path), encoded.as_slice().unwrap(), W, H);
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    // A flat, low-contrast frame: everything inside [0.40, 0.60].
    let flat = Array3::<f32>::from_shape_fn((H, W, 1), |(y, x, _)| {
        let ramp = x as f32 / (W - 1) as f32;
        let banding = ((y as f32 / 24.0).sin() * 0.5 + 0.5) * 0.02;
        0.40 + ramp * 0.18 + banding
    });
    write("examples/output/18_original.ppm", &flat);

    let params = HistogramParams::default();
    let before = histogram(flat.view(), &params).unwrap();
    print_histogram(
        "input histogram (256 bins over [0, 1], shown in 32 buckets)",
        &before,
    );

    // ── 1. Equalisation: cdf -> apply_lut, and that is all of it ─────
    let table = before.equalisation_lut();
    let equalised = apply_lut(
        flat.view(),
        table.row(0).to_owned().view(),
        &LutParams::default(),
    )
    .unwrap();
    write("examples/output/18_equalised.ppm", &equalised);
    let after = histogram(equalised.view(), &params).unwrap();
    print_histogram(
        "after equalisation — the same two calls, nothing else",
        &after,
    );

    let occupied = |h: &Histogram| (0..h.bins).filter(|&b| h.counts()[[0, b]] > 0).count();
    println!(
        "\noccupied bins: {} before, {} after",
        occupied(&before),
        occupied(&after)
    );

    // ── 2. Solarisation: a non-monotone table ────────────────────────
    // Neither tone_curve nor zone_system can express this: both are
    // monotone by construction, and the Sabattier effect is a fold.
    let solarise = Array1::from_shape_fn(256, |i| {
        let t = i as f32 / 255.0;
        if t < 0.55 { t / 0.55 } else { (1.0 - t) / 0.45 }
    });
    let solarised = apply_lut(flat.view(), solarise.view(), &LutParams::default()).unwrap();
    write("examples/output/18_solarised.ppm", &solarised);

    // ── 3. A film characteristic curve, tabulated ────────────────────
    // Toe, straight section, shoulder — the H&D curve — with no kernel
    // of its own. A table is enough.
    let film = Array1::from_shape_fn(256, |i| {
        let t = i as f32 / 255.0;
        let toe = 0.12;
        let shoulder = 0.82;
        if t < toe {
            // Compressed foot: shadows roll into black gradually.
            let u = t / toe;
            0.06 * u * u
        } else if t < shoulder {
            let u = (t - toe) / (shoulder - toe);
            0.06 + u * 0.80
        } else {
            // Shoulder, meeting the straight section smoothly.
            let u = (t - shoulder) / (1.0 - shoulder);
            0.86 + (1.0 - (1.0 - u) * (1.0 - u)) * 0.14
        }
    });
    let filmic = apply_lut(flat.view(), film.view(), &LutParams::default()).unwrap();
    write("examples/output/18_film_curve.ppm", &filmic);

    println!("\nwrote examples/output/18_*.ppm");

    // ── 4. What the clipping tallies are for ─────────────────────────
    let pushed = phaios_core::exposure::exposure(flat.view(), 1.5).unwrap();
    let clipped = histogram(pushed.view(), &params).unwrap();
    println!(
        "\nafter +1.5 EV: {} samples above the range, {} below, out of {}",
        clipped.above()[0],
        clipped.below()[0],
        clipped.total(0)
    );
    println!("  (counted apart from the end bins, so 'bright' and 'clipped' cannot be confused)");
}
