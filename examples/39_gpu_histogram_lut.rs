// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 39 — The analysis pair, `histogram` and `apply_lut`, on the
//! CUDA backend.
//!
//! The GPU twin of `examples/18_histogram_lut.rs`. Same deliberately
//! flat frame — a low-contrast gradient confined to `[0.40, 0.60]`, the
//! kind of thing a scan of a flat negative produces — the same four
//! output files and the same printed bar charts, computed by
//! `histogram_device` and `apply_lut_device`.
//!
//! The frame is uploaded **once**. Everything after that happens on the
//! card: the histogram is reduced there and only its counts come back,
//! the equalised image is transformed there and its histogram is taken
//! from the device buffer without a return trip, and the four finished
//! frames are the only images downloaded.
//!
//! `histogram` is the odd one out among the device kernels — the
//! crate's only reduction, so it has no resident form to chain: the
//! counts are small and their destination is a caller's display or an
//! auto-correction, both of which live on the host. It is also
//! deterministic for free. `docs/ffi.md` §6 promises it **bit-exact**
//! because the only float arithmetic in it is the bin assignment;
//! everything after that is integer counting, and integer addition
//! commutes, so the order in which the device's atomics complete cannot
//! change a total. `apply_lut` is promised bit-exact for the same
//! reason its CPU form is cheap: subtract, divide, multiply, truncate
//! and one linear interpolation, every one of them correctly rounded.
//! Both comparisons below are therefore `==`, with no tolerance.
//!
//! What to look for:
//! - The agreement table. Every row `true`, including the histogram
//!   row, which compares the bin counts and the three out-of-range
//!   tallies element for element — a reduction agreeing exactly is a
//!   stronger statement than a map agreeing exactly, because a
//!   reduction is where a race or a lost atomic would show.
//! - The printed histograms before and after equalisation. The input
//!   occupies a narrow spike; the output spans the axis. Two device
//!   calls and a table produce that, with no third component.
//! - The clipping tallies after +1.5 EV. Out-of-range samples are
//!   counted *separately* from the end bins, so a bright picture and a
//!   clipped one do not look alike — the flaw in most histogram
//!   displays, and one the device kernel reproduces exactly.
//! - `39_solarised.ppm` reverses above its midpoint: highlights come
//!   back down towards black, the Sabattier effect. That table is
//!   deliberately non-monotone, which is the one thing `tone_curve` and
//!   `zone_system` structurally cannot express.
//!
//! `39_original.ppm` is written from the *downloaded* copy of the
//! uploaded frame rather than from the host array, so that the file
//! also exercises the round trip: if it matches example 18's byte for
//! byte, upload and download moved the frame without touching it.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPMs are display-referred, and
//! that encode runs on the host exactly as in example 18 — the files
//! are meant to be byte-comparable, so nothing in the writing path may
//! differ either.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::{Array1, Array3};
use phaios_core::cuda::{self, kernels as k};
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

/// Whether two histograms agree exactly, in every field that carries a
/// measurement.
///
/// `Histogram` has no `PartialEq` — it is a public type whose equality
/// no consumer has asked for — so the comparison is spelled out here
/// rather than derived on the library type for one example's benefit.
fn histograms_equal(a: &Histogram, b: &Histogram) -> bool {
    a.counts() == b.counts()
        && a.below() == b.below()
        && a.above() == b.above()
        && a.non_finite() == b.non_finite()
        && a.bins == b.bins
        && a.channels == b.channels
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

    // A flat, low-contrast frame: everything inside [0.40, 0.60].
    let flat = Array3::<f32>::from_shape_fn((H, W, 1), |(y, x, _)| {
        let ramp = x as f32 / (W - 1) as f32;
        let banding = ((y as f32 / 24.0).sin() * 0.5 + 0.5) * 0.02;
        0.40 + ramp * 0.18 + banding
    });

    // The one upload. Everything below reads this buffer or a buffer
    // derived from it without leaving the card.
    let flat_dev = ctx.upload(flat.view()).unwrap();
    let round_tripped = ctx.download(&flat_dev).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/39_original.ppm"),
        round_tripped.view(),
        W,
        H,
    );

    let params = HistogramParams::default();
    let before_gpu = k::histogram_device(&flat_dev, &params).unwrap();
    let before_cpu = histogram(flat.view(), &params).unwrap();
    print_histogram(
        "input histogram, reduced on the device (256 bins over [0, 1], shown in 32 buckets)",
        &before_gpu,
    );

    // ── 1. Equalisation: histogram -> table -> apply_lut ─────────────
    // The table comes from the *device's* histogram, so the whole
    // transform is derived on the card; the CPU builds its own from its
    // own counts, and the two tables are compared below rather than
    // shared, which would have hidden a disagreement in the counts.
    let table_gpu = before_gpu.equalisation_lut();
    let table_cpu = before_cpu.equalisation_lut();
    let lut_params = LutParams::default();

    let equalised_dev = k::apply_lut_device(&flat_dev, table_gpu.row(0), &lut_params).unwrap();
    let equalised_gpu = ctx.download(&equalised_dev).unwrap();
    let equalised_cpu = apply_lut(flat.view(), table_cpu.row(0), &lut_params).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/39_equalised.ppm"),
        equalised_gpu.view(),
        W,
        H,
    );

    // The equalised frame is still resident, so its histogram costs no
    // transfer at all.
    let after_gpu = k::histogram_device(&equalised_dev, &params).unwrap();
    let after_cpu = histogram(equalised_cpu.view(), &params).unwrap();
    print_histogram(
        "after equalisation — the same two device calls, nothing else",
        &after_gpu,
    );

    let occupied = |h: &Histogram| (0..h.bins).filter(|&b| h.counts()[[0, b]] > 0).count();
    println!(
        "\noccupied bins: {} before, {} after",
        occupied(&before_gpu),
        occupied(&after_gpu)
    );

    // ── 2. Solarisation: a non-monotone table ────────────────────────
    let solarise = Array1::from_shape_fn(256, |i| {
        let t = i as f32 / 255.0;
        if t < 0.55 { t / 0.55 } else { (1.0 - t) / 0.45 }
    });
    let solarised_gpu = ctx
        .download(&k::apply_lut_device(&flat_dev, solarise.view(), &lut_params).unwrap())
        .unwrap();
    let solarised_cpu = apply_lut(flat.view(), solarise.view(), &lut_params).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/39_solarised.ppm"),
        solarised_gpu.view(),
        W,
        H,
    );

    // ── 3. A film characteristic curve, tabulated ────────────────────
    // Toe, straight section, shoulder — the H&D curve — with no kernel
    // of its own. A table is enough, on either backend.
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
    let filmic_gpu = ctx
        .download(&k::apply_lut_device(&flat_dev, film.view(), &lut_params).unwrap())
        .unwrap();
    let filmic_cpu = apply_lut(flat.view(), film.view(), &lut_params).unwrap();
    shared::write_ppm_grey_display(
        Path::new("examples/output/39_film_curve.ppm"),
        filmic_gpu.view(),
        W,
        H,
    );

    println!("\nwrote examples/output/39_*.ppm");

    // ── 4. What the clipping tallies are for ─────────────────────────
    let pushed_dev = k::exposure_device(&flat_dev, 1.5).unwrap();
    let clipped_gpu = k::histogram_device(&pushed_dev, &params).unwrap();
    let clipped_cpu = histogram(
        phaios_core::exposure::exposure(flat.view(), 1.5)
            .unwrap()
            .view(),
        &params,
    )
    .unwrap();
    println!(
        "\nafter +1.5 EV: {} samples above the range, {} below, out of {}",
        clipped_gpu.above()[0],
        clipped_gpu.below()[0],
        clipped_gpu.total(0)
    );
    println!("  (counted apart from the end bins, so 'bright' and 'clipped' cannot be confused)");

    // ── Agreement: the point of the example for a beta tester ────────
    let rows: [(&str, bool); 7] = [
        ("upload -> download round trip", flat == round_tripped),
        (
            "histogram of the input",
            histograms_equal(&before_cpu, &before_gpu),
        ),
        ("equalisation table from it", table_cpu == table_gpu),
        ("apply_lut: equalised", equalised_cpu == equalised_gpu),
        (
            "histogram of the equalised",
            histograms_equal(&after_cpu, &after_gpu),
        ),
        ("apply_lut: solarised", solarised_cpu == solarised_gpu),
        ("apply_lut: film curve", filmic_cpu == filmic_gpu),
    ];
    println!("\ndevice-resident analysis pair vs the CPU specification:");
    let mut all_exact = true;
    for (label, equal) in rows {
        all_exact &= equal;
        println!("  {label:<32} CPU == GPU: {equal}");
    }
    println!(
        "  histogram after +1.5 EV, tallies included: CPU == GPU: {}",
        histograms_equal(&clipped_cpu, &clipped_gpu)
    );
    all_exact &= histograms_equal(&clipped_cpu, &clipped_gpu);
    println!("\nevery comparison above bit-exact (docs/ffi.md §6 promises this): {all_exact}");
    println!(
        "diff against example 18: for f in original equalised solarised film_curve; do cmp examples/output/39_$f.ppm examples/output/18_$f.ppm; done"
    );
}
