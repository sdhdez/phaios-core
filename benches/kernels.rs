// SPDX-License-Identifier: GPL-3.0-or-later
//! Criterion benchmarks for phaios-core kernels.
//!
//! One benchmark per kernel on a synthetic 24 MP (4323 × 5765) f32
//! image. Input arrays are pre-allocated outside the timed loop.
//! Output allocation is included in the measured time (mirrors real
//! usage). Run with: `cargo bench` — the bench profile is already
//! optimised, so there is no `--release` flag to pass.
//!
//! Results are written to `target/criterion/`. Open
//! `target/criterion/report/index.html` in a browser for a full report.

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array3;
use phaios_core::bw::{
    ColorFilter, HslWeightedParams, LuminanceStandard, channel_mixer_bw, color_filter_bw, hsl_bw,
    luminance_bw,
};
use phaios_core::encode::encode_srgb;
use phaios_core::exposure::exposure;
use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};
use phaios_core::tone::{ToneCurveParams, ZoneParams, tone_curve, zone_system};
use std::collections::HashMap;
use std::hint::black_box;

const H: usize = 4323;
const W: usize = 5765;

/// Build a deterministic pseudo-random image in [0, 1).
///
/// A constant image is not a representative input: in `encode_srgb`
/// every pixel takes the same branch, in `zone_system` every pixel lands
/// at the same zone position, and in `local_contrast` every window has
/// zero variance. Those are the cheapest possible paths, so a flat
/// image flatters the measurement.
///
/// This is a plain xorshift64 rather than a crate: benches stay
/// dependency-free, and the same bytes are produced on every run and
/// every platform, so results are comparable across machines.
fn pseudo_random_image(h: usize, w: usize, c: usize) -> Array3<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    Array3::from_shape_simple_fn((h, w, c), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Top 24 bits → [0, 1) with exact f32 spacing.
        (state >> 40) as f32 / 16_777_216.0
    })
}

fn bench_exposure(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    c.bench_function("exposure/24MP/+1EV", |b| {
        b.iter(|| exposure(black_box(img.view()), black_box(1.0_f32)).unwrap())
    });
}

fn bench_luminance_bw(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    c.bench_function("luminance_bw/24MP/BT709", |b| {
        b.iter(|| luminance_bw(black_box(img.view()), LuminanceStandard::Bt709).unwrap())
    });
}

fn bench_channel_mixer_bw(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    c.bench_function("channel_mixer_bw/24MP", |b| {
        b.iter(|| channel_mixer_bw(black_box(img.view()), black_box([0.21, 0.72, 0.07])).unwrap())
    });
}

fn bench_color_filter_bw(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    c.bench_function("color_filter_bw/24MP/Red25A", |b| {
        b.iter(|| {
            color_filter_bw(
                black_box(img.view()),
                black_box(ColorFilter::Red25A),
                black_box(LuminanceStandard::Bt709),
            )
            .unwrap()
        })
    });
}

fn bench_hsl_bw(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    let params = HslWeightedParams::new(
        [0.3, -0.2, 0.5, 0.1, 0.0, -0.6, 0.2, -0.1],
        LuminanceStandard::Bt709,
        30.0,
    );
    c.bench_function("hsl_bw/24MP/8-bands", |b| {
        b.iter(|| hsl_bw(black_box(img.view()), black_box(&params)).unwrap())
    });
}

fn bench_zone_system(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let mut offsets = HashMap::new();
    offsets.insert(5_i32, 0.5_f32);
    let params = ZoneParams::new(offsets);
    c.bench_function("zone_system/24MP/1-zone-offset", |b| {
        b.iter(|| zone_system(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_tone_curve(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let params = ToneCurveParams::new(1.1, 0.02, 0.85);
    c.bench_function("tone_curve/24MP/slope-offset-power", |b| {
        b.iter(|| tone_curve(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_local_contrast(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let params = GuidedFilterParams::new(8, 0.01);
    c.bench_function("local_contrast/24MP/r=8", |b| {
        b.iter(|| {
            local_contrast(
                black_box(grey.view()),
                black_box(&params),
                black_box(0.5_f32),
            )
            .unwrap()
        })
    });
}

fn bench_encode_srgb(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    c.bench_function("encode_srgb/24MP", |b| {
        b.iter(|| encode_srgb(black_box(grey.view())).unwrap())
    });
}

criterion_group!(
    benches,
    bench_exposure,
    bench_luminance_bw,
    bench_channel_mixer_bw,
    bench_color_filter_bw,
    bench_hsl_bw,
    bench_zone_system,
    bench_tone_curve,
    bench_local_contrast,
    bench_encode_srgb,
);
criterion_main!(benches);
