// SPDX-License-Identifier: GPL-3.0-or-later
//! Criterion benchmarks for phaios-core kernels.
//!
//! One benchmark per kernel on a synthetic 24 MP (4323 × 5765) f32
//! image. Input arrays are pre-allocated outside the timed loop.
//! Output allocation is included in the measured time (mirrors real
//! usage). Run with: `cargo bench --release`.
//!
//! Results are written to `target/criterion/`. Open
//! `target/criterion/report/index.html` in a browser for a full report.

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array3;
use phaios_core::bw::{
    ColorFilter, LuminanceStandard, channel_mixer_bw, color_filter_bw, luminance_bw,
};
use phaios_core::encode::encode_srgb;
use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};
use phaios_core::tone::{ZoneParams, zone_system};
use std::collections::HashMap;
use std::hint::black_box;

const H: usize = 4323;
const W: usize = 5765;

fn bench_luminance_bw(c: &mut Criterion) {
    let img = Array3::<f32>::from_elem((H, W, 3), 0.5_f32);
    c.bench_function("luminance_bw/24MP/BT709", |b| {
        b.iter(|| luminance_bw(black_box(img.view()), LuminanceStandard::Bt709).unwrap())
    });
}

fn bench_channel_mixer_bw(c: &mut Criterion) {
    let img = Array3::<f32>::from_elem((H, W, 3), 0.5_f32);
    c.bench_function("channel_mixer_bw/24MP", |b| {
        b.iter(|| channel_mixer_bw(black_box(img.view()), black_box([0.21, 0.72, 0.07])).unwrap())
    });
}

fn bench_color_filter_bw(c: &mut Criterion) {
    let img = Array3::<f32>::from_elem((H, W, 3), 0.5_f32);
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

fn bench_zone_system(c: &mut Criterion) {
    let grey = Array3::<f32>::from_elem((H, W, 1), 0.18_f32);
    let mut offsets = HashMap::new();
    offsets.insert(5_i32, 0.5_f32);
    let params = ZoneParams::new(offsets);
    c.bench_function("zone_system/24MP/1-zone-offset", |b| {
        b.iter(|| zone_system(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_local_contrast(c: &mut Criterion) {
    let grey = Array3::<f32>::from_elem((H, W, 1), 0.18_f32);
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
    let grey = Array3::<f32>::from_elem((H, W, 1), 0.5_f32);
    c.bench_function("encode_srgb/24MP", |b| {
        b.iter(|| encode_srgb(black_box(grey.view())).unwrap())
    });
}

criterion_group!(
    benches,
    bench_luminance_bw,
    bench_channel_mixer_bw,
    bench_color_filter_bw,
    bench_zone_system,
    bench_local_contrast,
    bench_encode_srgb,
);
criterion_main!(benches);
