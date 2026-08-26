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
use phaios_core::blur::{BlurParams, BlurShape, blur};
use phaios_core::bw::{
    ColorFilter, HslWeightedParams, LuminanceStandard, channel_mixer_bw, color_filter_bw, hsl_bw,
    luminance_bw,
};
use phaios_core::encode::encode_srgb;
use phaios_core::exposure::exposure;
use phaios_core::film_grain::{GrainParams, film_grain};
use phaios_core::geometry::{
    CropParams, Orientation, ResizeFilter, ResizeParams, StraightenParams, crop, orient, resize,
    straighten,
};
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};
use phaios_core::histogram::{HistogramParams, histogram};
use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};
use phaios_core::lut::{LutParams, apply_lut};
use phaios_core::quantize::{Dither, QuantizeParams, quantize_u8, quantize_u16};
use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};
use phaios_core::split_toning::{SplitToningParams, split_toning};
use phaios_core::tone::{ToneCurveParams, ZoneParams, tone_curve, zone_system};
use phaios_core::vignette::{VignetteParams, vignette};
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

fn bench_geometry(c: &mut Criterion) {
    let img = pseudo_random_image(H, W, 3);
    let params = CropParams::new(W as u32 / 4, H as u32 / 4, W as u32 / 2, H as u32 / 2);
    c.bench_function("crop/24MP/centre-half", |b| {
        b.iter(|| crop(black_box(img.view()), black_box(&params)).unwrap())
    });
    c.bench_function("orient/24MP/rotate90", |b| {
        b.iter(|| orient(black_box(img.view()), black_box(Orientation::Rotate90)).unwrap())
    });
    let down = ResizeParams::new(2048, 1536, ResizeFilter::Area);
    c.bench_function("resize/24MP/to-2048-area", |b| {
        b.iter(|| resize(black_box(img.view()), black_box(&down)).unwrap())
    });
    let angle = StraightenParams::new(2.0);
    c.bench_function("straighten/24MP/2deg", |b| {
        b.iter(|| straighten(black_box(img.view()), black_box(&angle)).unwrap())
    });
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

fn bench_highlight_rolloff(c: &mut Criterion) {
    // Values spanning the shoulder, so the quadratic solve is actually
    // exercised rather than short-circuiting on the below-knee branch.
    let grey = pseudo_random_image(H, W, 1).mapv(|v| v * 4.0);
    let params = RolloffParams::new(0.7, 4.0);
    c.bench_function("highlight_rolloff/24MP/knee0.7-white4", |b| {
        b.iter(|| highlight_rolloff(black_box(grey.view()), black_box(&params)).unwrap())
    });
    // The default is a pure branch: worth measuring what a caller who
    // leaves it alone actually pays.
    let clip = RolloffParams::default();
    c.bench_function("highlight_rolloff/24MP/default-clip", |b| {
        b.iter(|| highlight_rolloff(black_box(grey.view()), black_box(&clip)).unwrap())
    });
}

fn bench_shadow_rolloff(c: &mut Criterion) {
    // Values inside the toe, so the cubic is exercised rather than
    // short-circuiting on the above-knee branch.
    let grey = pseudo_random_image(H, W, 1).mapv(|v| v * 0.2);
    let params = ShadowRolloffParams::new(0.2, 0.8);
    c.bench_function("shadow_rolloff/24MP/knee0.2-strength0.8", |b| {
        b.iter(|| shadow_rolloff(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_quantize(c: &mut Criterion) {
    // Display-referred input, as the kernel expects.
    let grey = pseudo_random_image(H, W, 1);
    let plain = QuantizeParams::default();
    let dithered = QuantizeParams::new(Dither::Tpdf, 20260825);
    c.bench_function("quantize/24MP/u8-plain", |b| {
        b.iter(|| quantize_u8(black_box(grey.view()), black_box(&plain)).unwrap())
    });
    c.bench_function("quantize/24MP/u8-dithered", |b| {
        b.iter(|| quantize_u8(black_box(grey.view()), black_box(&dithered)).unwrap())
    });
    c.bench_function("quantize/24MP/u16-dithered", |b| {
        b.iter(|| quantize_u16(black_box(grey.view()), black_box(&dithered)).unwrap())
    });
}

fn bench_analysis(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let rgb = pseudo_random_image(H, W, 3);

    let display = HistogramParams::default();
    c.bench_function("histogram/24MP/256bins-grey", |b| {
        b.iter(|| histogram(black_box(grey.view()), black_box(&display)).unwrap())
    });
    c.bench_function("histogram/24MP/256bins-rgb", |b| {
        b.iter(|| histogram(black_box(rgb.view()), black_box(&display)).unwrap())
    });
    // The analysis case: enough bins to resolve individual 16-bit codes.
    let fine = HistogramParams::new(65_536, 0.0, 1.0);
    c.bench_function("histogram/24MP/65536bins-grey", |b| {
        b.iter(|| histogram(black_box(grey.view()), black_box(&fine)).unwrap())
    });

    let lut = ndarray::Array1::from_shape_fn(256, |i| (i as f32 / 255.0).powf(0.7));
    let lp = LutParams::default();
    c.bench_function("apply_lut/24MP/256-entry", |b| {
        b.iter(|| {
            apply_lut(
                black_box(grey.view()),
                black_box(lut.view()),
                black_box(&lp),
            )
            .unwrap()
        })
    });
}

fn bench_blur(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    // Either side of the crossover: the direct path's cost grows with
    // sigma, the box path's does not.
    for sigma in [2.0_f32, 4.0, 5.9, 6.0, 16.0, 64.0] {
        let params = BlurParams::new(sigma, BlurShape::Gaussian);
        let path = if sigma < 6.0 { "direct" } else { "box" };
        c.bench_function(&format!("blur/24MP/sigma{sigma}-{path}"), |b| {
            b.iter(|| blur(black_box(grey.view()), black_box(&params)).unwrap())
        });
    }
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

fn bench_film_grain(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let params = GrainParams::new(0.25, 2.0, 20_260_815);
    c.bench_function("film_grain/24MP/size=2", |b| {
        b.iter(|| film_grain(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_split_toning(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let params = SplitToningParams::new([0.0, 0.03, -0.04], [0.0, -0.02, 0.05], 0.5, 0.1);
    c.bench_function("split_toning/24MP", |b| {
        b.iter(|| split_toning(black_box(grey.view()), black_box(&params)).unwrap())
    });
}

fn bench_vignette(c: &mut Criterion) {
    let grey = pseudo_random_image(H, W, 1);
    let params = VignetteParams::new(0.5, 0.7, 0.2);
    c.bench_function("vignette/24MP", |b| {
        b.iter(|| vignette(black_box(grey.view()), black_box(&params)).unwrap())
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
    bench_geometry,
    bench_exposure,
    bench_luminance_bw,
    bench_channel_mixer_bw,
    bench_color_filter_bw,
    bench_hsl_bw,
    bench_zone_system,
    bench_tone_curve,
    bench_highlight_rolloff,
    bench_shadow_rolloff,
    bench_quantize,
    bench_analysis,
    bench_blur,
    bench_local_contrast,
    bench_film_grain,
    bench_split_toning,
    bench_vignette,
    bench_encode_srgb,
);
criterion_main!(benches);
