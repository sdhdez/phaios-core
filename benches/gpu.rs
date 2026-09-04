// SPDX-License-Identifier: GPL-3.0-or-later
//! Criterion benchmarks for the CUDA backend, on the same synthetic
//! 24 MP (4323 × 5765) f32 image as `benches/kernels.rs`.
//!
//! Ids are the CPU ids under a `gpu/` prefix, so `cargo bench` reports
//! `exposure/24MP/+1EV` and `gpu/exposure/24MP/+1EV` side by side and a
//! reader can divide one by the other.
//!
//! **Measuring an asynchronous backend.** Kernel launches return before
//! the work is done, so a naive timer measures the launch queue. Each
//! benchmark therefore uses `iter_custom`: it issues `iters` launches
//! and then drains the stream once, which is also how a real pipeline
//! behaves — many kernels, one synchronisation at the end.
//!
//! The drain is a `download` of a **1 × 1 × 1** image. Any download
//! synchronises the stream, but downloading the 299 MB result would put
//! PCIe transfer inside the timer: measured that way the cheap kernels
//! read roughly four times their true cost, being about 75% bus traffic.
//!
//! Two kernels return host data by contract and so cannot avoid the
//! transfer — `quantize` copies back 24.9 MB of integer codes per call for
//! this single-channel frame (75 MB for a three-channel one) and
//! `histogram` a few KiB of counts. Their numbers include it, which is
//! what a caller actually pays.
//!
//! Without `--features cuda`, or with no usable device, this target
//! prints one line and exits 0 rather than failing.

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("gpu benches skipped: build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    gpu::run();
}

#[cfg(feature = "cuda")]
mod gpu {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use criterion::Criterion;
    use ndarray::Array3;

    use phaios_core::blur::{BlurParams, BlurShape};
    use phaios_core::bw::{ColorFilter, HslWeightedParams, LuminanceStandard};
    use phaios_core::cuda::{Context, DeviceImage, kernels as k};
    use phaios_core::film_grain::GrainParams;
    use phaios_core::geometry::{
        CropParams, Orientation, ResizeFilter, ResizeParams, StraightenParams,
    };
    use phaios_core::glow::GlowParams;
    use phaios_core::highlight_rolloff::RolloffParams;
    use phaios_core::histogram::HistogramParams;
    use phaios_core::local_contrast::GuidedFilterParams;
    use phaios_core::lut::LutParams;
    use phaios_core::quantize::{Dither, QuantizeParams};
    use phaios_core::shadow_rolloff::ShadowRolloffParams;
    use phaios_core::sharpen::SharpenParams;
    use phaios_core::split_toning::SplitToningParams;
    use phaios_core::tone::{ToneCurveParams, ZoneParams};
    use phaios_core::vignette::VignetteParams;

    const H: usize = 4323;
    const W: usize = 5765;

    /// Same generator and seed as `benches/kernels.rs`, so both backends
    /// are measured on identical bytes.
    fn pseudo_random_image(h: usize, w: usize, c: usize) -> Array3<f32> {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        Array3::from_shape_simple_fn((h, w, c), || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0
        })
    }

    /// Time `iters` launches, then drain the stream once.
    fn timed<F>(ctx: &Context, sync: &DeviceImage, iters: u64, mut launch: F) -> Duration
    where
        F: FnMut(),
    {
        let start = Instant::now();
        for _ in 0..iters {
            launch();
        }
        // Synchronises without moving the result across PCIe.
        let _ = ctx.download(sync).expect("drain");
        start.elapsed()
    }

    macro_rules! bench {
        ($c:expr, $ctx:expr, $sync:expr, $id:expr, $body:expr) => {
            $c.bench_function($id, |b| {
                b.iter_custom(|iters| {
                    timed($ctx, $sync, iters, || {
                        black_box($body);
                    })
                })
            });
        };
    }

    pub fn run() {
        if !phaios_core::cuda::available() {
            println!("gpu benches skipped: no usable CUDA device");
            return;
        }
        let ctx = match Context::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("gpu benches skipped: {e}");
                return;
            }
        };

        let mut c = Criterion::default().configure_from_args();
        let rgb = ctx
            .upload(pseudo_random_image(H, W, 3).view())
            .expect("upload");
        let luma = ctx
            .upload(pseudo_random_image(H, W, 1).view())
            .expect("upload");
        // One pixel, uploaded once: downloading it drains the stream.
        // benches/kernels.rs pre-scales these two so each kernel takes
        // its working branch rather than an identity fast path.
        let bright = ctx
            .upload(pseudo_random_image(H, W, 1).mapv(|v| v * 4.0).view())
            .expect("upload");
        let dark = ctx
            .upload(pseudo_random_image(H, W, 1).mapv(|v| v * 0.2).view())
            .expect("upload");
        let sync = ctx
            .upload(Array3::<f32>::zeros((1, 1, 1)).view())
            .expect("upload sync pixel");

        // ── geometry ────────────────────────────────────────────────
        let crop = CropParams::new(
            (W / 4) as u32,
            (H / 4) as u32,
            (W / 2) as u32,
            (H / 2) as u32,
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/crop/24MP/centre-half",
            k::crop_device(&rgb, &crop).unwrap()
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/orient/24MP/rotate90",
            k::orient_device(&rgb, Orientation::Rotate90).unwrap()
        );
        let rs = ResizeParams::new(2048, 1536, ResizeFilter::Area);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/resize/24MP/to-2048-area",
            k::resize_device(&rgb, &rs).unwrap()
        );
        let st = StraightenParams::new(2.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/straighten/24MP/2deg",
            k::straighten_device(&rgb, &st).unwrap()
        );

        // ── tone and colour ─────────────────────────────────────────
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/exposure/24MP/+1EV",
            k::exposure_device(&rgb, 1.0).unwrap()
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/luminance_bw/24MP/BT709",
            k::luminance_bw_device(&rgb, LuminanceStandard::Bt709).unwrap()
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/channel_mixer_bw/24MP",
            k::channel_mixer_bw_device(&rgb, [0.3, 0.59, 0.11]).unwrap()
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/color_filter_bw/24MP/Red25A",
            k::color_filter_bw_device(&rgb, ColorFilter::Red25A, LuminanceStandard::Bt709).unwrap()
        );
        let hsl = HslWeightedParams::new([0.2; 8], LuminanceStandard::Bt709, 25.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/hsl_bw/24MP/8-bands",
            k::hsl_bw_device(&rgb, &hsl).unwrap()
        );

        let mut offsets = std::collections::HashMap::new();
        offsets.insert(5_i32, 0.5_f32);
        let zones = ZoneParams::new(offsets);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/zone_system/24MP/1-zone-offset",
            k::zone_system_device(&luma, &zones).unwrap()
        );
        let tc = ToneCurveParams::new(1.2, 0.02, 1.1);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/tone_curve/24MP/slope-offset-power",
            k::tone_curve_device(&luma, &tc).unwrap()
        );
        let hr = RolloffParams::new(0.7, 4.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/highlight_rolloff/24MP/knee0.7-white4",
            k::highlight_rolloff_device(&bright, &hr).unwrap()
        );
        let sr = ShadowRolloffParams::new(0.2, 0.8);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/shadow_rolloff/24MP/knee0.2-strength0.8",
            k::shadow_rolloff_device(&dark, &sr).unwrap()
        );
        let vg = VignetteParams::new(0.4, 0.7, 0.5);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/vignette/24MP",
            k::vignette_device(&luma, &vg).unwrap()
        );
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/encode_srgb/24MP",
            k::encode_srgb_device(&luma).unwrap()
        );

        // ── spatial ─────────────────────────────────────────────────
        for (label, sigma) in [
            ("sigma2-direct", 2.0_f32),
            ("sigma16-box", 16.0),
            ("sigma64-box", 64.0),
        ] {
            let bp = BlurParams::new(sigma, BlurShape::Gaussian);
            bench!(
                c,
                &ctx,
                &sync,
                &format!("gpu/blur/24MP/{label}"),
                k::blur_device(&luma, &bp).unwrap()
            );
        }
        let glow = GlowParams::new(0.7, 12.0, 0.4);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/glow/24MP/halation",
            k::glow_device(&luma, &glow).unwrap()
        );
        let sp_params = SharpenParams::new(0.5, 1.2, 0.02);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/sharpen/24MP/capture",
            k::sharpen_device(&luma, &sp_params).unwrap()
        );
        let gf = GuidedFilterParams::new(8, 0.01);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/local_contrast/24MP/r=8",
            k::local_contrast_device(&luma, &gf, 0.5).unwrap()
        );
        let grain = GrainParams::new(0.3, 2.0, 20_260_828);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/film_grain/24MP/size=2",
            k::film_grain_device(&luma, &grain).unwrap()
        );
        let sp = SplitToningParams::new([0.02, -0.03, 0.0], [0.0, 0.03, 0.02], 0.5, 0.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/split_toning/24MP",
            k::split_toning_device(&luma, &sp).unwrap()
        );

        let ramp: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let lut = ndarray::Array1::from(ramp);
        let lp = LutParams::new(0.0, 1.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/apply_lut/24MP/256-entry",
            k::apply_lut_device(&luma, lut.view(), &lp).unwrap()
        );

        // ── terminal stages: these return host data by contract, so the
        //    device-to-host copy is part of what a caller pays.
        let qp = QuantizeParams::new(Dither::default(), 7);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/quantize/24MP/u8-plain",
            k::quantize_u8_device(&luma, &qp).unwrap()
        );
        let hp = HistogramParams::new(256, 0.0, 1.0);
        bench!(
            c,
            &ctx,
            &sync,
            "gpu/histogram/24MP/256bins-rgb",
            k::histogram_device(&rgb, &hp).unwrap()
        );

        c.final_summary();
    }
}
