// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 23 — GPU self-test: every CUDA kernel against its CPU oracle.
//!
//! The hosted CI runner has neither nvcc nor a device, so the CUDA
//! backend is the one part of this crate that CI can never check. This
//! example is the substitute. Anyone with a supported card runs
//!
//! ```text
//! cargo run --release --features cuda --example 23_gpu_selftest
//! ```
//!
//! and pastes the table into an issue, so the backend is verified by
//! whoever owns the hardware rather than only by the maintainer.
//!
//! Every `*_device` entry point in [`phaios_core::cuda::kernels`] gets
//! exactly one row, checked against the CPU function that is its
//! specification (`docs/ffi.md` §6: the CPU implementation is the
//! specification, never the other way round). The device-resident form
//! is the one exercised — upload once, run `*_device`, download —
//! because that is the path a real pipeline takes and the path where a
//! wrong buffer or a stale stream would hide.
//!
//! What a row asserts comes from `docs/ffi.md` §6, which is the source
//! of truth for the agreement class of each kernel:
//!
//! - **bit-exact** — kernels free of transcendentals, where every
//!   operation is correctly rounded on both sides and the PTX is built
//!   with `-fmad=false`. The check is `==` over the whole array. There
//!   is no tolerance here and there must never be one.
//! - **a committed `(rtol, atol)` bound** — kernels containing a
//!   transcendental, which IEEE-754 does not standardise. The check is
//!   the worst element expressed as a multiple of that bound; at or
//!   below 1.000 passes. The numbers are copied from §6, not invented
//!   here, so a driver update that regresses accuracy shows up as a
//!   failing row rather than being quietly absorbed.
//!
//! The exit status is the point: 0 when every kernel agrees, 1 when any
//! does not, so this doubles as the gate of a locally-installed CI. A
//! machine with no device prints why and exits 0 — unless
//! `PHAIOS_REQUIRE_GPU` is set, which makes the missing device itself a
//! failure. That is the same contract `tests/cuda_conformance.rs` uses,
//! and it is what stops "green" from ever meaning "silently skipped".

use std::process::ExitCode;

use ndarray::{Array1, Array3, s};
use phaios_core::cuda;
use phaios_core::error::PhaiosError;

// ── the documented agreement classes (docs/ffi.md §6) ────────────────────────

/// A committed `(rtol, atol)` pair, carrying the label the table prints.
///
/// Numbers and label live together so a row cannot advertise one bound
/// while the comparison uses another.
#[derive(Clone, Copy)]
struct Bound {
    rtol: f32,
    atol: f32,
    label: &'static str,
}

/// §6: the bound for kernels whose transcendental content is a small
/// fixed number of `powf`, `expf`, `log2f` or `cbrtf` calls per pixel.
/// `hsl_bw` makes eight and `zone_system` thirteen; both sit in this
/// class.
const ONE_TRANSCENDENTAL: Bound = Bound {
    rtol: 1e-5,
    atol: 1e-7,
    label: "1e-5/1e-7",
};

/// §6: `local_contrast`, which reformulates the CPU's global f64
/// summed-area tables as separable box filters.
const GUIDED_FILTER: Bound = Bound {
    rtol: 1e-4,
    atol: 1e-6,
    label: "1e-4/1e-6",
};

/// §6: `film_grain`'s Box–Muller half.
///
/// The splitmix64 hash underneath it is exact, but nothing in this file
/// exercises it: the row's bit-exact case is `intensity = 0`, which
/// `film_grain_device` answers with a device-to-device copy
/// (`src/cuda/kernels/grain.rs`) without launching the grain kernel at
/// all, so `pixel_hash` never runs here. The hash is asserted over a
/// coordinate grid by `hash_grid` in `tests/cuda_conformance.rs`, which
/// is where to look for it.
const BOX_MULLER: Bound = Bound {
    rtol: 1e-3,
    atol: 1e-5,
    label: "1e-3/1e-5",
};

/// Label for a kernel §6 promises bit-exact across backends.
const EXACT: &str = "bit-exact";

/// Used only to give a *magnitude* to a bit-exact kernel that has
/// failed. A passing bit-exact row always reads 0.000, so this pair
/// never changes a verdict — it only answers "how badly?" when the
/// answer matters.
const DIAGNOSTIC: Bound = ONE_TRANSCENDENTAL;

/// Rows that check more than the promise column can say on its own.
const EXTRA_NOTES: &[(&str, &str)] = &[
    (
        "zone_system",
        "the empty offset map is also required bit-exact (identity path)",
    ),
    (
        "local_contrast",
        "includes the 1e6 highlight bar, where the f32 variance cancels",
    ),
    (
        "film_grain",
        "the bound covers Box-Muller; the splitmix64 hash is asserted by \
         hash_grid in tests/cuda_conformance.rs, not here",
    ),
    (
        "vignette",
        "amount 0 included — a device copy, not a launch, so it tests the \
         identity path rather than the kernel",
    ),
    (
        "shadow_rolloff",
        "strength 0 included — likewise a device copy, not a launch",
    ),
    (
        "tone_curve",
        "power == 1 also required bit-exact; the bound covers power != 1",
    ),
    (
        "blur",
        "sigma 2.5 direct path, sigma 8 box path, sigma 0 required bit-exact",
    ),
    ("glow", "amount 0 identity also required bit-exact"),
    ("sharpen", "amount 0 identity also required bit-exact"),
];

// ── the oracle ───────────────────────────────────────────────────────────────

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound. `<= 1.0` means every element passes.
///
/// This mirrors `worst_violation` in `tests/cuda_conformance.rs`, and
/// deliberately keeps its two pieces of care. Non-finite values are
/// compared explicitly rather than arithmetically, because the helper
/// this descends from used `.zip(...).fold(0.0, f32::max)` and `f32::max`
/// returns the *other* operand when one side is NaN — so an all-NaN GPU
/// output scored 0.000, a perfect match. And the shapes are checked
/// rather than zipped, because `zip` stops at the shorter side, so an
/// empty or truncated output scored 0.000 too.
///
/// The one deliberate difference from the test helper: a shape mismatch
/// returns infinity instead of panicking. A self-test whose job is to
/// print 27 rows must not lose 26 of them to the first broken kernel.
/// Infinity preserves the property that matters — a shape mismatch can
/// never read as agreement.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32) -> f32 {
    if a.dim() != b.dim() {
        return f32::INFINITY;
    }
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let v = if x.is_nan() || y.is_nan() {
            // Agreeing on NaN is agreement; disagreeing about whether a
            // value is NaN at all is total disagreement, not zero.
            if x.is_nan() && y.is_nan() {
                0.0
            } else {
                f32::INFINITY
            }
        } else if x.is_infinite() || y.is_infinite() {
            if x == y { 0.0 } else { f32::INFINITY }
        } else {
            (x - y).abs() / (atol + rtol * x.abs())
        };
        // Plain `>`, so a NaN cannot sneak through here either.
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Deterministic pseudo-random image — the xorshift the conformance
/// suite and the benches use, so a reported number is comparable with
/// theirs. No RNG dependency, and no seed hidden in a thread-local.
fn pseudo_random_image(h: usize, w: usize, c: usize) -> Array3<f32> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    Array3::from_shape_simple_fn((h, w, c), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / 16_777_216.0
    })
}

// ── accumulating one kernel's result ─────────────────────────────────────────

/// One kernel's verdict, ready to print.
struct Row {
    kernel: &'static str,
    promise: &'static str,
    cases: usize,
    compared: usize,
    bit_exact: bool,
    /// Worst violation as a multiple of the bound used. `None` for
    /// integer-output kernels, where no such ratio exists.
    worst: Option<f32>,
    pass: bool,
    detail: Option<String>,
}

/// Accumulates the cases of a single kernel.
///
/// A kernel may mix classes — `tone_curve` is bit-exact at `power == 1`
/// and bounded elsewhere — so exact and bounded cases are recorded
/// separately and both must hold for the row to pass.
struct Check {
    cases: usize,
    /// Elements actually compared across every case in this row.
    ///
    /// A comparison over an empty array reports a worst violation of
    /// 0.000, which is indistinguishable in the table from "agreed
    /// everywhere". `crop` to a 0x0 output is a legitimate case and does
    /// exactly that, so the row footer says how much was really looked at
    /// rather than leaving the reader to assume.
    compared: usize,
    bit_exact: bool,
    worst: Option<f32>,
    passed: bool,
    detail: Option<String>,
}

impl Check {
    fn new() -> Self {
        Self {
            cases: 0,
            compared: 0,
            bit_exact: true,
            worst: None,
            passed: true,
            detail: None,
        }
    }

    /// Keep the first failure only. The table stays one line per kernel;
    /// a reader who needs more reruns the case named here.
    fn note(&mut self, message: String) {
        if self.detail.is_none() {
            self.detail = Some(message);
        }
    }

    fn record_worst(&mut self, v: f32) {
        let previous = self.worst.unwrap_or(0.0);
        // `>` rather than `max`, for the same reason the oracle uses it.
        self.worst = Some(if v > previous { v } else { previous });
    }

    /// A case §6 promises bit-exact. Byte equality or nothing.
    fn exact(&mut self, label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>) {
        self.cases += 1;
        self.compared += cpu.len().min(gpu.len());
        let v = worst_violation(cpu, gpu, DIAGNOSTIC.rtol, DIAGNOSTIC.atol);
        self.record_worst(v);
        if cpu != gpu {
            self.bit_exact = false;
            self.passed = false;
            let d = DIAGNOSTIC.label;
            self.note(format!(
                "{label}: not bit-exact ({v:.3}x the {d} diagnostic pair)"
            ));
        }
    }

    /// A case §6 gives a committed bound rather than exactness.
    fn bounded(&mut self, label: &str, cpu: &Array3<f32>, gpu: &Array3<f32>, bound: Bound) {
        self.cases += 1;
        self.compared += cpu.len().min(gpu.len());
        let v = worst_violation(cpu, gpu, bound.rtol, bound.atol);
        self.record_worst(v);
        if cpu != gpu {
            self.bit_exact = false;
        }
        // NaN is spelled out rather than written `!(v <= 1.0)`: a NaN
        // ratio must fail, and the negated comparison would not say so
        // as clearly.
        if v > 1.0 || v.is_nan() {
            self.passed = false;
            let bl = bound.label;
            self.note(format!("{label}: {v:.3}x the {bl} bound"));
        }
    }

    /// An integer-output case. There is no tolerance on a code value, so
    /// the only question is equality; the worst code difference is
    /// reported when it is not.
    fn exact_codes<T>(&mut self, label: &str, cpu: &Array3<T>, gpu: &Array3<T>)
    where
        T: Copy + PartialEq + Into<i64>,
    {
        self.cases += 1;
        self.compared += cpu.len().min(gpu.len());
        if cpu.dim() != gpu.dim() {
            self.bit_exact = false;
            self.passed = false;
            self.note(format!(
                "{label}: shape mismatch, {:?} against {:?}",
                cpu.dim(),
                gpu.dim()
            ));
            return;
        }
        let worst_code = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(&a, &b)| (a.into() - b.into()).abs())
            .max()
            .unwrap_or(0);
        if worst_code != 0 {
            self.bit_exact = false;
            self.passed = false;
            self.note(format!("{label}: worst code difference {worst_code}"));
        }
    }

    /// A case whose comparison is not over an array — `histogram`
    /// returns counts, edges and non-finite tallies, all integers.
    fn exact_flag(&mut self, label: &str, agrees: bool, compared: usize, what: &str) {
        self.cases += 1;
        self.compared += compared;
        if !agrees {
            self.bit_exact = false;
            self.passed = false;
            self.note(format!("{label}: {what}"));
        }
    }

    fn finish(self, kernel: &'static str, promise: &'static str) -> Row {
        Row {
            kernel,
            promise,
            cases: self.cases,
            compared: self.compared,
            bit_exact: self.bit_exact,
            worst: self.worst,
            pass: self.passed,
            detail: self.detail,
        }
    }
}

/// Run one kernel's cases and turn them into a row.
///
/// An error from a kernel is a failed row, not an abort: a caller
/// pasting this report needs the other 23 lines just as much.
fn run(
    kernel: &'static str,
    promise: &'static str,
    body: impl FnOnce(&mut Check) -> Result<(), PhaiosError>,
) -> Row {
    let mut check = Check::new();
    if let Err(e) = body(&mut check) {
        check.passed = false;
        check.bit_exact = false;
        check.note(format!("returned an error: {e}"));
    }
    check.finish(kernel, promise)
}

// ── the checks, roughly in pipeline order ───────────────────────────────────
// Geometry and restoration first, then the tonal stages; the optional
// and analysis kernels (blur, glow, sharpen, quantize, histogram,
// apply_lut) come last so the table ends with the terminal ones.

#[allow(clippy::too_many_lines)]
fn check_all(ctx: &cuda::Context) -> Vec<Row> {
    use phaios_core::{
        blur, bw, denoise, encode, exposure, film_grain, geometry, glow, highlight_rolloff,
        histogram, hot_pixels, local_contrast, lut, quantize, shadow_rolloff, sharpen,
        split_toning, tone, vignette,
    };

    let rgb = pseudo_random_image(257, 389, 3);
    let mono = pseudo_random_image(257, 389, 1);
    let mut rows = Vec::new();

    // ── geometry: pure index permutations and polynomial filters ─────────

    rows.push(run("crop", EXACT, |ck| {
        let dev = ctx.upload(rgb.view())?;
        // (x, y) is a column/row offset, so the bounds are w <= 389 and
        // h <= 257 for this 257x389 frame. The last two are the corner
        // cases: a single pixel at the far corner, and the zero-size
        // rectangle the kernel documents as legal.
        for (x, y, w, h) in [
            (0, 0, 389, 257),
            (13, 7, 100, 200),
            (388, 256, 1, 1),
            (17, 23, 0, 0),
        ] {
            let params = geometry::CropParams::new(x, y, w, h);
            let cpu = geometry::crop(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::crop_device(&dev, &params)?)?;
            ck.exact(&format!("crop {x},{y} {w}x{h}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("orient", EXACT, |ck| {
        use geometry::Orientation::{
            FlipHorizontal, FlipVertical, Normal, Rotate90, Rotate180, Rotate270, Transpose,
            Transverse,
        };
        let dev = ctx.upload(rgb.view())?;
        for o in [
            Normal,
            FlipHorizontal,
            Rotate180,
            FlipVertical,
            Transpose,
            Rotate90,
            Transverse,
            Rotate270,
        ] {
            let cpu = geometry::orient(rgb.view(), o)?;
            let gpu = ctx.download(&cuda::kernels::orient_device(&dev, o)?)?;
            ck.exact(&format!("{o:?}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("resize", EXACT, |ck| {
        use geometry::ResizeFilter::{Area, Bilinear, CatmullRom};
        let dev = ctx.upload(rgb.view())?;
        for filter in [Area, Bilinear, CatmullRom] {
            // One downscale and one upscale: Area and CatmullRom take
            // different paths in each direction.
            for (w, h) in [(97_u32, 61_u32), (512, 600)] {
                let params = geometry::ResizeParams::new(w, h, filter);
                let cpu = geometry::resize(rgb.view(), &params)?;
                let gpu = ctx.download(&cuda::kernels::resize_device(&dev, &params)?)?;
                ck.exact(&format!("{filter:?} {w}x{h}"), &cpu, &gpu);
            }
        }
        Ok(())
    }));

    rows.push(run("straighten", EXACT, |ck| {
        let dev = ctx.upload(rgb.view())?;
        // sin/cos is computed once on the host and shared with the
        // device, which is what makes this one exact (§6).
        for degrees in [-45.0_f32, -12.5, 0.0, 7.25, 45.0] {
            let params = geometry::StraightenParams::new(degrees);
            let cpu = geometry::straighten(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::straighten_device(&dev, &params)?)?;
            ck.exact(&format!("{degrees} deg"), &cpu, &gpu);
        }
        Ok(())
    }));

    // ── hot-pixel removal and denoise: right after orient, before ─────────
    // ── straighten/resize in each kernel's own documented order ───────────

    rows.push(run("hot_pixels", EXACT, |ck| {
        let dev = ctx.upload(mono.view())?;
        for (threshold, relative) in [(0.1_f32, 0.0_f32), (0.05, 0.05)] {
            let params = hot_pixels::HotPixelParams::new(threshold, relative);
            let cpu = hot_pixels::hot_pixels(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::hot_pixels_device(&dev, &params)?)?;
            ck.exact(&format!("t={threshold} rel={relative}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("denoise", GUIDED_FILTER.label, |ck| {
        let dev_mono = ctx.upload(mono.view())?;
        let dev_rgb = ctx.upload(rgb.view())?;
        for (radius, amount) in [(1_u32, 0.5_f32), (8, 1.0)] {
            let params =
                denoise::DenoiseParams::new(radius, 0.05, amount, bw::LuminanceStandard::Bt709);

            let cpu = denoise::denoise(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::denoise_device(&dev_mono, &params)?)?;
            ck.bounded(
                &format!("C=1 r={radius} a={amount}"),
                &cpu,
                &gpu,
                GUIDED_FILTER,
            );

            let cpu3 = denoise::denoise(rgb.view(), &params)?;
            let gpu3 = ctx.download(&cuda::kernels::denoise_device(&dev_rgb, &params)?)?;
            ck.bounded(
                &format!("C=3 r={radius} a={amount}"),
                &cpu3,
                &gpu3,
                GUIDED_FILTER,
            );
        }
        Ok(())
    }));

    // ── exposure ─────────────────────────────────────────────────────────

    rows.push(run("exposure", EXACT, |ck| {
        // Ragged and degenerate shapes on purpose: last-block bounds
        // handling is where launch-geometry bugs live.
        for (h, w, c) in [
            (64, 64, 1),
            (257, 389, 3),
            (1, 1, 1),
            (1, 4096, 1),
            (517, 1, 3),
        ] {
            let img = pseudo_random_image(h, w, c);
            let dev = ctx.upload(img.view())?;
            for stops in [-3.0_f32, -0.75, 0.0, 0.5, 4.0] {
                let cpu = exposure::exposure(img.view(), stops)?;
                let gpu = ctx.download(&cuda::kernels::exposure_device(&dev, stops)?)?;
                ck.exact(&format!("{h}x{w}x{c} at {stops} EV"), &cpu, &gpu);
            }
        }
        Ok(())
    }));

    // ── the four B&W conversions ─────────────────────────────────────────

    rows.push(run("luminance_bw", EXACT, |ck| {
        use bw::LuminanceStandard::{Bt601, Bt709, Bt2020};
        let dev = ctx.upload(rgb.view())?;
        for standard in [Bt601, Bt709, Bt2020] {
            let cpu = bw::luminance_bw(rgb.view(), standard)?;
            let gpu = ctx.download(&cuda::kernels::luminance_bw_device(&dev, standard)?)?;
            ck.exact(&format!("{standard:?}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("channel_mixer_bw", EXACT, |ck| {
        let dev = ctx.upload(rgb.view())?;
        for weights in [
            [0.2126_f32, 0.7152, 0.0722],
            [1.0, 0.0, 0.0],
            [-0.2, 1.4, -0.2],
            [0.0, 0.0, 0.0],
        ] {
            let cpu = bw::channel_mixer_bw(rgb.view(), weights)?;
            let gpu = ctx.download(&cuda::kernels::channel_mixer_bw_device(&dev, weights)?)?;
            ck.exact(&format!("{weights:?}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("color_filter_bw", EXACT, |ck| {
        use bw::ColorFilter::{Blue47C5, Green11X1, NoFilter, Orange21, Red25A, Yellow8K2};
        let standard = bw::LuminanceStandard::Bt709;
        let dev = ctx.upload(rgb.view())?;
        for filter in [NoFilter, Yellow8K2, Orange21, Red25A, Green11X1, Blue47C5] {
            let cpu = bw::color_filter_bw(rgb.view(), filter, standard)?;
            let gpu = ctx.download(&cuda::kernels::color_filter_bw_device(
                &dev, filter, standard,
            )?)?;
            ck.exact(&format!("{filter:?}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("hsl_bw", ONE_TRANSCENDENTAL.label, |ck| {
        // Bounded, not exact: the hue-band weighting is a Gaussian, so
        // there is an expf per band.
        let dev = ctx.upload(rgb.view())?;
        for (weights, sigma) in [
            ([0.0_f32, 0.0, 0.4, 0.2, 0.0, -0.5, 0.0, 0.0], 30.0_f32),
            ([1.0, -1.0, 0.5, 0.0, 0.25, 0.0, -0.75, 0.1], 12.0),
        ] {
            let params = bw::HslWeightedParams::new(weights, bw::LuminanceStandard::Bt709, sigma);
            let cpu = bw::hsl_bw(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::hsl_bw_device(&dev, &params)?)?;
            ck.bounded(&format!("sigma {sigma}"), &cpu, &gpu, ONE_TRANSCENDENTAL);
        }
        Ok(())
    }));

    // ── zone system ──────────────────────────────────────────────────────

    rows.push(run("zone_system", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(mono.view())?;

        let params = tone::ZoneParams::new(
            [(2, 0.5_f32), (3, -0.3), (7, 0.4), (9, -0.6)]
                .into_iter()
                .collect(),
        );
        let cpu = tone::zone_system(mono.view(), &params)?;
        let gpu = ctx.download(&cuda::kernels::zone_system_device(&dev, &params)?)?;
        ck.bounded("four offsets", &cpu, &gpu, ONE_TRANSCENDENTAL);

        // The identity path does no zone arithmetic at all, so it is
        // exact on both sides.
        let identity = tone::ZoneParams::new(std::collections::HashMap::new());
        let cpu = tone::zone_system(mono.view(), &identity)?;
        let gpu = ctx.download(&cuda::kernels::zone_system_device(&dev, &identity)?)?;
        ck.exact("empty offset map", &cpu, &gpu);
        Ok(())
    }));

    // ── local contrast ───────────────────────────────────────────────────

    rows.push(run("local_contrast", GUIDED_FILTER.label, |ck| {
        let dev = ctx.upload(mono.view())?;
        for (radius, eps, strength) in [(1_u32, 0.1_f32, 2.0_f32), (8, 0.01, 0.5), (64, 0.01, 0.7)]
        {
            let params = local_contrast::GuidedFilterParams::new(radius, eps);
            let cpu = local_contrast::local_contrast(mono.view(), &params, strength)?;
            let gpu = ctx.download(&cuda::kernels::local_contrast_device(
                &dev, &params, strength,
            )?)?;
            ck.bounded(
                &format!("r={radius} eps={eps} s={strength}"),
                &cpu,
                &gpu,
                GUIDED_FILTER,
            );
        }

        // The case that caught a real bug: a bright bar on a dark field,
        // where the variance is a subtraction of two ~1e12 numbers whose
        // true difference is near zero. Random [0, 1) input never
        // reaches this regime.
        let mut bar = Array3::<f32>::from_elem((81, 97, 1), 1e-4_f32);
        bar.slice_mut(s![30..34, 10..80, ..]).fill(1e6);
        let dev_bar = ctx.upload(bar.view())?;
        for radius in [1_u32, 2, 8] {
            let params = local_contrast::GuidedFilterParams::new(radius, 0.5);
            let cpu = local_contrast::local_contrast(bar.view(), &params, 0.5)?;
            let gpu = ctx.download(&cuda::kernels::local_contrast_device(
                &dev_bar, &params, 0.5,
            )?)?;
            ck.bounded(
                &format!("1e6 highlight bar, r={radius}"),
                &cpu,
                &gpu,
                GUIDED_FILTER,
            );
        }
        Ok(())
    }));

    // ── film grain ───────────────────────────────────────────────────────

    rows.push(run("film_grain", BOX_MULLER.label, |ck| {
        let dev = ctx.upload(mono.view())?;
        for (intensity, size, seed) in [
            (0.3_f32, 2.0_f32, 12_345_u64),
            (0.15, 1.0, 1),
            (1.0, 8.0, 7),
        ] {
            let params = film_grain::GrainParams::new(intensity, size, seed);
            let cpu = film_grain::film_grain(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::film_grain_device(&dev, &params)?)?;
            ck.bounded(
                &format!("i={intensity} size={size} seed={seed}"),
                &cpu,
                &gpu,
                BOX_MULLER,
            );
        }
        // Zero intensity is the identity, and identities are exact.
        let off = film_grain::GrainParams::new(0.0, 2.0, 7);
        let cpu = film_grain::film_grain(mono.view(), &off)?;
        let gpu = ctx.download(&cuda::kernels::film_grain_device(&dev, &off)?)?;
        ck.exact("intensity 0", &cpu, &gpu);
        Ok(())
    }));

    // ── split toning: the channel count returns here ─────────────────────

    rows.push(run("split_toning", ONE_TRANSCENDENTAL.label, |ck| {
        // Bounded: OKLab is a cbrt round trip.
        let dev = ctx.upload(mono.view())?;
        for (shadow, highlight, pivot, balance) in [
            (
                [0.0_f32, -0.02, -0.05],
                [0.0_f32, 0.03, 0.04],
                0.5_f32,
                0.2_f32,
            ),
            ([0.0, 0.05, -0.03], [0.0, -0.04, 0.06], 0.35, -0.5),
        ] {
            let params = split_toning::SplitToningParams::new(shadow, highlight, pivot, balance);
            let cpu = split_toning::split_toning(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::split_toning_device(&dev, &params)?)?;
            ck.bounded(
                &format!("pivot {pivot} balance {balance}"),
                &cpu,
                &gpu,
                ONE_TRANSCENDENTAL,
            );
        }
        Ok(())
    }));

    // ── vignette ─────────────────────────────────────────────────────────

    rows.push(run("vignette", EXACT, |ck| {
        let dev = ctx.upload(rgb.view())?;
        // amount may be negative (a brightening vignette); feather and
        // roundness are both confined to 0..=1.
        for (amount, feather, roundness) in [
            (0.5_f32, 1.0_f32, 0.0_f32),
            (0.5, 0.2, 1.0),
            (-0.8, 0.6, 0.5),
            (0.0, 1.0, 0.0),
        ] {
            let params = vignette::VignetteParams::new(amount, feather, roundness);
            let cpu = vignette::vignette(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::vignette_device(&dev, &params)?)?;
            ck.exact(&format!("a={amount} f={feather} r={roundness}"), &cpu, &gpu);
        }
        Ok(())
    }));

    // ── the characteristic curve: toe, straight section, shoulder ────────

    rows.push(run("shadow_rolloff", EXACT, |ck| {
        // A cubic in Horner form: multiply, add, subtract, one divide.
        let dev = ctx.upload(rgb.view())?;
        for (knee, strength) in [(0.1_f32, 1.0_f32), (0.25, 0.5), (0.1, 0.0)] {
            let params = shadow_rolloff::ShadowRolloffParams::new(knee, strength);
            let cpu = shadow_rolloff::shadow_rolloff(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::shadow_rolloff_device(&dev, &params)?)?;
            ck.exact(&format!("knee={knee} strength={strength}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("tone_curve", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(rgb.view())?;
        // power == 1 skips the powf entirely, so §6 promises exactness.
        for (slope, offset) in [(1.0_f32, 0.0_f32), (1.4, -0.05), (0.7, 0.1)] {
            let params = tone::ToneCurveParams::new(slope, offset, 1.0);
            let cpu = tone::tone_curve(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::tone_curve_device(&dev, &params)?)?;
            ck.exact(
                &format!("power 1, slope {slope} offset {offset}"),
                &cpu,
                &gpu,
            );
        }
        for power in [0.45_f32, 0.9, 2.2] {
            let params = tone::ToneCurveParams::new(1.1, 0.02, power);
            let cpu = tone::tone_curve(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::tone_curve_device(&dev, &params)?)?;
            ck.bounded(&format!("power {power}"), &cpu, &gpu, ONE_TRANSCENDENTAL);
        }
        Ok(())
    }));

    rows.push(run("highlight_rolloff", EXACT, |ck| {
        // A quadratic solve; IEEE-754-2008 §5.4.1 requires sqrt to be
        // correctly rounded, so the whole curve carries across.
        let dev = ctx.upload(rgb.view())?;
        for (knee, white) in [(0.8_f32, 1.0_f32), (0.5, 2.0), (0.0, 1.0)] {
            let params = highlight_rolloff::RolloffParams::new(knee, white);
            let cpu = highlight_rolloff::highlight_rolloff(rgb.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::highlight_rolloff_device(&dev, &params)?)?;
            ck.exact(&format!("knee={knee} white={white}"), &cpu, &gpu);
        }
        Ok(())
    }));

    // ── blur, glow and sharpen ──────────────────────────────────────────

    rows.push(run("blur", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(mono.view())?;
        // Both paths: below BOX_CROSSOVER_SIGMA the transfer is a direct
        // separable convolution, at or above it three box passes.
        for sigma in [2.5_f32, 8.0] {
            let params = blur::BlurParams::new(sigma, blur::BlurShape::Gaussian);
            let cpu = blur::blur(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::blur_device(&dev, &params)?)?;
            ck.bounded(&format!("sigma {sigma}"), &cpu, &gpu, ONE_TRANSCENDENTAL);
        }
        let identity = blur::BlurParams::new(0.0, blur::BlurShape::Gaussian);
        let cpu = blur::blur(mono.view(), &identity)?;
        let gpu = ctx.download(&cuda::kernels::blur_device(&dev, &identity)?)?;
        ck.exact("sigma 0", &cpu, &gpu);
        Ok(())
    }));

    rows.push(run("glow", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(mono.view())?;
        for (threshold, sigma, amount) in [(0.5_f32, 4.0_f32, 0.6_f32), (0.2, 12.0, 0.3)] {
            let params = glow::GlowParams::new(threshold, sigma, amount);
            let cpu = glow::glow(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::glow_device(&dev, &params)?)?;
            ck.bounded(
                &format!("t={threshold} sigma={sigma} a={amount}"),
                &cpu,
                &gpu,
                ONE_TRANSCENDENTAL,
            );
        }
        let off = glow::GlowParams::new(0.5, 4.0, 0.0);
        let cpu = glow::glow(mono.view(), &off)?;
        let gpu = ctx.download(&cuda::kernels::glow_device(&dev, &off)?)?;
        ck.exact("amount 0", &cpu, &gpu);
        Ok(())
    }));

    rows.push(run("sharpen", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(mono.view())?;
        // threshold 0 and non-zero; sigma below and above blur's box
        // crossover (5.9 direct / 6.0 box), so both device blur paths
        // are exercised through sharpen's own entry point.
        for (amount, sigma, threshold) in [(0.6_f32, 2.5_f32, 0.0_f32), (0.4, 12.0, 0.3)] {
            let params = sharpen::SharpenParams::new(amount, sigma, threshold);
            let cpu = sharpen::sharpen(mono.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::sharpen_device(&dev, &params)?)?;
            ck.bounded(
                &format!("a={amount} sigma={sigma} t={threshold}"),
                &cpu,
                &gpu,
                ONE_TRANSCENDENTAL,
            );
        }
        let off = sharpen::SharpenParams::new(0.0, 2.5, 0.1);
        let cpu = sharpen::sharpen(mono.view(), &off)?;
        let gpu = ctx.download(&cuda::kernels::sharpen_device(&dev, &off)?)?;
        ck.exact("amount 0", &cpu, &gpu);
        Ok(())
    }));

    // ── the display-referred tail ────────────────────────────────────────

    rows.push(run("encode_srgb", ONE_TRANSCENDENTAL.label, |ck| {
        let dev = ctx.upload(rgb.view())?;
        let cpu = encode::encode_srgb(rgb.view())?;
        let gpu = ctx.download(&cuda::kernels::encode_srgb_device(&dev)?)?;
        ck.bounded("random [0, 1)", &cpu, &gpu, ONE_TRANSCENDENTAL);

        // A ramp straddling 0.0031308, where the transfer switches from
        // the linear segment to the power segment. It is C-zero there
        // but not C-one, so it is the interesting neighbourhood.
        let ramp = Array3::from_shape_fn((1, 4096, 1), |(_, x, _)| x as f32 * (0.01 / 4095.0));
        let dev_ramp = ctx.upload(ramp.view())?;
        let cpu = encode::encode_srgb(ramp.view())?;
        let gpu = ctx.download(&cuda::kernels::encode_srgb_device(&dev_ramp)?)?;
        ck.bounded("ramp across 0.0031308", &cpu, &gpu, ONE_TRANSCENDENTAL);
        Ok(())
    }));

    // Display-referred input for the terminal kernels, deliberately
    // spilling outside [0, 1] so the clamping paths are exercised too.
    let display = rgb.mapv(|v| v * 1.2 - 0.1);

    rows.push(run("quantize_u8", EXACT, |ck| {
        let dev = ctx.upload(display.view())?;
        for (dither, seed) in [
            (quantize::Dither::Off, 0_u64),
            (quantize::Dither::Tpdf, 20_260_827),
        ] {
            let params = quantize::QuantizeParams::new(dither, seed);
            let cpu = quantize::quantize_u8(display.view(), &params)?;
            let gpu = cuda::kernels::quantize_u8_device(&dev, &params)?;
            ck.exact_codes(&format!("{dither:?} seed={seed}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows.push(run("quantize_u16", EXACT, |ck| {
        let dev = ctx.upload(display.view())?;
        for (dither, seed) in [
            (quantize::Dither::Off, 0_u64),
            (quantize::Dither::Tpdf, 20_260_827),
        ] {
            let params = quantize::QuantizeParams::new(dither, seed);
            let cpu = quantize::quantize_u16(display.view(), &params)?;
            let gpu = cuda::kernels::quantize_u16_device(&dev, &params)?;
            ck.exact_codes(&format!("{dither:?} seed={seed}"), &cpu, &gpu);
        }
        Ok(())
    }));

    // ── the reduction and the table lookup ───────────────────────────────

    rows.push(run("histogram", EXACT, |ck| {
        // Exact because the only float arithmetic is the bin
        // assignment; everything after it is integer counting, and
        // integer addition commutes whatever order the atomics land in.
        let dev = ctx.upload(display.view())?;
        for bins in [2_u32, 256, 4096, 65_536] {
            let params = histogram::HistogramParams::new(bins, 0.0, 1.0);
            let cpu = histogram::histogram(display.view(), &params)?;
            let gpu = cuda::kernels::histogram_device(&dev, &params)?;
            let agrees = cpu.counts() == gpu.counts()
                && cpu.below() == gpu.below()
                && cpu.above() == gpu.above()
                && cpu.non_finite() == gpu.non_finite();
            // Bin counts plus the three out-of-range tallies: what this
            // case actually compares, since it is not an array comparison.
            ck.exact_flag(
                &format!("{bins} bins"),
                agrees,
                cpu.counts().len() + 3,
                "counts, below, above or non_finite differ",
            );
        }

        // Non-finite and out-of-range samples must be classified the
        // same way on both backends, not merely counted the same.
        let edge = ndarray::array![[
            [-1.0_f32, 0.0, 1.0],
            [1.5, f32::INFINITY, f32::NEG_INFINITY],
            [f32::NAN, 0.5, 2.0],
        ]];
        let dev_edge = ctx.upload(edge.view())?;
        let params = histogram::HistogramParams::new(16, 0.0, 1.0);
        let cpu = histogram::histogram(edge.view(), &params)?;
        let gpu = cuda::kernels::histogram_device(&dev_edge, &params)?;
        let agrees = cpu.counts() == gpu.counts()
            && cpu.below() == gpu.below()
            && cpu.above() == gpu.above()
            && cpu.non_finite() == gpu.non_finite();
        ck.exact_flag(
            "NaN and infinity classification",
            agrees,
            cpu.counts().len() + 3,
            "classified differently",
        );
        Ok(())
    }));

    rows.push(run("apply_lut", EXACT, |ck| {
        let dev = ctx.upload(display.view())?;
        let table = Array1::from_shape_fn(256, |i| (i as f32 / 255.0).powf(0.45));
        for (min, max) in [(0.0_f32, 1.0_f32), (-0.1, 1.2)] {
            let params = lut::LutParams::new(min, max);
            let cpu = lut::apply_lut(display.view(), table.view(), &params)?;
            let gpu = ctx.download(&cuda::kernels::apply_lut_device(
                &dev,
                table.view(),
                &params,
            )?)?;
            ck.exact(&format!("range {min}..{max}"), &cpu, &gpu);
        }
        Ok(())
    }));

    rows
}

// ── reporting ────────────────────────────────────────────────────────────────

/// CUDA driver version, as the driver itself reports it.
///
/// §6 makes a driver update something that can move these numbers, so a
/// pasted report has to name it. `cudarc` wraps this call only at the
/// `sys` level, which is why it is made directly; it is reached only
/// after `cuda::devices()` has already loaded the driver successfully.
fn driver_version() -> String {
    let mut version: std::ffi::c_int = 0;
    // Safety: cuDriverGetVersion writes a single `int` through the
    // pointer and touches nothing else. The pointer is to a live local.
    let rc = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
    if rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        format!("{}.{}", version / 1000, (version % 1000) / 10)
    } else {
        "unknown".to_string()
    }
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<20} {:>5}  {:<9}  {:>9}  {:>9}  result",
        "kernel", "cases", "promise", "bit-exact", "worst*bnd"
    );
    println!("{}", "-".repeat(68));
    for row in rows {
        let worst = match row.worst {
            None => "-".to_string(),
            Some(v) if v.is_infinite() => "inf".to_string(),
            Some(v) => format!("{v:.3}"),
        };
        println!(
            "{:<20} {:>5}  {:<9}  {:>9}  {:>9}  {}",
            row.kernel,
            row.cases,
            row.promise,
            if row.bit_exact { "yes" } else { "no" },
            worst,
            if row.pass { "PASS" } else { "FAIL" }
        );
    }
    println!("{}", "-".repeat(68));
}

fn main() -> ExitCode {
    let require_gpu = std::env::var_os("PHAIOS_REQUIRE_GPU").is_some();

    println!("phaios-core GPU self-test");
    println!("crate version : {}", env!("CARGO_PKG_VERSION"));

    let devices = cuda::devices();
    if devices.is_empty() {
        println!("\nno CUDA device found: no driver, no card, or a broken installation.");
        println!("nothing was verified.");
        if require_gpu {
            println!("PHAIOS_REQUIRE_GPU is set, so a missing device is itself a failure.");
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }

    println!("cuda driver   : {}", driver_version());
    for d in &devices {
        println!(
            "device {}      : {} (cc {}.{}, supported: {})",
            d.ordinal, d.name, d.compute_capability.0, d.compute_capability.1, d.supported
        );
    }

    let ctx = match cuda::Context::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("\ndevice present but unusable: {e}");
            println!("nothing was verified.");
            if require_gpu {
                println!("PHAIOS_REQUIRE_GPU is set, so an unusable device is a failure.");
                return ExitCode::FAILURE;
            }
            return ExitCode::SUCCESS;
        }
    };
    println!("fingerprint   : {}", ctx.fingerprint());
    println!();

    let rows = check_all(&ctx);
    print_table(&rows);

    let failed: Vec<&Row> = rows.iter().filter(|r| !r.pass).collect();
    let cases: usize = rows.iter().map(|r| r.cases).sum();
    let exact = rows.iter().filter(|r| r.bit_exact).count();

    if !failed.is_empty() {
        println!("\nfailures:");
        for row in &failed {
            let detail = row.detail.as_deref().unwrap_or("no detail recorded");
            println!("  {:<20} {detail}", row.kernel);
        }
    }

    println!("\nnotes:");
    for (kernel, note) in EXTRA_NOTES {
        println!("  {kernel:<20} {note}");
    }

    println!("\nlegend:");
    println!("  promise    the agreement class docs/ffi.md section 6 commits for the kernel.");
    println!("  bit-exact  whether every case matched the CPU byte for byte.");
    println!("  worst*bnd  worst element as a multiple of the bound; <= 1.000 passes.");
    println!("             for a bit-exact row this is diagnostic only, measured against");
    println!("             (1e-5, 1e-7), and reads 0.000 whenever the row passes.");
    println!("  elements   how many were actually compared. A comparison over an empty");
    println!("             array also reports 0.000, so this is what separates \"agreed");
    println!("             everywhere\" from \"looked at nothing\" — `crop` to a 0x0");
    println!("             output is a legitimate case that compares no elements.");

    let compared: usize = rows.iter().map(|r| r.compared).sum();
    println!("\nelements compared: {compared}");
    let thin: Vec<&str> = rows
        .iter()
        .filter(|r| r.compared == 0)
        .map(|r| r.kernel)
        .collect();
    if !thin.is_empty() {
        println!("rows that compared nothing at all: {}", thin.join(", "));
    }
    println!(
        "\nsummary: {} kernels, {} pass, {} fail, {cases} cases, {exact} bit-exact",
        rows.len(),
        rows.len() - failed.len(),
        failed.len()
    );
    println!("fingerprint: {}", ctx.fingerprint());

    if failed.is_empty() {
        println!("result: PASS");
        ExitCode::SUCCESS
    } else {
        println!("result: FAIL");
        ExitCode::FAILURE
    }
}
