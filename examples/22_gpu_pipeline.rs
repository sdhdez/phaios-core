// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 22 — The device-resident chain: upload once, download once.
//!
//! Example 15 runs a *single* kernel through the per-call offload form
//! (`upload, run, download`). That form is convenient and, for a
//! pipeline, the wrong shape: a chain of n kernels pays n PCIe round
//! trips to compute a result that only needed two crossings.
//!
//! This example runs the whole v0.2 chain the other way — one
//! [`Context::upload`], eleven `*_device` kernels whose inputs and
//! outputs never leave the card, one [`Context::download`] — in the
//! this order:
//!
//! ```text
//! exposure → luminance_bw → zone_system → local_contrast → film_grain
//!   → split_toning → vignette → shadow_rolloff → tone_curve
//!   → highlight_rolloff → encode_srgb
//! ```
//!
//! That is not the canonical pipeline order, which puts
//! `shadow_rolloff` and `tone_curve` before `film_grain` and
//! `split_toning`. What this example demonstrates is residency, and the
//! drift table below is read the same way whichever order the eleven
//! stages run in.
//!
//! The same eleven stages are then written out twice more — once on the
//! CPU, once as per-call offload — because the whole point is the
//! comparison between the three.
//!
//! What to look for in the output:
//!
//! - **The drift table.** After every stage the device image is pulled
//!   back and compared with the CPU chain at the same point, so each row
//!   is *cumulative*: it carries everything upstream of it, and a
//!   bit-exact kernel downstream of a divergent one still prints a
//!   difference. Divergence enters at `local_contrast`, where the device
//!   replaces the CPU's global f64 summed-area tables with separable
//!   window sums: f64 for the L and L² statistics, Kahan-compensated f32
//!   for the a/b coefficient sums downstream. It is a few ULP there and it is still a few
//!   ULP eight stages later: the middle column, the committed
//!   whole-chain bound as a multiple, barely moves. **That** is the
//!   property residency needs — a long device-resident chain is no less
//!   faithful than a short one.
//! - **The first three rows, and why they prove less than they look
//!   like.** `exposure` (one multiply) and `luminance_bw` (one dot
//!   product) are documented bit-exact, so their zeros are a promise.
//!   `zone_system`'s is not: it evaluates `log2f` and `expf`, which are
//!   implementation-defined. The chart has only 24 distinct patch
//!   values and the two libms happen to agree exactly on all of them.
//!   On photographic input that row would show a few ULP like the rest.
//! - **The 8-bit verdict.** Agreement in f32 is not the question a
//!   photographer asks. The example quantises both chains the way
//!   `write_ppm` does and counts how many of the 294 912 codes differ,
//!   and by how much — a much smaller number than the f32 count above
//!   it, because a few ULP almost never crosses a rounding boundary.
//!   `22_pipeline_diff.ppm` is the map of the f32 disagreement: white
//!   wherever the two backends differ by so much as one ULP, black
//!   where they match to the bit. It is where the drift lives, and it
//!   follows the grain and the patch edges rather than being uniform.
//! - **The transfer economics.** At 24 MP the resident chain moves six
//!   floats per pixel across the bus and the offload chain fifty — the
//!   same arithmetic, eight times the traffic and eleven times the
//!   synchronisation. The timings below are measured, not modelled.
//!
//! `22_pipeline_gpu.ppm` and `22_pipeline_cpu.ppm` are the two results.
//! They end in `encode_srgb`, so they are already display-referred and
//! are written with the raw `write_ppm` rather than the `*_display`
//! helpers, which would apply the transfer a second time.
//!
//! With no CUDA device (or no NVIDIA driver at all) the example prints
//! why and exits 0, exactly as example 15 does.

#[path = "shared/mod.rs"]
mod shared;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use ndarray::{Array3, ArrayView3};
use phaios_core::bw::LuminanceStandard;
use phaios_core::cuda::{self, Context, DeviceImage, kernels as k};
use phaios_core::film_grain::GrainParams;
use phaios_core::highlight_rolloff::RolloffParams;
use phaios_core::local_contrast::GuidedFilterParams;
use phaios_core::shadow_rolloff::ShadowRolloffParams;
use phaios_core::split_toning::SplitToningParams;
use phaios_core::tone::{ToneCurveParams, ZoneParams};
use phaios_core::vignette::VignetteParams;

/// Stage names, in the order the three chains below apply them.
const STAGES: [&str; 11] = [
    "exposure",
    "luminance_bw",
    "zone_system",
    "local_contrast",
    "film_grain",
    "split_toning",
    "vignette",
    "shadow_rolloff",
    "tone_curve",
    "highlight_rolloff",
    "encode_srgb",
];

/// Export size, the same 24 MP frame example 15 and `benches/gpu.rs`
/// measure on, so the numbers here can be divided by theirs.
const BIG_H: usize = 4323;
const BIG_W: usize = 5765;

/// Every parameter the chain needs, built once and shared by all three
/// spellings of it. A real consumer does the same: the orchestrator in
/// the desktop app builds these once per pipeline run.
struct Grade {
    stops: f32,
    standard: LuminanceStandard,
    zones: ZoneParams,
    guided: GuidedFilterParams,
    strength: f32,
    grain: GrainParams,
    toning: SplitToningParams,
    vignette: VignetteParams,
    toe: ShadowRolloffParams,
    curve: ToneCurveParams,
    shoulder: RolloffParams,
}

impl Grade {
    /// A plausible print: half a stop up, a zone-system push with
    /// deepened shadows, moderate local contrast, fine grain, a selenium
    /// split, a soft vignette, and a toe/straight/shoulder that lands
    /// the whites below clipping.
    fn new() -> Self {
        let mut zones = HashMap::new();
        zones.insert(3_i32, -0.3_f32);
        zones.insert(8_i32, 0.4_f32);
        Self {
            stops: 0.5,
            standard: LuminanceStandard::Bt709,
            zones: ZoneParams::new(zones),
            guided: GuidedFilterParams::new(8, 0.01),
            strength: 0.6,
            grain: GrainParams::new(0.12, 1.5, 20_260_827),
            toning: SplitToningParams::new([0.0, -0.02, -0.04], [0.0, 0.02, 0.03], 0.5, 0.0),
            vignette: VignetteParams::new(0.35, 0.8, 0.1),
            toe: ShadowRolloffParams::new(0.18, 0.7),
            curve: ToneCurveParams::new(1.1, 0.0, 0.95),
            shoulder: RolloffParams::new(0.75, 3.0),
        }
    }
}

// ── The chain, three ways ────────────────────────────────────────────────────
//
// The stage list is written out three times on purpose. The differences
// between the three spellings — what each one allocates, and where the
// bus crossings sit — are the subject of this example, so collapsing
// them behind a shared abstraction would hide it.

/// The CPU chain. `tap` sees each intermediate, in stage order.
fn cpu_chain(
    img: ArrayView3<f32>,
    g: &Grade,
    tap: &mut dyn FnMut(usize, &Array3<f32>),
) -> Array3<f32> {
    let mut a = phaios_core::exposure::exposure(img, g.stops).unwrap();
    tap(0, &a);
    a = phaios_core::bw::luminance_bw(a.view(), g.standard).unwrap();
    tap(1, &a);
    a = phaios_core::tone::zone_system(a.view(), &g.zones).unwrap();
    tap(2, &a);
    a = phaios_core::local_contrast::local_contrast(a.view(), &g.guided, g.strength).unwrap();
    tap(3, &a);
    a = phaios_core::film_grain::film_grain(a.view(), &g.grain).unwrap();
    tap(4, &a);
    a = phaios_core::split_toning::split_toning(a.view(), &g.toning).unwrap();
    tap(5, &a);
    a = phaios_core::vignette::vignette(a.view(), &g.vignette).unwrap();
    tap(6, &a);
    a = phaios_core::shadow_rolloff::shadow_rolloff(a.view(), &g.toe).unwrap();
    tap(7, &a);
    a = phaios_core::tone::tone_curve(a.view(), &g.curve).unwrap();
    tap(8, &a);
    a = phaios_core::highlight_rolloff::highlight_rolloff(a.view(), &g.shoulder).unwrap();
    tap(9, &a);
    a = phaios_core::encode::encode_srgb(a.view()).unwrap();
    tap(10, &a);
    a
}

/// The device-resident chain: eleven kernels, no transfer between them.
///
/// Each `a = ...(&a, ...)` drops the previous stage's buffer as soon as
/// the next one exists, so the device working set is two images, not
/// eleven — the other half of what makes residency worth having.
///
/// `tap` is called with each intermediate *while it is still on the
/// device*. Passing a no-op tap leaves the chain exactly as a consumer
/// would write it; the drift table below passes one that downloads, and
/// so is deliberately kept out of the timed path.
fn gpu_chain(
    img: &DeviceImage,
    g: &Grade,
    tap: &mut dyn FnMut(usize, &DeviceImage),
) -> DeviceImage {
    let mut a = k::exposure_device(img, g.stops).unwrap();
    tap(0, &a);
    a = k::luminance_bw_device(&a, g.standard).unwrap();
    tap(1, &a);
    a = k::zone_system_device(&a, &g.zones).unwrap();
    tap(2, &a);
    a = k::local_contrast_device(&a, &g.guided, g.strength).unwrap();
    tap(3, &a);
    a = k::film_grain_device(&a, &g.grain).unwrap();
    tap(4, &a);
    a = k::split_toning_device(&a, &g.toning).unwrap();
    tap(5, &a);
    a = k::vignette_device(&a, &g.vignette).unwrap();
    tap(6, &a);
    a = k::shadow_rolloff_device(&a, &g.toe).unwrap();
    tap(7, &a);
    a = k::tone_curve_device(&a, &g.curve).unwrap();
    tap(8, &a);
    a = k::highlight_rolloff_device(&a, &g.shoulder).unwrap();
    tap(9, &a);
    a = k::encode_srgb_device(&a).unwrap();
    tap(10, &a);
    a
}

/// The same chain through the per-call offload entry points: every
/// stage uploads its input and downloads its result. Same arithmetic,
/// eleven round trips instead of one.
fn gpu_offload_chain(ctx: &Context, img: ArrayView3<f32>, g: &Grade) -> Array3<f32> {
    let mut a = k::exposure(ctx, img, g.stops).unwrap();
    a = k::luminance_bw(ctx, a.view(), g.standard).unwrap();
    a = k::zone_system(ctx, a.view(), &g.zones).unwrap();
    a = k::local_contrast(ctx, a.view(), &g.guided, g.strength).unwrap();
    a = k::film_grain(ctx, a.view(), &g.grain).unwrap();
    a = k::split_toning(ctx, a.view(), &g.toning).unwrap();
    a = k::vignette(ctx, a.view(), &g.vignette).unwrap();
    a = k::shadow_rolloff(ctx, a.view(), &g.toe).unwrap();
    a = k::tone_curve(ctx, a.view(), &g.curve).unwrap();
    a = k::highlight_rolloff(ctx, a.view(), &g.shoulder).unwrap();
    k::encode_srgb(ctx, a.view()).unwrap()
}

// ── Measurement ──────────────────────────────────────────────────────────────

/// Worst violation of `|x − y| <= atol + rtol·|x|`, as a multiple of the
/// bound: <= 1.0 means every element passes.
///
/// The same two-term form `tests/cuda_conformance.rs` uses, including
/// its NaN handling — a metric built on `f32::max` over a `zip` scores a
/// wholly-NaN or truncated output as a perfect match, which is how the
/// conformance oracle was wrong before it was fixed.
fn worst_violation(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32) -> f32 {
    assert_eq!(
        a.dim(),
        b.dim(),
        "shape mismatch: {:?} vs {:?}",
        a.dim(),
        b.dim()
    );
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let v = if x.is_nan() || y.is_nan() {
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
        if v > worst {
            worst = v;
        }
    }
    worst
}

/// Largest absolute difference, element for element.
///
/// Reduced with a plain `>` rather than `f32::max`, for the reason given
/// above: `f32::max` returns the *other* operand when one side is NaN,
/// so a fold over it reports a NaN-filled output as a perfect match.
fn max_abs_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    let mut worst = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > worst || d.is_nan() {
            worst = d;
        }
    }
    worst
}

/// The 8-bit code `shared::write_ppm` would emit for a value.
fn code(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Best of `reps` wall-clock timings, in milliseconds.
///
/// Best-of rather than mean: the quantity of interest is the cost of the
/// work, and every source of noise on a shared machine adds time.
fn best_ms(reps: usize, f: &mut dyn FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
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
    let grade = Grade::new();

    // ── The chart, both backends, stage by stage ─────────────────────
    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();

    let mut cpu_stages: Vec<Array3<f32>> = Vec::with_capacity(STAGES.len());
    let cpu = cpu_chain(rgb.view(), &grade, &mut |_, a| cpu_stages.push(a.clone()));

    // The resident chain, tapped after each stage. The tap downloads,
    // which is the one thing a resident chain exists not to do — it is
    // here to measure the chain, and the timing section below runs the
    // same function with a tap that does nothing.
    let uploaded = ctx.upload(rgb.view()).unwrap();
    let mut drift: Vec<(f32, f32, usize)> = Vec::with_capacity(STAGES.len());
    let mut channels: Vec<usize> = Vec::with_capacity(STAGES.len());
    let result = gpu_chain(&uploaded, &grade, &mut |i, d| {
        let host = ctx.download(d).unwrap();
        channels.push(host.dim().2);
        drift.push((
            max_abs_diff(&cpu_stages[i], &host),
            worst_violation(&cpu_stages[i], &host, 1e-3, 1e-5),
            host.iter()
                .zip(cpu_stages[i].iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count(),
        ));
    });
    let gpu = ctx.download(&result).unwrap();

    println!("eleven stages, one upload, one download — drift against the CPU chain");
    println!(
        "{:>2}  {:<18} {:>11} {:>10} {:>17}",
        "#", "after stage", "max |delta|", "bound", "elements differing"
    );
    for (i, name) in STAGES.iter().enumerate() {
        let (abs, viol, differing) = drift[i];
        let share = format!("{differing}/{}", cpu_stages[i].len());
        let note = if differing == 0 { "  (exact)" } else { "" };
        println!(
            "{:>2}  {name:<18} {abs:>11.3e} {viol:>10.4} {share:>17}{note}",
            i + 1
        );
    }
    println!(
        "  bound is the committed whole-chain (rtol 1e-3, atol 1e-5) from\n  \
         tests/cuda_conformance.rs, as a multiple: 1.0000 is the limit"
    );

    // ── The verdict in 8-bit codes, which is what reaches a file ─────
    let codes_differ = cpu
        .iter()
        .zip(gpu.iter())
        .filter(|(c, g)| code(**c) != code(**g))
        .count();
    let worst_code = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(c, g)| code(*c).abs_diff(code(*g)))
        .max()
        .unwrap_or(0);
    println!(
        "\n8-bit codes after encode_srgb: {codes_differ} of {} differ, \
         largest difference {worst_code} code(s)",
        cpu.len()
    );

    // ── Output ───────────────────────────────────────────────────────
    // Already display-referred (encode_srgb was stage 11), so write the
    // values as they are rather than through write_ppm_*_display.
    let write = |name: &str, img: &Array3<f32>| {
        let path = format!("examples/output/22_pipeline_{name}.ppm");
        shared::write_ppm(
            Path::new(&path),
            img.as_slice().expect("kernel output is contiguous"),
            shared::WIDTH,
            shared::HEIGHT,
        );
    };
    write("gpu", &gpu);
    write("cpu", &cpu);

    // White wherever the two backends disagree in f32 by so much as one
    // ULP, black where they match to the bit. Not an amplified
    // difference: the magnitudes are a few ULP everywhere and would
    // amplify to a uniform grey, where the *position* is what carries
    // information.
    let mut map = Array3::<f32>::zeros(cpu.dim());
    ndarray::Zip::from(&mut map)
        .and(&cpu)
        .and(&gpu)
        .for_each(|m, c, g| *m = f32::from(c.to_bits() != g.to_bits()));
    write("diff", &map);
    println!("wrote examples/output/22_pipeline_{{gpu,cpu,diff}}.ppm");

    // ── Transfer economics at 24 MP ──────────────────────────────────
    let n = BIG_H * BIG_W;
    let resident_floats = n * 3 + n * channels[STAGES.len() - 1];
    let mut offload_floats = 0_usize;
    let mut prev = 3_usize;
    for &c in &channels {
        offload_floats += n * prev + n * c;
        prev = c;
    }

    let big = Array3::<f32>::from_shape_fn((BIG_H, BIG_W, 3), |(y, x, c)| {
        // A gradient with a bright corner, so the tone stages and the
        // grain envelope all have something to do. The content does not
        // affect timing — every kernel here is branch-free per pixel —
        // but a constant image would be a poor advertisement.
        (x as f32 / BIG_W as f32) * 0.8 + (y as f32 / BIG_H as f32) * 0.2 + c as f32 * 0.02
    });

    // Warm the module cache and the JIT so the first PTX load is not
    // charged to the first timed run.
    {
        let d = ctx.upload(big.view()).unwrap();
        let out = gpu_chain(&d, &grade, &mut |_, _| {});
        let _ = ctx.download(&out).unwrap();
    }

    // Kernel launches are asynchronous, so a timer around the eleven
    // `*_device` calls alone would measure the launch queue. The final
    // `download` synchronises the stream, which is why it is inside the
    // timed region — and why one-synchronisation-per-frame is the
    // pattern the resident API is for.
    let resident_ms = best_ms(3, &mut || {
        let d = ctx.upload(big.view()).unwrap();
        let out = gpu_chain(&d, &grade, &mut |_, _| {});
        std::hint::black_box(ctx.download(&out).unwrap());
    });
    let offload_ms = best_ms(3, &mut || {
        std::hint::black_box(gpu_offload_chain(&ctx, big.view(), &grade));
    });
    let cpu_ms = best_ms(2, &mut || {
        std::hint::black_box(cpu_chain(big.view(), &grade, &mut |_, _| {}));
    });

    let mib = |floats: usize| floats as f64 * 4.0 / (1024.0 * 1024.0);
    println!(
        "\n{BIG_H} x {BIG_W} ({:.1} MP), eleven stages:",
        n as f64 / 1e6
    );
    println!(
        "{:<26} {:>10} {:>12} {:>16}",
        "", "time", "PCIe (MiB)", "synchronisations"
    );
    println!(
        "{:<26} {resident_ms:>9.1}ms {:>12.0} {:>16}",
        "device-resident",
        mib(resident_floats),
        1
    );
    println!(
        "{:<26} {offload_ms:>9.1}ms {:>12.0} {:>16}",
        "per-call offload",
        mib(offload_floats),
        STAGES.len()
    );
    println!("{:<26} {cpu_ms:>9.1}ms {:>12} {:>16}", "CPU", "-", "-");
    println!(
        "  the offload form does the same arithmetic and moves {:.1}x the bytes;\n  \
         residency is worth {:.2}x here, and more as the frame grows",
        offload_floats as f64 / resident_floats as f64,
        offload_ms / resident_ms
    );
}
