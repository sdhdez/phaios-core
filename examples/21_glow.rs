// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 21 — Halation, diffusion and veiling glare: one kernel.
//!
//! Demonstrates `glow`, and the claim that the three named effects are
//! the same operation at different parameters *and different pipeline
//! positions*:
//!
//! ```text
//! out = in + amount · blur(max(in − threshold, 0), σ)
//! ```
//!
//! Four renderings of a +2 EV checker:
//!
//! 1. `21_none` — the reference.
//! 2. `21_halation` — high threshold, moderate σ, applied **early**
//!    (right after exposure, before the tone stages) because halation
//!    happens in the emulsion at capture.
//! 3. `21_diffusion` — mid threshold, large σ, applied **late** (after
//!    the tone stages) because diffusion happens at the print.
//! 4. `21_glare` — no threshold, frame-spanning σ, applied **earliest**
//!    because veiling glare happens in the lens.
//!
//! What to look for:
//! - In image 2 the light patches carry a tight halo; the dark patches
//!   are untouched, because nothing below the threshold scatters.
//! - Image 4 has no halo at all — instead the *blacks lift*. That is the
//!   signature of glare, and the printed table shows why a tone curve
//!   cannot reproduce it: the lift depends on the brightness of the
//!   whole frame, which a per-pixel transfer cannot see.
//! - Images 2 and 3 use nearly the same numbers and look different,
//!   because position in the pipeline is doing the work.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::exposure::exposure;
use phaios_core::glow::{GlowParams, glow};
use phaios_core::tone::{ToneCurveParams, tone_curve};

fn write(name: &str, img: ndarray::ArrayView3<f32>) {
    let path = format!("examples/output/21_{name}.ppm");
    shared::write_ppm_grey_display(Path::new(&path), img, shared::WIDTH, shared::HEIGHT);
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let pushed = exposure(bw.view(), 2.0).unwrap();
    let grade = ToneCurveParams::new(1.15, 0.0, 0.9);

    // 1. Reference: exposure then tone, nothing scattered.
    let plain = tone_curve(pushed.view(), &grade).unwrap();
    write("none", plain.view());

    // 2. Halation — in the emulsion, so BEFORE the tone stages.
    let haloed = glow(pushed.view(), &GlowParams::new(0.8, 8.0, 0.35)).unwrap();
    write(
        "halation",
        tone_curve(haloed.view(), &grade).unwrap().view(),
    );

    // 3. Diffusion — at the print, so AFTER them.
    let diffused = glow(plain.view(), &GlowParams::new(0.5, 20.0, 0.4)).unwrap();
    write("diffusion", diffused.view());

    // 4. Veiling glare — in the lens, so earliest of all.
    let glared = glow(pushed.view(), &GlowParams::new(0.0, 400.0, 0.06)).unwrap();
    write("glare", tone_curve(glared.view(), &grade).unwrap().view());

    println!("wrote examples/output/21_*.ppm");

    // ── Why a tone curve cannot be veiling glare ─────────────────────
    // The same black pixel, in two frames that differ only elsewhere.
    println!("\nveiling glare: the black point lifts by an amount set by the WHOLE frame");
    println!(
        "{:>18} {:>14} {:>16}",
        "scene", "corner before", "corner after"
    );
    let n = 64_usize;
    for (label, peak) in [("mostly dark", 0.2_f32), ("bright subject", 4.0)] {
        let mut scene = Array3::<f32>::zeros((n, n, 1));
        for y in 0..8 {
            for x in 0..8 {
                scene[[y, x, 0]] = peak;
            }
        }
        let out = glow(scene.view(), &GlowParams::new(0.0, 200.0, 0.5)).unwrap();
        println!(
            "{label:>18} {:>14.5} {:>16.5}",
            scene[[n - 1, n - 1, 0]],
            out[[n - 1, n - 1, 0]]
        );
    }
    println!(
        "  the corner pixel is 0.0 in both frames, and ends up different —\n  \
         no per-pixel transfer can do that, which is the whole argument"
    );

    // ── And what the threshold buys ──────────────────────────────────
    println!("\nthreshold selects what scatters (sigma 6, amount 0.5):");
    let probe = Array3::<f32>::from_shape_fn((1, 5, 1), |(_, x, _)| x as f32 * 0.5);
    for threshold in [0.0_f32, 0.5, 1.0, 2.0] {
        let out = glow(probe.view(), &GlowParams::new(threshold, 6.0, 0.5)).unwrap();
        let deltas: Vec<String> = (0..5)
            .map(|i| format!("{:+.4}", out[[0, i, 0]] - probe[[0, i, 0]]))
            .collect();
        println!("  threshold {threshold:>4.1}: {}", deltas.join("  "));
    }
    println!("  (inputs 0.0 0.5 1.0 1.5 2.0 — a higher threshold scatters less)");
}
