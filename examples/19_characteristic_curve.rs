// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 19 — The characteristic curve: toe, straight section, shoulder.
//!
//! Demonstrates `shadow_rolloff` (the toe) and, more to the point, the
//! composition it completes. A film tone scale is not a straight line:
//! it compresses at both ends and holds contrast in the middle, and that
//! shape is what separates a photographic rendering from a linear one.
//!
//! Three kernels, already present, make it:
//!
//! ```text
//! shadow_rolloff  →  tone_curve  →  highlight_rolloff
//!      toe            straight          shoulder
//! ```
//!
//! Five renderings of a +2 EV colour checker are written:
//!
//! 1. `19_linear` — none of the three. A straight line, clipped at white.
//! 2. `19_toe_only` — the toe alone. Shadows deepen and close up.
//! 3. `19_shoulder_only` — the shoulder alone, for comparison.
//! 4. `19_characteristic` — all three: the full curve.
//! 5. `19_hard_toe` — the toe at full strength, where the deepest
//!    shadows lose their separation entirely. Included because knowing
//!    where a control stops being useful is part of knowing the control.
//!
//! What to look for:
//! - The dark patches in 1 and 4 side by side. In 1 they hold linear
//!   separation right down to black; in 4 they close together, which is
//!   what shadow detail looks like on film.
//! - The printed slope table is the argument in numbers. A
//!   characteristic curve has low slope at both ends and its highest in
//!   the midtones; a linear rendering has the same slope everywhere
//!   until it clips, at which point the slope drops to zero abruptly
//!   rather than gradually.
//! - The second table isolates the toe: two samples 0.02 apart just
//!   above black, and what is left of that separation at each toe
//!   strength.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPMs are display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::exposure::exposure;
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};
use phaios_core::shadow_rolloff::{ShadowRolloffParams, shadow_rolloff};
use phaios_core::tone::{ToneCurveParams, tone_curve};

/// Apply the three stages in pipeline order. Any of them may be the
/// identity, which is how the comparison images are made.
fn characteristic(
    img: ndarray::ArrayView3<f32>,
    toe: &ShadowRolloffParams,
    straight: &ToneCurveParams,
    shoulder: &RolloffParams,
) -> Array3<f32> {
    let a = shadow_rolloff(img, toe).unwrap();
    let b = tone_curve(a.view(), straight).unwrap();
    highlight_rolloff(b.view(), shoulder).unwrap()
}

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();
    let pushed = exposure(bw.view(), 2.0).unwrap();

    let no_toe = ShadowRolloffParams::default();
    let toe = ShadowRolloffParams::new(0.18, 0.8);
    let hard_toe = ShadowRolloffParams::new(0.30, 1.0);
    let flat = ToneCurveParams::new(1.0, 0.0, 1.0);
    let contrast = ToneCurveParams::new(1.2, 0.0, 1.0);
    let no_shoulder = RolloffParams::default();
    let shoulder = RolloffParams::new(0.7, 3.0);

    let configs: [(&str, ShadowRolloffParams, ToneCurveParams, RolloffParams); 5] = [
        ("linear", no_toe.clone(), flat.clone(), no_shoulder.clone()),
        ("toe_only", toe.clone(), flat.clone(), no_shoulder.clone()),
        (
            "shoulder_only",
            no_toe.clone(),
            flat.clone(),
            shoulder.clone(),
        ),
        (
            "characteristic",
            toe.clone(),
            contrast.clone(),
            shoulder.clone(),
        ),
        ("hard_toe", hard_toe.clone(), contrast, shoulder.clone()),
    ];

    for (name, t, c, sh) in configs.iter() {
        let out = characteristic(pushed.view(), t, c, sh);
        let path = format!("examples/output/19_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }
    println!("wrote examples/output/19_*.ppm");

    // ── The curve shape, in numbers ──────────────────────────────────
    // Local slope d(out)/d(in) at four scene values, spanning shadow to
    // highlight. The characteristic shape is low, high, high, low.
    let probes = [0.01_f32, 0.05, 0.40, 2.00];
    let step = 1e-4_f32;

    println!("\nlocal slope d(out)/d(in) — the shape of each rendering:");
    println!(
        "{:>16} {:>10} {:>10} {:>10} {:>10}",
        "", "in 0.01", "in 0.05", "in 0.40", "in 2.00"
    );
    for (name, t, c, sh) in configs.iter() {
        let mut cells = String::new();
        for p in probes {
            let pair = Array3::from_shape_vec((1, 2, 1), vec![p, p + step]).unwrap();
            let out = characteristic(pair.view(), t, c, sh);
            let slope = (out[[0, 1, 0]] - out[[0, 0, 0]]) / step;
            cells.push_str(&format!("{slope:>10.3}"));
        }
        println!("{name:>16}{cells}");
    }

    println!("\nthe toe compresses shadow separation — two samples 0.02 apart just above black:");
    for strength in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let pair = Array3::from_shape_vec((1, 2, 1), vec![0.0_f32, 0.02]).unwrap();
        let out = shadow_rolloff(pair.view(), &ShadowRolloffParams::new(0.2, strength)).unwrap();
        let sep = out[[0, 1, 0]] - out[[0, 0, 0]];
        println!(
            "  strength {strength:.2}: {sep:.5}  ({:.0}% of the input separation)",
            sep / 0.02 * 100.0
        );
    }
}
