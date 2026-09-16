// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 16 — Highlight roll-off: the clip-versus-shoulder decision.
//!
//! Demonstrates `highlight_rolloff`, the stage that decides what becomes
//! of the highlight headroom every earlier kernel carefully preserved.
//!
//! The colour checker is pushed +2 EV so that its lighter patches carry
//! genuine scene values above 1.0 — the situation a bright sky or a
//! specular highlight produces — and then finished five ways:
//!
//! 1. `hard_clip` — the default `(1.0, 1.0)`, identical to `np.clip`.
//! 2. `white_2` — knee 0.7, white point 2.0. One stop recovered.
//! 3. `white_4` — knee 0.7, white point 4.0. Two stops recovered.
//! 4. `white_8` — knee 0.7, white point 8.0. Three stops, visibly flatter
//!    near white: recovering range costs highlight contrast, always.
//! 5. `low_knee` — knee 0.3, white point 4.0. The shoulder starts in the
//!    midtones, which is usually too much.
//!
//! What to look for:
//! - In image 1 the top row of light patches is one flat white area with
//!   a hard border. In images 2–4 that border is gone and the patches
//!   separate again. That difference is the entire point of the kernel.
//! - Nothing below the knee moves in any of them. Compare the dark
//!   patches across all five files: they are bit-identical.
//! - Image 5 shows the cost of over-reaching: mid-grey has been dragged
//!   down along with the highlights, so the picture loses snap.
//!
//! The printed tables are as much the point as the images. The first
//! reports how many distinct output values survive from eight distinct
//! highlight inputs spanning 1.0 to 8.0: one under the hard clip, and
//! four, six or eight under a shoulder, according to how far its white
//! point reaches — everything above the white point is white. The
//! second samples the curve itself at the zone anchors.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::exposure::exposure;
use phaios_core::highlight_rolloff::{RolloffParams, highlight_rolloff};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // +2 EV, so the light patches genuinely exceed 1.0 and there is
    // something for the shoulder to recover.
    let pushed = exposure(bw.view(), 2.0).unwrap();

    let configs: [(&str, RolloffParams); 5] = [
        ("hard_clip", RolloffParams::default()),
        ("white_2", RolloffParams::new(0.7, 2.0)),
        ("white_4", RolloffParams::new(0.7, 4.0)),
        ("white_8", RolloffParams::new(0.7, 8.0)),
        ("low_knee", RolloffParams::new(0.3, 4.0)),
    ];

    for (name, params) in configs.iter() {
        let out = highlight_rolloff(pushed.view(), params).unwrap();
        let path = format!("examples/output/16_rolloff_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }
    println!("wrote examples/output/16_rolloff_*.ppm");

    // How much highlight detail each setting actually keeps. Eight scene
    // values from nominal white to three stops over; count how many
    // remain distinguishable after the stage.
    let probes: Vec<f32> = (0..8).map(|i| 2.0_f32.powf(i as f32 * 3.0 / 7.0)).collect();
    println!("\ndistinct output codes from 8 distinct highlights (1.0 .. 8.0):");
    for (name, params) in configs.iter() {
        let img = Array3::from_shape_vec((1, probes.len(), 1), probes.clone()).unwrap();
        let out = highlight_rolloff(img.view(), params).unwrap();
        let mut bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        bits.sort_unstable();
        bits.dedup();
        println!(
            "  {name:>10}: {} of 8 survive  (knee {:.2}, white {:.1})",
            bits.len(),
            params.knee,
            params.white_point
        );
    }

    // The curve itself, at the zone anchors, for the two-stop setting.
    let params = RolloffParams::new(0.7, 4.0);
    println!("\nknee 0.70, white 4.0 — sampled at the zone anchors:");
    for zone in 0..=12 {
        let linear = 0.18_f32 * 2.0_f32.powi(zone - 5);
        let sample = Array3::from_elem((1, 1, 1), linear);
        let out = highlight_rolloff(sample.view(), &params).unwrap()[[0, 0, 0]];
        let mark = if linear <= params.knee {
            ""
        } else {
            "  <- shoulder"
        };
        println!("  zone {zone:>2}: {linear:>9.4} -> {out:>8.5}{mark}");
    }
}
