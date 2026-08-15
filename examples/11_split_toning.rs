// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 11 — Split-toning in OKLab.
//!
//! Demonstrates `split_toning`, which takes the monochrome image back to
//! three channels by tinting shadows and highlights separately. Five
//! configurations:
//!
//! 1. Reference — no chroma in either tint; the output is neutral and
//!    numerically identical to the input, channel for channel.
//! 2. Sepia — a single warm tint in both slots, the toned-print look.
//! 3. Selenium — cool shadows, neutral highlights.
//! 4. Cross-process — cool shadows against warm highlights, the split
//!    that gives the technique its name.
//! 5. The same as 4 with `balance = 0.6`, moving the crossover down so
//!    the highlight tint claims more of the frame.
//!
//! What to look for:
//! - Image 1 proves the round trip: monochrome in, neutral out, no cast.
//!   Everything you see in images 2–5 is the tint, not conversion error.
//! - The neutral ramp in row 4 is where the crossover is visible. In
//!   image 4 the dark patches lean blue and the light ones lean amber,
//!   crossing near the middle; in image 5 the same crossing point sits
//!   two patches lower.
//! - The patches keep their relative brightness across all five images.
//!   That is the property OKLab buys: tinting in linear sRGB would have
//!   made the tinted patches lighter as well as coloured.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::split_toning::{SplitToningParams, split_toning};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, SplitToningParams); 5] = [
        (
            "reference",
            SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, 0.0),
        ),
        (
            "sepia",
            SplitToningParams::new([0.0, 0.03, 0.05], [0.0, 0.03, 0.05], 0.5, 0.0),
        ),
        (
            "selenium",
            SplitToningParams::new([0.0, 0.01, -0.05], [0.0, 0.0, 0.0], 0.5, 0.0),
        ),
        (
            "cross_process",
            SplitToningParams::new([0.0, -0.02, -0.05], [0.0, 0.03, 0.04], 0.5, 0.0),
        ),
        (
            "balance_high",
            SplitToningParams::new([0.0, -0.02, -0.05], [0.0, 0.03, 0.04], 0.5, 0.6),
        ),
    ];

    for (name, params) in configs {
        let out = split_toning(bw.view(), &params).unwrap();
        let path = format!("examples/output/11_toning_{name}.ppm");
        shared::write_ppm_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    // The reference configuration must be a faithful monochrome → RGB
    // round trip, or every tint above sits on top of a colour cast.
    let neutral = split_toning(
        bw.view(),
        &SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, 0.0),
    )
    .unwrap();
    let max_err = (0..shared::HEIGHT)
        .flat_map(|y| (0..shared::WIDTH).map(move |x| (y, x)))
        .flat_map(|(y, x)| (0..3).map(move |c| (y, x, c)))
        .map(|(y, x, c)| (neutral[[y, x, c]] - bw[[y, x, 0]]).abs())
        .fold(0.0_f32, f32::max);

    println!("wrote examples/output/11_toning_*.ppm");
    println!("untinted round trip (luminance -> OKLab -> linear sRGB): max error {max_err:.3e}");
}
