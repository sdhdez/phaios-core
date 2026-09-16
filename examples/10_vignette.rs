// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 10 — Radial vignette.
//!
//! Demonstrates `vignette` across the three controls:
//!
//! 1. Reference — `amount = 0`, the identity.
//! 2. Classic burn — `amount 0.5, feather 1.0, roundness 0.0`.
//! 3. Hard circle — the same amount with `feather 0.2`, so the
//!    transition is squeezed towards the corners.
//! 4. Rectangular — `roundness 1.0`, iso-lines parallel to the frame.
//! 5. Inverted — `amount −0.5`, corners lightened instead.
//!
//! What to look for:
//! - The chart is flat within each patch, which makes the falloff itself
//!   visible: the gradient you see inside a patch is entirely the
//!   vignette, since nothing else varies across it.
//! - Image 3 shows why `feather` matters more than it sounds: with a
//!   narrow transition the smoothstep still has zero slope at both ends,
//!   so even the "hard" setting has no banding edge.
//! - In image 4 the middle of each edge is as dark as the corners; in
//!   image 2 it is not. That is the whole difference between the
//!   Euclidean and Chebyshev distance measures.
//!
//! The output is passed through `encode_srgb` before it is written
//! (the terminal pipeline stage), so the PPM is display-referred.
//! Example 06 shows what skipping that stage looks like.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::vignette::{VignetteParams, vignette};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    let configs: [(&str, VignetteParams); 5] = [
        ("reference", VignetteParams::new(0.0, 0.5, 0.0)),
        ("classic", VignetteParams::new(0.5, 1.0, 0.0)),
        ("hard", VignetteParams::new(0.5, 0.2, 0.0)),
        ("rectangular", VignetteParams::new(0.5, 1.0, 1.0)),
        ("inverted", VignetteParams::new(-0.5, 1.0, 0.0)),
    ];

    for (name, params) in configs {
        let out = vignette(bw.view(), &params).unwrap();
        let path = format!("examples/output/10_vignette_{name}.ppm");
        shared::write_ppm_grey_display(Path::new(&path), out.view(), shared::WIDTH, shared::HEIGHT);
    }

    // Resolution independence: the same parameters on a half-size frame
    // must give the same factors at corresponding positions.
    let params = VignetteParams::new(0.5, 1.0, 0.0);
    let full = vignette(
        Array3::from_elem((shared::HEIGHT, shared::WIDTH, 1), 1.0_f32).view(),
        &params,
    )
    .unwrap();
    let half = vignette(
        Array3::from_elem((shared::HEIGHT / 2, shared::WIDTH / 2, 1), 1.0_f32).view(),
        &params,
    )
    .unwrap();
    let corner_delta = (full[[0, 0, 0]] - half[[0, 0, 0]]).abs();
    let centre_delta = (full[[shared::HEIGHT / 2, shared::WIDTH / 2, 0]]
        - half[[shared::HEIGHT / 4, shared::WIDTH / 4, 0]])
    .abs();

    println!("wrote examples/output/10_vignette_*.ppm");
    println!(
        "full frame vs half-size preview: corner delta {corner_delta:.2e}, centre delta {centre_delta:.2e}"
    );
}
