// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 05 — Local contrast via the guided filter.
//!
//! Demonstrates `local_contrast` at two strength levels:
//! 1. Unprocessed luminance (reference).
//! 2. `radius=8, eps=0.01, strength=0.5` — moderate enhancement.
//! 3. `radius=16, eps=0.01, strength=1.0` — aggressive enhancement.
//!
//! What to look for:
//! - The patch boundaries become sharper and develop a subtle
//!   "halo" at high strength — the classic unsharp-mask artefact.
//! - Inside flat patches the guided filter is the identity: the
//!   smooth version equals the input, so `L - guided = 0` and the
//!   output equals the input. This verifies correctness.
//! - The neutral ramp (row 4) should show no cross-patch bleeding
//!   of luminance, only edge enhancement at boundaries.

#[path = "shared/mod.rs"]
mod shared;

use std::path::Path;

use ndarray::Array3;
use phaios_core::bw::{LuminanceStandard, luminance_bw};
use phaios_core::local_contrast::{GuidedFilterParams, local_contrast};

fn main() {
    std::fs::create_dir_all("examples/output").unwrap();

    let raw = shared::synthetic_macbeth();
    let rgb = Array3::from_shape_vec((shared::HEIGHT, shared::WIDTH, 3), raw).unwrap();
    let bw = luminance_bw(rgb.view(), LuminanceStandard::Bt709).unwrap();

    // 1 — Unprocessed reference
    shared::write_ppm_grey(
        Path::new("examples/output/05_contrast_reference.ppm"),
        bw.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 2 — Moderate local contrast
    let params_moderate = GuidedFilterParams::new(8, 0.01);
    let out_moderate = local_contrast(bw.view(), &params_moderate, 0.5).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/05_contrast_moderate.ppm"),
        out_moderate.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    // 3 — Aggressive local contrast
    let params_aggressive = GuidedFilterParams::new(16, 0.01);
    let out_aggressive = local_contrast(bw.view(), &params_aggressive, 1.0).unwrap();
    shared::write_ppm_grey(
        Path::new("examples/output/05_contrast_aggressive.ppm"),
        out_aggressive.as_slice().unwrap(),
        shared::WIDTH,
        shared::HEIGHT,
    );

    println!("wrote examples/output/05_contrast_{{reference,moderate,aggressive}}.ppm");
}
