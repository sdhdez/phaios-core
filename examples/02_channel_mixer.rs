// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 02 — Channel mixer B&W conversion.
//!
//! Demonstrates `channel_mixer_bw` by writing three output images:
//! 1. Standard BT.709 weights (0.21, 0.72, 0.07) as a reference.
//! 2. Red-boosted weights (1.0, 0.0, 0.0) — infrared-like: reds bright,
//!    greens and blues very dark.
//! 3. Custom weights (0.0, 0.5, 0.5) — equal green+blue, no red.
//!
//! What to look for:
//! - Image 2 inverts the tonal relationship between red (#15) and green
//!   (#14) patches relative to image 1.
//! - Negative weights (not shown here but valid) produce inverted tones.

#[path = "shared/mod.rs"]
mod shared;

fn main() {
    todo!("implemented in Step 1 — run after kernels are in place")
}
