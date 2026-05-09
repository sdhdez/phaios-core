// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 01 — Standard luminance B&W conversion.
//!
//! Demonstrates `luminance_bw` with the default BT.709 weights on the
//! synthetic Macbeth chart. What to look for in the output:
//!
//! - The neutral patches (row 4) should form a smooth grey ramp.
//! - The green patch (#11, yellow-green) will render brighter than the
//!   red patch (#15) despite both being vivid — this is the green
//!   dominance of the BT.709 luminance weights (0.71 vs 0.21).
//! - Compare with examples 02 and 03 to see how different conversion
//!   methods change relative tonal values.

#[path = "shared/mod.rs"]
mod shared;

fn main() {
    todo!("implemented in Step 1 — run after kernels are in place")
}
