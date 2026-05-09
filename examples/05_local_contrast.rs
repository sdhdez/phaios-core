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

fn main() {
    todo!("implemented in Step 1 — run after kernels are in place")
}
