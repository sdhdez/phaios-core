// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 04 — Adams/Archer Zone System tone curve.
//!
//! Demonstrates `zone_system` with three configurations:
//! 1. No-op (empty params) — output == input.
//! 2. "Pull" Zone V: offset Zone V by −1 stop (darker shadows).
//! 3. "Push" Zone VII by +1 stop, "deepen" Zone III by −0.5 stops
//!    — a classic landscape / darkroom interpretation curve.
//!
//! What to look for:
//! - Configuration 1 and 3 are written side by side; compare the grey
//!   ramp in row 4 of the chart.
//! - The Gaussian blending means a Zone V offset also affects Zones IV
//!   and VI (σ = 0.8 zones) — the influence is gradual, not a sharp step.
//! - Patches near the target zone show the most change.

#[path = "shared/mod.rs"]
mod shared;

fn main() {
    todo!("implemented in Step 1 — run after kernels are in place")
}
