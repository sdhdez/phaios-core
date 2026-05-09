// SPDX-License-Identifier: GPL-3.0-or-later
//! Example 03 — Colour filter B&W simulation.
//!
//! Applies all six Wratten-style presets (None, Yellow #8 K2, Orange #21,
//! Red #25 A, Green #11 X1, Blue #47 C5) and writes one PPM per preset.
//!
//! What to look for:
//! - Red #25 A makes the blue patch (#03, blue sky) very dark and
//!   lightens reds (#09, #15). Classic infrared / dramatic sky effect.
//! - Green #11 X1 brightens the foliage patch (#04, #11, #14) and
//!   darkens reds and blues — natural landscape look.
//! - Blue #47 C5 effectively inverts what Red does: sky is bright, reds
//!   are dark. Used for atmospheric haze enhancement.

#[path = "shared/mod.rs"]
mod shared;

fn main() {
    todo!("implemented in Step 1 — run after kernels are in place")
}
