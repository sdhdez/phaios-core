// SPDX-License-Identifier: GPL-3.0-or-later
//! sRGB transfer encoding (terminal pipeline stage).
//!
//! Applies the IEC 61966-2-1 piecewise transfer function to convert
//! scene-referred linear f32 values to display-referred sRGB. This is
//! always the last kernel in the pipeline.
//!
//! Reference:
//! - IEC 61966-2-1:1999, "Multimedia systems and equipment — Colour
//!   measurement and management — Part 2-1: Colour management —
//!   Default RGB colour space — sRGB."
//!
//! Placeholder — implementation in Step 1.
