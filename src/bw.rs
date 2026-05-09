// SPDX-License-Identifier: GPL-3.0-or-later
//! Black-and-white conversion kernels.
//!
//! Three methods for v0.1:
//! 1. Standard luminance (ITU-R BT.601 / BT.709 / BT.2020 weights)
//! 2. Channel mixer (arbitrary user weights in −2..+2)
//! 3. Coloured-filter simulation (Wratten-style presets)
//!
//! References:
//! - ITU-R BT.709-6, "Parameter values for the HDTV standards for
//!   production and international programme exchange" (2015).
//! - ITU-R BT.601-7, "Studio encoding parameters of digital television
//!   for standard 4:3 and wide-screen 16:9 aspect ratios" (2011).
//! - ITU-R BT.2020-2, "Parameter values for ultra-high definition
//!   television systems for production and international programme
//!   exchange" (2015).
//! - Kodak Professional B+W Filters datasheet (Wratten 2 series).
//!
//! Placeholder — implementation in Step 1.
