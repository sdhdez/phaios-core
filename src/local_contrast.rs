// SPDX-License-Identifier: GPL-3.0-or-later
//! Local contrast enhancement via the He–Sun–Tang guided filter.
//!
//! Implements the O(1) integral-image formulation of the guided filter
//! and exposes `local_contrast(luminance, params, strength)` which
//! computes `L + strength * (L - guided_filter(L, L, r, eps))`.
//!
//! Reference:
//! - Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image Filtering,"
//!   *ECCV 2010*, LNCS 6311, pp. 1–14.
//!   Patent-free; see authors' project page.
//!
//! Placeholder — implementation in Step 1.
