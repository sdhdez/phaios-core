// SPDX-License-Identifier: GPL-3.0-or-later
//! Adams/Archer Zone System tone curve.
//!
//! Implements the eleven-zone system (0..=10) with the Gaussian-blending
//! modernisation from Phil Davis (1999). Each zone is one stop apart;
//! Zone V = middle grey at 18% reflectance.
//!
//! Given user-supplied offsets `{zone_index: stops}`, the output
//! luminance at each pixel is:
//!
//! ```text
//! zone_pos = 5 + log₂(max(L, ε) / 0.18)
//! total_offset = Σ_z  offset[z] · exp(-(zone_pos - z)² / (2·σ²))
//! L_out = L · 2^total_offset
//! ```
//!
//! where σ = 0.8 zones and ε = 1e-10 (guards against log₂(0)).
//!
//! References:
//! - Ansel Adams, *The Negative*, Little, Brown (1948), chapter 5.
//! - Phil Davis, *Beyond the Zone System*, 4th ed., Focal Press (1999).

use std::collections::HashMap;

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Zone System tone offsets.
///
/// Maps zone index (0..=10) to a stop offset in −3..+3. Zones not
/// present in the map are treated as having a 0-stop offset.
///
/// ```python
/// # Lift Zone V by half a stop, deepen Zone III by 0.3 stops
/// params = phaios_core.ZoneParams({5: 0.5, 3: -0.3})
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Default)]
pub struct ZoneParams {
    offsets: HashMap<i32, f32>,
}

#[pymethods]
impl ZoneParams {
    /// Create a new ``ZoneParams`` from a dict mapping zone index to stop offset.
    #[new]
    pub fn new(offsets: HashMap<i32, f32>) -> Self {
        Self { offsets }
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        let mut pairs: Vec<(i32, f32)> = self.offsets.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|(k, _)| *k);
        let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        format!("ZoneParams({{{}}})", inner.join(", "))
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Middle grey reference: 18% reflectance.
const MIDDLE_GREY: f32 = 0.18;

/// Gaussian σ in zone-position space (Davis 1999).
const SIGMA: f32 = 0.8;

/// 2·σ² denominator — precomputed.
const TWO_SIGMA_SQ: f32 = 2.0 * SIGMA * SIGMA;

/// Guard against log₂(0).
const L_EPSILON: f32 = 1e-10;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Apply the Adams/Archer Zone System tone curve.
///
/// Eleven zones (0..=10), each one stop apart; Zone V = middle grey
/// at 18% reflectance. Offsets are blended via a Gaussian in
/// zone-position space (σ = 0.8 zones, Davis 1999).
///
/// Input shape: `(H, W, 1)` — linear luminance. Any memory layout is
/// accepted (strided views and Fortran-order arrays included); the
/// output is always freshly allocated and C-contiguous.
/// Output shape: `(H, W, 1)` — tone-adjusted luminance.
///
/// Reference: Ansel Adams, *The Negative*, Little, Brown (1948), ch. 5;
/// modernised in Davis, *Beyond the Zone System*, Focal Press (1999).
///
/// # Errors
/// Returns [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn zone_system(img: ArrayView3<f32>, params: &ZoneParams) -> Result<Array3<f32>, PhaiosError> {
    if img.shape()[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "zone_system expects (H, W, 1) luminance input, got shape {:?}",
            img.shape()
        )));
    }

    let offsets: Vec<(f32, f32)> = params
        .offsets
        .iter()
        .map(|(&z, &off)| (z as f32, off))
        .collect();

    let (h, w, _) = img.dim();
    let mut out = Array3::<f32>::zeros((h, w, 1));

    // No offsets → identity. Hoisted out of the pixel loop.
    if offsets.is_empty() {
        out.assign(&img);
        return Ok(out);
    }

    ndarray::Zip::from(&mut out).and(img).par_for_each(|o, &l| {
        let l_pos = l.max(L_EPSILON);
        let zone_pos = 5.0 + (l_pos / MIDDLE_GREY).log2();
        let total: f32 = offsets
            .iter()
            .map(|(z, off)| {
                let d = zone_pos - z;
                off * (-d * d / TWO_SIGMA_SQ).exp()
            })
            .sum();
        *o = l * 2.0_f32.powf(total);
    });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn no_op_params_are_identity() {
        let img = array![[[0.18_f32]], [[0.36_f32]]];
        let params = ZoneParams::default();
        let out = zone_system(img.view(), &params).unwrap();
        for (&expected, &got) in img.iter().zip(out.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn zone_v_plus_one_stop_doubles_middle_grey() {
        // Middle grey at Zone V → +1 stop should approximately double the value.
        // The Gaussian blend means the multiplier is slightly less than 2.0
        // (Gaussian peak is 1.0 at zone_pos == 5, so total_offset ≈ 1.0).
        let img = array![[[MIDDLE_GREY]]];
        let mut offsets = HashMap::new();
        offsets.insert(5_i32, 1.0_f32);
        let params = ZoneParams::new(offsets);
        let out = zone_system(img.view(), &params).unwrap();
        let ratio = out[[0, 0, 0]] / MIDDLE_GREY;
        // At zone_pos = 5 (exactly Zone V), Gaussian peak = exp(0) = 1.0.
        // So output = 0.18 * 2^1 = 0.36 exactly.
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "expected ~2×, got ratio {ratio:.4} (value {})",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn shape_error_on_rgb_input() {
        let img = Array3::<f32>::zeros((4, 4, 3));
        assert!(zone_system(img.view(), &ZoneParams::default()).is_err());
    }

    #[test]
    fn accepts_non_contiguous_input() {
        // Regression: the kernel used to call `as_slice().expect(...)`,
        // which panicked on any strided view. Reaching Python that panic
        // became `pyo3_runtime.PanicException`, which does not inherit
        // from `Exception` and so escapes ordinary caller error handling.
        let img =
            Array3::<f32>::from_shape_fn((8, 4, 1), |(y, x, _)| (y * 4 + x) as f32 / 32.0 + 0.01);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        assert!(
            !strided.is_standard_layout(),
            "test setup: view should not be C-contiguous"
        );

        let mut offsets = HashMap::new();
        offsets.insert(5_i32, 0.5_f32);
        let params = ZoneParams::new(offsets);

        let out = zone_system(strided, &params).unwrap();
        assert_eq!(out.dim(), (4, 4, 1));
        assert!(out.is_standard_layout(), "output must be C-contiguous");

        // Must agree bit-for-bit with the same data laid out contiguously.
        let expected = zone_system(strided.to_owned().view(), &params).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn identity_path_accepts_non_contiguous_input() {
        // The empty-offsets fast path takes a different code branch.
        let img = Array3::<f32>::from_shape_fn((6, 3, 1), |(y, x, _)| (y + x) as f32);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let out = zone_system(strided, &ZoneParams::default()).unwrap();
        assert_eq!(out, strided.to_owned());
    }
}
