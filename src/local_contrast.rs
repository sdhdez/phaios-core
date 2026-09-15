// SPDX-License-Identifier: GPL-3.0-or-later
//! Local contrast enhancement via the He–Sun–Tang guided filter.
//!
//! Implements the O(1) integral-image (summed-area table) formulation of
//! the guided filter in self-guided mode (guide = input = luminance) and
//! exposes it as a local-contrast kernel:
//!
//! ```text
//! output = L + strength · (L − guided_filter(L, radius, eps))
//! ```
//!
//! The guided filter is edge-preserving: it smooths flat regions while
//! leaving edges intact, making the residual `L − q` a clean
//! high-frequency (detail) signal rather than a halo-ridden Gaussian
//! unsharp mask.
//!
//! **Algorithm (self-guided, integral-image formulation)**
//!
//! For a single-channel input L and window radius r:
//! 1. Build SATs (summed-area tables) of L and L².
//! 2. Per pixel: `mean_L`, `mean_L2 = mean(L²)`, `var_L = mean_L2 − mean_L²`.
//! 3. `a = var_L / (var_L + ε)`. Convention: 0/0 → 0 (flat region → no edge).
//! 4. `b = mean_L · (1 − a)`.
//! 5. Build SATs of a and b; per pixel: `mean_a`, `mean_b`.
//! 6. `q = mean_a · L + mean_b`.
//!
//! Boundary windows are clamped to the image extent (replicate-border
//! padding semantics). All accumulation is done in f64 to avoid
//! precision loss in the SATs.
//!
//! **Memory.** Four full-resolution f64 tables would dominate the
//! footprint if they were all live at once, so each is dropped as soon
//! as the values derived from it exist: the tables of L and L² are
//! released before those of a and b are built. Peak scratch is about
//! 24 bytes per pixel.
//!
//! Reference: Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image
//! Filtering," *ECCV 2010*, LNCS 6311, pp. 1–14. The authors' extended
//! version is IEEE *TPAMI* 35(6), 2013, pp. 1397–1409.
//!
//! # Provenance and prior art
//!
//! Two things a future contributor needs to know, neither of which is a
//! legal opinion — this is a record of what was checked, not advice.
//!
//! **Do not port the authors' MATLAB.** Their reference implementation
//! (`guided-filter-code-v1`) carries no licence file and its readme
//! restricts it: "This code is for academic purpose only. Not for
//! commercial/industrial activities." That is incompatible with GPLv3
//! and with distribution on crates.io and PyPI. The code here is an
//! independent reimplementation from the paper's published equations,
//! which is what that readme explicitly invites, and it differs in
//! formulation as well as language: summed-area tables with four-corner
//! queries rather than the reference's `cumsum` box filter, specialised
//! to the self-guided case `I = p`, with f64 accumulation and a variance
//! clamp the reference has no counterpart for. The CUDA path is a third
//! formulation again. Keep it that way.
//!
//! **Patents.** A search of the granted-patent record found nothing
//! claiming guided image filtering itself, and the three closest
//! Microsoft filings naming these authors — US 8,625,888 (variable
//! kernel size image matting), US 8,386,964 (interactive image matting)
//! and US 8,855,411 (opacity measurement using a global pixel set) —
//! were each read against this kernel and none of their claims cover it;
//! all three are matting patents requiring elements this code has no
//! analogue of. That is a search result, not a clearance opinion, and an
//! earlier version of this comment overstated it as the bare assertion
//! "Patent-free."

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;
use crate::integral::{sat, window_sum};

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for the guided filter.
///
/// ```python
/// params = phaios_core.GuidedFilterParams(radius=8, eps=0.01)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct GuidedFilterParams {
    /// Filter radius in pixels. The window is `(2r+1) × (2r+1)`.
    #[pyo3(get, set)]
    pub radius: u32,
    /// Regularisation term ε. Controls the degree of smoothing.
    /// Larger values → more smoothing, less edge preservation.
    #[pyo3(get, set)]
    pub eps: f32,
}

#[pymethods]
impl GuidedFilterParams {
    /// Create new ``GuidedFilterParams``.
    #[new]
    pub fn new(radius: u32, eps: f32) -> Self {
        Self { radius, eps }
    }

    /// Two ``GuidedFilterParams`` are equal when both fields match.
    ///
    /// Consumers compare parameter objects to decide whether a cached
    /// render is still valid.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.radius == other.radius && self.eps.to_bits() == other.eps.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "GuidedFilterParams(radius={}, eps={})",
            self.radius, self.eps
        )
    }
}

// ── Guided filter (internal) ──────────────────────────────────────────────────

/// Apply the He–Sun–Tang guided filter (self-guided, 2-D).
///
/// Not exposed to Python. Called by [`local_contrast`].
fn guided_filter(img: ArrayView2<f32>, radius: u32, eps: f32) -> Result<Array2<f32>, PhaiosError> {
    let (h, w) = img.dim();
    let r = radius as usize;
    let eps_f64 = eps as f64;

    // Per-pixel linear-model coefficients for the window centred on that
    // pixel. Held as f32: they are averaged through another SAT, and the
    // f64 precision only matters while accumulating.
    let mut a_arr = crate::alloc::zeros2::<f32>((h, w))?;
    let mut b_arr = crate::alloc::zeros2::<f32>((h, w))?;

    {
        let sat_l = sat(img, |v| v as f64)?;
        let sat_l2 = sat(img, |v| {
            let d = v as f64;
            d * d
        })?;

        ndarray::Zip::indexed(&mut a_arr)
            .and(&mut b_arr)
            .par_for_each(|(y, x), a_out, b_out| {
                let (sum_l, area) = window_sum(&sat_l, y, x, r, h, w);
                let (sum_l2, _) = window_sum(&sat_l2, y, x, r, h, w);
                let mean_l = sum_l / area;
                let mean_l2 = sum_l2 / area;
                // mean(L²) − mean(L)² is a cancelling subtraction: for a
                // large image of large values the SAT rounding error can
                // exceed the true variance and drive it below zero, which
                // would make `a` negative (model inverted) or greater than
                // one (model over-driven). Clamp: no measurable structure
                // means a = 0, i.e. pure smoothing.
                let var_l = (mean_l2 - mean_l * mean_l).max(0.0);
                // Convention: 0/0 → 0 (flat region, no edge to preserve).
                let a = if var_l + eps_f64 > 0.0 {
                    var_l / (var_l + eps_f64)
                } else {
                    0.0
                };
                *a_out = a as f32;
                *b_out = (mean_l * (1.0 - a)) as f32;
            });
        // sat_l and sat_l2 die here, before the tables of a and b are
        // built — two full-resolution f64 buffers that would otherwise
        // stay live to the end of the function.
    }

    // Average overlapping windows: SAT of a and b, then per-pixel mean.
    // Each coefficient array is released as soon as its table exists.
    let sat_a = sat(a_arr.view(), |v| v as f64)?;
    drop(a_arr);
    let sat_b = sat(b_arr.view(), |v| v as f64)?;
    drop(b_arr);

    let mut out = crate::alloc::zeros2::<f32>((h, w))?;
    ndarray::Zip::indexed(&mut out)
        .and(img)
        .par_for_each(|(y, x), o, &l| {
            let (sum_a, area) = window_sum(&sat_a, y, x, r, h, w);
            let (sum_b, _) = window_sum(&sat_b, y, x, r, h, w);
            let mean_a = sum_a / area;
            let mean_b = sum_b / area;
            *o = (mean_a * l as f64 + mean_b) as f32;
        });

    Ok(out)
}

// ── Public kernel ─────────────────────────────────────────────────────────────

/// Validate shape and parameters. Shared verbatim by the CPU kernel and
/// the CUDA kernel in `src/cuda/` (feature-gated), so both backends
/// reject exactly the same inputs with exactly the same messages.
pub(crate) fn validate(
    shape: &[usize],
    params: &GuidedFilterParams,
    strength: f32,
) -> Result<(), PhaiosError> {
    if shape[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "local_contrast expects (H, W, 1) luminance input, got shape {shape:?}"
        )));
    }
    if !params.eps.is_finite() || params.eps < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "eps is {}, expected a finite value >= 0",
            params.eps
        )));
    }
    if !strength.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "strength is {strength}, expected a finite value"
        )));
    }
    Ok(())
}

/// Enhance local contrast using the guided filter.
///
/// Computes `output = L + strength · (L − guided_filter(L, radius, ε))`.
///
/// The `guided_filter` produces a smooth (low-frequency) version of L.
/// The difference `L − q` is the high-frequency detail. `strength`
/// controls how much detail is added back:
/// - 0.0 → no change
/// - 1.0 → standard unsharp mask
/// - > 1.0 → over-sharpening
///
/// Input shape: `(H, W, 1)` — any memory layout. Output shape:
/// `(H, W, 1)`, freshly allocated and C-contiguous.
///
/// A `radius` larger than the image is harmless: windows are clamped to
/// the image extent, so every window becomes the whole image.
///
/// Reference: He, Sun, Tang, "Guided Image Filtering," ECCV 2010.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
/// - [`PhaiosError::Parameter`] if `eps` is negative or non-finite, or
///   `strength` is non-finite. A negative `eps` makes `a = var/(var+ε)`
///   singular wherever the local variance approaches `−ε`, producing
///   infinities in the output.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn local_contrast(
    img: ArrayView3<f32>,
    params: &GuidedFilterParams,
    strength: f32,
) -> Result<Array3<f32>, PhaiosError> {
    validate(img.shape(), params, strength)?;
    let (h, w, _) = img.dim();
    let img2d = img.index_axis(ndarray::Axis(2), 0);
    let smooth = guided_filter(img2d, params.radius, params.eps)?;
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;
    ndarray::Zip::from(out.slice_mut(ndarray::s![.., .., 0]))
        .and(&img2d)
        .and(&smooth)
        .par_for_each(|o, &l, &q| {
            *o = l + strength * (l - q);
        });
    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn const_img(v: f32, h: usize, w: usize) -> Array3<f32> {
        Array3::from_elem((h, w, 1), v)
    }

    #[test]
    fn guided_filter_constant_image() {
        // On a constant image, the guided filter must return the same constant.
        let img2d = ndarray::Array2::from_elem((16, 16), 0.5_f32);
        let q = guided_filter(img2d.view(), 4, 0.01).unwrap();
        for &v in q.iter() {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "constant image guided filter: expected 0.5, got {v}"
            );
        }
    }

    #[test]
    fn guided_filter_identity_radius_zero() {
        // With radius=0, window = 1×1; mean_L = L; var_L = 0.
        // a = 0/(0+eps) = 0 (or 0/0 = 0 by convention), b = mean_L = L.
        // Output q = 0·L + L = L. Identity.
        let img2d = ndarray::Array2::from_shape_fn((8, 8), |(y, x)| (y * 8 + x) as f32 / 64.0);
        let q = guided_filter(img2d.view(), 0, 0.0).unwrap();
        for (&l, &qv) in img2d.iter().zip(q.iter()) {
            assert!(
                (qv - l).abs() < 1e-5,
                "radius=0 guided filter should be identity; got |{qv}-{l}|={}",
                (qv - l).abs()
            );
        }
    }

    #[test]
    fn local_contrast_constant_image_unchanged() {
        // On a constant image, the detail component (L - q) is 0, so output = L.
        let img = const_img(0.3, 32, 32);
        let params = GuidedFilterParams::new(4, 0.01);
        let out = local_contrast(img.view(), &params, 0.5).unwrap();
        for &v in out.iter() {
            assert!(
                (v - 0.3).abs() < 1e-4,
                "constant image local_contrast: expected 0.3, got {v}"
            );
        }
    }

    #[test]
    fn shape_error_on_rgb_input() {
        let img = Array3::<f32>::zeros((4, 4, 3));
        let params = GuidedFilterParams::new(2, 0.01);
        assert!(local_contrast(img.view(), &params, 1.0).is_err());
    }

    #[test]
    fn rejects_invalid_parameters() {
        let img = const_img(0.5, 8, 8);
        // Negative eps makes a = var/(var+eps) singular near var = −eps.
        assert!(matches!(
            local_contrast(img.view(), &GuidedFilterParams::new(4, -0.01), 1.0).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
        assert!(matches!(
            local_contrast(img.view(), &GuidedFilterParams::new(4, f32::NAN), 1.0).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
        assert!(matches!(
            local_contrast(img.view(), &GuidedFilterParams::new(4, 0.01), f32::INFINITY)
                .unwrap_err(),
            PhaiosError::Parameter(_)
        ));
        // eps = 0 stays legal: it is the "no regularisation" limit and the
        // radius-0 identity test relies on it.
        assert!(local_contrast(img.view(), &GuidedFilterParams::new(4, 0.0), 1.0).is_ok());
        // A radius larger than the image is legal — windows clamp.
        assert!(local_contrast(img.view(), &GuidedFilterParams::new(9999, 0.01), 1.0).is_ok());
    }

    #[test]
    fn params_compare_by_value() {
        assert!(GuidedFilterParams::new(8, 0.01).__eq__(&GuidedFilterParams::new(8, 0.01)));
        assert!(!GuidedFilterParams::new(8, 0.01).__eq__(&GuidedFilterParams::new(9, 0.01)));
        assert!(!GuidedFilterParams::new(8, 0.01).__eq__(&GuidedFilterParams::new(8, 0.02)));
    }

    #[test]
    fn guided_filter_preserves_large_constant() {
        // SAT entries grow with both pixel magnitude and image area, so a
        // bright image is where the cancelling variance subtraction is
        // least well conditioned. A constant must survive it exactly.
        let img2d = ndarray::Array2::from_elem((128, 128), 1.0e7_f32);
        let q = guided_filter(img2d.view(), 8, 0.01).unwrap();
        for &v in q.iter() {
            assert!(
                (v - 1.0e7).abs() < 1.0,
                "constant 1e7 image: guided filter returned {v}"
            );
        }
    }
}
