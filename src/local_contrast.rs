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
//! Filtering," *ECCV 2010*, LNCS 6311, pp. 1–14. Patent-free.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

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

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "GuidedFilterParams(radius={}, eps={})",
            self.radius, self.eps
        )
    }
}

// ── Summed-area table helpers ─────────────────────────────────────────────────

/// Column-block width for the vertical prefix-sum pass.
///
/// Wide enough that each rayon task does useful work, narrow enough that
/// a block's two active rows stay in cache while the pass walks down them.
const SAT_COL_BLOCK: usize = 512;

/// Build a 2-D summed-area table of `f(data)` in f64.
///
/// Two passes, each parallel:
/// 1. Horizontal prefix sums — rows are independent.
/// 2. Vertical prefix sums — columns are independent, so the table is
///    split into blocks of [`SAT_COL_BLOCK`] columns and each block is
///    accumulated row by row. Walking row-major inside a block keeps the
///    traversal cache-friendly; a column-at-a-time pass would stride
///    across the whole row pitch on every step.
///
/// `f` maps each input sample before accumulation, which lets the caller
/// build the table of L² without materialising a full-resolution copy of
/// the squared image.
///
/// Accumulation is f64 throughout: a window statistic is the difference
/// of two large partial sums, and an f32 table would lose exactly the low
/// bits that the variance is computed from.
fn sat<F>(data: ArrayView2<f32>, f: F) -> Array2<f64>
where
    F: Fn(f32) -> f64 + Sync + Send,
{
    let (h, _) = data.dim();
    let mut s = Array2::<f64>::zeros(data.raw_dim());

    ndarray::Zip::from(s.rows_mut())
        .and(data.rows())
        .par_for_each(|mut srow, drow| {
            let mut acc = 0.0_f64;
            for (o, &v) in srow.iter_mut().zip(drow.iter()) {
                acc += f(v);
                *o = acc;
            }
        });

    s.axis_chunks_iter_mut(Axis(1), SAT_COL_BLOCK)
        .into_par_iter()
        .for_each(|mut block| {
            for y in 1..h {
                let (top, mut bottom) = block.view_mut().split_at(Axis(0), y);
                let previous = top.row(y - 1);
                let mut current = bottom.row_mut(0);
                current += &previous;
            }
        });

    s
}

/// Query a rectangular window sum from a SAT.
///
/// Window covers rows `[y.saturating_sub(r), y.add(r).min(h-1)]` and
/// columns `[x.saturating_sub(r), x.add(r).min(w-1)]`.
/// Returns `(sum, area)`.
#[inline]
fn window_sum(s: &Array2<f64>, y: usize, x: usize, r: usize, h: usize, w: usize) -> (f64, f64) {
    let y1 = y.saturating_sub(r);
    let x1 = x.saturating_sub(r);
    let y2 = (y + r).min(h - 1);
    let x2 = (x + r).min(w - 1);

    let br = s[[y2, x2]];
    let tl = if y1 > 0 && x1 > 0 {
        s[[y1 - 1, x1 - 1]]
    } else {
        0.0
    };
    let tr = if y1 > 0 { s[[y1 - 1, x2]] } else { 0.0 };
    let bl = if x1 > 0 { s[[y2, x1 - 1]] } else { 0.0 };

    let sum = br - tr - bl + tl;
    let area = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f64;
    (sum, area)
}

// ── Guided filter (internal) ──────────────────────────────────────────────────

/// Apply the He–Sun–Tang guided filter (self-guided, 2-D).
///
/// Not exposed to Python. Called by [`local_contrast`].
fn guided_filter(img: ArrayView2<f32>, radius: u32, eps: f32) -> Array2<f32> {
    let (h, w) = img.dim();
    let r = radius as usize;
    let eps_f64 = eps as f64;

    // Per-pixel linear-model coefficients for the window centred on that
    // pixel. Held as f32: they are averaged through another SAT, and the
    // f64 precision only matters while accumulating.
    let mut a_arr = Array2::<f32>::zeros((h, w));
    let mut b_arr = Array2::<f32>::zeros((h, w));

    {
        let sat_l = sat(img, |v| v as f64);
        let sat_l2 = sat(img, |v| {
            let d = v as f64;
            d * d
        });

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
    let sat_a = sat(a_arr.view(), |v| v as f64);
    drop(a_arr);
    let sat_b = sat(b_arr.view(), |v| v as f64);
    drop(b_arr);

    let mut out = Array2::<f32>::zeros((h, w));
    ndarray::Zip::indexed(&mut out)
        .and(img)
        .par_for_each(|(y, x), o, &l| {
            let (sum_a, area) = window_sum(&sat_a, y, x, r, h, w);
            let (sum_b, _) = window_sum(&sat_b, y, x, r, h, w);
            let mean_a = sum_a / area;
            let mean_b = sum_b / area;
            *o = (mean_a * l as f64 + mean_b) as f32;
        });

    out
}

// ── Public kernel ─────────────────────────────────────────────────────────────

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
/// Input shape: `(H, W, 1)`. Output shape: `(H, W, 1)`.
///
/// Reference: He, Sun, Tang, "Guided Image Filtering," ECCV 2010.
///
/// # Errors
/// Returns [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn local_contrast(
    img: ArrayView3<f32>,
    params: &GuidedFilterParams,
    strength: f32,
) -> Result<Array3<f32>, PhaiosError> {
    if img.shape()[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "local_contrast expects (H, W, 1) luminance input, got shape {:?}",
            img.shape()
        )));
    }
    let (h, w, _) = img.dim();
    let img2d = img.index_axis(ndarray::Axis(2), 0);
    let smooth = guided_filter(img2d, params.radius, params.eps);
    let mut out = Array3::<f32>::zeros((h, w, 1));
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
        let q = guided_filter(img2d.view(), 4, 0.01);
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
        let q = guided_filter(img2d.view(), 0, 0.0);
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
    fn sat_matches_naive_prefix_sum() {
        // The table is built in two parallel passes over column blocks;
        // this pins it to the textbook definition
        // S[y,x] = Σ_{j≤y, i≤x} f[j,i]. The image is deliberately wider
        // than SAT_COL_BLOCK so more than one block is exercised, and the
        // column count is not a multiple of it so the ragged last block is
        // covered too.
        let (h, w) = (40, SAT_COL_BLOCK + 137);
        let img =
            ndarray::Array2::from_shape_fn((h, w), |(y, x)| ((y * 31 + x * 17) % 97) as f32 / 97.0);

        let table = sat(img.view(), |v| v as f64);

        let mut naive = ndarray::Array2::<f64>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let v = img[[y, x]] as f64;
                let above = if y > 0 { naive[[y - 1, x]] } else { 0.0 };
                let left = if x > 0 { naive[[y, x - 1]] } else { 0.0 };
                let diag = if y > 0 && x > 0 {
                    naive[[y - 1, x - 1]]
                } else {
                    0.0
                };
                naive[[y, x]] = v + above + left - diag;
            }
        }

        for (a, b) in table.iter().zip(naive.iter()) {
            assert!((a - b).abs() < 1e-9, "SAT mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn sat_applies_its_mapping_function() {
        // The L² table is built by mapping during accumulation rather than
        // materialising a squared copy of the image.
        let img = ndarray::Array2::from_shape_fn((8, 8), |(y, x)| (y + x) as f32);
        let squared = sat(img.view(), |v| (v as f64) * (v as f64));
        let expected: f64 = img.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!((squared[[7, 7]] - expected).abs() < 1e-9);
    }

    #[test]
    fn guided_filter_preserves_large_constant() {
        // SAT entries grow with both pixel magnitude and image area, so a
        // bright image is where the cancelling variance subtraction is
        // least well conditioned. A constant must survive it exactly.
        let img2d = ndarray::Array2::from_elem((128, 128), 1.0e7_f32);
        let q = guided_filter(img2d.view(), 8, 0.01);
        for &v in q.iter() {
            assert!(
                (v - 1.0e7).abs() < 1.0,
                "constant 1e7 image: guided filter returned {v}"
            );
        }
    }
}
