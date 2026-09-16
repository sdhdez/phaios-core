// SPDX-License-Identifier: GPL-3.0-or-later
//! Integral images (summed-area tables).
//!
//! Shared by the kernels that need box statistics in time independent of
//! the window size: [`crate::local_contrast`] for the guided filter's
//! local mean and variance, and [`crate::film_grain`] for the
//! band-pass that gives grain its size.
//!
//! Internal to the crate — the layout of a table is an implementation
//! detail, and exposing it would freeze it.
//!
//! Reference: Franklin C. Crow, "Summed-area tables for texture
//! mapping", *SIGGRAPH '84*, pp. 207–212.

use ndarray::parallel::prelude::*;

use crate::error::PhaiosError;
use ndarray::{Array2, ArrayView2, Axis};

/// Column-block width for the vertical prefix-sum pass.
///
/// Wide enough that each rayon task does useful work, narrow enough that
/// a block's two active rows stay in cache while the pass walks down them.
pub(crate) const SAT_COL_BLOCK: usize = 512;

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
pub(crate) fn sat<F>(data: ArrayView2<f32>, f: F) -> Result<Array2<f64>, PhaiosError>
where
    F: Fn(f32) -> f64 + Sync + Send,
{
    let (h, w) = data.dim();
    let mut s = crate::alloc::zeros2::<f64>((h, w))?;

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

    Ok(s)
}

/// Query a rectangular window sum from a SAT.
///
/// Window covers rows `[y.saturating_sub(r), y.add(r).min(h-1)]` and
/// columns `[x.saturating_sub(r), x.add(r).min(w-1)]`.
/// Returns `(sum, area)`.
#[inline]
pub(crate) fn window_sum(
    s: &Array2<f64>,
    y: usize,
    x: usize,
    r: usize,
    h: usize,
    w: usize,
) -> (f64, f64) {
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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

        let table = sat(img.view(), |v| v as f64).unwrap();

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
        let squared = sat(img.view(), |v| (v as f64) * (v as f64)).unwrap();
        let expected: f64 = img.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!((squared[[7, 7]] - expected).abs() < 1e-9);
    }

    #[test]
    fn window_sum_clamps_at_the_borders() {
        // A window centred at a corner shrinks to the pixels that
        // exist, and reports the area it actually summed so the
        // caller's mean stays correct. This is not replicate padding:
        // no edge sample is counted twice.
        let img = ndarray::Array2::<f32>::ones((5, 5));
        let table = sat(img.view(), |v| v as f64).unwrap();

        let (sum, area) = window_sum(&table, 0, 0, 2, 5, 5);
        assert_eq!(area, 9.0, "corner window should be 3x3");
        assert!((sum - 9.0).abs() < 1e-9);

        let (sum, area) = window_sum(&table, 2, 2, 2, 5, 5);
        assert_eq!(area, 25.0, "centre window should cover the whole image");
        assert!((sum - 25.0).abs() < 1e-9);

        let (_, area) = window_sum(&table, 2, 2, 99, 5, 5);
        assert_eq!(area, 25.0, "an oversized radius clamps to the image");
    }
}
