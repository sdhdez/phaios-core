// SPDX-License-Identifier: GPL-3.0-or-later
//! Hot-pixel removal — a conditional (switching) median filter for RAW
//! sensor defects.
//!
//! ```text
//! m     = median9(window)              // 3×3, index-clamped at the border
//! limit = threshold + relative · |m|
//! out   = if |p − m| > limit { m } else { p }
//! ```
//!
//! # Why a hot pixel is an outlier, not noise
//!
//! A stuck or thermally-excited photosite reads as a single-pixel value
//! unrelated to its neighbourhood — an impulse (salt-and-pepper) outlier,
//! not the additive, spatially-correlated noise a Gaussian blur is built
//! to attenuate. Gonzalez & Woods, *Digital Image Processing*, 4th ed.
//! (Pearson, 2018), §5.2 "Salt-and-Pepper Noise" (p.370) names the
//! model; §5.3 "Restoration in the Presence of Noise Only — Spatial
//! Filtering" → "Order-Statistic Filters" → "Median Filter" (p.378) is
//! why an order-statistic filter is the right tool: unlike a mean, a
//! median is unmoved by a single arbitrarily-extreme sample in an
//! odd-sized window, so long as that sample is a minority of the window.
//! Tukey, *Exploratory Data Analysis*, Addison-Wesley (1977), is the
//! standard source for median smoothing more generally.
//!
//! The *conditional*/"switching" refinement below — replace only when
//! the centre sample deviates from the window median by more than a
//! criterion, otherwise keep it unchanged — is a well-established family
//! in the impulse-noise restoration literature; cited here for the
//! concept only, the same posture [`crate::local_contrast`] takes toward
//! the guided filter's originators and [`crate::sharpen`] takes toward
//! Polesel et al.
//!
//! # The two-term criterion
//!
//! `limit = threshold + relative · |m|` has an absolute term and a term
//! proportional to the local median's own magnitude. Photon shot noise
//! grows with signal (its standard deviation scales with the square root
//! of the photon count, so it still grows relative to a fixed baseline),
//! so a single absolute threshold that correctly targets a midtone hot
//! pixel is either too loose in the shadows or too tight in the
//! highlights. `relative` lets the criterion widen with local
//! brightness; `relative = 0.0` (the default) reproduces the
//! absolute-only form.
//!
//! # Order
//!
//! Right after [`crate::geometry::orient`], before
//! [`crate::geometry::straighten`] and [`crate::geometry::resize`]: a
//! bad photosite is a single extreme sample, and any resampling filter
//! (rotation, scaling) mixes it into its neighbours, smearing a
//! one-pixel defect into a blob before it can be told apart from real
//! detail. Also before this crate's planned `denoise` stage: an
//! edge-aware smoother reads a hot pixel's own extreme local contrast as
//! structure to protect, which would leave the defect largely intact
//! rather than removing it.
//!
//! # Range
//!
//! Output is always one of the nine samples already present in the
//! window — `out ∈ [min(window), max(window)]`, stronger than merely
//! "never clamps": both branches of the kernel (keep, replace) select an
//! existing sample rather than compute a new one. Channels are filtered
//! independently, so a defect in one channel of a multi-channel input
//! cannot affect the others. Finite in ⇒ finite out: every output sample
//! is copied unchanged from the input, and the one subtraction
//! (`|p − m|`) is silent on non-finite input, as elsewhere in this
//! crate.
//!
//! # Determinism
//!
//! The median step is comparison-and-selection only — [`median9`] is a
//! fixed 19-comparator network of `f32::min`/`f32::max` pairs, with **no
//! arithmetic** at all. That is a strictly stronger guarantee than
//! "correctly-rounded arithmetic on both backends", the bound most of
//! this crate's bit-exact kernels rely on. `f32::min`/`f32::max` and
//! CUDA's `fminf`/`fmaxf` both implement IEEE-754-2008 `minNum`/`maxNum`
//! semantics (return the finite operand when exactly one input is NaN),
//! and the comparator sequence never branches on the *value* being
//! compared, only on fixed indices, so host and device execute the
//! identical sequence of operations regardless of input — they agree
//! bit-for-bit even when the window holds a NaN. The kernel's one
//! arithmetic step, `|p − m|` and the comparison against `limit =
//! threshold + relative · |m|`, is ordinary correctly-rounded `f32`
//! arithmetic on both backends (no fused multiply-add). The whole kernel
//! is bit-exact between CPU and CUDA.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`hot_pixels`].
///
/// ```python
/// # Replace samples that deviate from their 3x3 median by more than an
/// # absolute term plus a term proportional to the median's magnitude.
/// params = phaios_core.HotPixelParams(threshold=0.04, relative=0.02)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HotPixelParams {
    /// Absolute term of the replace-vs-keep criterion, in the same units
    /// as the input. `0.0` is legal — it means "replace unconditionally"
    /// (an unconditional 3×3 median) — but no finite value is an
    /// identity for arbitrary input, so unlike `relative` there is no
    /// default.
    #[pyo3(get, set)]
    pub threshold: f32,
    /// Relative term, scaling the tolerance with the local median's own
    /// magnitude (`relative · |m|`): photon shot noise grows with
    /// signal, so a fixed absolute tolerance that is correct in the
    /// midtones is either too loose in the shadows or too tight in the
    /// highlights. `0.0` (the default) reproduces the absolute-only
    /// criterion.
    #[pyo3(get, set)]
    pub relative: f32,
}

#[pymethods]
impl HotPixelParams {
    /// Create new ``HotPixelParams``.
    #[new]
    #[pyo3(signature = (threshold, relative = 0.0))]
    pub fn new(threshold: f32, relative: f32) -> Self {
        Self {
            threshold,
            relative,
        }
    }

    /// Two ``HotPixelParams`` are equal when both fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.threshold.to_bits() == other.threshold.to_bits()
            && self.relative.to_bits() == other.relative.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "HotPixelParams(threshold={}, relative={})",
            self.threshold, self.relative
        )
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate hot-pixel parameters. Shared by the CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate(params: &HotPixelParams) -> Result<(), PhaiosError> {
    if !params.threshold.is_finite() || params.threshold < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "threshold is {}, expected a finite value of at least 0",
            params.threshold
        )));
    }
    if !params.relative.is_finite() || params.relative < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "relative is {}, expected a finite value of at least 0",
            params.relative
        )));
    }
    Ok(())
}

/// Sort a pair in place: `v[i]` becomes the smaller, `v[j]` the larger,
/// using only `f32::min`/`f32::max` (no arithmetic, no branch on the
/// runtime value beyond the two comparisons `min`/`max` make
/// themselves).
#[inline]
fn sort2(v: &mut [f32; 9], i: usize, j: usize) {
    let (lo, hi) = (v[i].min(v[j]), v[i].max(v[j]));
    v[i] = lo;
    v[j] = hi;
}

/// The median of nine values, via a fixed 19-comparator sorting network.
///
/// Minimal for extracting only the median (a full sort of nine needs 25
/// comparators) — the network commonly attributed to S. M. Smith (1996)
/// and widely reproduced in image-processing code, e.g. as `opt_med9` in
/// N. Devillard's public-domain "Fast median search: an ANSI C
/// implementation" (1998). Every comparator is one `(i, j)` pair below,
/// one per line, in a clearly delimited block: this exact sequence
/// transcribes verbatim into the CUDA twin's `fminf`/`fmaxf` calls, so
/// both backends run the identical sequence of operations.
///
/// No arithmetic anywhere in this function — only `f32::min`/`f32::max`,
/// which implement IEEE-754-2008 `minNum`/`maxNum`: when exactly one of
/// the two operands is NaN, the finite operand is returned. Because the
/// comparator sequence is fixed (it never branches on the value being
/// compared, only on the constant indices above), the host and a CUDA
/// twin built from `fminf`/`fmaxf` execute the identical sequence of
/// operations for any input, including one that contains a NaN — so the
/// two backends agree bit-for-bit even then. A NaN in the window does
/// not propagate to the result: the first comparator that touches its
/// slot replaces it with a copy of whatever it was compared against.
#[must_use]
pub(crate) fn median9(mut v: [f32; 9]) -> f32 {
    // ── Comparator network (19 pairs) — DO NOT REORDER ──────────────────
    sort2(&mut v, 1, 2);
    sort2(&mut v, 4, 5);
    sort2(&mut v, 7, 8);
    sort2(&mut v, 0, 1);
    sort2(&mut v, 3, 4);
    sort2(&mut v, 6, 7);
    sort2(&mut v, 1, 2);
    sort2(&mut v, 4, 5);
    sort2(&mut v, 7, 8);
    sort2(&mut v, 0, 3);
    sort2(&mut v, 5, 8);
    sort2(&mut v, 4, 7);
    sort2(&mut v, 3, 6);
    sort2(&mut v, 1, 4);
    sort2(&mut v, 2, 5);
    sort2(&mut v, 4, 7);
    sort2(&mut v, 4, 2);
    sort2(&mut v, 6, 4);
    sort2(&mut v, 4, 2);
    // ── End comparator network ───────────────────────────────────────────

    v[4]
}

/// Remove hot pixels with a conditional (switching) 3×3 median.
///
/// Computes, independently per channel, `m = median9(window)` over the
/// 3×3 neighbourhood with each of the four ±1 offsets index-clamped
/// independently to `[0, h−1]` × `[0, w−1]` (duplicate-edge padding, not
/// a shrinking window — see the module documentation), then replaces the
/// centre sample with `m` when `|p − m| > threshold + relative · |m|`,
/// and otherwise leaves it unchanged.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous. Empty input (`H`, `W` or `C` is `0`)
/// returns the empty array of the same shape.
///
/// Order-sensitive; see the module documentation. Bit-exact across
/// backends; see the module documentation and [`median9`].
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `threshold` or `relative` is negative
///   or not finite.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn hot_pixels(
    img: ArrayView3<f32>,
    params: &HotPixelParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, c) = img.dim();
    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
    if h == 0 || w == 0 || c == 0 {
        return Ok(out);
    }

    let threshold = params.threshold;
    let relative = params.relative;

    ndarray::Zip::indexed(out.rows_mut()).par_for_each(|(y, x), mut px| {
        // Each offset clamped independently — duplicate-edge padding, a
        // fixed nine-input window even at the corners. This clamp and
        // the row-major gather order below must match the CUDA kernel
        // exactly (step 2) for the two backends to agree bit-for-bit.
        let y0 = y.saturating_sub(1);
        let y2 = (y + 1).min(h - 1);
        let x0 = x.saturating_sub(1);
        let x2 = (x + 1).min(w - 1);

        for ch in 0..c {
            let window = [
                img[[y0, x0, ch]],
                img[[y0, x, ch]],
                img[[y0, x2, ch]],
                img[[y, x0, ch]],
                img[[y, x, ch]],
                img[[y, x2, ch]],
                img[[y2, x0, ch]],
                img[[y2, x, ch]],
                img[[y2, x2, ch]],
            ];
            let p = img[[y, x, ch]];
            let m = median9(window);
            let limit = threshold + relative * m.abs();
            px[ch] = if (p - m).abs() > limit { m } else { p };
        }
    });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_the_boundary_values() {
        assert!(validate(&HotPixelParams::new(0.0, 0.0)).is_ok());
        assert!(validate(&HotPixelParams::new(1.0, 1.0)).is_ok());
    }

    #[test]
    fn validate_rejects_out_of_domain_threshold() {
        for threshold in [-0.1_f32, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&HotPixelParams::new(threshold, 0.0)).unwrap_err();
            assert!(
                err.to_string().contains("threshold is"),
                "threshold={threshold}: {err}"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_domain_relative() {
        for relative in [-0.1_f32, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&HotPixelParams::new(0.05, relative)).unwrap_err();
            assert!(
                err.to_string().contains("relative is"),
                "relative={relative}: {err}"
            );
        }
    }

    // ── median9 ──────────────────────────────────────────────────────────────

    #[test]
    fn median9_of_an_increasing_sequence() {
        let window = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(median9(window).to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn median9_of_a_decreasing_sequence() {
        let window = [9.0_f32, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(median9(window).to_bits(), 5.0_f32.to_bits());
    }

    #[test]
    fn median9_of_a_constant_window() {
        let window = [3.5_f32; 9];
        assert_eq!(median9(window).to_bits(), 3.5_f32.to_bits());
    }

    /// Sweeps which of the nine positions holds a duplicate (of its
    /// neighbour), so every comparator in the network is exercised at
    /// least once with a tied pair — a case the strictly-monotonic
    /// increasing/decreasing patterns above never trigger.
    #[test]
    fn median9_with_one_duplicate_at_each_position() {
        let base = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for i in 0..9 {
            let mut window = base;
            window[i] = base[(i + 1) % 9];
            let mut sorted = window;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected = sorted[4];
            assert_eq!(
                median9(window).to_bits(),
                expected.to_bits(),
                "duplicate at position {i}: window={window:?}"
            );
        }
    }

    #[test]
    fn median9_of_a_reversed_arbitrary_window() {
        let window = [5.0_f32, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0];
        let mut sorted = window;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = sorted[4];
        assert_eq!(
            median9(window).to_bits(),
            expected.to_bits(),
            "window={window:?}"
        );

        let mut reversed = window;
        reversed.reverse();
        assert_eq!(
            median9(reversed).to_bits(),
            expected.to_bits(),
            "reversed={reversed:?}"
        );
    }

    /// `median9` against a sort-based reference over 10,000 fixed-seed
    /// pseudorandom windows (reusing [`crate::film_grain::splitmix64`],
    /// the crate's other deterministic hash — not for its noise
    /// properties here, just as a reproducible, dependency-free source
    /// of numbers). The hand-picked patterns above are the ones a
    /// transcription mistake is most likely to get wrong first; this is
    /// the general-case check.
    #[test]
    fn median9_matches_a_sort_based_reference_over_10000_random_windows() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64; // fixed seed, reproducible
        for i in 0..10_000_u32 {
            let mut window = [0.0_f32; 9];
            for slot in window.iter_mut() {
                state = crate::film_grain::splitmix64(state);
                let bits24 = (state >> 40) as i32; // 0..=0xFF_FFFF
                *slot = (bits24 - 0x0080_0000) as f32 / 8192.0;
            }
            let mut sorted = window;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let expected = sorted[4];
            let got = median9(window);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "case {i}: window={window:?} expected {expected} got {got}"
            );
        }
    }

    /// Pins the module documentation's NaN claim: a single NaN among the
    /// nine inputs must not poison the result, since the CUDA twin's
    /// `fminf`/`fmaxf` shares exactly this `minNum`/`maxNum` behaviour.
    #[test]
    fn median9_returns_a_finite_value_when_exactly_one_input_is_nan() {
        let mut window = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        window[3] = f32::NAN;
        assert!(!median9(window).is_nan(), "single NaN must not propagate");
    }

    // ── HotPixelParams ───────────────────────────────────────────────────────

    #[test]
    fn params_equality_repr_and_defaults() {
        let a = HotPixelParams::new(0.05, 0.1);
        assert!(a.__eq__(&HotPixelParams::new(0.05, 0.1)));
        assert!(!a.__eq__(&HotPixelParams::new(0.05, 0.2)));
        assert!(!a.__eq__(&HotPixelParams::new(0.06, 0.1)));
        assert_eq!(a.__repr__(), "HotPixelParams(threshold=0.05, relative=0.1)");

        let default_relative = HotPixelParams::new(0.05, 0.0);
        assert_eq!(default_relative.relative, 0.0);
    }
}
