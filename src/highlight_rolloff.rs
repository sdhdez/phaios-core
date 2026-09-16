// SPDX-License-Identifier: GPL-3.0-or-later
//! Highlight roll-off — the explicit choice between clipping and a shoulder.
//!
//! Every stage before this one preserves values above 1.0: an exposure
//! push, a specular reflection and the sun itself all leave the pipeline
//! carrying scene-referred highlights with nowhere to go on a display.
//! Something has to map them into the displayable range, and until this
//! kernel existed that something was whoever called `np.clip` — a hard
//! clip, which flattens every value above 1.0 onto the same white and
//! leaves a visible edge where the image crosses it.
//!
//! Photographic emulsion does not behave that way. Density approaches
//! maximum along a *shoulder*, compressing highlight detail rather than
//! discarding it, and that gradual approach is a large part of why film
//! highlights read as they do (Hunt, *The Reproduction of Colour*, 6th
//! ed., Wiley 2004, §8.3, on the shape of the characteristic curve).
//!
//! This kernel puts the decision in the caller's hands with two
//! photographic parameters:
//!
//! - `knee` — the value below which nothing changes at all;
//! - `white_point` — the scene value that becomes pure white.
//!
//! The default (`1.0`, `1.0`) reproduces a hard clip exactly, so adding
//! the stage to a pipeline changes nothing until it is asked to.
//!
//! ## The curve
//!
//! Between the knee and the white point the transfer is a quadratic
//! Bézier with control points `(k, k)`, `(1, 1)` and `(W, 1)`. Those
//! control points are not arbitrary: the tangent at the start is the
//! line `y = x` and the tangent at the end is `y = 1`, and those two
//! lines meet at `(1, 1)`. The curve therefore leaves the identity at
//! slope 1 and arrives at white at slope 0, so neither the knee nor the
//! white point produces a visible edge, and the result is C¹ across the
//! whole range.
//!
//! Writing `x(t)` and `y(t)` for the Bézier components and solving the
//! quadratic `x(t) = x` for `t` gives the closed form used below. The
//! vertical component collapses pleasantly:
//!
//! ```text
//! y = 1 − (1 − k)(1 − t)²
//! ```
//!
//! Reference for the construction: Farin, *Curves and Surfaces for CAGD*,
//! 5th ed., Morgan Kaufmann (2002), §4.2 (Bézier curves) and §3.3
//! (the tangent-intersection form of the control polygon).
//!
//! ## Determinism
//!
//! The evaluation uses only addition, subtraction, multiplication,
//! division and square root. All five are correctly rounded by
//! IEEE-754-2008 §5.4.1, so this kernel is bit-exact across backends —
//! see `docs/ffi.md` §6.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`highlight_rolloff`].
///
/// ```python
/// # Hold two stops of highlight detail above the knee
/// params = phaios_core.RolloffParams(knee=0.75, white_point=4.0)
///
/// # The default is a hard clip at 1.0, identical to np.clip(x, None, 1.0)
/// params = phaios_core.RolloffParams()
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct RolloffParams {
    /// Where compression begins, 0..=1.
    ///
    /// Values at or below the knee pass through untouched, so this is
    /// the promise that midtones are not disturbed. Lowering it recruits
    /// more of the tonal range into the shoulder, which buys smoother
    /// highlights at the cost of some contrast just below white.
    #[pyo3(get, set)]
    pub knee: f32,
    /// The scene value that becomes pure white, ≥ 1.0.
    ///
    /// `1.0` means "clip at 1.0" and gives back the hard clip. `4.0`
    /// means a value two stops above nominal white is what finally
    /// reaches 1.0, so those two stops of highlight survive as detail
    /// instead of collapsing to a flat patch. Anything above the white
    /// point is white.
    #[pyo3(get, set)]
    pub white_point: f32,
}

#[pymethods]
impl RolloffParams {
    /// Create new ``RolloffParams``.
    #[new]
    #[pyo3(signature = (knee = 1.0, white_point = 1.0))]
    pub fn new(knee: f32, white_point: f32) -> Self {
        Self { knee, white_point }
    }

    /// Two ``RolloffParams`` are equal when both fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.knee.to_bits() == other.knee.to_bits()
            && self.white_point.to_bits() == other.white_point.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "RolloffParams(knee={}, white_point={})",
            self.knee, self.white_point
        )
    }
}

impl Default for RolloffParams {
    /// A hard clip at 1.0 — the behaviour callers had before this kernel.
    fn default() -> Self {
        Self {
            knee: 1.0,
            white_point: 1.0,
        }
    }
}

// ── Curve ─────────────────────────────────────────────────────────────────────

/// Evaluate the shoulder for one sample.
///
/// Shared verbatim in structure by the CUDA kernel; any change here must
/// be mirrored in `src/cuda/ptx/highlight_rolloff.cu`.
#[inline]
// The negated comparison is deliberate and must not be "simplified" to
// `x <= knee`: the two differ precisely on NaN, which fails every
// comparison. `!(NaN > k)` is true, so a non-finite sample takes the
// pass-through branch instead of falling into the solve — and the CUDA
// kernel is written the same way, so both backends agree on it. Writing
// the positive form on one side and not the other is the exact bug
// `hsl_bw` had (docs/ffi.md §1).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn rolloff_sample(x: f32, knee: f32, white: f32) -> f32 {
    if !(x > knee) {
        return x;
    }
    if x >= white {
        return 1.0;
    }

    // x(t) = (W + k − 2)t² + 2(1 − k)t + k, solved for t at this x.
    let a = white + knee - 2.0;
    let b = 2.0 * (1.0 - knee);
    let c = knee - x;

    let t = if a == 0.0 {
        // W = 2 − k: the quadratic degenerates to a line. This is the
        // classic parabolic shoulder, and it is a legal configuration
        // rather than an edge case, so it gets the exact linear solve
        // instead of a division by zero.
        -c / b
    } else {
        // The `+` root is the one in [0, 1]; `a` may be either sign, but
        // the discriminant is non-negative throughout the valid domain.
        let disc = b * b - 4.0 * a * c;
        (-b + disc.max(0.0).sqrt()) / (2.0 * a)
    };

    let one_minus_t = 1.0 - t;
    1.0 - (1.0 - knee) * one_minus_t * one_minus_t
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate roll-off parameters. Shared by the CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate(params: &RolloffParams) -> Result<(), PhaiosError> {
    if !params.knee.is_finite() || !(0.0..=1.0).contains(&params.knee) {
        return Err(PhaiosError::Parameter(format!(
            "knee is {}, expected a value in 0..=1",
            params.knee
        )));
    }
    if !params.white_point.is_finite() || params.white_point < 1.0 {
        return Err(PhaiosError::Parameter(format!(
            "white_point is {}, expected a finite value >= 1.0",
            params.white_point
        )));
    }
    Ok(())
}

/// Roll highlights off into `[knee, 1.0]` instead of clipping them.
///
/// Below `knee` the transfer is the identity. Between `knee` and
/// `white_point` it follows the quadratic Bézier described in the module
/// documentation, leaving the identity at slope 1 and reaching exactly
/// 1.0 at `white_point` with slope 0. At and above `white_point` the
/// result is 1.0.
///
/// The output is therefore never greater than 1.0 and never less than
/// the input's own value below the knee — this is the stage that ends
/// the pipeline's highlight headroom, deliberately and where the caller
/// can see it.
///
/// Values below zero are left alone: clamping the black end is the
/// caller's decision and belongs to a different stage.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous. The curve is applied per channel, so on a
/// split-toned image an extreme highlight desaturates towards white as
/// each channel saturates in turn — the same thing film does, and
/// usually what is wanted.
///
/// Order-sensitive: this is the last stage that operates on linear
/// scene-referred data, immediately before [`crate::encode::encode_srgb`].
/// Running it after the transfer encoding would compress the wrong
/// quantity, and running it before the tone stages would let them lift
/// values back above 1.0 afterwards.
///
/// The default parameters are a hard clip at 1.0, so inserting this
/// stage with defaults reproduces `clip(x, 0, 1)` for the highlight end
/// exactly.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `knee` is outside 0..=1, or if
///   `white_point` is not finite or is below 1.0.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn highlight_rolloff(
    img: ArrayView3<f32>,
    params: &RolloffParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let RolloffParams { knee, white_point } = *params;
    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;

    ndarray::Zip::from(&mut out)
        .and(img)
        .par_for_each(|o, &v| *o = rolloff_sample(v, knee, white_point));

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn apply(v: f32, knee: f32, white: f32) -> f32 {
        let img = array![[[v]]];
        highlight_rolloff(img.view(), &RolloffParams::new(knee, white)).unwrap()[[0, 0, 0]]
    }

    #[test]
    fn default_is_a_hard_clip() {
        let p = RolloffParams::default();
        assert_eq!(p.knee, 1.0);
        assert_eq!(p.white_point, 1.0);
        for v in [-0.5, 0.0, 0.25, 0.999, 1.0, 1.5, 100.0] {
            let got = apply(v, 1.0, 1.0);
            let want = v.min(1.0);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "default must clip exactly: {v} -> {got}, want {want}"
            );
        }
    }

    #[test]
    fn endpoints_are_exact() {
        for (k, w) in [(0.8, 2.0), (0.5, 4.0), (0.0, 1.0), (0.95, 16.0)] {
            assert_eq!(apply(k, k, w), k, "f(knee) must be knee exactly");
            assert_eq!(apply(w, k, w), 1.0, "f(white) must be 1.0 exactly");
        }
    }

    #[test]
    fn never_exceeds_one_and_is_monotonic() {
        for (k, w) in [(0.8, 2.0), (0.5, 4.0), (0.7, 8.0), (0.2, 1.0)] {
            let mut prev = f32::NEG_INFINITY;
            for i in 0..=20_000 {
                let x = i as f32 / 1000.0; // 0 .. 20
                let y = apply(x, k, w);
                assert!(y <= 1.0, "knee={k} white={w}: f({x}) = {y} exceeds 1.0");
                assert!(y >= prev, "knee={k} white={w}: not monotonic at {x}");
                prev = y;
            }
        }
    }

    #[test]
    fn below_the_knee_is_untouched() {
        // Bit-exact identity, not merely close: the promise is that
        // midtones do not move at all.
        for v in [-1.0, 0.0, 0.18, 0.5, 0.7999] {
            assert_eq!(
                apply(v, 0.8, 4.0).to_bits(),
                v.to_bits(),
                "value below the knee must be untouched: {v}"
            );
        }
    }

    #[test]
    fn joins_the_identity_at_slope_one() {
        // A kink at the knee would show as a visible edge in a gradient.
        let (k, w) = (0.8_f32, 4.0_f32);
        let h = 1e-4;
        let slope_below = (apply(k, k, w) - apply(k - h, k, w)) / h;
        let slope_above = (apply(k + h, k, w) - apply(k, k, w)) / h;
        assert!(
            (slope_below - 1.0).abs() < 1e-2,
            "slope below the knee should be 1, got {slope_below}"
        );
        assert!(
            (slope_above - 1.0).abs() < 5e-2,
            "slope above the knee should also be ~1, got {slope_above}"
        );
    }

    #[test]
    fn lands_on_white_flat() {
        // Slope 0 at the white point means no edge where clipping starts.
        let (k, w) = (0.8_f32, 4.0_f32);
        let h = 1e-3;
        let slope = (apply(w, k, w) - apply(w - h, k, w)) / h;
        assert!(
            slope.abs() < 1e-2,
            "the curve should arrive at white flat, slope {slope}"
        );
    }

    #[test]
    fn parabolic_case_is_handled_without_dividing_by_zero() {
        // white = 2 − knee makes the quadratic coefficient exactly zero.
        let k = 0.6_f32;
        let w = 2.0 - k;
        for i in 0..=100 {
            let x = k + (w - k) * i as f32 / 100.0;
            let y = apply(x, k, w);
            assert!(y.is_finite(), "degenerate case produced {y} at {x}");
            assert!((k - 1e-6..=1.0 + 1e-6).contains(&y));
        }
        assert_eq!(apply(w, k, w), 1.0);
    }

    #[test]
    fn non_finite_samples_pass_through() {
        // Finiteness is a precondition (docs/ffi.md §1); what matters is
        // that the two backends agree, which needs the negated test in
        // `rolloff_sample` rather than a fall-through into the solve.
        assert!(apply(f32::NAN, 0.8, 4.0).is_nan());
        assert_eq!(apply(f32::INFINITY, 0.8, 4.0), 1.0);
        assert_eq!(apply(f32::NEG_INFINITY, 0.8, 4.0), f32::NEG_INFINITY);
    }

    #[test]
    fn rejects_out_of_domain_parameters() {
        let img = array![[[0.5_f32]]];
        for (k, w) in [
            (1.5, 2.0),
            (-0.1, 2.0),
            (f32::NAN, 2.0),
            (0.5, 0.5),
            (0.5, f32::NAN),
            (0.5, f32::INFINITY),
        ] {
            assert!(
                highlight_rolloff(img.view(), &RolloffParams::new(k, w)).is_err(),
                "knee={k} white={w} should be rejected"
            );
        }
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((6, 8, 3), |(y, x, c)| {
            (y * 8 + x) as f32 / 10.0 + c as f32
        });
        let strided = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned = strided.to_owned();
        let params = RolloffParams::new(0.7, 3.0);
        let a = highlight_rolloff(strided, &params).unwrap();
        let b = highlight_rolloff(owned.view(), &params).unwrap();
        assert_eq!(a, b);
        assert!(a.is_standard_layout());
    }

    #[test]
    fn empty_input_is_accepted() {
        let img = Array3::<f32>::zeros((0, 5, 1));
        let out = highlight_rolloff(img.view(), &RolloffParams::new(0.8, 2.0)).unwrap();
        assert_eq!(out.dim(), (0, 5, 1));
    }

    #[test]
    fn params_equality_and_repr() {
        let a = RolloffParams::new(0.8, 2.0);
        let b = RolloffParams::new(0.8, 2.0);
        let c = RolloffParams::new(0.8, 2.5);
        assert!(a.__eq__(&b));
        assert!(!a.__eq__(&c));
        assert_eq!(a.__repr__(), "RolloffParams(knee=0.8, white_point=2)");
    }
}
