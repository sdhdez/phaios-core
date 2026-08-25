// SPDX-License-Identifier: GPL-3.0-or-later
//! Shadow roll-off — the toe, and the counterpart to
//! [`crate::highlight_rolloff`].
//!
//! A photographic emulsion does not hold full contrast right down to
//! zero exposure. Below some threshold the density gradient falls away,
//! so the deepest shadows lose separation and run together into black
//! instead of being cut off at a hard floor. That region is the **toe**
//! of the characteristic curve, and it is one of the two things that
//! makes a film tone scale look unlike a straight line (Ferdinand Hurter
//! and Vero C. Driffield, "Photo-Chemical Investigations and a New
//! Method of Determination of the Sensitiveness of Photographic Plates",
//! *Journal of the Society of Chemical Industry* 9 (May 1890), p. 455 —
//! the paper the H&D curve is named for; the modern treatment is Hunt,
//! *The Reproduction of Colour*, 6th ed., Wiley 2004, §8.3).
//!
//! The other is the shoulder, which [`crate::highlight_rolloff`] already
//! provides. Between them sits the straight section, whose slope is the
//! contrast index — and that is exactly what
//! [`crate::tone::tone_curve`]'s slope and power control. So the whole
//! three-part shape is a composition of stages that already exist:
//!
//! ```text
//! shadow_rolloff  →  tone_curve  →  highlight_rolloff
//!      toe            straight          shoulder
//! ```
//!
//! Measured on that composition with a moderate setting (`examples/19`
//! prints the table), the slope `d(out)/d(in)` runs about 0.44 at an
//! input of 0.01, 1.09 at 0.05, 1.20 at 0.40 and 0.03 at 2.00 —
//! compressed at both ends, contrast held in the middle. At black
//! itself the slope is lower still, around 0.24. That is the
//! characteristic shape, built from three independently testable
//! kernels rather than one opaque film model.
//!
//! ## What this is not
//!
//! It is not a densitometric H&D model. It does not know about
//! base-plus-fog, D-max, or a named emulsion, and it does not claim a
//! gamma in the sense a densitometer would measure. It compresses the
//! shadow end of a linear tone scale in a controlled, monotone,
//! C¹-continuous way. For a genuine measured emulsion curve, tabulate
//! the real data and use [`crate::lut::apply_lut`] — which is why that
//! kernel exists.
//!
//! ## The curve
//!
//! On `[0, knee]` the transfer is the cubic Hermite fixed by four
//! conditions: it passes through the origin, leaves it with slope
//! `1 − strength`, and meets the identity at `knee` in both value and
//! slope. Writing `u = x / knee`:
//!
//! ```text
//! out = knee · [ (s−1)u³ + 2(1−s)u² + s·u ],   s = 1 − strength
//! ```
//!
//! `s = 1` collapses this to `out = x`, so `strength = 0` is the
//! identity — and the kernel short-circuits to a literal passthrough
//! there rather than relying on `knee · (x / knee)` to round back to `x`,
//! which it does not always do.
//!
//! Monotonicity holds for every `strength` in `0..=1`: the derivative is
//! `(1−s)(4u − 3u²) + s`, and `4u − 3u²` is non-negative across the
//! interval, so the slope never drops below `s`.
//!
//! ## Determinism
//!
//! Multiplication, addition, subtraction and one division. Every
//! operation is correctly rounded, so this kernel is bit-exact across
//! backends — see `docs/ffi.md` §6.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`shadow_rolloff`].
///
/// ```python
/// # A moderate toe over the bottom fifth of the scale
/// params = phaios_core.ShadowRolloffParams(knee=0.2, strength=0.6)
///
/// # The default leaves the image alone
/// params = phaios_core.ShadowRolloffParams()
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ShadowRolloffParams {
    /// Input value above which nothing changes, 0..=1.
    ///
    /// The toe occupies `[0, knee]`, and everything above it is returned
    /// bit-identical — the knee is a hard boundary on *where* the curve
    /// acts, not a fade. Larger values recruit more of the tonal range
    /// into the compression, which softens a wider band of shadows; the
    /// cost is paid inside that band, in the tones just below the knee
    /// that were previously untouched, not above it.
    #[pyo3(get, set)]
    pub knee: f32,
    /// How hard the shadows are compressed, 0..=1.
    ///
    /// This is one minus the slope at black. `0.0` is the identity and
    /// the default. `1.0` takes the slope at black to zero, so tones near
    /// zero run together completely — the deepest shadows become a single
    /// black rather than a graded near-black.
    #[pyo3(get, set)]
    pub strength: f32,
}

#[pymethods]
impl ShadowRolloffParams {
    /// Create new ``ShadowRolloffParams``.
    #[new]
    #[pyo3(signature = (knee = 0.2, strength = 0.0))]
    pub fn new(knee: f32, strength: f32) -> Self {
        Self { knee, strength }
    }

    /// Two ``ShadowRolloffParams`` are equal when both fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.knee.to_bits() == other.knee.to_bits()
            && self.strength.to_bits() == other.strength.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "ShadowRolloffParams(knee={}, strength={})",
            self.knee, self.strength
        )
    }
}

impl Default for ShadowRolloffParams {
    /// A knee at 0.2 with no compression — the identity.
    fn default() -> Self {
        Self {
            knee: 0.2,
            strength: 0.0,
        }
    }
}

// ── Curve ─────────────────────────────────────────────────────────────────────

/// Evaluate the toe for one sample.
///
/// Mirrored statement for statement by `src/cuda/ptx/shadow_rolloff.cu`.
#[inline]
// The negated comparisons are deliberate. `!(x < knee)` is true for NaN,
// so a non-finite sample passes straight through instead of entering the
// polynomial — and the device kernel is written the same way, so the two
// cannot disagree. Writing the positive form on one side only is the bug
// `hsl_bw` had (docs/ffi.md §1).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn shadow_sample(x: f32, knee: f32, strength: f32) -> f32 {
    if !(strength > 0.0) || !(knee > 0.0) || !(x < knee) {
        // No compression asked for, no region to compress in, the sample
        // is above the knee, or it is NaN.
        return x;
    }

    let s = 1.0 - strength;
    if x <= 0.0 {
        // −∞ must be returned before the multiply. At `strength == 1.0`
        // the slope `s` is exactly zero, and `0.0 * −∞` is NaN — so the
        // continuation would silently turn an infinity into a NaN, and
        // into a *different* NaN on each backend, which is precisely the
        // divergence the negated comparisons above exist to prevent.
        // Found by review; the original test used `strength = 0.5`,
        // where `s` is non-zero and the bug cannot appear.
        if x.is_infinite() {
            return x;
        }
        // Below zero the curve continues with its slope at the origin,
        // which keeps the transfer C¹ there. Negative samples are
        // non-physical — a negative channel-mixer weight can produce
        // them — but they must still be ordered and reproducible.
        return s * x;
    }

    let u = x / knee;
    // Horner, and the same grouping on both backends: a different
    // association would round differently and cost bit-exactness.
    knee * (u * (s + u * (2.0 * (1.0 - s) + u * (s - 1.0))))
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate toe parameters. Shared by the CPU and CUDA backends so both
/// reject the same inputs with the same messages.
pub(crate) fn validate(params: &ShadowRolloffParams) -> Result<(), PhaiosError> {
    if !params.knee.is_finite() || !(0.0..=1.0).contains(&params.knee) {
        return Err(PhaiosError::Parameter(format!(
            "knee is {}, expected a value in 0..=1",
            params.knee
        )));
    }
    if !params.strength.is_finite() || !(0.0..=1.0).contains(&params.strength) {
        return Err(PhaiosError::Parameter(format!(
            "strength is {}, expected a value in 0..=1",
            params.strength
        )));
    }
    Ok(())
}

/// Roll the deepest shadows off into black instead of holding full
/// contrast down to zero.
///
/// Above `knee` the transfer is the identity. Below it the curve bends
/// away, reaching the origin with slope `1 − strength`, so shadow
/// separation is compressed and the tones run together as they approach
/// black. At `knee` the join is C¹ in both value and slope, so the
/// transition does not show as a crease in a gradient.
///
/// The effect on separation is the point: with `knee = 0.2`, two samples
/// 0.02 apart just above black stay 0.02 apart at `strength = 0`, close
/// to 0.012 at `strength = 0.5`, and under 0.004 at `strength = 1.0`.
///
/// This is the toe of the characteristic curve.
/// [`crate::highlight_rolloff::highlight_rolloff`] is the shoulder, and
/// [`crate::tone::tone_curve`] sets the slope of the straight section
/// between them; applied in that order they compose the classic
/// three-part film tone scale. For a *measured* emulsion curve rather
/// than a parametric one, tabulate the data and use
/// [`crate::lut::apply_lut`].
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous.
///
/// Applied per channel, and on a split-toned image that makes the
/// shadows *more* saturated, not less — the opposite of what the
/// shoulder does. The shoulder pushes every channel towards the same
/// ceiling, so they converge; the toe compresses them towards zero,
/// where the same absolute compression is a larger relative one for the
/// darker channel, so their ratios widen. Measured on a pixel of
/// `(0.10, 0.12, 0.14)` at `knee = 0.2, strength = 0.8`, the chroma
/// ratio `(max − min) / max` goes from 0.286 to 0.384. Toned shadows
/// therefore deepen *and* intensify; if that is unwanted, apply the toe
/// before `split_toning` rather than after.
///
/// Order-sensitive: this belongs at the start of the tone stages, on
/// linear scene-referred data, before the contrast is set. Applying it
/// after `tone_curve` would compress shadows the curve had already
/// placed, which is a different and harder-to-predict picture.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `knee` or `strength` is outside
///   0..=1, or is not finite.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn shadow_rolloff(
    img: ArrayView3<f32>,
    params: &ShadowRolloffParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let ShadowRolloffParams { knee, strength } = *params;
    let mut out = Array3::<f32>::zeros(img.dim());

    ndarray::Zip::from(&mut out)
        .and(img)
        .par_for_each(|o, &v| *o = shadow_sample(v, knee, strength));

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn apply(v: f32, knee: f32, strength: f32) -> f32 {
        let img = array![[[v]]];
        shadow_rolloff(img.view(), &ShadowRolloffParams::new(knee, strength)).unwrap()[[0, 0, 0]]
    }

    #[test]
    fn default_is_the_exact_identity() {
        let p = ShadowRolloffParams::default();
        assert_eq!(p.strength, 0.0);
        for v in [-0.5, 0.0, 0.001, 0.1, 0.2, 0.5, 1.0, 100.0] {
            let got = shadow_rolloff(array![[[v]]].view(), &p).unwrap()[[0, 0, 0]];
            assert_eq!(
                got.to_bits(),
                v.to_bits(),
                "the default must not move {v}, got {got}"
            );
        }
    }

    #[test]
    fn strength_zero_is_exact_at_every_knee() {
        // knee · (x / knee) does not always round back to x, so the
        // identity has to be a real short-circuit, not an accident.
        for knee in [0.05_f32, 0.1, 0.2, 0.3, 0.7, 1.0] {
            for i in 0..1000 {
                let v = i as f32 / 1000.0;
                assert_eq!(
                    apply(v, knee, 0.0).to_bits(),
                    v.to_bits(),
                    "knee={knee} moved {v}"
                );
            }
        }
    }

    #[test]
    fn endpoints_are_exact() {
        for (knee, strength) in [(0.2_f32, 0.5_f32), (0.1, 1.0), (0.5, 0.8), (1.0, 0.3)] {
            assert_eq!(apply(0.0, knee, strength), 0.0, "the origin is fixed");
            assert_eq!(
                apply(knee, knee, strength).to_bits(),
                knee.to_bits(),
                "the knee is fixed"
            );
        }
    }

    #[test]
    fn above_the_knee_is_untouched() {
        for v in [0.2001_f32, 0.3, 0.5, 1.0, 4.0, 1e6] {
            assert_eq!(apply(v, 0.2, 1.0).to_bits(), v.to_bits(), "moved {v}");
        }
    }

    #[test]
    fn monotonic_and_bounded_for_every_strength() {
        for knee in [0.1_f32, 0.2, 0.5, 1.0] {
            for step in 0..=10 {
                let strength = step as f32 / 10.0;
                let mut prev = f32::NEG_INFINITY;
                for i in 0..=5000 {
                    let x = i as f32 / 5000.0 * knee;
                    let y = apply(x, knee, strength);
                    assert!(y >= prev, "knee={knee} strength={strength} dips at {x}");
                    assert!(
                        (-1e-7..=knee + 1e-7).contains(&y),
                        "knee={knee} strength={strength}: f({x}) = {y} left the interval"
                    );
                    prev = y;
                }
            }
        }
    }

    #[test]
    fn slope_at_black_is_one_minus_strength() {
        // The parameter has to mean what it says.
        let h = 1e-5_f32;
        for strength in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let measured = (apply(h, 0.2, strength) - apply(0.0, 0.2, strength)) / h;
            let want = 1.0 - strength;
            assert!(
                (measured - want).abs() < 2e-3,
                "strength={strength}: slope at black is {measured}, want {want}"
            );
        }
    }

    #[test]
    fn joins_the_identity_at_the_knee_without_a_crease() {
        let (knee, strength) = (0.2_f32, 0.8_f32);
        let h = 1e-4;
        let below = (apply(knee, knee, strength) - apply(knee - h, knee, strength)) / h;
        let above = (apply(knee + h, knee, strength) - apply(knee, knee, strength)) / h;
        assert!(
            (below - 1.0).abs() < 2e-2 && (above - 1.0).abs() < 1e-3,
            "slope must be 1 on both sides of the knee: {below} below, {above} above"
        );
    }

    /// The reason the kernel exists: shadow *separation* shrinks. A test
    /// that only checked the curve passes through its endpoints would
    /// pass on a kernel that did nothing in between.
    #[test]
    fn shadow_separation_is_compressed() {
        let (knee, gap) = (0.2_f32, 0.02_f32);
        let mut previous = f32::INFINITY;
        for strength in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let separation = apply(gap, knee, strength) - apply(0.0, knee, strength);
            assert!(
                separation < previous,
                "raising strength to {strength} must compress further: \
                 {separation} is not below {previous}"
            );
            previous = separation;
        }
        // And the endpoints of that sweep are the documented figures.
        assert!((apply(gap, knee, 0.0) - gap).abs() < 1e-6);
        assert!(
            apply(gap, knee, 1.0) < gap * 0.25,
            "full strength should compress ~5x"
        );
    }

    #[test]
    fn shadows_darken_rather_than_lift() {
        // Non-negative samples only. Below zero the continuation is
        // `s · x`, which moves a negative sample *towards* zero — that is
        // the price of keeping the transfer C¹ there, and it applies to
        // values that are already non-physical.
        //
        // The toe deepens real shadows while compressing them; a version
        // that lifted them would be a fog/black-lift control, a
        // different thing entirely.
        for i in 1..200 {
            let x = i as f32 / 1000.0;
            assert!(
                apply(x, 0.2, 0.7) <= x,
                "the toe must not lift {x} to {}",
                apply(x, 0.2, 0.7)
            );
        }
    }

    #[test]
    fn negative_samples_stay_ordered_and_continuous() {
        // The C¹ continuation below zero: slope 1 − strength.
        let strength = 0.6_f32;
        let s = 1.0 - strength;
        for v in [-1.0_f32, -0.1, -0.001] {
            let got = apply(v, 0.2, strength);
            assert!((got - s * v).abs() < 1e-7, "{v} -> {got}, want {}", s * v);
            assert!(got < 0.0, "negatives must stay negative");
        }
    }

    /// Swept across `strength`, because the interesting case is
    /// `strength == 1.0`: there the continuation slope is exactly zero
    /// and `0.0 * −∞` is NaN, so `−∞` silently became a NaN — and a
    /// differently-shaped NaN on each backend. Testing one strength
    /// missed it entirely.
    #[test]
    fn non_finite_samples_pass_through_at_every_strength() {
        for step in 0..=10 {
            let strength = step as f32 / 10.0;
            assert!(
                apply(f32::NAN, 0.2, strength).is_nan(),
                "NaN must stay NaN at strength {strength}"
            );
            assert_eq!(
                apply(f32::INFINITY, 0.2, strength),
                f32::INFINITY,
                "+inf must pass through at strength {strength}"
            );
            assert_eq!(
                apply(f32::NEG_INFINITY, 0.2, strength),
                f32::NEG_INFINITY,
                "-inf must pass through at strength {strength}, not become NaN"
            );
        }
    }

    #[test]
    fn zero_knee_is_the_identity() {
        for v in [-1.0_f32, 0.0, 0.5, 2.0] {
            assert_eq!(apply(v, 0.0, 1.0).to_bits(), v.to_bits());
        }
    }

    #[test]
    fn rejects_out_of_domain_parameters() {
        let img = array![[[0.1_f32]]];
        for (k, s) in [
            (1.5, 0.5),
            (-0.1, 0.5),
            (f32::NAN, 0.5),
            (0.2, 1.5),
            (0.2, -0.1),
            (0.2, f32::NAN),
            (0.2, f32::INFINITY),
        ] {
            assert!(
                shadow_rolloff(img.view(), &ShadowRolloffParams::new(k, s)).is_err(),
                "knee={k} strength={s} should be rejected"
            );
        }
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((6, 8, 3), |(y, x, c)| {
            ((y * 8 + x) % 40) as f32 / 200.0 + c as f32 * 0.01
        });
        let strided = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned = strided.to_owned();
        let params = ShadowRolloffParams::new(0.2, 0.7);
        let a = shadow_rolloff(strided, &params).unwrap();
        let b = shadow_rolloff(owned.view(), &params).unwrap();
        assert_eq!(a, b);
        assert!(a.is_standard_layout());
    }

    #[test]
    fn empty_input_is_accepted() {
        let img = Array3::<f32>::zeros((0, 4, 1));
        assert_eq!(
            shadow_rolloff(img.view(), &ShadowRolloffParams::new(0.2, 0.5))
                .unwrap()
                .dim(),
            (0, 4, 1)
        );
    }

    /// The doc claims the toe *increases* shadow chroma, which is the
    /// opposite of what the shoulder does and was stated backwards in
    /// the first draft. Pin the direction so it cannot invert unnoticed.
    #[test]
    fn the_toe_intensifies_toned_shadows_rather_than_washing_them_out() {
        let px = array![[[0.10_f32, 0.12, 0.14]]];
        let out = shadow_rolloff(px.view(), &ShadowRolloffParams::new(0.2, 0.8)).unwrap();

        let chroma = |r: f32, g: f32, b: f32| {
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            (mx - mn) / mx
        };
        let before = chroma(0.10, 0.12, 0.14);
        let after = chroma(out[[0, 0, 0]], out[[0, 0, 1]], out[[0, 0, 2]]);
        assert!(
            after > before,
            "the toe must widen channel ratios, not narrow them: {before} -> {after}"
        );
        // And every channel still darkened, so this is intensification,
        // not a lift.
        for c in 0..3 {
            assert!(out[[0, 0, c]] < px[[0, 0, c]]);
        }
    }

    #[test]
    fn params_equality_and_repr() {
        let a = ShadowRolloffParams::new(0.2, 0.5);
        assert!(a.__eq__(&ShadowRolloffParams::new(0.2, 0.5)));
        assert!(!a.__eq__(&ShadowRolloffParams::new(0.2, 0.6)));
        assert_eq!(a.__repr__(), "ShadowRolloffParams(knee=0.2, strength=0.5)");
        assert_eq!(ShadowRolloffParams::default().strength, 0.0);
    }
}
