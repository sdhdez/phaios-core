// SPDX-License-Identifier: GPL-3.0-or-later
//! Arbitrary tone transfer through a 1-D lookup table.
//!
//! One kernel that subsumes a family. Give it a table and it becomes
//! whatever transfer that table describes:
//!
//! | To get | Build the table from |
//! |---|---|
//! | An arbitrary curve from a UI spline | the spline, sampled |
//! | Histogram equalisation | [`crate::histogram::Histogram::cdf`] |
//! | Histogram matching | one image's CDF composed with another's inverse |
//! | A film characteristic curve | tabulated H&D data |
//! | Solarisation (Sabattier) | a deliberately non-monotone table |
//!
//! That last row is the one the existing tone kernels structurally
//! cannot reach. [`crate::tone::tone_curve`] is a single power function
//! and [`crate::tone::zone_system`] blends offsets in log space; both are
//! monotone by construction, which is usually a virtue and occasionally
//! the exact thing standing between you and a darkroom effect. A lookup
//! table has no such opinion, so this kernel deliberately does **not**
//! check monotonicity.
//!
//! # Why a table rather than more curve kernels
//!
//! Every curve shape a front-end wants is a different formula, and
//! adding one kernel per formula would grow the API without bound while
//! still never covering the one the next photographer asks for. Sampling
//! whatever the front-end already draws into a table moves the shape to
//! where the shape is decided, and leaves the core with one small,
//! testable, bit-exact evaluation.
//!
//! # The cost, stated plainly
//!
//! A table is *data*, not a parameter. Two floats reproduce a
//! [`crate::tone::ToneCurveParams`]; reproducing a LUT means storing the
//! whole table alongside the settings. A consumer that promises exact
//! reproduction must write the table into its sidecar, not just a
//! reference to whichever spline widget produced it. See
//! `docs/export.md` §6.
//!
//! # Determinism
//!
//! Evaluation is one subtraction, one multiply, a floor, and a linear
//! interpolation — multiply and add. Every operation is exact or
//! correctly rounded, so the kernel is bit-exact across backends.

use ndarray::{Array3, ArrayView1, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameters ────────────────────────────────────────────────────────────────

/// Domain of the table passed to [`apply_lut`].
///
/// ```python
/// # A table covering the display range
/// params = phaios_core.LutParams()
///
/// # A table covering three stops of highlight headroom
/// params = phaios_core.LutParams(min=0.0, max=8.0)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct LutParams {
    /// Input value mapped to the table's first entry.
    ///
    /// Anything at or below this takes the first entry — the table is
    /// clamped, not extrapolated. Extrapolating a curve the photographer
    /// drew past the range they drew it over invents data.
    #[pyo3(get, set)]
    pub min: f32,
    /// Input value mapped to the table's last entry.
    #[pyo3(get, set)]
    pub max: f32,
}

#[pymethods]
impl LutParams {
    /// Create new ``LutParams``.
    #[new]
    #[pyo3(signature = (min = 0.0, max = 1.0))]
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Two ``LutParams`` are equal when both bounds match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.min.to_bits() == other.min.to_bits() && self.max.to_bits() == other.max.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!("LutParams(min={}, max={})", self.min, self.max)
    }
}

impl Default for LutParams {
    fn default() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

// ── Evaluation ────────────────────────────────────────────────────────────────

/// Look one sample up in the table, with linear interpolation.
///
/// Mirrored statement for statement by `src/cuda/ptx/lut.cu`.
#[inline]
// `!(t > 0.0)` rather than `t <= 0.0`: deliberate, and the same reasoning
// as `highlight_rolloff`. NaN is handled explicitly above, but keeping the
// negated form means a NaN that ever reached here would clamp rather than
// index out of bounds.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn lut_sample(value: f32, lut: &[f32], min: f32, max: f32) -> f32 {
    if value.is_nan() {
        // Propagate the *caller's* NaN bit pattern, not a fresh canonical
        // one: `f32::NAN` is positive, and every NaN ordinary f32
        // arithmetic produces on x86 is negative, so returning the
        // constant made the CPU disagree with the device — which returns
        // `value` — on the common case. Mapping NaN to an end entry would
        // be worse still, inventing a position on the curve for a sample
        // that has none.
        return value;
    }

    let n = lut.len();
    let last = n - 1;
    let t = (value - min) / (max - min) * last as f32;

    if !(t > 0.0) {
        // At or below the domain, or a negative t. Clamped, not
        // extrapolated.
        return lut[0];
    }
    if t >= last as f32 {
        return lut[last];
    }

    let i = t as usize; // floor, and t is in (0, last) so this is < last
    let frac = t - i as f32;
    // Written as a convex combination rather than `a + frac * (b - a)`.
    // The difference form overflows to infinity when two adjacent entries
    // are more than f32::MAX apart, producing a non-finite result from a
    // table `validate` certified as entirely finite. This form is bounded
    // by max(|a|, |b|), and still reproduces both knots exactly: at
    // frac = 0 it is `1·a + 0·b`, at frac = 1 it is `0·a + 1·b`.
    (1.0 - frac) * lut[i] + frac * lut[i + 1]
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate the table and its domain. Shared by the CPU and CUDA
/// backends so both reject the same inputs with the same messages.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn validate(lut: ArrayView1<f32>, params: &LutParams) -> Result<(), PhaiosError> {
    if lut.len() < 2 {
        return Err(PhaiosError::Parameter(format!(
            "lut has {} entries, expected at least 2",
            lut.len()
        )));
    }
    if !params.min.is_finite() || !params.max.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "domain is ({}, {}), expected finite bounds",
            params.min, params.max
        )));
    }
    if !(params.max > params.min) {
        return Err(PhaiosError::Parameter(format!(
            "domain is ({}, {}), expected max > min",
            params.min, params.max
        )));
    }
    // As in `histogram`: a span wider than f32 can represent overflows to
    // infinity, after which every sample maps to the first table entry.
    if !(params.max - params.min).is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "domain ({}, {}) spans more than f32 can represent",
            params.min, params.max
        )));
    }
    // A non-finite entry would poison every sample that interpolates
    // against it, and the caller would see it as a corrupt image rather
    // than as a bad table. Cheap to check: the table is small next to the
    // image, and this runs once per call, not once per pixel.
    if let Some((i, v)) = lut.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(PhaiosError::Parameter(format!(
            "lut[{i}] is {v}, expected every entry to be finite"
        )));
    }
    Ok(())
}

/// Apply a 1-D lookup table as a tone transfer.
///
/// The table's entries are spread evenly across `[params.min,
/// params.max]`; a sample between two entries is linearly interpolated,
/// and one outside the domain takes the nearest end entry. The table is
/// applied to every channel alike, which is what a black-and-white tone
/// curve is; per-channel tables are not supported, and a caller who
/// wants them should apply the kernel to each channel separately.
///
/// The table is **not** required to be monotone. Non-monotone tables are
/// how solarisation and other darkroom reversal effects are expressed,
/// and forbidding them would remove the one thing this kernel can do
/// that [`crate::tone::tone_curve`] cannot.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous.
///
/// Order-sensitive in the same way as any tone stage: the table was
/// drawn against some assumed input range, and `params` has to describe
/// that range. A table sampled from a display-referred curve applied to
/// linear data is a different picture, and a wrong one.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if the table has fewer than two entries,
///   contains a non-finite value, or the domain bounds are not finite
///   with `max > min`.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn apply_lut(
    img: ArrayView3<f32>,
    lut: ArrayView1<f32>,
    params: &LutParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate(lut, params)?;

    // A contiguous slice for the hot loop. The table is small, and a
    // caller passing a strided view of one should not pay for it per
    // pixel.
    // `as_standard_layout`, not `to_owned`: `to_owned` preserves the
    // source's memory order, so a reversed view stays negative-stride and
    // `as_slice` returns None a second time — the review found that the
    // `expect` then fired as a PanicException across the FFI, on caller
    // layout, which CLAUDE.md §2 forbids by name. `Context::upload` uses
    // `as_standard_layout` for exactly this reason. Reversed tables are
    // not exotic: `np.flip(cdf)` is the documented way to build a
    // histogram-matching transfer.
    let compact = lut.as_standard_layout();
    let table: &[f32] = compact
        .as_slice()
        .expect("as_standard_layout guarantees a contiguous C-order array");

    let LutParams { min, max } = *params;
    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;
    ndarray::Zip::from(&mut out)
        .and(img)
        .par_for_each(|o, &v| *o = lut_sample(v, table, min, max));

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3, array};

    fn identity_lut(n: usize) -> Array1<f32> {
        Array1::from_shape_fn(n, |i| i as f32 / (n - 1) as f32)
    }

    fn apply(v: f32, lut: &Array1<f32>, min: f32, max: f32) -> f32 {
        apply_lut(array![[[v]]].view(), lut.view(), &LutParams::new(min, max)).unwrap()[[0, 0, 0]]
    }

    #[test]
    fn an_identity_table_is_the_identity() {
        // Not approximately: an identity ramp sampled at its own knots
        // must reproduce them exactly.
        let lut = identity_lut(256);
        for i in 0..256 {
            let v = i as f32 / 255.0;
            let got = apply(v, &lut, 0.0, 1.0);
            assert!((got - v).abs() < 1e-6, "identity table moved {v} to {got}");
        }
    }

    #[test]
    fn endpoints_hit_the_end_entries_exactly() {
        let lut = array![0.25_f32, 0.5, 0.75];
        assert_eq!(apply(0.0, &lut, 0.0, 1.0), 0.25);
        assert_eq!(apply(1.0, &lut, 0.0, 1.0), 0.75);
        assert_eq!(apply(0.5, &lut, 0.0, 1.0), 0.5, "the middle knot");
    }

    #[test]
    fn interpolation_is_linear_between_knots() {
        let lut = array![0.0_f32, 1.0]; // two entries: a straight ramp
        for (v, want) in [
            (0.0, 0.0),
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
            (1.0, 1.0),
        ] {
            let got = apply(v, &lut, 0.0, 1.0);
            assert!((got - want).abs() < 1e-6, "at {v}: got {got}, want {want}");
        }
    }

    #[test]
    fn outside_the_domain_clamps_rather_than_extrapolating() {
        let lut = array![0.2_f32, 0.8];
        assert_eq!(apply(-5.0, &lut, 0.0, 1.0), 0.2);
        assert_eq!(apply(5.0, &lut, 0.0, 1.0), 0.8);
        assert_eq!(apply(f32::NEG_INFINITY, &lut, 0.0, 1.0), 0.2);
        assert_eq!(apply(f32::INFINITY, &lut, 0.0, 1.0), 0.8);
    }

    #[test]
    fn nan_propagates_rather_than_being_invented_away() {
        let lut = identity_lut(16);
        assert!(apply(f32::NAN, &lut, 0.0, 1.0).is_nan());
    }

    #[test]
    fn a_non_monotone_table_is_allowed() {
        // Solarisation: the transfer reverses above the midpoint. This is
        // the case tone_curve and zone_system structurally cannot express.
        let lut = Array1::from_shape_fn(256, |i| {
            let t = i as f32 / 255.0;
            if t < 0.5 { t * 2.0 } else { (1.0 - t) * 2.0 }
        });
        assert!(
            apply_lut(
                Array3::<f32>::from_elem((4, 4, 1), 0.75).view(),
                lut.view(),
                &LutParams::default()
            )
            .is_ok()
        );
        // Below and above the turning point map to the same output.
        let low = apply(0.25, &lut, 0.0, 1.0);
        let high = apply(0.75, &lut, 0.0, 1.0);
        assert!(
            (low - high).abs() < 1e-3,
            "the fold should be symmetric: {low} vs {high}"
        );
    }

    #[test]
    fn a_custom_domain_rescales_the_table() {
        let lut = array![0.0_f32, 1.0];
        // Domain 0..4: an input of 2.0 is the midpoint.
        assert!((apply(2.0, &lut, 0.0, 4.0) - 0.5).abs() < 1e-6);
        assert_eq!(apply(4.0, &lut, 0.0, 4.0), 1.0);
        assert_eq!(apply(0.0, &lut, 0.0, 4.0), 0.0);
    }

    #[test]
    fn two_entry_and_large_tables_both_work() {
        for n in [2_usize, 3, 17, 256, 4096, 65536] {
            let lut = identity_lut(n);
            let got = apply(0.5, &lut, 0.0, 1.0);
            assert!((got - 0.5).abs() < 1e-4, "n={n}: 0.5 -> {got}");
        }
    }

    #[test]
    fn accepts_any_layout_and_a_strided_table() {
        let img = Array3::<f32>::from_shape_fn((10, 14, 3), |(y, x, c)| {
            ((y * 14 + x + c) % 100) as f32 / 99.0
        });
        let lut = identity_lut(64);
        let strided_img = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned_img = strided_img.to_owned();
        let p = LutParams::default();
        assert_eq!(
            apply_lut(strided_img, lut.view(), &p).unwrap(),
            apply_lut(owned_img.view(), lut.view(), &p).unwrap()
        );

        // A strided table must give the same answer as its compacted copy.
        let wide = identity_lut(128);
        let strided_lut = wide.slice(ndarray::s![..;2]);
        let compact = strided_lut.to_owned();
        assert_eq!(
            apply_lut(img.view(), strided_lut, &p).unwrap(),
            apply_lut(img.view(), compact.view(), &p).unwrap()
        );
    }

    #[test]
    fn empty_image_is_accepted() {
        let lut = identity_lut(8);
        let img = Array3::<f32>::zeros((0, 3, 1));
        assert_eq!(
            apply_lut(img.view(), lut.view(), &LutParams::default())
                .unwrap()
                .dim(),
            (0, 3, 1)
        );
    }

    #[test]
    fn rejects_bad_tables_and_domains() {
        let img = array![[[0.5_f32]]];
        let good = identity_lut(8);

        assert!(apply_lut(img.view(), array![0.5_f32].view(), &LutParams::default()).is_err());
        assert!(
            apply_lut(
                img.view(),
                Array1::<f32>::zeros(0).view(),
                &LutParams::default()
            )
            .is_err()
        );
        assert!(
            apply_lut(
                img.view(),
                array![0.0_f32, f32::NAN].view(),
                &LutParams::default()
            )
            .is_err()
        );
        assert!(
            apply_lut(
                img.view(),
                array![0.0_f32, f32::INFINITY].view(),
                &LutParams::default()
            )
            .is_err()
        );
        for p in [
            LutParams::new(1.0, 0.0),
            LutParams::new(0.0, 0.0),
            LutParams::new(f32::NAN, 1.0),
            LutParams::new(0.0, f32::INFINITY),
            LutParams::new(-3.0e38, 3.0e38),
            LutParams::new(f32::MIN, f32::MAX),
        ] {
            assert!(apply_lut(img.view(), good.view(), &p).is_err(), "{p:?}");
        }
    }

    /// Finding 13: no CPU test used a non-zero `LutParams.min`, so
    /// dropping the `- min` term passed every Rust and Python test. Only
    /// the GPU conformance suite covered it, and that does not run in CI.
    #[test]
    fn a_non_zero_domain_minimum_shifts_the_lookup() {
        let lut = array![0.0_f32, 1.0];
        // Domain [2, 4]: 2 is the bottom, 3 the middle, 4 the top. Without
        // the `- min` term, 3 would land at 3/(4-2) = 1.5 -> clamped to 1.
        assert_eq!(apply(2.0, &lut, 2.0, 4.0), 0.0);
        assert!((apply(3.0, &lut, 2.0, 4.0) - 0.5).abs() < 1e-6);
        assert_eq!(apply(4.0, &lut, 2.0, 4.0), 1.0);
        // And a negative minimum, where the sign of the shift matters.
        assert!((apply(0.0, &lut, -1.0, 1.0) - 0.5).abs() < 1e-6);
        assert_eq!(apply(-1.0, &lut, -1.0, 1.0), 0.0);
    }

    /// A reversed table is the documented way to build a
    /// histogram-matching transfer (`np.flip(cdf)`), and it reaches the
    /// non-contiguous path with a *negative* stride — which `to_owned`
    /// does not compact, so the previous fallback panicked.
    #[test]
    fn a_reversed_table_works_and_matches_its_compacted_copy() {
        let img =
            Array3::<f32>::from_shape_fn((5, 7, 1), |(y, x, _)| ((y * 7 + x) % 32) as f32 / 31.0);
        let wide = identity_lut(64);
        let reversed = wide.slice(ndarray::s![..;-1]);
        assert!(
            reversed.as_slice().is_none(),
            "the test needs a genuinely non-contiguous view"
        );
        let compacted = reversed.to_owned();
        let p = LutParams::default();
        let a = apply_lut(img.view(), reversed, &p).unwrap();
        let b = apply_lut(img.view(), compacted.view(), &p).unwrap();
        assert_eq!(a, b, "a reversed table must give its compacted answer");
        // And it really is a reversal: an identity ramp reversed inverts.
        let mid = apply_lut(
            array![[[0.25_f32]]].view(),
            wide.slice(ndarray::s![..;-1]),
            &p,
        )
        .unwrap()[[0, 0, 0]];
        assert!(
            (mid - 0.75).abs() < 0.02,
            "reversed identity should invert: {mid}"
        );
    }

    /// Interpolating between two finite but enormously separated entries
    /// must stay finite: the difference form `a + frac*(b - a)` overflows
    /// there, producing a non-finite result from a table `validate`
    /// certified as entirely finite.
    #[test]
    fn interpolation_cannot_overflow_between_finite_entries() {
        let lut = array![-3.0e38_f32, 3.0e38];
        for i in 0..=10 {
            let v = i as f32 / 10.0;
            let got = apply(v, &lut, 0.0, 1.0);
            assert!(
                got.is_finite(),
                "interpolating a finite table produced {got} at {v}"
            );
            assert!(got.abs() <= 3.0e38, "and it must stay within the knots");
        }
    }

    /// NaN must come back with the caller's own bit pattern. Returning a
    /// fresh `f32::NAN` (which is positive) made the CPU disagree with
    /// the device on every negative NaN — which is what ordinary f32
    /// arithmetic produces on x86.
    #[test]
    fn nan_keeps_its_own_bit_pattern() {
        let lut = identity_lut(16);
        let negative_nan = f32::from_bits(0xFFC0_0000);
        assert!(negative_nan.is_nan() && negative_nan.is_sign_negative());
        let out = apply(negative_nan, &lut, 0.0, 1.0);
        assert_eq!(
            out.to_bits(),
            negative_nan.to_bits(),
            "the caller's NaN must be propagated, not replaced"
        );
    }

    #[test]
    fn params_equality_and_repr() {
        let a = LutParams::new(0.0, 1.0);
        assert!(a.__eq__(&LutParams::default()));
        assert!(!a.__eq__(&LutParams::new(0.0, 4.0)));
        assert_eq!(a.__repr__(), "LutParams(min=0, max=1)");
    }
}
