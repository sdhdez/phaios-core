// SPDX-License-Identifier: GPL-3.0-or-later
//! Unsharp masking — a plain, haloing, threshold-gated Gaussian sharpen.
//!
//! ```text
//! blurred = blur_σ(img)
//! detail  = img − blurred
//! T       = soft_gate(detail, threshold)     [see "Soft gate", below]
//! out     = img + amount · T · detail
//! ```
//!
//! At `threshold = 0.0` this is exactly Gonzalez & Woods, *Digital Image
//! Processing*, 4th ed. (Pearson, 2018), §3.6 "Unsharp Masking and
//! Highboost Filtering": `out = img + amount·(img − blur_σ(img))`, where
//! `amount` is their highboost constant `k` (`k = 1` standard unsharp
//! masking, `k > 1` highboost).
//!
//! `threshold` gates the residual so flat, near-noise-level regions are
//! not amplified — the concept behind A. Polesel, G. Ramponi, V. J.
//! Mathews, "Image enhancement via adaptive unsharp masking," *IEEE
//! Trans. Image Processing* 9(3), pp.505–510, March 2000 (DOI
//! 10.1109/83.826787): gain that depends on local detail rather than a
//! single fixed constant. The gate's exact nonlinearity below is this
//! crate's own construction, not theirs — cited for the concept, the
//! same posture [`crate::local_contrast`] already takes toward the
//! guided filter's originators.
//!
//! # Soft gate, not a hard cut
//!
//! A hard threshold (zero below it, full gain above) is discontinuous
//! *in the data*: two backends computing `detail` by different
//! accumulation orders can disagree by a few ULP (see [`crate::blur`],
//! `docs/ffi.md` §6), and a step turns that disagreement into a
//! full-amplitude flip of whether a pixel is gated at all. [`crate::glow`]
//! takes the same position, for the same reason, on the raw pixel value
//! rather than a derived signal — and because this gate reads a
//! *derived*, higher-frequency signal, more pixels sit close to the
//! boundary than they would for `glow`, making the soft knee matter more
//! here, not less.
//!
//! The gate is a Hermite smoothstep, `u²(3 − 2u)`, ramping over
//! `[threshold, SOFT_KNEE_SPAN · threshold]` — C¹ at both ends, so a
//! bounded (not zero) disagreement between backends near the boundary
//! stays bounded in the gate too. `threshold = 0.0` is special-cased to
//! `T ≡ 1.0` unconditionally: the ramp would otherwise have zero width,
//! and evaluating it would divide by that zero.
//!
//! `SOFT_KNEE_SPAN` is a free constant, not a fourth parameter, in the
//! company of [`crate::blur`]'s own `TRUNCATION` and `BOX_CROSSOVER_SIGMA`
//! and `split_toning`'s fixed crossover half-width.
//!
//! # Threshold is in detail units, not pixel units
//!
//! `threshold` gates `|img − blur(img)|`, a high-frequency residual —
//! not the raw pixel value the way [`crate::glow`]'s `threshold` does.
//! Useful values are therefore typically much smaller than a `glow`
//! threshold; porting one across kernels verbatim is a mistake.
//!
//! # Order
//!
//! Recommended after [`crate::local_contrast`], before
//! [`crate::film_grain`]: sharpening amplifies noise, so it belongs
//! before grain is added, for the same reason grain itself follows the
//! tone/detail stages rather than preceding them. Not fixed in the
//! canonical pipeline order any more than [`crate::blur`] or
//! [`crate::glow`] are — all three are optional-position effects
//! documented per-kernel rather than pipeline-fixed.
//!
//! # Range
//!
//! Never clamps. The overshoot and undershoot this produces at an edge
//! is Gonzalez & Woods §3.6's own ringing behaviour, not a defect this
//! crate suppresses.
//!
//! # Determinism
//!
//! Inherits [`crate::blur`]'s: bounded, not bit-exact. See
//! `docs/ffi.md` §6.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::blur::{BlurParams, BlurShape};
use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`sharpen`].
///
/// ```python
/// # A modest capture sharpen, ignoring near-flat noise
/// params = phaios_core.SharpenParams(amount=0.5, sigma=1.2, threshold=0.02)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct SharpenParams {
    /// How much of the gated detail is added back. `0.0` is the
    /// identity and the default.
    ///
    /// Non-negative: this kernel adds detail, it does not remove it —
    /// for smoothing use [`crate::blur`], or a negative
    /// [`crate::local_contrast`] `strength`.
    #[pyo3(get, set)]
    pub amount: f32,
    /// Standard deviation of the blur that `detail` is measured
    /// against, in pixels. `0.0` is the identity.
    ///
    /// [`crate::blur`]'s own domain, bounded above by its `MAX_SIGMA` —
    /// delegated, not restated.
    #[pyo3(get, set)]
    pub sigma: f32,
    /// Detail magnitude below which amplification fades out. `0.0`
    /// (the default) amplifies every pixel equally.
    ///
    /// In units of `|img − blur(img)|`, a high-frequency residual — not
    /// the raw pixel value the way [`crate::glow`]'s threshold is.
    /// Useful values are typically much smaller than a `glow` threshold.
    #[pyo3(get, set)]
    pub threshold: f32,
}

#[pymethods]
impl SharpenParams {
    /// Create new ``SharpenParams``.
    #[new]
    #[pyo3(signature = (amount = 0.0, sigma = 0.0, threshold = 0.0))]
    pub fn new(amount: f32, sigma: f32, threshold: f32) -> Self {
        Self {
            amount,
            sigma,
            threshold,
        }
    }

    /// Two ``SharpenParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.amount.to_bits() == other.amount.to_bits()
            && self.sigma.to_bits() == other.sigma.to_bits()
            && self.threshold.to_bits() == other.threshold.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SharpenParams(amount={}, sigma={}, threshold={})",
            self.amount, self.sigma, self.threshold
        )
    }
}

impl Default for SharpenParams {
    /// `amount = 0`: the identity.
    fn default() -> Self {
        Self {
            amount: 0.0,
            sigma: 0.0,
            threshold: 0.0,
        }
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// End of the soft-knee ramp, as a multiple of `threshold`: the gate
/// rises from `0` at `d = threshold` to `1` at `d = SOFT_KNEE_SPAN ×
/// threshold`. See the module documentation.
const SOFT_KNEE_SPAN: f32 = 2.0;

/// Validate sharpen parameters. Shared by the CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate(params: &SharpenParams) -> Result<(), PhaiosError> {
    if !params.amount.is_finite() || params.amount < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "amount is {}, expected a finite value of at least 0 (this kernel adds detail; for smoothing use blur or a negative local_contrast strength)",
            params.amount
        )));
    }
    if !params.threshold.is_finite() || params.threshold < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "threshold is {}, expected a finite value of at least 0",
            params.threshold
        )));
    }
    // σ is the blur's own parameter and its own domain.
    crate::blur::validate(&BlurParams::new(params.sigma, BlurShape::Gaussian))
}

/// The soft threshold gate: a Hermite smoothstep of `|d|`, ramping from
/// `0` at `threshold` to `1` at `SOFT_KNEE_SPAN · threshold`.
///
/// `threshold == 0.0` is special-cased to `1.0` unconditionally,
/// including at `d == 0.0` — the ramp would otherwise have zero width,
/// and `(|d| − 0) / 0` would produce NaN rather than the identity gate a
/// zero threshold is meant to give.
///
/// Even in `d`: only the *magnitude* of the detail gates it, so a dark
/// edge and a bright edge of matching strength amplify equally.
#[inline]
pub(crate) fn soft_gate(d: f32, threshold: f32) -> f32 {
    if threshold == 0.0 {
        return 1.0;
    }
    let span = (SOFT_KNEE_SPAN - 1.0) * threshold;
    let u = ((d.abs() - threshold) / span).clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Sharpen with a threshold-gated Gaussian unsharp mask.
///
/// Computes `out = img + amount · soft_gate(detail, threshold) · detail`
/// where `detail = img − blur_σ(img)`. See the module documentation for
/// the gate's shape and the citations.
///
/// `amount = 0.0` or `sigma = 0.0` is the exact identity, so inserting
/// the stage changes nothing until it is asked for.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous. Channels are filtered independently, since
/// `detail` is measured against [`crate::blur`] applied to `img`
/// directly.
///
/// Order-sensitive; see the module documentation. Never clamps: the
/// overshoot and undershoot at an edge is unsharp masking's own
/// behaviour, not a defect to suppress.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `amount` or `threshold` is negative
///   or not finite, or if `sigma` is outside [`crate::blur`]'s domain.
/// - [`PhaiosError::Allocation`] if the intermediates exceed the
///   backend's single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn sharpen(img: ArrayView3<f32>, params: &SharpenParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, c) = img.dim();
    if params.amount == 0.0 || params.sigma == 0.0 || h == 0 || w == 0 || c == 0 {
        let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
        out.assign(&img);
        return Ok(out);
    }

    let blurred = crate::blur::blur(img, &BlurParams::new(params.sigma, BlurShape::Gaussian))?;

    let amount = params.amount;
    let threshold = params.threshold;
    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
    ndarray::Zip::from(&mut out)
        .and(img)
        .and(&blurred)
        .par_for_each(|o, &v, &b| {
            let detail = v - b;
            let t = soft_gate(detail, threshold);
            *o = v + amount * t * detail;
        });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── soft_gate ────────────────────────────────────────────────────────────

    #[test]
    fn soft_gate_is_one_at_zero_threshold_including_zero_detail() {
        for d in [0.0_f32, -3.0, 0.2, 100.0] {
            assert_eq!(
                soft_gate(d, 0.0),
                1.0,
                "d={d} threshold=0 must give T=1 exactly"
            );
        }
    }

    #[test]
    fn soft_gate_is_zero_at_the_threshold_and_one_at_the_ramp_end() {
        let threshold = 0.4_f32;
        assert_eq!(soft_gate(threshold, threshold), 0.0);
        assert_eq!(soft_gate(-threshold, threshold), 0.0);

        let ramp_end = SOFT_KNEE_SPAN * threshold;
        assert!((soft_gate(ramp_end, threshold) - 1.0).abs() < 1e-6);
        assert!((soft_gate(-ramp_end, threshold) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn soft_gate_is_flat_below_the_threshold_and_above_the_ramp() {
        let threshold = 0.2_f32;
        assert_eq!(soft_gate(0.0, threshold), 0.0);
        assert_eq!(soft_gate(0.1, threshold), 0.0);
        assert_eq!(soft_gate(10.0, threshold), 1.0);
    }

    #[test]
    fn soft_gate_is_even_in_detail() {
        let threshold = 0.3_f32;
        for d in [0.05_f32, 0.3, 0.45, 0.6, 1.5] {
            assert_eq!(soft_gate(d, threshold), soft_gate(-d, threshold), "d={d}");
        }
    }

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_the_boundary_values() {
        assert!(validate(&SharpenParams::new(0.0, 0.0, 0.0)).is_ok());
        assert!(validate(&SharpenParams::new(0.0, crate::blur::MAX_SIGMA, 0.0)).is_ok());
    }

    #[test]
    fn validate_rejects_out_of_domain_amount() {
        for amount in [-0.1_f32, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&SharpenParams::new(amount, 1.0, 0.0)).unwrap_err();
            assert!(
                err.to_string().contains("amount is"),
                "amount={amount}: {err}"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_domain_threshold() {
        for threshold in [-0.1_f32, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&SharpenParams::new(0.5, 1.0, threshold)).unwrap_err();
            assert!(
                err.to_string().contains("threshold is"),
                "threshold={threshold}: {err}"
            );
        }
    }

    /// `sigma` is not sharpen's own parameter to validate — it is
    /// delegated to `blur::validate` so both backends' rejection message
    /// is identical without `sharpen::validate` restating it.
    #[test]
    fn validate_delegates_sigma_message_to_blur_verbatim() {
        for bad_sigma in [
            -1.0_f32,
            f32::NAN,
            f32::INFINITY,
            crate::blur::MAX_SIGMA * 2.0,
        ] {
            let sharpen_msg = validate(&SharpenParams::new(0.5, bad_sigma, 0.0))
                .unwrap_err()
                .to_string();
            let blur_msg = crate::blur::validate(&BlurParams::new(bad_sigma, BlurShape::Gaussian))
                .unwrap_err()
                .to_string();
            assert_eq!(sharpen_msg, blur_msg, "sigma={bad_sigma}");
        }
    }

    // ── SharpenParams ────────────────────────────────────────────────────────

    #[test]
    fn params_equality_repr_and_defaults() {
        let a = SharpenParams::new(0.5, 1.2, 0.02);
        assert!(a.__eq__(&SharpenParams::new(0.5, 1.2, 0.02)));
        assert!(!a.__eq__(&SharpenParams::new(0.5, 1.2, 0.03)));
        assert_eq!(
            a.__repr__(),
            "SharpenParams(amount=0.5, sigma=1.2, threshold=0.02)"
        );
        let d = SharpenParams::default();
        assert_eq!((d.amount, d.sigma, d.threshold), (0.0, 0.0, 0.0));
    }
}
