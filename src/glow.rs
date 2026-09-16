// SPDX-License-Identifier: GPL-3.0-or-later
//! Light scattering — halation, diffusion and veiling glare.
//!
//! One kernel, because they are one operation:
//!
//! ```text
//! out = in + amount · blur(max(in − threshold, 0), σ)
//! ```
//!
//! Light that exceeds the threshold is spread and added back. What
//! separates the three named effects is the parameters and **where in
//! the pipeline the call sits** — not the maths — so a kernel each would
//! have been three copies of this one.
//!
//! | Effect | Where it happens | threshold | σ | Pipeline position |
//! |---|---|---|---|---|
//! | **Veiling glare** | the lens | 0 — all light scatters | very large | earliest: before the emulsion sees anything |
//! | **Halation** | the emulsion | high — only bright areas | moderate | after `exposure`, before the tone stages |
//! | **Diffusion** | the print | mid | large | after the tone stages |
//!
//! # The three, in more detail
//!
//! **Veiling glare** is the scattered light of an uncoated or dirty
//! lens. With `threshold = 0` every part of the scene contributes, and
//! with σ spanning the frame the sum is effectively the scene's mean — so
//! the blacks lift by an amount that depends on how bright the *whole
//! frame* is. That is why a tone curve cannot reproduce it: a per-pixel
//! transfer has no access to the rest of the image. It is a large part of
//! why vintage lenses look the way they do.
//!
//! **Halation** is light passing through the emulsion, reflecting off
//! the film base and re-exposing from behind, haloing bright areas. It is
//! the reason anti-halation backing exists. Because it happens *at
//! capture*, it belongs early — before the tone curve, not after.
//!
//! **Diffusion** is the printing-side counterpart: the Imagon, a
//! diffusion filter, a stocking over the lens or the enlarger. Same
//! maths, opposite end of the pipeline.
//!
//! # Additive, and why
//!
//! The scattered light is *added* rather than exchanged, so the total
//! can exceed 1.0. That is deliberate and matches the crate's pipeline:
//! headroom is preserved through every stage and spent once, explicitly,
//! at [`crate::highlight_rolloff`]. A conserving form
//! (`(1 − amount)·in + amount·blur`) would darken the source to pay for
//! the halo, which is right for a diffuser and wrong for halation, where
//! the highlight is already saturated and the halo is genuinely extra
//! density. Additive suits all three closely enough, and the alternative
//! is one `tone_curve` away.
//!
//! # What this is not
//!
//! Not an unsharp mask. `amount` is required to be non-negative, so this
//! cannot subtract a blurred copy to sharpen; detail enhancement is
//! [`crate::local_contrast`], which uses an edge-aware filter rather than
//! a Gaussian and does not halo.
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

/// Parameters for [`glow`].
///
/// ```python
/// # Halation: bright areas only, moderate spread, applied early
/// params = phaios_core.GlowParams(threshold=0.8, sigma=8.0, amount=0.35)
///
/// # Veiling glare: everything scatters, frame-wide, applied earlier still
/// params = phaios_core.GlowParams(threshold=0.0, sigma=400.0, amount=0.06)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GlowParams {
    /// Level above which light scatters. `0.0` means all of it does.
    ///
    /// Subtracted rather than used as a hard cut — the weight is
    /// `max(in − threshold, 0)` — so the contribution fades in smoothly
    /// and a bright region does not acquire an outline at the threshold.
    #[pyo3(get, set)]
    pub threshold: f32,
    /// Standard deviation of the scatter, in pixels.
    ///
    /// Small for a tight halo, frame-spanning for glare. Being in pixels,
    /// it is resolution-dependent: scale it with the image when working
    /// on a preview.
    #[pyo3(get, set)]
    pub sigma: f32,
    /// How much scattered light is added back. `0.0` is the identity and
    /// the default.
    ///
    /// Non-negative: this kernel adds light, it does not subtract it.
    #[pyo3(get, set)]
    pub amount: f32,
}

#[pymethods]
impl GlowParams {
    /// Create new ``GlowParams``.
    #[new]
    #[pyo3(signature = (threshold = 0.0, sigma = 8.0, amount = 0.0))]
    pub fn new(threshold: f32, sigma: f32, amount: f32) -> Self {
        Self {
            threshold,
            sigma,
            amount,
        }
    }

    /// Two ``GlowParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.threshold.to_bits() == other.threshold.to_bits()
            && self.sigma.to_bits() == other.sigma.to_bits()
            && self.amount.to_bits() == other.amount.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "GlowParams(threshold={}, sigma={}, amount={})",
            self.threshold, self.sigma, self.amount
        )
    }
}

impl Default for GlowParams {
    /// `amount = 0`: the identity.
    fn default() -> Self {
        Self {
            threshold: 0.0,
            sigma: 8.0,
            amount: 0.0,
        }
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate glow parameters. Shared by the CPU and CUDA backends so both
/// reject the same inputs with the same messages.
pub(crate) fn validate(params: &GlowParams) -> Result<(), PhaiosError> {
    if !params.threshold.is_finite() || params.threshold < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "threshold is {}, expected a finite value of at least 0",
            params.threshold
        )));
    }
    if !params.amount.is_finite() || params.amount < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "amount is {}, expected a finite value of at least 0 \
             (this kernel adds light; for detail enhancement use local_contrast)",
            params.amount
        )));
    }
    // σ is the blur's own parameter and its own domain.
    crate::blur::validate(&BlurParams::new(params.sigma, BlurShape::Gaussian))
}

/// The light that scatters: `max(in − threshold, 0)`.
#[inline]
pub(crate) fn scatter_weight(value: f32, threshold: f32) -> f32 {
    let excess = value - threshold;
    if excess > 0.0 { excess } else { 0.0 }
}

/// Spread light above `threshold` and add it back.
///
/// Computes `out = in + amount · blur(max(in − threshold, 0), sigma)`.
/// The output may exceed 1.0 — deliberately, since the pipeline carries
/// headroom to [`crate::highlight_rolloff`] rather than clamping early.
///
/// `amount = 0.0` is the exact identity, so inserting the stage changes
/// nothing until it is asked for.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous. Channels scatter independently, so on a
/// toned image a coloured highlight glows in its own colour.
///
/// Order-sensitive, and the position is the whole difference between the
/// three effects this expresses — see the module documentation. Applying
/// halation late, after the tone curve, is a different and less
/// filmlike picture than applying it at capture where it belongs.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `threshold` or `amount` is negative
///   or not finite, or if `sigma` is outside [`crate::blur`]'s domain.
/// - [`PhaiosError::Allocation`] if the intermediates exceed the
///   backend's single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn glow(img: ArrayView3<f32>, params: &GlowParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, c) = img.dim();
    if params.amount == 0.0 || h == 0 || w == 0 || c == 0 {
        let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
        out.assign(&img);
        return Ok(out);
    }

    let threshold = params.threshold;
    let mut weight = crate::alloc::zeros3::<f32>((h, w, c))?;
    ndarray::Zip::from(&mut weight)
        .and(img)
        .par_for_each(|o, &v| *o = scatter_weight(v, threshold));

    let spread = crate::blur::blur(
        weight.view(),
        &BlurParams::new(params.sigma, BlurShape::Gaussian),
    )?;
    drop(weight);

    let amount = params.amount;
    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
    ndarray::Zip::from(&mut out)
        .and(img)
        .and(&spread)
        .par_for_each(|o, &v, &s| *o = v + amount * s);

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn bright_spot(n: usize) -> Array3<f32> {
        Array3::<f32>::from_shape_fn(
            (n, n, 1),
            |(y, x, _)| {
                if y == n / 2 && x == n / 2 { 1.0 } else { 0.05 }
            },
        )
    }

    #[test]
    fn amount_zero_is_the_exact_identity() {
        let img = bright_spot(9);
        for threshold in [0.0_f32, 0.5, 2.0] {
            let out = glow(img.view(), &GlowParams::new(threshold, 8.0, 0.0)).unwrap();
            for (a, b) in img.iter().zip(out.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn a_threshold_above_everything_is_also_the_identity() {
        // Nothing exceeds the threshold, so nothing scatters.
        let img = bright_spot(15);
        let out = glow(img.view(), &GlowParams::new(2.0, 4.0, 0.5)).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-7, "{a} became {b}");
        }
    }

    #[test]
    fn glow_only_adds_light() {
        let img = bright_spot(21);
        let out = glow(img.view(), &GlowParams::new(0.5, 3.0, 0.4)).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(*b >= *a - 1e-7, "glow darkened {a} to {b}");
        }
    }

    /// The halo is the point: light appears *beside* a bright region,
    /// where there was none. A kernel that merely brightened the region
    /// itself would pass a "gets brighter" test.
    ///
    /// The source is a 3x3 block rather than a single pixel — one pixel
    /// of excess light spread over σ = 4 raises its surroundings by only
    /// a few percent, so a single-pixel source measures the tolerance
    /// rather than the kernel.
    #[test]
    fn light_appears_beside_a_bright_spot() {
        let n = 31;
        let img = Array3::<f32>::from_shape_fn((n, n, 1), |(y, x, _)| {
            let (dy, dx) = (y.abs_diff(n / 2), x.abs_diff(n / 2));
            if dy <= 1 && dx <= 1 { 1.0 } else { 0.05 }
        });
        let out = glow(img.view(), &GlowParams::new(0.5, 4.0, 0.6)).unwrap();

        // Four pixels away from the spot, well outside it.
        let before = img[[n / 2, n / 2 + 4, 0]];
        let after = out[[n / 2, n / 2 + 4, 0]];
        assert!(
            after > before * 1.10,
            "no halo four pixels out: {before} -> {after}"
        );

        // And it falls off with distance, as a spread should.
        let near = out[[n / 2, n / 2 + 2, 0]] - img[[n / 2, n / 2 + 2, 0]];
        let far = out[[n / 2, n / 2 + 8, 0]] - img[[n / 2, n / 2 + 8, 0]];
        assert!(
            near > far,
            "halo should fall off: {near} at 2px, {far} at 8px"
        );
    }

    /// Veiling glare: with no threshold and a frame-spanning σ, the black
    /// point lifts by an amount that depends on the *whole frame's*
    /// brightness — which is exactly what a per-pixel tone curve cannot
    /// do, and the reason this kernel exists rather than a curve preset.
    #[test]
    fn veiling_glare_lifts_black_in_proportion_to_scene_brightness() {
        let n = 33;
        let dark = Array3::<f32>::from_shape_fn(
            (n, n, 1),
            |(y, x, _)| {
                if y < 3 && x < 3 { 0.2 } else { 0.0 }
            },
        );
        let bright =
            Array3::<f32>::from_shape_fn(
                (n, n, 1),
                |(y, x, _)| {
                    if y < 3 && x < 3 { 4.0 } else { 0.0 }
                },
            );
        let params = GlowParams::new(0.0, 200.0, 0.5);

        // Sample a corner far from the bright patch in both frames.
        let lift = |img: &Array3<f32>| {
            glow(img.view(), &params).unwrap()[[n - 1, n - 1, 0]] - img[[n - 1, n - 1, 0]]
        };
        let (a, b) = (lift(&dark), lift(&bright));
        assert!(a > 0.0, "glare must lift black at all: {a}");
        assert!(
            b > a * 5.0,
            "a brighter scene must lift black further: {a} vs {b}"
        );
    }

    #[test]
    fn the_weight_is_a_soft_knee_not_a_hard_cut() {
        // A hard cut would give the halo an outline at the threshold.
        assert_eq!(scatter_weight(0.4, 0.5), 0.0);
        assert_eq!(scatter_weight(0.5, 0.5), 0.0);
        assert!((scatter_weight(0.6, 0.5) - 0.1).abs() < 1e-7);
        assert!((scatter_weight(1.5, 0.5) - 1.0).abs() < 1e-7);
        // Continuous at the threshold: no step.
        assert!(scatter_weight(0.5 + 1e-6, 0.5) < 1e-5);
    }

    #[test]
    fn a_nan_sample_is_contained_and_does_not_spread() {
        // `scatter_weight` is written `if excess > 0.0 { excess } else
        // { 0.0 }` rather than with `max`, specifically so that a NaN
        // sample fails the test and contributes 0 to the blur. The same
        // reasoning, and the same shape of expression, is repeated in
        // src/cuda/ptx/glow.cu — where it is likewise the only thing
        // standing between one dead pixel and a ruined frame.
        //
        // Nothing exercised it on either backend: every glow test used
        // finite input, and the non-finite values in this module's other
        // tests are out-of-domain *parameters*, not samples.
        //
        // Invert the comparison and the NaN enters the blur. Because the
        // blur is separable, it becomes a NaN row after the horizontal
        // pass and a NaN frame after the vertical one — a single dead
        // sensor pixel would destroy the whole image.
        assert_eq!(scatter_weight(f32::NAN, 0.2), 0.0, "NaN must scatter 0");
        assert_eq!(
            scatter_weight(f32::NEG_INFINITY, 0.2),
            0.0,
            "-inf is below any threshold"
        );

        let mut img = Array3::<f32>::from_elem((9, 9, 1), 0.5);
        img[[4, 4, 0]] = f32::NAN;
        let out = glow(img.view(), &GlowParams::new(0.2, 2.0, 1.0)).unwrap();

        let non_finite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 1,
            "the NaN spread: {non_finite} of 81 outputs are non-finite"
        );
        assert!(out[[4, 4, 0]].is_nan(), "the poisoned sample stays NaN");

        // +inf is deliberately *not* contained: it passes `> 0`, which is
        // arithmetically right — it really is light above the threshold —
        // and `docs/ffi.md` §1 leaves any non-finite sample unspecified.
        // Containment here is a property of NaN specifically, which is
        // what makes the `>` form load-bearing rather than incidental.
        let mut img = Array3::<f32>::from_elem((9, 9, 1), 0.5);
        img[[4, 4, 0]] = f32::INFINITY;
        let out = glow(img.view(), &GlowParams::new(0.2, 2.0, 1.0)).unwrap();
        assert!(
            !out[[0, 0, 0]].is_finite(),
            "an infinite sample is expected to scatter, unlike a NaN"
        );
    }

    #[test]
    fn output_may_exceed_one_by_design() {
        // Headroom is carried to highlight_rolloff, not clamped here.
        let img = Array3::<f32>::from_elem((9, 9, 1), 1.0);
        let out = glow(img.view(), &GlowParams::new(0.0, 2.0, 0.5)).unwrap();
        assert!(out.iter().any(|v| *v > 1.0), "glow should not clamp");
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((10, 12, 3), |(y, x, c)| {
            ((y * 12 + x + c) % 20) as f32 / 19.0
        });
        let strided = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned = strided.to_owned();
        let p = GlowParams::new(0.3, 2.0, 0.4);
        assert_eq!(glow(strided, &p).unwrap(), glow(owned.view(), &p).unwrap());
    }

    #[test]
    fn empty_input_is_accepted() {
        let img = Array3::<f32>::zeros((0, 4, 3));
        assert_eq!(
            glow(img.view(), &GlowParams::new(0.0, 4.0, 0.5))
                .unwrap()
                .dim(),
            (0, 4, 3)
        );
    }

    #[test]
    fn rejects_out_of_domain_parameters() {
        let img = array![[[0.5_f32]]];
        for (t, s, a) in [
            (-0.1, 4.0, 0.5),
            (f32::NAN, 4.0, 0.5),
            (0.0, 4.0, -0.1),
            (0.0, 4.0, f32::NAN),
            (0.0, -1.0, 0.5),
            (0.0, f32::INFINITY, 0.5),
        ] {
            assert!(
                glow(img.view(), &GlowParams::new(t, s, a)).is_err(),
                "threshold={t} sigma={s} amount={a} should be rejected"
            );
        }
    }

    #[test]
    fn params_equality_and_repr() {
        let a = GlowParams::new(0.8, 8.0, 0.35);
        assert!(a.__eq__(&GlowParams::new(0.8, 8.0, 0.35)));
        assert!(!a.__eq__(&GlowParams::new(0.8, 8.0, 0.4)));
        assert_eq!(
            a.__repr__(),
            "GlowParams(threshold=0.8, sigma=8, amount=0.35)"
        );
        assert_eq!(GlowParams::default().amount, 0.0);
    }
}
