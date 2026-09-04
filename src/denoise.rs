// SPDX-License-Identifier: GPL-3.0-or-later
//! Guided-filter denoise — self-guided per channel, or cross-guided from
//! a shared luminance guide for RGB.
//!
//! ```text
//! // self-guided (C = 1, and every C other than 3)
//! q     = guided_filter(p, radius, eps)     // reused verbatim from local_contrast
//! out   = p + (−amount) · (p − q)           // amount=0 → p; amount=1 → q
//!
//! // cross-guided (C = 3), guide I = luminance_bw(img, standard)
//! a_c   = cov_w(I, p_c) / (var_w(I) + eps)  // 0/0 → 0, local_contrast's own convention
//! b_c   = mean_w(p_c) − a_c · mean_w(I)
//! q_c   = mean_w(a_c) · I + mean_w(b_c)     // uses the GUIDE, not p_c
//! out_c = p_c + (−amount) · (p_c − q_c)
//! ```
//! `eps = noise_sigma²`. `cov_w`/`var_w`/`mean_w` are window statistics
//! over a `(2·radius+1) × (2·radius+1)` box, via the same integral-image
//! machinery `local_contrast` uses.
//!
//! # Self-guided is an alias, worth exploiting
//!
//! On a single channel, `out = p + (−amount)·(p−q)` is exactly
//! [`crate::local_contrast::local_contrast`]'s own combine,
//! `l + strength·(l−q)`, with `strength = −amount` — the same `q`,
//! because this kernel calls the very same
//! [`crate::local_contrast::guided_filter`] it does. Negating one
//! operand of a floating-point multiply is an exact sign flip
//! (`fl((−a)·x) = −fl(a·x)` for any `a`, `x`: correctly-rounded
//! multiplication is symmetric about zero), and `y − z` is defined as
//! `y + (−z)`, so the two expressions round identically at every step.
//! Consequently `denoise` at any `radius`/`noise_sigma`/`amount`, on a
//! `(H, W, 1)` image, is **bit-exact** with
//! `local_contrast(img, GuidedFilterParams::new(radius, noise_sigma²),
//! −amount)`. Pinned in `tests/kernels.rs` by
//! `denoise_single_channel_is_bit_exact_with_local_contrast_at_negative_amount`.
//!
//! What the `C = 3` path adds is genuinely new code, not a parameter
//! change: a **cross-guided** filter (He–Sun–Tang §3.4) that takes its
//! edges from a shared luminance guide rather than each channel's own
//! signal, so a channel with little structure of its own (a colour cast
//! over an otherwise flat sky, say) is smoothed according to the
//! *guide's* edges, not mistaken structure in its own noise.
//!
//! # `noise_sigma`
//!
//! `eps = noise_sigma²` ties the guided filter's regularisation term to
//! an estimate of the input's noise variance, the reading He, Sun and
//! Tang give ε in their colour-guide formulation. `noise_sigma = 0.0` is
//! the no-regularisation limit — legal (mirrors `eps = 0.0` in
//! [`crate::local_contrast`]) but not on its own an identity; `amount =
//! 0.0` is the only true identity switch.
//!
//! # Memory
//!
//! Self-guided calls [`crate::local_contrast::guided_filter`] once per
//! channel, sequentially — its own documented peak, about 24 bytes per
//! pixel, is this path's peak too, independent of channel count, since
//! one channel's scratch is fully released before the next channel's is
//! built.
//!
//! Cross-guided builds the shared guide `I` (4 B/px, live for the whole
//! call) and two shared f64 SATs of `I` and `I²` (16 B/px, live for the
//! whole call, queried once per channel), then processes channels
//! sequentially. Per channel: two transient first-stage f64 SATs (of
//! `p_c` and `I·p_c`, 16 B/px, dropped once the linear coefficients `a_c`
//! `b_c` exist) — building the `I·p_c` SAT costs a brief 4 B/px
//! elementwise-product scratch array, since [`crate::integral::sat`]
//! takes one input array and this crate does not add a two-array
//! variant for one caller — then the two f32 coefficient arrays (8 B/px,
//! dropped once their own SATs exist) and two second-stage f64 SATs
//! (16 B/px, dropped once the channel's output is written). Each
//! channel's transient scratch is released before the next channel's is
//! built, so the whole-call peak is the shared 20 B/px plus one
//! channel's own worst moment (first-stage coefficients, 24 B/px) —
//! **≈44 B/px, ≈1.06 GB at 24 MP** — not the ≈60 B/px a design that kept
//! every channel's coefficients live at once would reach.
//!
//! # Order
//!
//! Native resolution, right after [`crate::hot_pixels`], before
//! [`crate::geometry::straighten`]/[`crate::geometry::resize`], and
//! before `exposure`. Two reasons to precede resampling and exposure:
//! `noise_sigma` is a per-pixel physical quantity, and resampling mixes
//! neighbours with filter-dependent weights, changing per-output-pixel
//! noise statistics unrelated to the physical model it is meant to
//! describe; and exposure is a multiply, so running after it would
//! couple `noise_sigma` to the caller's creative stop choice rather than
//! the sensor's own noise. Must follow `hot_pixels`: the guided filter
//! is edge-preserving, so an uncorrected hot pixel's extreme local
//! variance reads as structure to protect, and — since `a_c`/`b_c` are
//! window-averaged — smears a wrong coefficient up to `radius` away.
//!
//! # Range and channels
//!
//! Never clamps. `C == 3` is cross-guided; every other channel count,
//! including `C == 1`, is self-guided per channel — no count is
//! rejected. Finite in ⇒ finite out, on the same terms
//! [`crate::local_contrast`] documents: sums of finite values, ratios
//! guarded by `+ eps`, the 0/0 → 0 convention avoiding NaN at `eps = 0`.
//!
//! Reference: Kaiming He, Jian Sun, Xiaoou Tang, "Guided Image
//! Filtering," IEEE *Transactions on Pattern Analysis and Machine
//! Intelligence* 35(6), 2013, pp. 1397–1409, §3.4 (the colour-guide/
//! colour-filtering-process formulation this module's cross-guided path
//! follows). See [`crate::local_contrast`] for the self-guided ECCV 2010
//! formulation this module's self-guided path reuses verbatim, and its
//! provenance note on the reference MATLAB implementation's licence.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::bw::{LuminanceStandard, luminance_bw};
use crate::error::PhaiosError;
use crate::integral::{sat, window_sum};
use crate::local_contrast::guided_filter;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`denoise`].
///
/// ```python
/// # A gentle denoise tuned to an estimated noise level.
/// params = phaios_core.DenoiseParams(radius=4, noise_sigma=0.01, amount=0.6)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct DenoiseParams {
    /// Filter radius in pixels. The window is `(2r+1) × (2r+1)`.
    ///
    /// [`crate::local_contrast::GuidedFilterParams`]'s own domain:
    /// unvalidated, any `u32` is legal — a radius larger than the image
    /// is harmless, since windows clamp to the image extent.
    #[pyo3(get, set)]
    pub radius: u32,
    /// Estimated noise standard deviation, in the input's own units.
    /// `eps = noise_sigma²` regularises the guided filter — see the
    /// module documentation. `0.0` (the default) is the
    /// no-regularisation limit: legal, but on its own not an identity.
    #[pyo3(get, set)]
    pub noise_sigma: f32,
    /// Blend between the input and the filtered base term. `0.0` (the
    /// default) is the exact identity; `1.0` is the full guided-filter
    /// base term.
    #[pyo3(get, set)]
    pub amount: f32,
    /// Luminance standard for the `C == 3` guide. Ignored when the
    /// input is not three channels.
    #[pyo3(get, set)]
    pub standard: LuminanceStandard,
}

#[pymethods]
impl DenoiseParams {
    /// Create new ``DenoiseParams``.
    #[new]
    #[pyo3(signature = (radius = 0, noise_sigma = 0.0, amount = 0.0, standard = LuminanceStandard::Bt709))]
    pub fn new(radius: u32, noise_sigma: f32, amount: f32, standard: LuminanceStandard) -> Self {
        Self {
            radius,
            noise_sigma,
            amount,
            standard,
        }
    }

    /// Two ``DenoiseParams`` are equal when all four fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.radius == other.radius
            && self.noise_sigma.to_bits() == other.noise_sigma.to_bits()
            && self.amount.to_bits() == other.amount.to_bits()
            && self.standard == other.standard
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "DenoiseParams(radius={}, noise_sigma={}, amount={}, standard={:?})",
            self.radius, self.noise_sigma, self.amount, self.standard
        )
    }
}

impl Default for DenoiseParams {
    /// `amount = 0`: the identity.
    fn default() -> Self {
        Self {
            radius: 0,
            noise_sigma: 0.0,
            amount: 0.0,
            standard: LuminanceStandard::Bt709,
        }
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate denoise parameters. Shared by the CPU kernel and its CUDA
/// twin, so both backends reject exactly the same inputs with exactly
/// the same messages.
pub(crate) fn validate(params: &DenoiseParams) -> Result<(), PhaiosError> {
    if !params.noise_sigma.is_finite() || params.noise_sigma < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "noise_sigma is {}, expected a finite value >= 0",
            params.noise_sigma
        )));
    }
    if !params.amount.is_finite() || !(0.0..=1.0).contains(&params.amount) {
        return Err(PhaiosError::Parameter(format!(
            "amount is {}, expected a finite value in 0..=1",
            params.amount
        )));
    }
    Ok(())
}

/// Self-guided path: every channel filtered independently through the
/// reused [`guided_filter`]. Covers `C == 1` (the common case) and any
/// `C` other than 3 — there is nothing special about one channel; it is
/// simply this loop's `C == 1` case.
///
/// Writes `out = p + (−amount) · (p − q)` — see the module documentation
/// for why this exact form, not `p − amount · (p − q)`, is what makes
/// the single-channel path bit-exact with `local_contrast`.
fn self_guided(
    img: ArrayView3<f32>,
    radius: u32,
    eps: f32,
    amount: f32,
    out: &mut Array3<f32>,
) -> Result<(), PhaiosError> {
    let (_, _, c) = img.dim();
    let neg_amount = -amount;
    for ch in 0..c {
        let p = img.slice(ndarray::s![.., .., ch]);
        let q = guided_filter(p, radius, eps)?;
        ndarray::Zip::from(out.slice_mut(ndarray::s![.., .., ch]))
            .and(&p)
            .and(&q)
            .par_for_each(|o, &pv, &qv| {
                *o = pv + neg_amount * (pv - qv);
            });
    }
    Ok(())
}

/// Cross-guided path (`C == 3`): a shared luminance guide `I`, edges
/// taken from the guide rather than each channel's own signal. See the
/// module documentation for the formula and the memory accounting.
fn cross_guided(
    img: ArrayView3<f32>,
    radius: u32,
    eps: f32,
    amount: f32,
    standard: LuminanceStandard,
    out: &mut Array3<f32>,
) -> Result<(), PhaiosError> {
    let (h, w, _) = img.dim();
    let r = radius as usize;
    let eps_f64 = eps as f64;
    let neg_amount = -amount;

    let guide = luminance_bw(img, standard)?;
    let guide2d = guide.index_axis(ndarray::Axis(2), 0);

    // Shared across all three channels: dropped only once every
    // channel's coefficients have been computed.
    let sat_i = sat(guide2d, |v| v as f64)?;
    let sat_i2 = sat(guide2d, |v| {
        let d = v as f64;
        d * d
    })?;

    for ch in 0..3 {
        let p_c = img.slice(ndarray::s![.., .., ch]);

        let mut a_arr = crate::alloc::zeros2::<f32>((h, w))?;
        let mut b_arr = crate::alloc::zeros2::<f32>((h, w))?;
        {
            let sat_p = sat(p_c, |v| v as f64)?;
            let sat_ip = {
                // I·p_c has no single-array SAT of its own — `sat` maps
                // one input array, and this crate does not add a
                // two-array variant for this one caller. Materialise the
                // elementwise product, build its SAT, then let the
                // product array die at the end of this block, before the
                // per-pixel coefficient pass below runs.
                let mut ip = crate::alloc::zeros2::<f32>((h, w))?;
                ndarray::Zip::from(&mut ip)
                    .and(&guide2d)
                    .and(&p_c)
                    .par_for_each(|o, &iv, &pv| *o = iv * pv);
                sat(ip.view(), |v| v as f64)?
            };

            ndarray::Zip::indexed(&mut a_arr)
                .and(&mut b_arr)
                .par_for_each(|(y, x), a_out, b_out| {
                    let (sum_i, area) = window_sum(&sat_i, y, x, r, h, w);
                    let (sum_i2, _) = window_sum(&sat_i2, y, x, r, h, w);
                    let (sum_p, _) = window_sum(&sat_p, y, x, r, h, w);
                    let (sum_ip, _) = window_sum(&sat_ip, y, x, r, h, w);

                    let mean_i = sum_i / area;
                    let mean_i2 = sum_i2 / area;
                    let mean_p = sum_p / area;
                    let mean_ip = sum_ip / area;

                    // Same cancelling-subtraction clamp as
                    // local_contrast's own var_l: a large-magnitude guide
                    // can drive the SAT rounding error past the true
                    // variance.
                    let var_i = (mean_i2 - mean_i * mean_i).max(0.0);
                    let cov = mean_ip - mean_i * mean_p;

                    // Convention: 0/0 → 0 (flat guide, no edge to key
                    // off), the same branch local_contrast's `a` uses.
                    let a = if var_i + eps_f64 > 0.0 {
                        cov / (var_i + eps_f64)
                    } else {
                        0.0
                    };
                    let b = mean_p - a * mean_i;

                    *a_out = a as f32;
                    *b_out = b as f32;
                });
            // sat_p and sat_ip die here, before sat_a/sat_b are built.
        }

        let sat_a = sat(a_arr.view(), |v| v as f64)?;
        drop(a_arr);
        let sat_b = sat(b_arr.view(), |v| v as f64)?;
        drop(b_arr);

        ndarray::Zip::indexed(out.slice_mut(ndarray::s![.., .., ch]))
            .and(&guide2d)
            .and(&p_c)
            .par_for_each(|(y, x), o, &iv, &pv| {
                let (sum_a, area) = window_sum(&sat_a, y, x, r, h, w);
                let (sum_b, _) = window_sum(&sat_b, y, x, r, h, w);
                let mean_a = sum_a / area;
                let mean_b = sum_b / area;
                let q = (mean_a * iv as f64 + mean_b) as f32;
                *o = pv + neg_amount * (pv - q);
            });
        // sat_a and sat_b die here, before the next channel's sat_p/
        // sat_ip are built — no two channels' scratch overlap.
    }

    Ok(())
}

/// Denoise with the He–Sun–Tang guided filter.
///
/// `C == 3` uses a cross-guided filter with edges from a shared
/// luminance guide; every other channel count, including `C == 1`, is
/// self-guided per channel. See the module documentation for both
/// formulas, the `noise_sigma` → `eps` mapping, the self-guided/
/// `local_contrast` alias, memory, order and range notes.
///
/// Input shape: `(H, W, C)`, any channel count, any memory layout.
/// Output shape: `(H, W, C)`, freshly allocated and C-contiguous. Empty
/// input (`H`, `W` or `C` is `0`) returns the empty array of the same
/// shape.
///
/// `amount = 0.0` is the exact identity.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `noise_sigma` is negative or not
///   finite, or `amount` is outside `0..=1` or not finite.
/// - [`PhaiosError::Allocation`] if an intermediate exceeds the
///   backend's single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn denoise(img: ArrayView3<f32>, params: &DenoiseParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, c) = img.dim();
    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
    if h == 0 || w == 0 || c == 0 {
        return Ok(out);
    }

    let eps = params.noise_sigma * params.noise_sigma;
    let amount = params.amount;

    if c == 3 {
        cross_guided(img, params.radius, eps, amount, params.standard, &mut out)?;
    } else {
        self_guided(img, params.radius, eps, amount, &mut out)?;
    }

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film_grain::splitmix64;

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn validate_accepts_the_boundary_values() {
        assert!(validate(&DenoiseParams::new(0, 0.0, 0.0, LuminanceStandard::Bt709)).is_ok());
        assert!(validate(&DenoiseParams::new(0, 0.0, 1.0, LuminanceStandard::Bt709)).is_ok());
    }

    #[test]
    fn validate_rejects_out_of_domain_noise_sigma() {
        for noise_sigma in [-0.1_f32, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&DenoiseParams::new(
                0,
                noise_sigma,
                0.0,
                LuminanceStandard::Bt709,
            ))
            .unwrap_err();
            assert!(
                err.to_string().contains("noise_sigma is"),
                "noise_sigma={noise_sigma}: {err}"
            );
        }
    }

    #[test]
    fn validate_rejects_out_of_domain_amount() {
        for amount in [-0.1_f32, 1.1, f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let err = validate(&DenoiseParams::new(
                0,
                0.0,
                amount,
                LuminanceStandard::Bt709,
            ))
            .unwrap_err();
            assert!(
                err.to_string().contains("amount is"),
                "amount={amount}: {err}"
            );
        }
    }

    // ── DenoiseParams ────────────────────────────────────────────────────────

    #[test]
    fn params_equality_repr_and_defaults() {
        let a = DenoiseParams::new(4, 0.02, 0.6, LuminanceStandard::Bt601);
        assert!(a.__eq__(&DenoiseParams::new(4, 0.02, 0.6, LuminanceStandard::Bt601)));
        assert!(!a.__eq__(&DenoiseParams::new(5, 0.02, 0.6, LuminanceStandard::Bt601)));
        assert!(!a.__eq__(&DenoiseParams::new(4, 0.03, 0.6, LuminanceStandard::Bt601)));
        assert!(!a.__eq__(&DenoiseParams::new(4, 0.02, 0.7, LuminanceStandard::Bt601)));
        assert!(!a.__eq__(&DenoiseParams::new(4, 0.02, 0.6, LuminanceStandard::Bt709)));
        assert_eq!(
            a.__repr__(),
            "DenoiseParams(radius=4, noise_sigma=0.02, amount=0.6, standard=Bt601)"
        );

        let d = DenoiseParams::default();
        assert_eq!(d.radius, 0);
        assert_eq!(d.noise_sigma, 0.0);
        assert_eq!(d.amount, 0.0);
        assert_eq!(d.standard, LuminanceStandard::Bt709);
    }

    // ── cross-guided naive reference ────────────────────────────────────────

    /// A direct O(n·r²) implementation of the cross-guided formula:
    /// explicit window loops (no SATs, no `window_sum`) and f64
    /// accumulation throughout — an independent oracle sharing no
    /// arithmetic structure with [`cross_guided`]'s SAT-based
    /// implementation, per this crate's practice that an empirical sweep
    /// against an independent reference catches what code review alone
    /// does not.
    fn naive_cross_guided(
        img: &Array3<f32>,
        radius: u32,
        eps: f32,
        amount: f32,
        standard: LuminanceStandard,
    ) -> Array3<f32> {
        let (h, w, c) = img.dim();
        assert_eq!(c, 3);
        let r = radius as usize;
        let eps_f64 = eps as f64;
        let neg_amount = -amount as f64;
        let guide = luminance_bw(img.view(), standard).unwrap();

        let mut a = Array3::<f64>::zeros((h, w, 3));
        let mut b = Array3::<f64>::zeros((h, w, 3));
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            for x in 0..w {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);

                let mut n = 0.0_f64;
                let mut sum_i = 0.0_f64;
                let mut sum_i2 = 0.0_f64;
                let mut sum_p = [0.0_f64; 3];
                let mut sum_ip = [0.0_f64; 3];
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        let iv = guide[[yy, xx, 0]] as f64;
                        n += 1.0;
                        sum_i += iv;
                        sum_i2 += iv * iv;
                        for ch in 0..3 {
                            let pv = img[[yy, xx, ch]] as f64;
                            sum_p[ch] += pv;
                            sum_ip[ch] += iv * pv;
                        }
                    }
                }
                let mean_i = sum_i / n;
                let var_i = (sum_i2 / n - mean_i * mean_i).max(0.0);
                for ch in 0..3 {
                    let mean_p = sum_p[ch] / n;
                    let mean_ip = sum_ip[ch] / n;
                    let cov = mean_ip - mean_i * mean_p;
                    let a_c = if var_i + eps_f64 > 0.0 {
                        cov / (var_i + eps_f64)
                    } else {
                        0.0
                    };
                    a[[y, x, ch]] = a_c;
                    b[[y, x, ch]] = mean_p - a_c * mean_i;
                }
            }
        }

        let mut out = Array3::<f32>::zeros((h, w, 3));
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            for x in 0..w {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);

                let mut n = 0.0_f64;
                let mut sum_a = [0.0_f64; 3];
                let mut sum_b = [0.0_f64; 3];
                for yy in y0..=y1 {
                    for xx in x0..=x1 {
                        n += 1.0;
                        for ch in 0..3 {
                            sum_a[ch] += a[[yy, xx, ch]];
                            sum_b[ch] += b[[yy, xx, ch]];
                        }
                    }
                }
                let iv = guide[[y, x, 0]] as f64;
                for ch in 0..3 {
                    let mean_a = sum_a[ch] / n;
                    let mean_b = sum_b[ch] / n;
                    let q = mean_a * iv + mean_b;
                    let pv = img[[y, x, ch]] as f64;
                    out[[y, x, ch]] = (pv + neg_amount * (pv - q)) as f32;
                }
            }
        }

        out
    }

    /// `cross_guided` (SAT-based) against [`naive_cross_guided`] (window
    /// loops, no SATs) on a small fixed-seed pseudorandom RGB image, over
    /// several radii and (noise_sigma, amount) pairs. Relative agreement
    /// within `1e-5`, with a small absolute floor for values near zero.
    #[test]
    fn cross_guided_matches_a_naive_windowed_reference() {
        let (h, w) = (11, 13);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let img = Array3::<f32>::from_shape_fn((h, w, 3), |_| {
            state = splitmix64(state);
            let bits24 = (state >> 40) as i32; // 0..=0xFF_FFFF
            (bits24 - 0x0080_0000) as f32 / 8_388_608.0 // roughly [-1, 1)
        });

        for radius in 1..=3_u32 {
            for &(noise_sigma, amount) in &[(0.0_f32, 1.0_f32), (0.05, 0.6), (0.4, 1.0)] {
                let eps = noise_sigma * noise_sigma;
                let params =
                    DenoiseParams::new(radius, noise_sigma, amount, LuminanceStandard::Bt709);
                let got = denoise(img.view(), &params).unwrap();
                let want = naive_cross_guided(&img, radius, eps, amount, LuminanceStandard::Bt709);

                for (&g, &w_) in got.iter().zip(want.iter()) {
                    let diff = (g - w_).abs();
                    let rel = diff / w_.abs().max(1e-6);
                    assert!(
                        rel < 1e-5 || diff < 1e-6,
                        "radius={radius} noise_sigma={noise_sigma} amount={amount}: \
                         got {g}, want {w_} (diff {diff}, rel {rel})"
                    );
                }
            }
        }
    }

    /// A constant image drives `var(I)` to (near) zero everywhere; at
    /// `eps = 0` the denominator `var(I) + eps` is exactly `0.0` for
    /// every pixel, exercising the 0/0 → 0 convention rather than a
    /// division that could produce NaN or infinity.
    #[test]
    fn cross_guided_zero_over_zero_case_yields_finite_output() {
        let img = Array3::<f32>::from_elem((9, 9, 3), 0.4_f32);
        let params = DenoiseParams::new(2, 0.0, 1.0, LuminanceStandard::Bt709);
        let out = denoise(img.view(), &params).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "expected finite output on the 0/0 case, got a non-finite value"
        );
    }
}
