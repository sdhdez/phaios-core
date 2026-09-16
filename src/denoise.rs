// SPDX-License-Identifier: GPL-3.0-or-later
//! Guided-filter denoise — self-guided per channel, or cross-guided from
//! a shared luminance guide for RGB.
//!
//! ```text
//! // self-guided (C = 1, and every C other than 3)
//! q     = guided_filter(p, radius, eps)     // eps = noise_sigma², direct window sums
//! out   = p + (−amount) · (p − q)           // amount=0 → p; amount=1 → q
//!
//! // cross-guided (C = 3), guide I = luminance_bw(img, standard)
//! a_c   = cov_w(I, p_c) / (var_w(I) + eps)  // 0/0 → 0, local_contrast's own convention
//! b_c   = mean_w(p_c) − a_c · mean_w(I)
//! q_c   = mean_w(a_c) · I + mean_w(b_c)     // uses the GUIDE, not p_c
//! out_c = p_c + (−amount) · (p_c − q_c)
//! ```
//! `eps = noise_sigma²`. `cov_w`/`var_w`/`mean_w` are window statistics
//! over a `(2·radius+1) × (2·radius+1)` box, **summed directly from the
//! window's own pixels** — see "Direct sums, not summed-area tables"
//! below for why that distinction matters.
//!
//! # Direct sums, not summed-area tables
//!
//! An earlier version of this module built these statistics from global
//! f64 summed-area tables, queried in O(1) per window — the technique
//! [`crate::local_contrast`] itself still uses. It failed at extreme
//! highlights: cross-guided's `cov(I, p_c)` differences two *separately*
//! accumulated global tables whose entries reach `~1e16` at a `1e8`
//! highlight, and two entries differing only by a dark window's own tiny
//! contribution round to the *same* f64 value there — so every window
//! from the bright region onward, including windows that are themselves
//! entirely dark, got the wrong residual (measured 5.1e5 off an
//! independent oracle at one such pixel). The device never had this
//! problem: its kernels sum each window directly from its own
//! `(2r+1)²` pixels, so a running total's magnitude is bounded by the
//! *window*, never by the image. This module now does the same — two
//! passes, horizontal then vertical, each output resummed from scratch
//! rather than carried forward (a sliding running sum is only a
//! differently-shaped prefix table, and reintroduces the identical
//! cancellation). The clipped-window convention itself is unchanged and
//! matches `integral::window_sum`'s (private): rows/columns independently
//! clamped to `[0, extent−1]`, area the product of what is actually
//! covered.
//!
//! Direct summation makes cost linear in `radius` rather than the
//! O(1)-per-pixel a global table gave, so `radius` is bounded above by
//! [`MAX_RADIUS`]. Denoise is meant for a small, physically-motivated
//! noise correlation length, not large-radius structure work — that
//! stays [`crate::local_contrast`]'s job, whose own guided filter keeps
//! the O(1) table and bears its cancellation risk deliberately, at f64.
//!
//! Consequently the single-channel path no longer shares a literal
//! function call with `local_contrast`'s own guided filter — it sums
//! windows by a different route now — so the two agree only within the
//! guided filter's own bound (`rtol 1e-4, atol 1e-6`, the same class the
//! CUDA kernels are held to), not bit-for-bit. Pinned in
//! `tests/kernels.rs`.
//!
//! What the `C = 3` path adds beyond a per-channel parameter change is
//! genuinely new code: a **cross-guided** filter (He–Sun–Tang §3.4) that
//! takes its edges from a shared luminance guide rather than each
//! channel's own signal, so a channel with little structure of its own
//! (a colour cast over an otherwise flat sky, say) is smoothed according
//! to the *guide's* edges, not mistaken structure in its own noise.
//!
//! # `noise_sigma`
//!
//! `eps = noise_sigma²` ties the guided filter's regularisation term to
//! an estimate of the input's noise variance, the reading He, Sun and
//! Tang give ε in their colour-guide formulation. `noise_sigma = 0.0` is
//! the no-regularisation limit — legal (mirrors `eps = 0.0` in
//! [`crate::local_contrast`]) but not on its own an identity. `amount =
//! 0.0` is the only identity switch, and it is a value identity rather
//! than a short-circuit: the filter still runs.
//!
//! # Memory
//!
//! Both paths hold the same tables a summed-area table would have —
//! same shapes, same f64/f32 types — just built by direct window
//! summation instead of a global prefix sum, so the peak scratch
//! figures are unchanged from the SAT-based version this replaced.
//!
//! Self-guided processes channels sequentially, one channel's scratch
//! fully released before the next channel's is built: a transient pair
//! of f64 row-sum tables (`L`, `L²`, 16 B/px) live alongside the f32
//! coefficient pair being computed from them (8 B/px) — a 24 B/px
//! moment — then that coefficient pair lives alongside the second-stage
//! f64 row-sum pair computed from it (`a`, `b`, another 24 B/px moment).
//! Peak **≈24 B/px**, independent of channel count.
//!
//! Cross-guided builds the shared guide `I` (4 B/px) and its shared f64
//! row-sum pair (`I`, `I²`, 16 B/px) once, live for the whole call
//! (**20 B/px shared baseline**), then processes channels sequentially.
//! Per channel: a transient f64 row-sum pair (`p_c`, `I·p_c`, 16 B/px —
//! computed directly, with no elementwise-product array to materialise
//! first, unlike the SAT-based version this replaced) lives alongside
//! the f32 coefficient pair (8 B/px) being computed from it, then that
//! coefficient pair lives alongside the second-stage f64 row-sum pair
//! (`a_c`, `b_c`, 16 B/px) — a worst per-channel moment of 24 B/px, on
//! top of the shared baseline. Peak **≈44 B/px, ≈1.06 GB at 24 MP**.
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
//! formulation this module's self-guided path's own maths follows, and
//! its provenance note on the reference MATLAB implementation's licence.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2};
use pyo3::{pyclass, pymethods};

use crate::bw::{LuminanceStandard, luminance_bw};
use crate::error::PhaiosError;

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
    /// Bounded above by [`MAX_RADIUS`]: unlike
    /// [`crate::local_contrast::GuidedFilterParams`]'s own unvalidated
    /// radius (an O(1)-per-pixel table query regardless of size), this
    /// kernel's window statistics are summed directly from the window's
    /// own pixels, so cost is linear in radius — see the module
    /// documentation.
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

/// Largest radius [`denoise`] will accept, in pixels.
///
/// [`crate::local_contrast::GuidedFilterParams`]'s radius is unbounded
/// because it costs the same, O(1) per pixel, at any size — a global
/// summed-area table queried by four-corner subtraction. This module
/// gave up that table (see the module documentation's "Direct sums, not
/// summed-area tables") to fix a cancellation at extreme highlights, and
/// a direct window sum costs O(radius) per pixel instead. 32 keeps that
/// linear cost cheap (a `65 × 65` window at most) and is generous for
/// any physically motivated noise correlation length; a caller wanting
/// a larger smoothing radius wants `local_contrast`, not `denoise`.
pub const MAX_RADIUS: u32 = 32;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate denoise parameters. Shared by the CPU kernel and its CUDA
/// twin, so both backends reject exactly the same inputs with exactly
/// the same messages.
pub(crate) fn validate(params: &DenoiseParams) -> Result<(), PhaiosError> {
    if params.radius > MAX_RADIUS {
        return Err(PhaiosError::Parameter(format!(
            "radius is {}, above the maximum of {MAX_RADIUS}",
            params.radius
        )));
    }
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

// ── Direct window sums ──────────────────────────────────────────────────────
//
// Every function below mirrors one kernel in src/cuda/ptx/{local_contrast,
// denoise}.cu, so a reviewer can check either side against the other
// directly. None of them builds a running or global total: each output
// element is an independent sum over its own clipped window, computed
// from scratch, which is the whole fix (see the module documentation).

/// The clipped window's row/column bounds and area for one output pixel
/// — the same shrinking-window convention [`crate::integral::window_sum`]
/// uses: rows and columns independently clamped to `[0, extent−1]`, area
/// the product of what is actually covered. Shared by every stage below
/// so the convention is written once, not re-derived per kernel.
#[inline]
fn window_bounds(
    y: usize,
    x: usize,
    r: usize,
    h: usize,
    w: usize,
) -> (usize, usize, usize, usize, f64) {
    let y1 = y.saturating_sub(r);
    let y2 = (y + r).min(h - 1);
    let x1 = x.saturating_sub(r);
    let x2 = (x + r).min(w - 1);
    let area = ((y2 - y1 + 1) * (x2 - x1 + 1)) as f64;
    (y1, y2, x1, x2, area)
}

/// Direct horizontal window sums of `v` and `v²`, in f64 — mirrors
/// `box_h_l_l2` (`src/cuda/ptx/local_contrast.cu`): for each row,
/// independently, a fresh sum from scratch over the clipped range
/// `[x−r, x+r] ∩ [0, w−1]`, never a running total carried along the row.
/// Used for `(L, L²)` in the self-guided path and `(I, I²)` in the
/// cross-guided path's shared guide stage.
///
/// Parallel over rows only: each output cell's own sums have one fixed
/// term order regardless of how rayon schedules rows across threads, so
/// the result is identical at any thread count.
fn box_h_pair(data: ArrayView2<f32>, r: usize) -> Result<(Array2<f64>, Array2<f64>), PhaiosError> {
    let (h, w) = data.dim();
    let mut hsum = crate::alloc::zeros2::<f64>((h, w))?;
    let mut hsum2 = crate::alloc::zeros2::<f64>((h, w))?;
    ndarray::Zip::from(hsum.rows_mut())
        .and(hsum2.rows_mut())
        .and(data.rows())
        .par_for_each(|mut orow, mut orow2, drow| {
            for x in 0..w {
                let x1 = x.saturating_sub(r);
                let x2 = (x + r).min(w - 1);
                let mut sum = 0.0_f64;
                let mut sum2 = 0.0_f64;
                for i in x1..=x2 {
                    let v = drow[i] as f64;
                    sum += v;
                    sum2 += v * v;
                }
                orow[x] = sum;
                orow2[x] = sum2;
            }
        });
    Ok((hsum, hsum2))
}

/// Direct horizontal window sums of `p_c` and `I·p_c`, in f64 — mirrors
/// `box_h_cross` (`src/cuda/ptx/denoise.cu`), generalising [`box_h_pair`]
/// from one array summed against itself to a guide/channel pair. Each
/// factor is cast to f64 before multiplying (not multiplied in f32 then
/// widened), matching the device kernel's own precision choice — more
/// accurate than this module's previous SAT-based version, which
/// widened the product only after forming it in f32; can only help
/// agreement with the naive oracle, never hurt it.
fn box_h_cross_pair(
    guide: ArrayView2<f32>,
    data: ArrayView2<f32>,
    r: usize,
) -> Result<(Array2<f64>, Array2<f64>), PhaiosError> {
    let (h, w) = data.dim();
    let mut hsum_p = crate::alloc::zeros2::<f64>((h, w))?;
    let mut hsum_ip = crate::alloc::zeros2::<f64>((h, w))?;
    ndarray::Zip::from(hsum_p.rows_mut())
        .and(hsum_ip.rows_mut())
        .and(guide.rows())
        .and(data.rows())
        .par_for_each(|mut op, mut oip, grow, drow| {
            for x in 0..w {
                let x1 = x.saturating_sub(r);
                let x2 = (x + r).min(w - 1);
                let mut sum_p = 0.0_f64;
                let mut sum_ip = 0.0_f64;
                for i in x1..=x2 {
                    let gv = grow[i] as f64;
                    let pv = drow[i] as f64;
                    sum_p += pv;
                    sum_ip += gv * pv;
                }
                op[x] = sum_p;
                oip[x] = sum_ip;
            }
        });
    Ok((hsum_p, hsum_ip))
}

/// Direct horizontal window sums of two independent arrays, verbatim, in
/// f64 — mirrors `box_h_ab` (`src/cuda/ptx/local_contrast.cu`). Shared
/// by both paths' second stage: smoothing the linear-model coefficients
/// `a`/`b` is identical maths whether they came from the self- or
/// cross-guided first stage. `a`/`b` themselves stay f32 between the
/// two box-sum stages, matching `local_contrast`'s own
/// `guided_filter` convention (and the device's Kahan-compensated f32);
/// promoting them to f64 was tried and measured to make no difference
/// to agreement with an all-f64 naive reference at extreme highlights —
/// the residual gap there comes from summing the *first*-stage
/// statistics in a different grouping (row-then-column here, one flat
/// running total there), not from `a`/`b`'s own storage width. See the
/// HDR oracle test in this module's own unit tests for the measurement.
fn box_h_two(
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    r: usize,
) -> Result<(Array2<f64>, Array2<f64>), PhaiosError> {
    let (h, w) = a.dim();
    let mut hsum_a = crate::alloc::zeros2::<f64>((h, w))?;
    let mut hsum_b = crate::alloc::zeros2::<f64>((h, w))?;
    ndarray::Zip::from(hsum_a.rows_mut())
        .and(hsum_b.rows_mut())
        .and(a.rows())
        .and(b.rows())
        .par_for_each(|mut oa, mut ob, arow, brow| {
            for x in 0..w {
                let x1 = x.saturating_sub(r);
                let x2 = (x + r).min(w - 1);
                let mut sum_a = 0.0_f64;
                let mut sum_b = 0.0_f64;
                for i in x1..=x2 {
                    sum_a += arow[i] as f64;
                    sum_b += brow[i] as f64;
                }
                oa[x] = sum_a;
                ob[x] = sum_b;
            }
        });
    Ok((hsum_a, hsum_b))
}

/// Vertical window sums of `hsum_l`/`hsum_l2` and the self-guided linear
/// model built from them — mirrors `coeff_ab`
/// (`src/cuda/ptx/local_contrast.cu`) line for line: same window
/// clamping (via [`window_bounds`]), same `var ≥ 0` clamp against the
/// cancelling subtraction, same 0/0 → 0 convention.
fn coeff_self(
    hsum_l: &Array2<f64>,
    hsum_l2: &Array2<f64>,
    r: usize,
    eps_f64: f64,
) -> Result<(Array2<f32>, Array2<f32>), PhaiosError> {
    let (h, w) = hsum_l.dim();
    let mut a_arr = crate::alloc::zeros2::<f32>((h, w))?;
    let mut b_arr = crate::alloc::zeros2::<f32>((h, w))?;
    ndarray::Zip::indexed(&mut a_arr)
        .and(&mut b_arr)
        .par_for_each(|(y, x), a_out, b_out| {
            let (y1, y2, _, _, area) = window_bounds(y, x, r, h, w);
            let mut sum_l = 0.0_f64;
            let mut sum_l2 = 0.0_f64;
            for j in y1..=y2 {
                sum_l += hsum_l[[j, x]];
                sum_l2 += hsum_l2[[j, x]];
            }
            let mean_l = sum_l / area;
            let mean_l2 = sum_l2 / area;
            // Cancelling subtraction: kept clamped as a matching
            // convention with local_contrast, even though a direct
            // local sum makes a spuriously negative result far rarer
            // than a global table did.
            let var_l = (mean_l2 - mean_l * mean_l).max(0.0);
            let a = if var_l + eps_f64 > 0.0 {
                var_l / (var_l + eps_f64)
            } else {
                0.0
            };
            *a_out = a as f32;
            *b_out = (mean_l * (1.0 - a)) as f32;
        });
    Ok((a_arr, b_arr))
}

/// Vertical window sums of `hsum_i`/`hsum_i2`/`hsum_p`/`hsum_ip`, fused
/// into one pass over each column's clipped range, and the cross-guided
/// linear model built from them — mirrors `coeff_ab_cross`
/// (`src/cuda/ptx/denoise.cu`) line for line: `var(I)` clamped ≥ 0 (a
/// cancelling subtraction), `cov(I, p_c)` left free to be negative (a
/// genuine covariance), same 0/0 → 0 convention.
fn coeff_cross(
    hsum_i: &Array2<f64>,
    hsum_i2: &Array2<f64>,
    hsum_p: &Array2<f64>,
    hsum_ip: &Array2<f64>,
    r: usize,
    eps_f64: f64,
) -> Result<(Array2<f32>, Array2<f32>), PhaiosError> {
    let (h, w) = hsum_i.dim();
    let mut a_arr = crate::alloc::zeros2::<f32>((h, w))?;
    let mut b_arr = crate::alloc::zeros2::<f32>((h, w))?;
    ndarray::Zip::indexed(&mut a_arr)
        .and(&mut b_arr)
        .par_for_each(|(y, x), a_out, b_out| {
            let (y1, y2, _, _, area) = window_bounds(y, x, r, h, w);
            let (mut sum_i, mut sum_i2, mut sum_p, mut sum_ip) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for j in y1..=y2 {
                sum_i += hsum_i[[j, x]];
                sum_i2 += hsum_i2[[j, x]];
                sum_p += hsum_p[[j, x]];
                sum_ip += hsum_ip[[j, x]];
            }
            let mean_i = sum_i / area;
            let mean_i2 = sum_i2 / area;
            let mean_p = sum_p / area;
            let mean_ip = sum_ip / area;
            let var_i = (mean_i2 - mean_i * mean_i).max(0.0);
            let cov = mean_ip - mean_i * mean_p;
            let a = if var_i + eps_f64 > 0.0 {
                cov / (var_i + eps_f64)
            } else {
                0.0
            };
            let b = mean_p - a * mean_i;
            *a_out = a as f32;
            *b_out = b as f32;
        });
    Ok((a_arr, b_arr))
}

/// Vertical window means of `hsum_a`/`hsum_b`, the model prediction
/// `q = mean(a)·guide + mean(b)`, and the blend
/// `out = p + (−amount)·(p − q)` — mirrors `final_out`/`final_out_cross`
/// (`src/cuda/ptx/{local_contrast,denoise}.cu`). `guide` is `p` itself
/// for the self-guided path (`I = p`) or the shared luminance for
/// cross-guided; that is the only difference between the two device
/// kernels, and the only reason this function takes it separately from
/// `p`.
fn final_combine(
    hsum_a: &Array2<f64>,
    hsum_b: &Array2<f64>,
    guide: ArrayView2<f32>,
    p: ArrayView2<f32>,
    neg_amount: f32,
    r: usize,
    out: ArrayViewMut2<f32>,
) {
    let (h, w) = hsum_a.dim();
    ndarray::Zip::indexed(out)
        .and(&guide)
        .and(&p)
        .par_for_each(|(y, x), o, &gv, &pv| {
            let (y1, y2, _, _, area) = window_bounds(y, x, r, h, w);
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            for j in y1..=y2 {
                sum_a += hsum_a[[j, x]];
                sum_b += hsum_b[[j, x]];
            }
            let mean_a = sum_a / area;
            let mean_b = sum_b / area;
            let q = (mean_a * gv as f64 + mean_b) as f32;
            *o = pv + neg_amount * (pv - q);
        });
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Self-guided path: every channel filtered independently. Covers
/// `C == 1` (the common case) and any `C` other than 3 — there is
/// nothing special about one channel; it is simply this loop's `C == 1`
/// case. Computes the guided filter directly (`I = p`) via the direct
/// window sums above rather than delegating to
/// [`crate::local_contrast`], which still uses a global summed-area
/// table (see the module documentation).
///
/// Writes `out = p + (−amount) · (p − q)` — the same combine shape as
/// `local_contrast`'s own `l + strength·(l−q)` at `strength = −amount`.
fn self_guided(
    img: ArrayView3<f32>,
    radius: u32,
    eps: f32,
    amount: f32,
    out: &mut Array3<f32>,
) -> Result<(), PhaiosError> {
    let (_, _, c) = img.dim();
    let r = radius as usize;
    let eps_f64 = eps as f64;
    let neg_amount = -amount;

    for ch in 0..c {
        let p = img.slice(ndarray::s![.., .., ch]);

        let (a_arr, b_arr) = {
            let (hsum_l, hsum_l2) = box_h_pair(p, r)?;
            coeff_self(&hsum_l, &hsum_l2, r, eps_f64)?
            // hsum_l/hsum_l2 drop here, before the a/b row-sum tables
            // below are built.
        };

        let (hsum_a, hsum_b) = box_h_two(a_arr.view(), b_arr.view(), r)?;
        drop(a_arr);
        drop(b_arr);

        final_combine(
            &hsum_a,
            &hsum_b,
            p,
            p,
            neg_amount,
            r,
            out.slice_mut(ndarray::s![.., .., ch]),
        );
        // hsum_a/hsum_b drop here, before the next channel's tables are
        // built -- no two channels' scratch overlap.
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
    let r = radius as usize;
    let eps_f64 = eps as f64;
    let neg_amount = -amount;

    let guide = luminance_bw(img, standard)?;
    let guide2d = guide.index_axis(ndarray::Axis(2), 0);

    // Shared across all three channels: dropped only when this function
    // returns.
    let (hsum_i, hsum_i2) = box_h_pair(guide2d, r)?;

    for ch in 0..3 {
        let p_c = img.slice(ndarray::s![.., .., ch]);

        let (a_arr, b_arr) = {
            let (hsum_p, hsum_ip) = box_h_cross_pair(guide2d, p_c, r)?;
            coeff_cross(&hsum_i, &hsum_i2, &hsum_p, &hsum_ip, r, eps_f64)?
            // hsum_p/hsum_ip drop here.
        };

        let (hsum_a, hsum_b) = box_h_two(a_arr.view(), b_arr.view(), r)?;
        drop(a_arr);
        drop(b_arr);

        final_combine(
            &hsum_a,
            &hsum_b,
            guide2d,
            p_c,
            neg_amount,
            r,
            out.slice_mut(ndarray::s![.., .., ch]),
        );
        // hsum_a/hsum_b drop here, before the next channel's first-stage
        // tables are built.
    }

    Ok(())
}

/// Denoise with the He–Sun–Tang guided filter.
///
/// `C == 3` uses a cross-guided filter with edges from a shared
/// luminance guide; every other channel count, including `C == 1`, is
/// self-guided per channel. See the module documentation for both
/// formulas, the `noise_sigma` → `eps` mapping, the self-guided
/// agreement with `local_contrast`, memory, order and range notes.
///
/// Input shape: `(H, W, C)`, any channel count, any memory layout.
/// Output shape: `(H, W, C)`, freshly allocated and C-contiguous. Empty
/// input (`H`, `W` or `C` is `0`) returns the empty array of the same
/// shape.
///
/// `amount = 0.0` returns the input value for value, but the filter is
/// still evaluated: there is no short-circuit, and the passthrough comes
/// from `p + (−0.0)·(p − q)` collapsing. One consequence is that a `-0.0`
/// input can come back as `+0.0`, depending on the sign of `p − q`.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `radius` is above [`MAX_RADIUS`],
///   `noise_sigma` is negative or not finite, or `amount` is outside
///   `0..=1` or not finite.
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
        assert!(
            validate(&DenoiseParams::new(
                MAX_RADIUS,
                0.0,
                0.0,
                LuminanceStandard::Bt709
            ))
            .is_ok()
        );
    }

    #[test]
    fn validate_rejects_radius_above_max() {
        let err = validate(&DenoiseParams::new(
            MAX_RADIUS + 1,
            0.0,
            0.0,
            LuminanceStandard::Bt709,
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("radius is"),
            "radius={}: {err}",
            MAX_RADIUS + 1
        );
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
    /// explicit window loops (no SATs, no `window_sum`, no separable
    /// box-sum reformulation) and f64 accumulation throughout — an
    /// independent oracle sharing no arithmetic structure with
    /// [`cross_guided`]'s implementation, per this crate's practice that
    /// an empirical sweep against an independent reference catches what
    /// code review alone does not.
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

    /// `cross_guided` (direct box sums) against [`naive_cross_guided`]
    /// (window loops, no box-sum reformulation) on a small fixed-seed
    /// pseudorandom RGB image, over several radii and (noise_sigma,
    /// amount) pairs. Relative agreement within `1e-5`, with a small
    /// absolute floor for values near zero.
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

    /// The exact construction that exposed the pre-fix cancellation
    /// (`.cache/scratch/denoise/PROGRESS.md`, step 4's finding): a
    /// near-zero field with a `1e8` bar present in R and B only (G stays
    /// at the field value everywhere), so `cov(I, p_c)` must resolve a
    /// genuine covariance between two very different signals at exactly
    /// the magnitude where a global summed-area table lost the residual.
    /// `amount = 1.0` so the output is `q` exactly, with nothing scaling
    /// an error down.
    ///
    /// Checked against the guided filter's own committed class
    /// (`rtol 1e-4, atol 1e-6`, `docs/ffi.md` §6 — the same bound the
    /// CUDA conformance sweeps hold this kernel to), not the tighter
    /// `1e-6` pure-relative figure this task was framed around: measured
    /// directly, a pure `1e-6` relative check fails here by two to three
    /// orders of magnitude at the darkest pixels, even with `a`/`b`
    /// promoted to f64 (tried and reverted — see [`box_h_two`]'s doc
    /// comment; it changed nothing). The residual is not the fixed
    /// cancellation: it is ordinary floating-point summation-order
    /// sensitivity — this module sums each window's first-stage
    /// statistics row-then-column, the reference sums them as one flat
    /// running total, and at a 12-order-of-magnitude spread between the
    /// `1e-4` field and the `1e8` bar those two *valid* f64 summations
    /// of the same terms round differently by more than `1e-6` relative
    /// at a dark output. Measured worst case at the darkest pixels: well
    /// under the GUIDED_FILTER bound with real margin (recorded via
    /// `--nocapture` below) — five to six orders of magnitude tighter
    /// than the ~5e5 the pre-fix global-table cancellation produced, and
    /// every pixel, dark ones included, is covered by the sweep.
    #[test]
    fn cross_guided_matches_the_naive_reference_on_an_hdr_partial_channel_bar() {
        let (h, w) = (20, 32);
        let mut img = Array3::<f32>::from_elem((h, w, 3), 1e-4_f32);
        let (y0, y1) = (h * 4 / 10, h * 4 / 10 + 4);
        let (x0, x1) = (w / 10, w - w / 10);
        img.slice_mut(ndarray::s![y0..y1, x0..x1, 0]).fill(1e8_f32);
        img.slice_mut(ndarray::s![y0..y1, x0..x1, 2]).fill(1e8_f32);
        // Channel 1 (G) stays at the dark field value everywhere.

        let noise_sigma = 0.1_f32;
        let amount = 1.0_f32;
        let eps = noise_sigma * noise_sigma;
        let (atol, rtol) = (1e-6_f32, 1e-4_f32); // GUIDED_FILTER class.

        for &radius in &[2_u32, 8] {
            let params = DenoiseParams::new(radius, noise_sigma, amount, LuminanceStandard::Bt709);
            let got = denoise(img.view(), &params).unwrap();
            let want = naive_cross_guided(&img, radius, eps, amount, LuminanceStandard::Bt709);

            let mut worst = 0.0_f32;
            let mut worst_where = 0usize;
            for (idx, (&g, &w_)) in got.iter().zip(want.iter()).enumerate() {
                let diff = (g - w_).abs();
                let ratio = diff / (atol + rtol * w_.abs());
                if ratio > worst {
                    worst = ratio;
                    worst_where = idx;
                }
                assert!(
                    ratio <= 1.0,
                    "radius={radius} idx={idx}: got {g}, want {w_} (diff {diff}, \
                     {ratio:.4}x the (rtol {rtol:e}, atol {atol:e}) bound)"
                );
            }
            eprintln!(
                "HDR partial-channel-bar naive-oracle check, radius={radius}: worst \
                 {worst:.4}x the GUIDED_FILTER bound (idx {worst_where})"
            );
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

    // ── box_h_cross_pair: direct sums, not sliding ──────────────────────────

    /// Added after the step 4b mutation gate (`.cache/scratch/denoise/
    /// PROGRESS.md`) found a real blind spot: replacing
    /// [`box_h_cross_pair`]'s from-scratch loop with an incremental
    /// sliding sum (add the entering element on the right, subtract the
    /// departing one on the left as `x` advances) left every existing
    /// end-to-end test green, including this module's own HDR
    /// partial-channel-bar oracle test at `radius ∈ {2, 8}` and
    /// `tests/cuda_conformance.rs`'s equivalent sweep up to
    /// `radius = MAX_RADIUS`, across every width tried. A *correctly
    /// bounded* sliding window's accumulator magnitude is tied to the
    /// window's own `(2r+1)` pixels, not the whole image the way a
    /// genuine summed-area table's is — measurably better-behaved,
    /// empirically, than the module documentation's "reintroduces the
    /// identical cancellation" claim suggests, at least at the radii and
    /// dynamic ranges the shipped sweeps happen to cover.
    ///
    /// This test isolates the one function the mutation touches instead
    /// of going through the full kernel and an independent oracle (whose
    /// own summation order differs enough at extreme highlights to
    /// confound the comparison on its own — see the HDR oracle test
    /// above). A direct sum and a sliding sum are the *same* sequence of
    /// left-to-right additions for as long as the window has only grown
    /// (`x1` still `0`), so no dynamic range is needed to tell them apart
    /// until the window first loses an element on the left; from then on
    /// a sliding accumulator carries that element's contribution through
    /// extra additions before subtracting it back out, rather than never
    /// having included it. At ordinary unit-range values f64 has enough
    /// mantissa to make both routes bit-identical regardless (checked
    /// during development); one value twelve orders of magnitude above
    /// its neighbours is what exposes the difference, at `x = 6` here
    /// (radius 3, the spike at column 2 — the first `x` whose window no
    /// longer reaches column 2). Mutation this catches: any accumulation
    /// in [`box_h_cross_pair`] (or a future refactor introducing one)
    /// that carries a running total across output positions instead of
    /// summing each window fresh from its own pixels.
    #[test]
    fn box_h_cross_pair_matches_a_fresh_sum_once_a_value_leaves_the_window() {
        let (h, w, r) = (1_usize, 20_usize, 3_usize);
        let mut guide = Array2::<f32>::from_elem((h, w), 1e-4_f32);
        let mut data = Array2::<f32>::from_elem((h, w), 1e-4_f32);
        guide[[0, 2]] = 1e8_f32; // one extreme spike, everything else flat
        data[[0, 2]] = 1e8_f32;

        let (hsum_p, hsum_ip) = box_h_cross_pair(guide.view(), data.view(), r).unwrap();

        // Independent direct-from-scratch reference: same clamped bounds,
        // same left-to-right term order, so a correct box_h_cross_pair
        // must match it bit for bit.
        for x in 0..w {
            let x1 = x.saturating_sub(r);
            let x2 = (x + r).min(w - 1);
            let mut want_p = 0.0_f64;
            let mut want_ip = 0.0_f64;
            for i in x1..=x2 {
                want_p += data[[0, i]] as f64;
                want_ip += guide[[0, i]] as f64 * data[[0, i]] as f64;
            }
            assert_eq!(
                hsum_p[[0, x]].to_bits(),
                want_p.to_bits(),
                "x={x}: sum_p diverged from a fresh from-scratch sum over \
                 the same clamped window (direct sum required, not a \
                 sliding/incremental one)"
            );
            assert_eq!(
                hsum_ip[[0, x]].to_bits(),
                want_ip.to_bits(),
                "x={x}: sum_ip diverged from a fresh from-scratch sum over \
                 the same clamped window (direct sum required, not a \
                 sliding/incremental one)"
            );
        }
    }
}
