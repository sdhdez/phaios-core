// SPDX-License-Identifier: GPL-3.0-or-later
//! Gaussian blur — the primitive the scattering kernels are built on.
//!
//! Useful alone for softening, and the reason it exists first: halation,
//! diffusion and veiling glare are all *blur, weighted and added back*,
//! so none of them can be written until this is.
//!
//! # Two paths, and why
//!
//! A Gaussian is separable, so both paths filter rows then columns. What
//! differs is how each 1-D pass is computed, and the choice is forced by
//! measurement rather than taste:
//!
//! - **σ < 6: direct convolution** with a sampled, normalised Gaussian
//!   truncated at ±4σ. Exact by construction, and 49 taps per axis at
//!   the top of this range.
//! - **σ ≥ 6: three box passes**, widths chosen so their variances sum
//!   to σ². Cost is independent of σ, where direct convolution grows
//!   without bound: at σ = 32 a truncated kernel would need 386 taps per
//!   axis.
//!
//! The crossover sits where the two *cost* the same, because on accuracy
//! the direct path always wins — it has no σ quantisation and no shape
//! error at all. Measured on a 24 MP frame: direct is 50 ms at σ = 2 and
//! 71 ms at σ = 3.9, about 1.3 ms per tap, so it reaches the box path's
//! flat 90 ms at roughly σ = 6. Below that the direct path is both more
//! accurate *and* faster; above it the box path's independence from σ
//! takes over — 90 ms whether σ is 6 or 64.
//!
//! Three successive box filters converge on a Gaussian by the central
//! limit theorem — the classical result, and the basis of Kovesi's
//! *Fast Almost-Gaussian Filtering* (DICTA 2010).
//!
//! # Why the box path cannot simply be used everywhere
//!
//! Because at small σ it cannot deliver the σ that was asked for. A box of odd width `w` has variance `(w² − 1)/12`, and variances
//! add in series, so three width-3 boxes give σ = 1.414 and **nothing
//! smaller is reachable at all**. Just above that floor the achievable
//! σ values are sparse: a request for 2.0 lands on 2.16, an 8% error in
//! a parameter the caller can see. By σ = 6 the reachable set is dense
//! enough to stay within 1% — see `MAX_WIDTH_RATIO` for the second
//! constraint the search needs.
//!
//! Two things that look like fixes and are not, both measured:
//!
//! - **More box passes do not help small σ.** They *raise* the floor:
//!   1.414 for three passes, 1.633 for four, 1.826 for five. Extra
//!   passes buy shape accuracy at large σ, not reach at small σ.
//! - **Matching σ alone is the wrong objective.** An unconstrained
//!   search for widths hitting σ = 2 exactly returns `[1, 1, 7]` — the
//!   right variance from a single box and two passes that do nothing,
//!   with none of the Gaussian's shape. Every width must be at least 3.
//!
//! # Determinism
//!
//! Bounded, not bit-exact — the same status as
//! [`crate::local_contrast`], and for the same reason: the running sums
//! accumulate in a different order on a device than on a host. See
//! `docs/ffi.md` §6.
//!
//! On **non-finite** input the two backends diverge below the crossover,
//! and `docs/ffi.md` §1 records it: the direct path's Kahan compensation
//! computes `∞ − ∞ = NaN` and poisons the rest of the sum, so the device
//! returns NaN where the host returns ±∞. At or above the crossover both
//! backends now accumulate in f64 and agree — a sliding window subtracts,
//! so `∞` becomes NaN on *both* sides. Finite samples are a precondition
//! of the whole crate either way.

use ndarray::{Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

/// Requested σ at or above which the box approximation is used.
///
/// Set where the two paths cost the same on a 24 MP frame, since the
/// direct path is strictly more accurate below it — see the module
/// documentation.
pub const BOX_CROSSOVER_SIGMA: f32 = 6.0;

/// Where the direct Gaussian is truncated, in units of σ. At ±4σ the
/// discarded tail is under 1e-4 of the total weight.
const TRUNCATION: f32 = 4.0;

// ── Parameter types ───────────────────────────────────────────────────────────

/// Blur kernel shape.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlurShape {
    /// Isotropic Gaussian. The default, and the right choice for
    /// diffusion, halation and glare, where the point is a smooth
    /// falloff rather than a recognisable aperture.
    #[default]
    Gaussian,
}

/// Parameters for [`blur`].
///
/// ```python
/// params = phaios_core.BlurParams(sigma=8.0)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct BlurParams {
    /// Standard deviation in pixels. `0.0` is the identity.
    ///
    /// Measured in pixels of the image as given, so a blur applied to a
    /// half-size preview is *not* the same picture as the same σ on the
    /// full frame — unlike [`crate::vignette`], which is
    /// resolution-independent by construction. Scale σ with the image if
    /// you are previewing.
    ///
    /// Bounded above by [`MAX_SIGMA`].
    #[pyo3(get, set)]
    pub sigma: f32,
    /// Kernel shape. Default [`BlurShape::Gaussian`].
    #[pyo3(get, set)]
    pub shape: BlurShape,
}

#[pymethods]
impl BlurParams {
    /// Create new ``BlurParams``.
    #[new]
    #[pyo3(signature = (sigma = 0.0, shape = BlurShape::Gaussian))]
    pub fn new(sigma: f32, shape: BlurShape) -> Self {
        Self { sigma, shape }
    }

    /// Two ``BlurParams`` are equal when both fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.sigma.to_bits() == other.sigma.to_bits() && self.shape == other.shape
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "BlurParams(sigma={}, shape=BlurShape.{:?})",
            self.sigma, self.shape
        )
    }
}

impl Default for BlurParams {
    /// σ = 0: the identity.
    fn default() -> Self {
        Self {
            sigma: 0.0,
            shape: BlurShape::Gaussian,
        }
    }
}

// ── Kernel construction ───────────────────────────────────────────────────────

/// Standard deviation of a series of box filters of the given odd widths.
///
/// A box of odd width `w` has variance `(w² − 1)/12`, and independent
/// variances add.
#[must_use]
pub(crate) fn box_series_sigma(widths: &[usize]) -> f32 {
    let var: f64 = widths.iter().map(|&w| ((w * w) as f64 - 1.0) / 12.0).sum();
    var.sqrt() as f32
}

/// Largest ratio permitted between the widest and narrowest box.
///
/// Without it the search degenerates: minimising σ error alone picks
/// `[3, 3, 17]` for σ = 5, which has the best σ match available (0.66%)
/// and is *almost a single box* — two passes that barely filter and one
/// that does all the work, so the result carries a box's hard edges
/// rather than a Gaussian's falloff. Measured shape error against a true
/// Gaussian: 0.0223 for that triple against 0.0131 for a balanced one at
/// the same σ error, and at σ = 8 the gap is 0.0099 against 0.0033.
///
/// Three at a time is enough freedom to hit σ closely — the worst σ
/// error over 4…64 stays under 1% — while forcing every pass to do real
/// work, which is what makes the central limit theorem apply.
const MAX_WIDTH_RATIO: usize = 3;

/// Largest standard deviation any blur will accept, in pixels.
///
/// Choosing the box widths searches a space that grows with σ, and σ
/// arrives from the caller. Before this bound that search was cubic *and*
/// unbounded:
/// σ = 3000 took 4.6 s, σ = 10 000 did not return at all, and because the
/// kernel runs under `py.detach` there was no way to interrupt it from
/// Python — a caller-controlled parameter that hangs the calling thread.
///
/// 4096 px of standard deviation is far past any frame this crate is built
/// for; a 24 MP image is 6000 px on its long edge, and a blur at this σ
/// returns a near-constant field. With the search pruned to its feasible
/// range the worst case inside the bound is a few tens of milliseconds.
pub const MAX_SIGMA: f32 = 4096.0;

/// Three odd widths, each at least 3 and within [`MAX_WIDTH_RATIO`] of
/// one another, whose box series best matches `sigma`.
///
/// Searched directly rather than taken from a closed form. The usual
/// two-width formula picks a worse triple than a search does (for σ = 2
/// it gives 1.826 where 2.16 is available), and the search runs once per
/// call, not once per pixel.
pub(crate) fn box_widths(sigma: f32) -> [usize; 3] {
    // Variances add, so this is a search for three odd widths with
    //     a² + b² + c² = 12σ² + 3
    // as nearly as odd integers allow. The objective stays in σ space
    // rather than sum space: √ is concave, so the triple closest in *sum*
    // is not always the triple closest in σ.
    let target = 12.0 * f64::from(sigma) * f64::from(sigma) + 3.0;

    // The width ceiling is part of the *result*, not just a stopping rule,
    // and is kept bit-for-bit as it was. It excludes triples that sit at
    // the MAX_WIDTH_RATIO ceiling even when they tie on σ: at σ = 9 both
    // [15, 15, 23] and [9, 13, 27] sum to 979 and so match σ identically,
    // but the first is balanced and the second is the near-a-single-box
    // shape MAX_WIDTH_RATIO exists to keep out. Widening this changes the
    // picture, so it does not move.
    let hi = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() as usize).max(3) + 8;

    // The lower bound *is* a pure pruning, and is exact. With
    // 3 ≤ a ≤ b ≤ c ≤ min(MAX_WIDTH_RATIO·a, hi) the largest sum any `a`
    // can reach is 19a², so an `a` falling short of the target there is
    // beaten by the balanced triple near √(target/3) — which is always
    // inside the range and always near-exact. Two widths of slack absorbs
    // the rounding to odd. This is what turns a cubic sweep starting at 3
    // into a quadratic one starting near σ.
    let lo = ((target / 19.0).sqrt().floor() as usize)
        .saturating_sub(2)
        .max(3);

    let mut best = [3_usize; 3];
    let mut best_err = f32::INFINITY;
    let mut a = lo | 1;
    while a <= hi {
        // `3a` is odd because `a` is, but `hi` need not be — round the
        // ceiling down so it can never contribute an even width.
        let c_max = {
            let m = (a * MAX_WIDTH_RATIO).min(hi);
            if m.is_multiple_of(2) { m - 1 } else { m }
        };
        let mut b = a;
        while b <= c_max {
            // Second, `c`. For fixed `a` and `b` the series σ is strictly
            // increasing in `c`, so |σ_series − σ| is strictly V-shaped and
            // only the odd widths bracketing the exact solution — or the
            // ends of the legal range, when the solution falls outside it —
            // can win. Evaluated ascending with a strict improvement test,
            // this picks exactly the triple a full scan over `c` would.
            let rest = target - (a * a) as f64 - (b * b) as f64;
            let exact = if rest > 0.0 { rest.sqrt() } else { 0.0 };
            let mid = (exact.floor() as usize).max(1) | 1;
            let mut cands = [b, mid.saturating_sub(2) | 1, mid, mid + 2, c_max];
            cands.sort_unstable();
            let mut prev = 0;
            for c in cands {
                if c == prev || c < b || c > c_max {
                    continue;
                }
                prev = c;
                let err = (box_series_sigma(&[a, b, c]) - sigma).abs();
                if err < best_err {
                    best_err = err;
                    best = [a, b, c];
                }
            }
            b += 2;
        }
        a += 2;
    }
    best
}

/// Sampled, normalised Gaussian weights truncated at ±`TRUNCATION`·σ.
///
/// Normalising the *sampled* weights rather than using the continuous
/// density is what makes a constant image survive exactly: the taps sum
/// to one by construction.
pub(crate) fn gaussian_weights(sigma: f32) -> Vec<f32> {
    let radius = (TRUNCATION * sigma).ceil().max(1.0) as usize;
    let two_sigma_sq = 2.0 * (sigma as f64) * (sigma as f64);
    let mut w: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let d = i as f64 - radius as f64;
            (-d * d / two_sigma_sq).exp()
        })
        .collect();
    let sum: f64 = w.iter().sum();
    for v in &mut w {
        *v /= sum;
    }
    w.into_iter().map(|v| v as f32).collect()
}

// ── 1-D passes ────────────────────────────────────────────────────────────────

/// Convolve one lane against `weights`, clamping at the borders.
///
/// A lane is a `(len, channels)` slice of the image — a row for the
/// horizontal pass, a column for the vertical one — so the same code
/// serves both without transposing anything.
fn conv_lane(src: ArrayView2<'_, f32>, mut dst: ArrayViewMut2<'_, f32>, weights: &[f32]) {
    let (n, c) = (src.shape()[0], src.shape()[1]);
    if n == 0 {
        return;
    }
    let radius = (weights.len() - 1) / 2;
    let last = n - 1;
    for i in 0..n {
        for ch in 0..c {
            let mut acc = 0.0_f64;
            for (k, &w) in weights.iter().enumerate() {
                let idx = (i + k).saturating_sub(radius).min(last);
                acc += f64::from(w) * f64::from(src[[idx, ch]]);
            }
            dst[[i, ch]] = acc as f32;
        }
    }
}

/// Box-filter one lane with a sliding sum, clamping at the borders.
///
/// The accumulator is `f64` so a long lane cannot drift: an f32 running
/// sum over six thousand samples loses low bits that the next pass would
/// then spread across the frame.
fn box_lane(src: ArrayView2<'_, f32>, mut dst: ArrayViewMut2<'_, f32>, radius: usize) {
    let (n, c) = (src.shape()[0], src.shape()[1]);
    if n == 0 {
        return;
    }
    let last = n - 1;
    let width = (2 * radius + 1) as f64;
    for ch in 0..c {
        let at = |i: isize| f64::from(src[[i.clamp(0, last as isize) as usize, ch]]);
        let mut acc: f64 = (-(radius as isize)..=(radius as isize)).map(at).sum();
        dst[[0, ch]] = (acc / width) as f32;
        for i in 1..n {
            acc += at(i as isize + radius as isize);
            acc -= at(i as isize - radius as isize - 1);
            dst[[i, ch]] = (acc / width) as f32;
        }
    }
}

/// Filter along rows, then along columns.
///
/// Lanes are independent, so each pass runs across them in parallel —
/// the same shape of parallelism every other kernel here uses. Nothing
/// is transposed: `axis_iter` over axis 0 yields rows and over axis 1
/// yields columns, both as `(len, channels)` views.
/// Takes a *view*, not an owned array, so the first pass can read the
/// caller's memory directly. `Zip` walks any layout, so a strided or
/// Fortran-order input costs nothing but the read; materialising it first
/// would cost a full-resolution copy — 100 MB on a 24 MP frame — and, worse,
/// would allocate the caller's *logical* shape before any bound was checked.
fn separable<F>(img: ArrayView3<'_, f32>, pass: F) -> Result<Array3<f32>, PhaiosError>
where
    F: Fn(ArrayView2<'_, f32>, ArrayViewMut2<'_, f32>) + Sync + Send,
{
    let (h, w, c) = img.dim();
    let mut mid = crate::alloc::zeros3::<f32>((h, w, c))?;
    if h == 0 || w == 0 || c == 0 {
        return Ok(mid);
    }

    ndarray::Zip::from(mid.axis_iter_mut(Axis(0)))
        .and(img.axis_iter(Axis(0)))
        .par_for_each(|d, s| pass(s, d));

    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
    ndarray::Zip::from(out.axis_iter_mut(Axis(1)))
        .and(mid.axis_iter(Axis(1)))
        .par_for_each(|d, s| pass(s, d));

    Ok(out)
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate blur parameters. Shared by the CPU and CUDA backends so both
/// reject the same inputs with the same messages.
pub(crate) fn validate(params: &BlurParams) -> Result<(), PhaiosError> {
    if !params.sigma.is_finite() || params.sigma < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "sigma is {}, expected a finite value of at least 0",
            params.sigma
        )));
    }
    if params.sigma > MAX_SIGMA {
        return Err(PhaiosError::Parameter(format!(
            "sigma is {}, above the maximum of {MAX_SIGMA}",
            params.sigma
        )));
    }
    Ok(())
}

/// Blur an image with an isotropic Gaussian of standard deviation
/// `sigma`, in pixels.
///
/// `sigma = 0.0` is the exact identity. Borders clamp, so a constant
/// image is preserved everywhere including its edges.
///
/// Below σ = [`BOX_CROSSOVER_SIGMA`] the transfer is a direct separable
/// convolution against sampled Gaussian weights — exact, with no σ
/// quantisation. At or above it, three box passes whose variances sum to
/// σ², where the realised σ is within 1% of the request and the cost no
/// longer grows with radius.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)`, C-contiguous. Channels are filtered independently.
///
/// Order-sensitive, and the direction depends on what the blur is *for*:
/// as a capture-side effect (halation) it belongs early, on linear
/// scene-referred data; as a print-side one (diffusion) it belongs after
/// the tone stages. Blurring display-referred data is a different picture
/// from blurring linear data — light adds linearly, and only the linear
/// frame gets that right.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `sigma` is negative, not finite, or
///   above [`MAX_SIGMA`].
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn blur(img: ArrayView3<f32>, params: &BlurParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, c) = img.dim();
    // Bound the caller's *logical* shape before touching a single pixel of
    // it. A zero-stride numpy view reaches gigabytes from four bytes of
    // storage, so a guard placed after the copy is not a guard: it reports
    // the right error having already committed the memory.
    crate::alloc::check_shape::<f32>((h, w, c))?;

    if params.sigma == 0.0 || h == 0 || w == 0 || c == 0 {
        // Identity: a fresh C-contiguous copy, matching every other
        // kernel's fast path.
        let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
        out.assign(&img);
        return Ok(out);
    }

    match params.shape {
        BlurShape::Gaussian => {
            if params.sigma < BOX_CROSSOVER_SIGMA {
                let weights = gaussian_weights(params.sigma);
                separable(img, |s, d| conv_lane(s, d, &weights))
            } else {
                let widths = box_widths(params.sigma);
                let mut cur: Option<Array3<f32>> = None;
                for wdt in widths {
                    let r = (wdt - 1) / 2;
                    // The first pass reads the caller's view; later ones read
                    // the previous pass's output. Matched rather than
                    // `map_or`, so the borrow of `cur` ends before it is
                    // reassigned.
                    cur = Some(match cur.as_ref() {
                        Some(prev) => separable(prev.view(), |s, d| box_lane(s, d, r))?,
                        None => separable(img, |s, d| box_lane(s, d, r))?,
                    });
                }
                Ok(cur.expect("three box passes always run at least once"))
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn blurred(img: &Array3<f32>, sigma: f32) -> Array3<f32> {
        blur(img.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap()
    }

    #[test]
    fn sigma_zero_is_the_exact_identity() {
        let img =
            Array3::<f32>::from_shape_fn((5, 7, 3), |(y, x, c)| (y * 7 + x + c) as f32 / 20.0);
        let out = blurred(&img, 0.0);
        for (a, b) in img.iter().zip(out.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn a_constant_image_survives_at_every_sigma() {
        // Borders clamp, so this must hold at the edges too — the usual
        // place a blur leaks darkness in.
        for sigma in [0.5_f32, 1.0, 2.0, 3.9, 4.0, 8.0, 20.0] {
            let img = Array3::<f32>::from_elem((9, 11, 1), 0.375);
            let out = blurred(&img, sigma);
            let worst = out.iter().map(|v| (v - 0.375).abs()).fold(0.0, f32::max);
            assert!(worst < 2e-6, "sigma {sigma}: constant drifted by {worst}");
        }
    }

    #[test]
    /// A blur redistributes light; it must not create or destroy it.
    ///
    /// Only true while the kernel fits inside the frame. Borders clamp,
    /// so an impulse close enough to an edge loses the tail that falls
    /// outside — the frame here is sized so that cannot happen, which is
    /// the honest way to test the property rather than loosening the
    /// tolerance until a leaking blur would pass too.
    fn energy_is_preserved_while_the_kernel_fits() {
        for sigma in [1.0_f32, 2.0, 5.0, 8.0] {
            let n = (4.0 * sigma).ceil() as usize * 2 + 9;
            let img = Array3::<f32>::from_shape_fn((n, n, 1), |(y, x, _)| {
                if y == n / 2 && x == n / 2 { 1.0 } else { 0.0 }
            });
            let sum: f32 = blurred(&img, sigma).iter().sum();
            assert!(
                (sum - 1.0).abs() < 2e-3,
                "sigma {sigma} in a {n}x{n} frame: energy {sum}"
            );
        }
    }

    /// The complementary fact, stated so it cannot be mistaken later for
    /// a bug: with clamped borders an impulse near an edge *does* lose
    /// light, and that is what clamping means.
    #[test]
    fn energy_leaks_at_a_border_as_clamping_implies() {
        let img =
            Array3::<f32>::from_shape_fn(
                (16, 16, 1),
                |(y, x, _)| {
                    if y == 8 && x == 8 { 1.0 } else { 0.0 }
                },
            );
        let sum: f32 = blurred(&img, 5.0).iter().sum();
        assert!(
            sum < 0.95,
            "a sigma-5 kernel overruns a 16x16 frame, so light must be lost: {sum}"
        );
    }

    #[test]
    fn the_result_is_symmetric_about_a_centred_impulse() {
        let n = 21;
        let img = Array3::<f32>::from_shape_fn((n, n, 1), |(y, x, _)| {
            if y == n / 2 && x == n / 2 { 1.0 } else { 0.0 }
        });
        for sigma in [1.5_f32, 4.0, 6.0] {
            let out = blurred(&img, sigma);
            for y in 0..n {
                for x in 0..n {
                    let mirrored = out[[n - 1 - y, n - 1 - x, 0]];
                    assert!(
                        (out[[y, x, 0]] - mirrored).abs() < 1e-6,
                        "sigma {sigma}: asymmetric at ({y}, {x})"
                    );
                }
            }
        }
    }

    #[test]
    fn box_widths_are_odd_at_least_three_and_match_sigma() {
        for i in 0..=240 {
            let sigma = BOX_CROSSOVER_SIGMA + i as f32 * 0.25;
            let w = box_widths(sigma);
            for width in w {
                assert!(
                    width >= 3 && width % 2 == 1,
                    "sigma {sigma}: bad width {width}"
                );
            }
            let realised = box_series_sigma(&w);
            let rel = (realised - sigma).abs() / sigma;
            assert!(
                rel < 0.01,
                "sigma {sigma}: realised {realised}, {rel:.3} relative error"
            );
        }
    }

    /// The widths must stay balanced. Minimising σ error alone returns
    /// `[3, 3, 17]` for σ = 5 — best σ match, nearly a single box, and
    /// visibly not a Gaussian. This pins the constraint that prevents it.
    #[test]
    fn box_widths_stay_balanced() {
        for i in 0..=240 {
            let sigma = BOX_CROSSOVER_SIGMA + i as f32 * 0.25;
            let w = box_widths(sigma);
            let (lo, hi) = (w.iter().min().unwrap(), w.iter().max().unwrap());
            assert!(
                hi <= &(lo * MAX_WIDTH_RATIO),
                "sigma {sigma}: widths {w:?} exceed the {MAX_WIDTH_RATIO}:1 ratio"
            );
        }
        assert_ne!(
            box_widths(5.0),
            [3, 3, 17],
            "the degenerate triple must be excluded"
        );
    }

    #[test]
    fn the_box_floor_is_what_the_docs_claim() {
        // Three width-3 boxes, and nothing smaller is reachable.
        assert!((box_series_sigma(&[3, 3, 3]) - std::f32::consts::SQRT_2).abs() < 1e-5);
        assert!(box_series_sigma(&[3, 3, 5]) > box_series_sigma(&[3, 3, 3]));
    }

    #[test]
    fn gaussian_weights_are_normalised_and_symmetric() {
        for sigma in [0.5_f32, 1.0, 2.0, 3.9] {
            let w = gaussian_weights(sigma);
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "sigma {sigma}: weights sum to {sum}"
            );
            for i in 0..w.len() / 2 {
                assert!((w[i] - w[w.len() - 1 - i]).abs() < 1e-9);
            }
        }
    }

    /// The realised standard deviation of a blur, measured from its own
    /// impulse response: √(Σ v·d² / Σ v) about the centre.
    ///
    /// A single-row image keeps this cheap — under clamping the vertical
    /// pass over one row is the identity — and 12σ of width keeps the
    /// finite support of three box passes (roughly ±3σ) clear of the
    /// edges, so no weight is lost to clamping.
    fn realised_sigma(sigma: f32) -> f32 {
        let n = ((12.0 * sigma).ceil() as usize) | 1;
        let mut img = Array3::<f32>::zeros((1, n, 1));
        img[[0, n / 2, 0]] = 1.0;
        let out = blur(img.view(), &BlurParams::new(sigma, BlurShape::Gaussian)).unwrap();

        let centre = (n / 2) as f64;
        let (mut m0, mut m2) = (0.0_f64, 0.0_f64);
        for x in 0..n {
            let v = f64::from(out[[0, x, 0]]);
            let d = x as f64 - centre;
            m0 += v;
            m2 += v * d * d;
        }
        (m2 / m0).sqrt() as f32
    }

    #[test]
    fn the_realised_sigma_matches_the_request_on_both_paths() {
        // `blur`'s doc comment promises the box path lands "within 1% of
        // the request". Nothing asserted it. That matters more than an
        // ordinary untested claim, because `box_widths` is host-side
        // Rust the CUDA path also calls, so both backends would realise
        // the *same* wrong σ and the cross-backend conformance suite
        // could never see it. An oracle on the output is the only thing
        // that can.
        //
        // Both paths are covered, the direct convolution included: it is
        // exact by construction, so if it ever drifted the fault would be
        // in this measurement rather than in the kernel.
        //
        // Measured at HEAD: the conv path is within 0.04%, and the box
        // path is exact except at the σ = 6 crossover, where the nearest
        // odd-width triple gives 6.0553 — 0.92%, the whole margin the 1%
        // promise has.
        for sigma in [1.0_f32, 2.0, 4.0, 5.9] {
            let ratio = realised_sigma(sigma) / sigma;
            assert!(
                (ratio - 1.0).abs() < 0.01,
                "conv path: sigma {sigma} realised as {} ({:.4}x)",
                realised_sigma(sigma),
                ratio
            );
        }
        for sigma in [BOX_CROSSOVER_SIGMA, 8.0, 12.0, 32.0, 64.0, 200.0] {
            let ratio = realised_sigma(sigma) / sigma;
            assert!(
                (ratio - 1.0).abs() < 0.01,
                "box path: sigma {sigma} realised as {} ({:.4}x)",
                realised_sigma(sigma),
                ratio
            );
        }
    }

    #[test]
    fn the_two_paths_agree_across_the_crossover() {
        // The crossover must not be a visible seam: σ just below it uses
        // the direct convolution and just above it three box passes, and
        // a caller sweeping a slider through 6.0 should see no step.
        let below = realised_sigma(BOX_CROSSOVER_SIGMA - 0.01);
        let above = realised_sigma(BOX_CROSSOVER_SIGMA);
        assert!(
            (above - below).abs() < 0.1,
            "realised sigma steps from {below} to {above} across the crossover"
        );
    }

    #[test]
    fn a_larger_sigma_blurs_further() {
        let n = 41;
        let img = Array3::<f32>::from_shape_fn((n, n, 1), |(y, x, _)| {
            if y == n / 2 && x == n / 2 { 1.0 } else { 0.0 }
        });
        let peak = |s: f32| blurred(&img, s)[[n / 2, n / 2, 0]];
        let mut prev = f32::INFINITY;
        for sigma in [0.5_f32, 1.0, 2.0, 4.0, 6.0] {
            let p = peak(sigma);
            assert!(
                p < prev,
                "sigma {sigma}: peak {p} did not fall below {prev}"
            );
            prev = p;
        }
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((8, 10, 3), |(y, x, c)| {
            ((y * 10 + x + c) % 30) as f32 / 29.0
        });
        let strided = img.slice(ndarray::s![..;2, ..;2, ..]);
        let owned = strided.to_owned();
        let p = BlurParams::new(2.0, BlurShape::Gaussian);
        assert_eq!(blur(strided, &p).unwrap(), blur(owned.view(), &p).unwrap());
    }

    #[test]
    fn empty_and_degenerate_shapes_are_accepted() {
        for shape in [(0, 5, 1), (5, 0, 1), (1, 1, 1), (1, 9, 3), (9, 1, 3)] {
            let img = Array3::<f32>::zeros(shape);
            let out = blurred(&img, 3.0);
            assert_eq!(out.dim(), shape);
        }
    }

    #[test]
    fn rejects_bad_sigma() {
        let img = array![[[0.5_f32]]];
        for s in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(blur(img.view(), &BlurParams::new(s, BlurShape::Gaussian)).is_err());
        }
    }

    /// The unpruned cubic scan `box_widths` replaced, kept verbatim as the
    /// oracle for the pruned one.
    fn exhaustive_box_widths(sigma: f32) -> [usize; 3] {
        let hi = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() as usize).max(3) + 8;
        let mut best = [3_usize; 3];
        let mut best_err = f32::INFINITY;
        let mut a = 3;
        while a <= hi {
            let mut b = a;
            while b <= hi {
                let mut c = b;
                while c <= hi {
                    if c <= a * MAX_WIDTH_RATIO {
                        let err = (box_series_sigma(&[a, b, c]) - sigma).abs();
                        if err < best_err {
                            best_err = err;
                            best = [a, b, c];
                        }
                    }
                    c += 2;
                }
                b += 2;
            }
            a += 2;
        }
        best
    }

    #[test]
    fn the_pruned_search_returns_what_an_exhaustive_one_would() {
        // The pruning is exact, not heuristic: it discards only triples that
        // cannot win. If that is ever wrong the picture changes silently, so
        // it is checked against the scan it replaced rather than argued for.
        let mut s = BOX_CROSSOVER_SIGMA;
        while s <= 48.0 {
            assert_eq!(box_widths(s), exhaustive_box_widths(s), "sigma = {s}");
            s += 0.25;
        }
        for s in [64.0, 96.5, 128.0, 200.25] {
            assert_eq!(box_widths(s), exhaustive_box_widths(s), "sigma = {s}");
        }
    }

    #[test]
    fn a_huge_sigma_is_refused_rather_than_searched_forever() {
        // `blur(sigma = 1e4)` used to run an unbounded cubic search under
        // `py.detach`: no result, no error, and no way to interrupt it from
        // Python. A caller-controlled parameter must not do that.
        let img = array![[[0.5_f32]]];
        for s in [MAX_SIGMA * 1.001, 1e4, 1e5, f32::MAX] {
            assert!(
                blur(img.view(), &BlurParams::new(s, BlurShape::Gaussian)).is_err(),
                "sigma = {s} should be refused"
            );
        }
        // The bound itself must still work, and promptly.
        assert!(blur(img.view(), &BlurParams::new(MAX_SIGMA, BlurShape::Gaussian)).is_ok());
    }

    #[test]
    fn an_oversized_logical_shape_is_refused_before_it_is_materialised() {
        // A zero-stride broadcast: four bytes of real storage behind a shape
        // implying 43 TB. The guard has to fire on the shape alone.
        //
        // The size is chosen so the test *bites*. `blur` used to open with
        // `img.to_owned()` and only meet a bound further down, which returned
        // exactly this error having already committed the caller's full
        // logical shape — so a merely-large shape passes either way on a
        // machine with the RAM to absorb it. At 43 TB the copy cannot
        // succeed: before the fix this aborts the process through
        // `handle_alloc_error`, which is the failure being prevented.
        let base = Array3::<f32>::zeros((1, 1, 1));
        let huge = base
            .broadcast((2_000_000, 2_000_000, 3))
            .expect("(1,1,1) broadcasts to anything");
        let err = blur(huge, &BlurParams::new(2.0, BlurShape::Gaussian)).unwrap_err();
        assert!(
            matches!(err, PhaiosError::Allocation(_)),
            "expected an allocation error, got {err:?}"
        );

        // The same view under the limit must still work, and still not be
        // materialised: 4 MB logical, one real pixel behind it.
        let ok = base
            .broadcast((1_000, 1_000, 1))
            .expect("(1,1,1) broadcasts to anything");
        let out = blur(ok, &BlurParams::new(2.0, BlurShape::Gaussian)).unwrap();
        assert_eq!(out.dim(), (1_000, 1_000, 1));
    }

    #[test]
    fn params_equality_and_repr() {
        let a = BlurParams::new(2.0, BlurShape::Gaussian);
        assert!(a.__eq__(&BlurParams::new(2.0, BlurShape::Gaussian)));
        assert!(!a.__eq__(&BlurParams::new(3.0, BlurShape::Gaussian)));
        assert_eq!(
            a.__repr__(),
            "BlurParams(sigma=2, shape=BlurShape.Gaussian)"
        );
        assert_eq!(BlurParams::default().sigma, 0.0);
    }
}
