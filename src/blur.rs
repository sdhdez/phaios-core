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
//! On **non-finite** input the two backends diverge outright, and
//! `docs/ffi.md` §1 records it: the device's Kahan compensation computes
//! `∞ − ∞ = NaN` and poisons the rest of the sum, so it returns NaN
//! where the host returns ±∞. Finite samples are a precondition of the
//! whole crate, and dropping the compensation to paper over this would
//! cost real accuracy on the input that is actually in scope.

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

/// Three odd widths, each at least 3 and within [`MAX_WIDTH_RATIO`] of
/// one another, whose box series best matches `sigma`.
///
/// Searched directly rather than taken from a closed form. The usual
/// two-width formula picks a worse triple than a search does (for σ = 2
/// it gives 1.826 where 2.16 is available), and the search runs once per
/// call, not once per pixel.
pub(crate) fn box_widths(sigma: f32) -> [usize; 3] {
    // A width beyond this cannot improve the match: the series already
    // overshoots σ with the smallest legal partners.
    let hi = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() as usize).max(3) + 8;
    let mut best = [3_usize; 3];
    let mut best_err = f32::INFINITY;
    let mut a = 3;
    while a <= hi {
        let mut b = a;
        while b <= hi {
            let mut c = b;
            while c <= hi {
                // `a` is the smallest and `c` the largest, since the loops
                // are non-decreasing.
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
fn separable<F>(img: &Array3<f32>, pass: F) -> Result<Array3<f32>, PhaiosError>
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
/// - [`PhaiosError::Parameter`] if `sigma` is negative or not finite.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn blur(img: ArrayView3<f32>, params: &BlurParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let owned = img.to_owned();
    let (h, w, c) = owned.dim();
    if params.sigma == 0.0 || h == 0 || w == 0 || c == 0 {
        // Identity: a fresh C-contiguous copy, matching every other
        // kernel's fast path.
        let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
        out.assign(&owned);
        return Ok(out);
    }

    match params.shape {
        BlurShape::Gaussian => {
            if params.sigma < BOX_CROSSOVER_SIGMA {
                let weights = gaussian_weights(params.sigma);
                separable(&owned, |s, d| conv_lane(s, d, &weights))
            } else {
                let widths = box_widths(params.sigma);
                let mut cur = owned;
                for wdt in widths {
                    let r = (wdt - 1) / 2;
                    cur = separable(&cur, |s, d| box_lane(s, d, r))?;
                }
                Ok(cur)
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
