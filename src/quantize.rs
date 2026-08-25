// SPDX-License-Identifier: GPL-3.0-or-later
//! Quantisation to integer codes, with optional dither.
//!
//! The terminal stage. Everything upstream is continuous `f32`; a file
//! holds integers, and something has to choose them. Doing it with a
//! bare `round` is what produces **banding**: the quantisation error of
//! a smooth gradient is itself smooth, so it collects into visible
//! contour lines rather than dispersing. Grey gradients are exactly
//! where the eye is best at spotting those steps, which makes this a
//! black-and-white problem more than a colour one.
//!
//! The fix is older than digital imaging: perturb each sample by
//! sub-LSB noise *before* rounding, so the error becomes noise instead
//! of structure. With a triangular probability density of ±1 LSB the
//! quantisation error is rendered independent of the signal and its
//! variance made constant — the standard result for subtractive-free
//! dither (Lipshitz, Wannamaker and Vanderkooy, "Quantization and
//! Dither: A Theoretical Survey", *Journal of the Audio Engineering
//! Society* 40(5), 1992, pp. 355–375; the argument is signal-theoretic
//! and transfers to images unchanged).
//!
//! Trading a contour for a little noise is a good trade at 8 bits and
//! an irrelevant one at 16. It is also nearly free if grain is already
//! present: `film_grain` at any visible intensity dithers the image as
//! a side effect, and `Dither::Off` is then the right setting.
//!
//! ## Determinism
//!
//! The dither is not random. It is `splitmix64` keyed on the pixel's
//! coordinates and the caller's `seed` — the same hash `film_grain`
//! uses — so it depends on position rather than on evaluation order.
//! Same seed, same bytes, at any thread count and on either backend.
//! There is no global RNG here any more than anywhere else in the crate.
//!
//! ## Input range
//!
//! Input is expected display-referred, i.e. the output of
//! [`crate::encode::encode_srgb`], nominally in `[0, 1]`. Values outside
//! that range are clamped to the representable code range rather than
//! wrapping, and NaN maps to 0.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;
use crate::film_grain::pixel_hash;

// ── Parameter types ───────────────────────────────────────────────────────────

/// Dither strategy for [`quantize_u8`] / [`quantize_u16`].
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dither {
    /// No dither: round to nearest. Banding is possible in smooth
    /// gradients, which at 16 bits is academic and at 8 bits is the most
    /// common visible defect in black-and-white output.
    ///
    /// The default, so quantisation does nothing surprising unasked.
    ///
    /// Named `Off` rather than `None` because the Python surface has to
    /// spell it: `Dither.None` is a syntax error in Python, `None` being
    /// a keyword.
    #[default]
    Off,
    /// Triangular PDF dither of ±1 LSB, keyed on pixel position.
    ///
    /// Replaces banding with fine noise of constant variance. Use it for
    /// 8-bit export, and for 16-bit only if the image is destined for
    /// heavy further grading.
    Tpdf,
}

/// Parameters for [`quantize_u8`] and [`quantize_u16`].
///
/// ```python
/// # 16-bit archival export, no dither needed
/// params = phaios_core.QuantizeParams()
///
/// # 8-bit web export, dithered
/// params = phaios_core.QuantizeParams(phaios_core.Dither.Tpdf, seed=20260825)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct QuantizeParams {
    /// Dither strategy. Default [`Dither::Off`].
    #[pyo3(get, set)]
    pub dither: Dither,
    /// Seed for the dither pattern. Ignored when `dither` is
    /// [`Dither::Off`].
    ///
    /// Explicit, like every other source of randomness in the crate: the
    /// same seed reproduces the same file, and the seed belongs in the
    /// settings that travel with the image.
    #[pyo3(get, set)]
    pub seed: u64,
}

#[pymethods]
impl QuantizeParams {
    /// Create new ``QuantizeParams``.
    #[new]
    #[pyo3(signature = (dither = Dither::Off, seed = 0))]
    pub fn new(dither: Dither, seed: u64) -> Self {
        Self { dither, seed }
    }

    /// Two ``QuantizeParams`` are equal when both fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.dither == other.dither && self.seed == other.seed
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "QuantizeParams(dither=Dither.{:?}, seed={})",
            self.dither, self.seed
        )
    }
}

// ── Dither ────────────────────────────────────────────────────────────────────

/// Triangular deviate on `[-1, 1]` from 64 hashed bits.
///
/// Two independent uniforms drawn from disjoint 24-bit fields — 24 bits
/// being the full `f32` mantissa, so nothing is discarded — and summed
/// after centring. The sum of two independent uniforms is triangular,
/// which is the distribution that makes the quantisation error's
/// variance independent of the signal.
///
/// Shared in structure with `src/cuda/ptx/quantize.cu`; both use only
/// exact integer arithmetic and correctly-rounded f32 operations, so
/// they agree bit for bit.
#[inline]
pub(crate) fn triangular_dither(bits: u64) -> f32 {
    const SCALE: f32 = 1.0 / 16_777_216.0; // 2^-24
    let u1 = ((bits >> 40) as f32) * SCALE;
    let u2 = (((bits >> 16) & 0x00FF_FFFF) as f32) * SCALE;
    (u1 - 0.5) + (u2 - 0.5)
}

/// Quantise one sample to an integer code in `0..=max_code`.
///
/// The rounding is written `floor(v + 0.5)` rather than `round(v)` on
/// purpose: `floor` and the addition are each exact or correctly
/// rounded, so the composition is reproducible without depending on a
/// library rounding routine.
#[inline]
// Negated comparisons are deliberate: NaN fails every comparison, so
// `!(v > 0.0)` sends it to code 0 on both backends. The positive form
// would leave NaN to an undefined float-to-integer cast, which is
// exactly the class of divergence the v0.2 audit found in `hsl_bw`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn quantize_sample(value: f32, dither: f32, max_code: f32) -> f32 {
    let v = value * max_code + dither;
    if !(v > 0.0) {
        // Negative, zero, or NaN.
        return 0.0;
    }
    if v >= max_code {
        return max_code;
    }
    let rounded = (v + 0.5).floor();
    if rounded > max_code {
        max_code
    } else {
        rounded
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// The generic body. `max_code` is `2^bits − 1` as an `f32`; both 255
/// and 65535 are exactly representable, and so is every code below them.
fn quantize_into<T, F>(
    img: ArrayView3<f32>,
    params: &QuantizeParams,
    max_code: f32,
    convert: F,
) -> Array3<T>
where
    T: Clone + num_traits_zero::Zero + Send + Sync,
    F: Fn(f32) -> T + Sync + Send,
{
    let (h, w, c) = img.dim();
    let mut out = Array3::<T>::from_elem((h, w, c), T::zero());
    let dithered = params.dither == Dither::Tpdf;
    let seed = params.seed;

    ndarray::Zip::indexed(&mut out)
        .and(img)
        .par_for_each(|(y, x, ch), o, &v| {
            let d = if dithered {
                // Keyed on the channel as well as the position, so the
                // three channels of a toned image are not given the same
                // perturbation (which would tint the noise).
                triangular_dither(pixel_hash(
                    seed ^ (ch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    x as u64,
                    y as u64,
                ))
            } else {
                0.0
            };
            *o = convert(quantize_sample(v, d, max_code));
        });

    out
}

/// Minimal local stand-in for `num_traits::Zero`, to avoid a dependency
/// for two impls.
mod num_traits_zero {
    /// Types that have an additive identity.
    pub trait Zero {
        /// The additive identity.
        fn zero() -> Self;
    }
    impl Zero for u8 {
        fn zero() -> Self {
            0
        }
    }
    impl Zero for u16 {
        fn zero() -> Self {
            0
        }
    }
}

/// Quantise display-referred `f32` to 8-bit codes.
///
/// Each sample is scaled to `0..=255`, optionally perturbed by ±1 LSB of
/// triangular dither, and rounded. Values outside `[0, 1]` clamp to the
/// end codes; NaN maps to 0.
///
/// Eight bits is where dither earns its keep: without it a smooth sky
/// bands visibly. Pass [`Dither::Tpdf`] with a seed unless the image
/// already carries grain.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)` `u8`, C-contiguous.
///
/// Order-sensitive: terminal. The input must already be display-referred
/// — that is, [`crate::encode::encode_srgb`] has run. Quantising linear
/// data throws away most of the shadow range, since linear code 1 of 255
/// is already well above black.
///
/// # Errors
/// Infallible for every 3-D input; the `Result` is for signature
/// consistency with the other kernels and for future parameters.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn quantize_u8(
    img: ArrayView3<f32>,
    params: &QuantizeParams,
) -> Result<Array3<u8>, PhaiosError> {
    Ok(quantize_into(img, params, 255.0, |v| v as u8))
}

/// Quantise display-referred `f32` to 16-bit codes.
///
/// As [`quantize_u8`] but to `0..=65535`. This is the archival default:
/// 16 bits leaves enough headroom that dither is a refinement rather
/// than a necessity, and enough precision that a consumer can grade the
/// file further without tearing it.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. Output:
/// `(H, W, C)` `u16`, C-contiguous.
///
/// Order-sensitive: terminal, after [`crate::encode::encode_srgb`].
///
/// # Errors
/// Infallible for every 3-D input; the `Result` is for signature
/// consistency with the other kernels.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn quantize_u16(
    img: ArrayView3<f32>,
    params: &QuantizeParams,
) -> Result<Array3<u16>, PhaiosError> {
    Ok(quantize_into(img, params, 65535.0, |v| v as u16))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn u8_of(v: f32, params: &QuantizeParams) -> u8 {
        quantize_u8(array![[[v]]].view(), params).unwrap()[[0, 0, 0]]
    }

    #[test]
    fn endpoints_and_midpoint_land_where_they_should() {
        let p = QuantizeParams::default();
        assert_eq!(u8_of(0.0, &p), 0);
        assert_eq!(u8_of(1.0, &p), 255);
        assert_eq!(u8_of(0.5, &p), 128); // 127.5 rounds up
        let img = array![[[0.0_f32, 0.5, 1.0]]];
        let out = quantize_u16(img.view(), &p).unwrap();
        assert_eq!(out[[0, 0, 0]], 0);
        assert_eq!(out[[0, 0, 2]], 65535);
    }

    #[test]
    fn out_of_range_clamps_and_nan_is_zero() {
        let p = QuantizeParams::default();
        assert_eq!(u8_of(-0.5, &p), 0);
        assert_eq!(u8_of(-1e30, &p), 0);
        assert_eq!(u8_of(1.5, &p), 255);
        assert_eq!(u8_of(1e30, &p), 255);
        assert_eq!(u8_of(f32::INFINITY, &p), 255);
        assert_eq!(u8_of(f32::NEG_INFINITY, &p), 0);
        assert_eq!(u8_of(f32::NAN, &p), 0);
    }

    #[test]
    fn undithered_is_plain_rounding() {
        let p = QuantizeParams::default();
        for i in 0..=255_u32 {
            let v = i as f32 / 255.0;
            assert_eq!(u8_of(v, &p), i as u8, "code {i} must round-trip");
        }
    }

    #[test]
    fn dither_is_deterministic_from_the_seed() {
        let img =
            Array3::<f32>::from_shape_fn((16, 16, 1), |(y, x, _)| (y * 16 + x) as f32 / 256.0);
        let p = QuantizeParams::new(Dither::Tpdf, 12345);
        let a = quantize_u8(img.view(), &p).unwrap();
        let b = quantize_u8(img.view(), &p).unwrap();
        assert_eq!(a, b, "same seed must give the same bytes");

        let c = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 12346)).unwrap();
        assert_ne!(a, c, "a different seed must give a different pattern");
    }

    #[test]
    fn dither_moves_codes_by_at_most_one() {
        // The whole promise: it perturbs, it does not distort.
        let img = Array3::<f32>::from_shape_fn((32, 32, 1), |(y, x, _)| {
            ((y * 32 + x) % 256) as f32 / 255.0
        });
        let plain = quantize_u8(img.view(), &QuantizeParams::default()).unwrap();
        let dithered = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 7)).unwrap();
        for (p, d) in plain.iter().zip(dithered.iter()) {
            let delta = i32::from(*d) - i32::from(*p);
            assert!(
                delta.abs() <= 1,
                "dither must stay within one code: {p} -> {d}"
            );
        }
    }

    /// The reason the kernel exists: on a ramp too shallow to resolve at
    /// 8 bits, plain rounding produces flat plateaux (bands) while dither
    /// produces a mixture that carries the gradient.
    #[test]
    fn dither_breaks_up_banding() {
        // A ramp covering only two 8-bit codes across 256 pixels.
        let img = Array3::<f32>::from_shape_fn((1, 256, 1), |(_, x, _)| {
            (100.0 + x as f32 / 255.0) / 255.0
        });
        let plain = quantize_u8(img.view(), &QuantizeParams::default()).unwrap();
        let dithered = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 99)).unwrap();

        let plain_transitions = plain
            .iter()
            .zip(plain.iter().skip(1))
            .filter(|(a, b)| a != b)
            .count();
        let dithered_transitions = dithered
            .iter()
            .zip(dithered.iter().skip(1))
            .filter(|(a, b)| a != b)
            .count();

        assert!(
            plain_transitions <= 1,
            "plain rounding should give one hard step, got {plain_transitions}"
        );
        assert!(
            dithered_transitions > 20,
            "dither should disperse the step into many, got {dithered_transitions}"
        );
    }

    #[test]
    fn dither_preserves_the_mean() {
        // Triangular dither is zero-mean, so it must not shift exposure.
        let img = Array3::<f32>::from_elem((64, 64, 1), 0.5019608); // exactly 128/255
        let plain = quantize_u8(img.view(), &QuantizeParams::default()).unwrap();
        let dithered = quantize_u8(img.view(), &QuantizeParams::new(Dither::Tpdf, 4)).unwrap();
        let mean = |a: &Array3<u8>| a.iter().map(|v| f64::from(*v)).sum::<f64>() / a.len() as f64;
        assert!(
            (mean(&dithered) - mean(&plain)).abs() < 0.05,
            "dither must be zero-mean: {} vs {}",
            mean(&dithered),
            mean(&plain)
        );
    }

    #[test]
    fn dither_none_ignores_the_seed() {
        let img = Array3::<f32>::from_shape_fn((8, 8, 1), |(y, x, _)| (y + x) as f32 / 16.0);
        let a = quantize_u8(img.view(), &QuantizeParams::new(Dither::Off, 1)).unwrap();
        let b = quantize_u8(img.view(), &QuantizeParams::new(Dither::Off, 999)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((6, 9, 3), |(y, x, c)| {
            ((y * 9 + x) * 3 + c) as f32 / 200.0
        });
        let strided = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned = strided.to_owned();
        let p = QuantizeParams::new(Dither::Tpdf, 5);
        let a = quantize_u8(strided, &p).unwrap();
        let b = quantize_u8(owned.view(), &p).unwrap();
        assert_eq!(a, b, "layout must not change the result");
        assert!(a.is_standard_layout());
    }

    #[test]
    fn empty_input_is_accepted() {
        let img = Array3::<f32>::zeros((0, 4, 3));
        assert_eq!(
            quantize_u8(img.view(), &QuantizeParams::default())
                .unwrap()
                .dim(),
            (0, 4, 3)
        );
        assert_eq!(
            quantize_u16(img.view(), &QuantizeParams::default())
                .unwrap()
                .dim(),
            (0, 4, 3)
        );
    }

    #[test]
    fn params_equality_and_repr() {
        let a = QuantizeParams::new(Dither::Tpdf, 7);
        assert!(a.__eq__(&QuantizeParams::new(Dither::Tpdf, 7)));
        assert!(!a.__eq__(&QuantizeParams::new(Dither::Off, 7)));
        assert!(!a.__eq__(&QuantizeParams::new(Dither::Tpdf, 8)));
        assert_eq!(a.__repr__(), "QuantizeParams(dither=Dither.Tpdf, seed=7)");
        assert_eq!(QuantizeParams::default().dither, Dither::Off);
    }

    /// Bounded, zero-mean, and *triangular* — the distribution is the
    /// claim, not a detail. Variance discriminates it: a triangular
    /// deviate on [-1, 1] (the sum of two uniforms) has variance 1/6,
    /// where a uniform one on the same interval would have 1/3. Without
    /// this, swapping TPDF for RPDF passes every CPU test, and only the
    /// GPU conformance suite notices — which it would stop doing the
    /// moment someone "fixed" the .cu to match.
    #[test]
    fn dither_is_bounded_centred_and_triangular() {
        let n = 400_000;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for i in 0..n {
            let d = f64::from(triangular_dither(pixel_hash(42, i as u64, 0)));
            assert!((-1.0..=1.0).contains(&d), "dither out of range: {d}");
            sum += d;
            sum_sq += d * d;
        }
        let mean = sum / f64::from(n);
        let variance = sum_sq / f64::from(n) - mean * mean;

        assert!(mean.abs() < 0.005, "dither mean should be ~0, got {mean}");
        assert!(
            (variance - 1.0 / 6.0).abs() < 0.01,
            "dither variance should be 1/6 = 0.1667 (triangular); got {variance:.4}. \
             0.333 would mean a uniform deviate, which does not decorrelate \
             the quantisation error the way TPDF does."
        );
    }
}
