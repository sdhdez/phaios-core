// SPDX-License-Identifier: GPL-3.0-or-later
//! Black-and-white conversion kernels.
//!
//! Four conversion methods:
//!
//! 1. **Standard luminance** — weighted sum Y = w·RGB using ITU-R
//!    BT.601, BT.709 (default), or BT.2020 luminance coefficients.
//! 2. **Channel mixer** — arbitrary user weights (wR, wG, wB) in −2..+2,
//!    allowing infrared-style effects via negative values.
//! 3. **Coloured-filter simulation** — multiply RGB by a Wratten-style
//!    per-channel transmission vector, then collapse with a chosen
//!    luminance standard.
//! 4. **Hue-weighted luminance** — scale each pixel's luminance by a
//!    weight interpolated around the hue circle from eight band
//!    centres, the digital equivalent of a continuously tunable filter
//!    set. See [`hsl_bw`].
//!
//! References:
//! - ITU-R BT.709-6, "Parameter values for the HDTV standards for
//!   production and international programme exchange" (2015), Table 1.
//! - ITU-R BT.601-7, "Studio encoding parameters of digital television
//!   for standard 4:3 and wide-screen 16:9 aspect ratios" (2011), §2.5.
//! - ITU-R BT.2020-2, "Parameter values for ultra-high definition
//!   television systems for production and international programme
//!   exchange" (2015), Table 4.
//! - Kodak Wratten Gelatin Filters datasheet, Publication B3-203 (5th
//!   ed.): spectral transmission curves for Wratten 2 series filters.

use ndarray::{Array3, ArrayView3, s};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Luminance standard ───────────────────────────────────────────────────────

/// ITU-R luminance standard for B&W conversion.
///
/// Selects which RGB primaries' Y-row coefficients to use when computing
/// the perceptual luminance of a scene-referred linear f32 image.
/// `Bt709` is the default and the correct choice for sRGB-primary data
/// (the vast majority of consumer RAW pipelines).
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum LuminanceStandard {
    /// ITU-R BT.601 weights: `(0.2990, 0.5870, 0.1140)`.
    ///
    /// Defined for SDTV primaries. Appropriate for digitised film or
    /// legacy SDTV content.
    Bt601 = 0,
    /// ITU-R BT.709 weights: `(0.2126, 0.7152, 0.0722)`.
    ///
    /// Defined for HDTV / sRGB primaries. The default for consumer
    /// digital cameras.
    #[default]
    Bt709 = 1,
    /// ITU-R BT.2020 weights: `(0.2627, 0.6780, 0.0593)`.
    ///
    /// Defined for UHDTV wide-gamut primaries. Use for 4K/8K content.
    Bt2020 = 2,
}

impl LuminanceStandard {
    /// Returns the `[wR, wG, wB]` luminance weights for this standard.
    #[must_use]
    pub fn weights(self) -> [f32; 3] {
        match self {
            Self::Bt601 => [0.2990, 0.5870, 0.1140],
            Self::Bt709 => [0.2126, 0.7152, 0.0722],
            Self::Bt2020 => [0.2627, 0.6780, 0.0593],
        }
    }
}

// ── Colour filter ────────────────────────────────────────────────────────────

/// Wratten-style coloured-filter preset for B&W contrast control.
///
/// Each variant represents a gel filter placed in front of the lens.
/// The filter attenuates light by channel, shifting the relative tonal
/// values of differently coloured subjects.
///
/// Channel transmission values are sampled at the centroid wavelengths
/// of a nominal sRGB camera's R/G/B channels (~620, ~540, ~450 nm) from
/// the Kodak Wratten Gelatin Filters datasheet, B3-203 (5th ed.).
/// These are approximations — real Wratten curves are continuous spectra.
///
/// **Note:** real filters reduce total exposure. This kernel does not
/// apply exposure compensation — do that in the exposure kernel.
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ColorFilter {
    /// No filtration — unit transmission `(1.00, 1.00, 1.00)`.
    #[default]
    NoFilter = 0,
    /// Yellow #8 K2 — `(1.00, 0.90, 0.30)`.
    /// Moderate sky darkening; the classic outdoor landscape filter.
    Yellow8K2 = 1,
    /// Orange #21 — `(1.00, 0.55, 0.10)`.
    /// Strong sky darkening; separates foliage from sky.
    Orange21 = 2,
    /// Red #25 A — `(1.00, 0.10, 0.02)`.
    /// Very dark sky; bright snow; near-infrared look.
    Red25A = 3,
    /// Green #11 X1 — `(0.20, 1.00, 0.30)`.
    /// Natural foliage rendering; darkens skin tones.
    Green11X1 = 4,
    /// Blue #47 C5 — `(0.10, 0.30, 1.00)`.
    /// Haze enhancement; inverts the effect of the red filter.
    Blue47C5 = 5,
}

impl ColorFilter {
    /// Returns the `[tR, tG, tB]` channel transmission vector.
    #[must_use]
    pub fn transmission(self) -> [f32; 3] {
        match self {
            Self::NoFilter => [1.00, 1.00, 1.00],
            Self::Yellow8K2 => [1.00, 0.90, 0.30],
            Self::Orange21 => [1.00, 0.55, 0.10],
            Self::Red25A => [1.00, 0.10, 0.02],
            Self::Green11X1 => [0.20, 1.00, 0.30],
            Self::Blue47C5 => [0.10, 0.30, 1.00],
        }
    }
}

// ── Shape validation ─────────────────────────────────────────────────────────

/// Validate `(H, W, 3)` shape. Shared by CPU and CUDA backends so both
/// reject the same inputs with the same message.
pub(crate) fn validate_rgb(img: ArrayView3<f32>) -> Result<(), PhaiosError> {
    validate_rgb_shape(img.shape())
}

/// Shape-only form of [`validate_rgb`], for callers holding a device
/// image rather than a host array.
///
/// The CUDA entry points cannot pass an `ArrayView3`, and used to repeat
/// this check by hand — three copies whose messages agreed only because
/// `{:?}` on a slice and `[{h}, {w}, {c}]` happen to render the same.
/// Both backends must reject identical inputs with identical messages
/// (`docs/ffi.md` §4); one implementation is how that is guaranteed
/// rather than merely observed.
pub(crate) fn validate_rgb_shape(shape: &[usize]) -> Result<(), PhaiosError> {
    if shape[2] != 3 {
        return Err(PhaiosError::Shape(format!(
            "expected (H, W, 3) RGB array, got shape {shape:?}"
        )));
    }
    Ok(())
}

// ── Kernels ──────────────────────────────────────────────────────────────────

/// Convert a linear RGB image to greyscale using standard luminance weights.
///
/// Computes `Y = wR·R + wG·G + wB·B` where the weights are selected by
/// `standard`. The default (`Bt709`) is correct for sRGB-primary data.
///
/// Input shape: `(H, W, 3)` — linear scene-referred f32 RGB.
/// Output shape: `(H, W, 1)` — linear luminance.
///
/// Reference: ITU-R BT.709-6 (2015), Part 2, item 3.2.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn luminance_bw(
    img: ArrayView3<f32>,
    standard: LuminanceStandard,
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    let (h, w, _) = img.dim();
    let lw = standard.weights();
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;
    ndarray::Zip::from(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .and(img.slice(s![.., .., 1]))
        .and(img.slice(s![.., .., 2]))
        .par_for_each(|y, &r, &g, &b| {
            *y = lw[0] * r + lw[1] * g + lw[2] * b;
        });
    Ok(out)
}

/// Convert a linear RGB image to greyscale using arbitrary channel weights.
///
/// Computes `Y = wR·R + wG·G + wB·B` with caller-supplied weights.
/// Weights may be negative (range −2..+2 is conventional, enabling
/// infrared-like inversions) and need not sum to one.
///
/// Input shape: `(H, W, 3)`. Output shape: `(H, W, 1)`.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn channel_mixer_bw(
    img: ArrayView3<f32>,
    weights: [f32; 3],
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    let (h, w, _) = img.dim();
    let [wr, wg, wb] = weights;
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;
    ndarray::Zip::from(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .and(img.slice(s![.., .., 1]))
        .and(img.slice(s![.., .., 2]))
        .par_for_each(|y, &r, &g, &b| {
            *y = wr * r + wg * g + wb * b;
        });
    Ok(out)
}

/// Convert a linear RGB image to greyscale using a Wratten-style filter.
///
/// Applies the filter's per-channel transmission vector to the input,
/// then collapses to luminance using `standard`. The net operation is
/// a single dot product with combined weights `(tR·wR, tG·wG, tB·wB)`.
///
/// Input shape: `(H, W, 3)`. Output shape: `(H, W, 1)`.
///
/// Reference: Kodak Wratten Gelatin Filters datasheet, B3-203 (5th ed.).
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn color_filter_bw(
    img: ArrayView3<f32>,
    filter: ColorFilter,
    standard: LuminanceStandard,
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    let (h, w, _) = img.dim();
    let t = filter.transmission();
    let lw = standard.weights();
    let cw = [t[0] * lw[0], t[1] * lw[1], t[2] * lw[2]];
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;
    ndarray::Zip::from(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .and(img.slice(s![.., .., 1]))
        .and(img.slice(s![.., .., 2]))
        .par_for_each(|y, &r, &g, &b| {
            *y = cw[0] * r + cw[1] * g + cw[2] * b;
        });
    Ok(out)
}

// ── HSL-weighted conversion ──────────────────────────────────────────────────

/// Centre of each hue band, in degrees.
///
/// The eight bands a photographer expects from a B&W mixer. They are not
/// evenly spaced. Red, orange and yellow sit 30° apart, and so do blue,
/// purple and magenta. The 60° gaps are yellow to green, green to aqua,
/// aqua to blue, and magenta back to red. The finer spacing is where
/// skin, foliage and sky separations matter most.
pub const HUE_BAND_CENTRES_DEG: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

/// Names of the eight hue bands, in the order of [`HUE_BAND_CENTRES_DEG`].
pub const HUE_BAND_NAMES: [&str; 8] = [
    "red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta",
];

/// Parameters for [`hsl_bw`].
///
/// ```python
/// # Darken blues (sky), lift yellows (foliage in autumn light)
/// params = phaios_core.HslWeightedParams([0, 0, 0.6, 0, 0, -0.7, 0, 0])
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HslWeightedParams {
    /// Per-band luminance multiplier, in the order of
    /// [`HUE_BAND_CENTRES_DEG`]: red, orange, yellow, green, aqua, blue,
    /// purple, magenta. −1..+1 is the useful range; not clamped.
    #[pyo3(get, set)]
    pub hue_weights: [f32; 8],
    /// Luminance standard used for the base greyscale value.
    #[pyo3(get, set)]
    pub standard: LuminanceStandard,
    /// Gaussian width in hue space, in degrees. Larger values blend
    /// neighbouring bands together; 30° makes adjacent bands overlap at
    /// roughly half weight.
    #[pyo3(get, set)]
    pub sigma_deg: f32,
}

#[pymethods]
impl HslWeightedParams {
    /// Create new ``HslWeightedParams``.
    #[new]
    #[pyo3(signature = (hue_weights, standard = LuminanceStandard::Bt709, sigma_deg = 30.0))]
    pub fn new(hue_weights: [f32; 8], standard: LuminanceStandard, sigma_deg: f32) -> Self {
        Self {
            hue_weights,
            standard,
            sigma_deg,
        }
    }

    /// Two ``HslWeightedParams`` are equal when all fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.standard == other.standard
            && self.sigma_deg.to_bits() == other.sigma_deg.to_bits()
            && self
                .hue_weights
                .iter()
                .zip(other.hue_weights.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "HslWeightedParams(hue_weights={:?}, standard={:?}, sigma_deg={})",
            self.hue_weights, self.standard, self.sigma_deg
        )
    }
}

impl Default for HslWeightedParams {
    fn default() -> Self {
        Self {
            hue_weights: [0.0; 8],
            standard: LuminanceStandard::Bt709,
            sigma_deg: 30.0,
        }
    }
}

/// Hue in degrees `[0, 360)` and chroma ratio `[0, 1]` of a linear RGB triple.
///
/// The hue is the usual hexagonal-projection hue shared by HSL and HSV.
///
/// The second value is `(max − min) / max` — HSV-style saturation, also
/// called the chroma ratio. HSL saturation, `(max − min) / (1 − |2L − 1|)`,
/// is deliberately *not* used here; see [`hsl_bw`] for why.
///
/// Negative components are clamped to zero first. White balance can push
/// a channel slightly below zero, and a negative `min` would otherwise
/// inflate the chroma above 1.
#[inline]
fn hue_and_chroma_ratio(r: f32, g: f32, b: f32) -> (f32, f32) {
    let r = r.max(0.0);
    let g = g.max(0.0);
    let b = b.max(0.0);

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Written as a positive test on purpose. The complement form
    // (`if delta <= 0.0 || max <= 0.0 { return }`) is equivalent for every
    // finite input, but a NaN delta — which an all-infinite pixel produces,
    // since inf - inf is NaN — fails *both* comparisons and so falls
    // through into the hue branch, poisoning the result. The CUDA kernel
    // always used the positive form, and the two backends therefore
    // disagreed on non-finite pixels (audit finding: CPU 0.0, GPU inf).
    if !(delta > 0.0 && max > 0.0) {
        // Neutral, black, or non-finite: hue is undefined, and a zero
        // chroma ratio means the caller's weights have no effect anyway.
        return (0.0, 0.0);
    }

    let hue = if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let hue = if hue < 0.0 { hue + 360.0 } else { hue };
    (hue, delta / max)
}

/// Shortest angular distance between two hues, in degrees `[0, 180]`.
///
/// Hue is circular: 350° is 10° from red, not 350° from it.
#[inline]
fn hue_distance_deg(a: f32, b: f32) -> f32 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 { 360.0 - d } else { d }
}

/// Validate HSL-weighted parameters. Shared by CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate_hsl(params: &HslWeightedParams) -> Result<(), PhaiosError> {
    if !params.sigma_deg.is_finite() || params.sigma_deg <= 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "sigma_deg is {}, expected a finite value > 0",
            params.sigma_deg
        )));
    }
    for (name, w) in HUE_BAND_NAMES.iter().zip(params.hue_weights.iter()) {
        if !w.is_finite() {
            return Err(PhaiosError::Parameter(format!(
                "hue weight for {name} is {w}, expected a finite value"
            )));
        }
    }
    Ok(())
}

/// Convert a linear RGB image to greyscale with per-hue-band weighting.
///
/// Computes a base luminance with the chosen standard, then scales it by
/// `1 + Σ_i w_i · G_i(hue) · chroma`, where `G_i` is a Gaussian centred
/// on band `i` in hue space. The result is clamped at zero: a band
/// weight of −1 takes a fully saturated pixel of that hue to black, and
/// nothing goes negative.
///
/// The band contributions are summed in a fixed array order, so the
/// result is bit-reproducible (see `docs/ffi.md` §6).
///
/// # Saturation measure
///
/// The modulation uses the **chroma ratio** `(max − min) / max`, not HSL
/// saturation `(max − min) / (1 − |2L − 1|)`, for two reasons:
///
/// 1. **Scale invariance.** The chroma ratio is unchanged by a change of
///    exposure, so adjusting exposure upstream does not silently alter
///    the B&W conversion. HSL saturation depends on lightness, so it
///    would.
/// 2. **Scene-referred data.** HSL saturation is degenerate above
///    L = 1: the denominator passes through zero and goes negative, and
///    scene-referred highlights routinely exceed 1.0.
///
/// A practical consequence: a bright saturated yellow gets a strong
/// response here, where HSL saturation would have judged it barely
/// saturated at all because of its high lightness.
///
/// Input shape: `(H, W, 3)`, any memory layout.
/// Output shape: `(H, W, 1)`, C-contiguous.
///
/// Reference: the hue/chroma geometry is the standard hexagonal
/// projection given in Joblove & Greenberg, "Color spaces for computer
/// graphics", *SIGGRAPH '78*, pp. 20–25.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
/// - [`PhaiosError::Parameter`] if `sigma_deg` is not finite and
///   positive, or any weight is not finite.
/// - [`PhaiosError::Allocation`] if the output exceeds the backend's
///   single-allocation limit.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn hsl_bw(
    img: ArrayView3<f32>,
    params: &HslWeightedParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    validate_hsl(params)?;

    let (h, w, _) = img.dim();
    let lw = params.standard.weights();
    let weights = params.hue_weights;
    let two_sigma_sq = 2.0 * params.sigma_deg * params.sigma_deg;

    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;
    ndarray::Zip::from(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .and(img.slice(s![.., .., 1]))
        .and(img.slice(s![.., .., 2]))
        .par_for_each(|y, &r, &g, &b| {
            let base = lw[0] * r + lw[1] * g + lw[2] * b;
            let (hue, chroma) = hue_and_chroma_ratio(r, g, b);

            let mut multiplier = 0.0_f32;
            for (band, &weight) in HUE_BAND_CENTRES_DEG.iter().zip(weights.iter()) {
                let d = hue_distance_deg(hue, *band);
                multiplier += weight * (-d * d / two_sigma_sq).exp();
            }

            *y = (base * (1.0 + multiplier * chroma)).max(0.0);
        });

    Ok(out)
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rgb_pixel(r: f32, g: f32, b: f32) -> Array3<f32> {
        array![[[r, g, b]]]
    }

    #[test]
    fn bt709_weights_sum_to_one() {
        let w = LuminanceStandard::Bt709.weights();
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "BT.709 weights sum = {sum}");
    }

    /// The coefficients of all three standards, retyped from the ITU-R
    /// recommendations rather than read from `weights()`.
    ///
    /// Only BT.709 was pinned anywhere before this. BT.601 and BT.2020
    /// were swept by tests that check shape and dtype alone, so any
    /// coefficient in either row could be transposed with the whole
    /// suite green — and `weights()` is host-side Rust that the CUDA
    /// path also calls (`cuda/kernels/elementwise.rs`, `hsl.rs`), so
    /// both backends move together and cross-backend comparison cannot
    /// see it either. Nothing but an independent table can.
    ///
    /// Reference: ITU-R BT.601-7 (2011) §2.5.1; BT.709-6 (2015) §3;
    /// BT.2020-2 (2015) Table 4.
    const REFERENCE_WEIGHTS: [(LuminanceStandard, [f32; 3]); 3] = [
        (LuminanceStandard::Bt601, [0.2990, 0.5870, 0.1140]),
        (LuminanceStandard::Bt709, [0.2126, 0.7152, 0.0722]),
        (LuminanceStandard::Bt2020, [0.2627, 0.6780, 0.0593]),
    ];

    #[test]
    fn each_standard_applies_its_own_coefficients() {
        for (standard, want) in REFERENCE_WEIGHTS {
            assert_eq!(standard.weights(), want, "{standard:?} table");

            // And the kernel must actually use them: a pure primary
            // converts to exactly its own coefficient.
            for (channel, w) in want.iter().enumerate() {
                let mut rgb = [0.0_f32; 3];
                rgb[channel] = 1.0;
                let img = rgb_pixel(rgb[0], rgb[1], rgb[2]);
                let got = luminance_bw(img.view(), standard).unwrap()[[0, 0, 0]];
                assert!(
                    (got - w).abs() < 1e-6,
                    "{standard:?} on pure channel {channel}: got {got}, want {w}"
                );
            }
        }
    }

    #[test]
    fn every_standard_is_neutral_preserving() {
        // Each row sums to 1, so a neutral grey survives conversion
        // unchanged. This catches a transposed digit even where the
        // individual coefficient still looks plausible.
        for (standard, _) in REFERENCE_WEIGHTS {
            let sum: f32 = standard.weights().iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "{standard:?} weights sum to {sum}, expected 1"
            );

            let grey = rgb_pixel(0.18, 0.18, 0.18);
            let got = luminance_bw(grey.view(), standard).unwrap()[[0, 0, 0]];
            assert!(
                (got - 0.18).abs() < 1e-6,
                "{standard:?} shifted 18% grey to {got}"
            );
        }
    }

    #[test]
    fn bt709_red_luminance() {
        let img = rgb_pixel(1.0, 0.0, 0.0);
        let out = luminance_bw(img.view(), LuminanceStandard::Bt709).unwrap();
        let diff = (out[[0, 0, 0]] - 0.2126_f32).abs();
        assert!(diff < 1e-6, "got {}, expected 0.2126", out[[0, 0, 0]]);
    }

    #[test]
    fn bt709_green_luminance() {
        let img = rgb_pixel(0.0, 1.0, 0.0);
        let out = luminance_bw(img.view(), LuminanceStandard::Bt709).unwrap();
        let diff = (out[[0, 0, 0]] - 0.7152_f32).abs();
        assert!(diff < 1e-6, "got {}, expected 0.7152", out[[0, 0, 0]]);
    }

    #[test]
    fn channel_mixer_red_only() {
        let img = rgb_pixel(0.7, 0.3, 0.5);
        let out = channel_mixer_bw(img.view(), [1.0, 0.0, 0.0]).unwrap();
        let diff = (out[[0, 0, 0]] - 0.7_f32).abs();
        assert!(
            diff < 1e-7,
            "expected red channel 0.7, got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn red_filter_on_pure_blue() {
        let img = rgb_pixel(0.0, 0.0, 1.0);
        let out =
            color_filter_bw(img.view(), ColorFilter::Red25A, LuminanceStandard::Bt709).unwrap();
        assert!(
            out[[0, 0, 0]] < 0.05,
            "red filter + pure blue should be nearly black, got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn shape_error_on_single_channel() {
        let img = Array3::<f32>::zeros((4, 4, 1));
        assert!(luminance_bw(img.view(), LuminanceStandard::Bt709).is_err());
    }

    // ── HSL-weighted ─────────────────────────────────────────────────────────

    #[test]
    fn hue_of_primaries_and_secondaries() {
        let cases = [
            ((1.0, 0.0, 0.0), 0.0),
            ((1.0, 1.0, 0.0), 60.0),
            ((0.0, 1.0, 0.0), 120.0),
            ((0.0, 1.0, 1.0), 180.0),
            ((0.0, 0.0, 1.0), 240.0),
            ((1.0, 0.0, 1.0), 300.0),
        ];
        for ((r, g, b), expected) in cases {
            let (hue, chroma) = hue_and_chroma_ratio(r, g, b);
            assert!(
                (hue - expected).abs() < 1e-3,
                "rgb({r}, {g}, {b}): expected hue {expected}, got {hue}"
            );
            assert!((chroma - 1.0).abs() < 1e-6, "expected full chroma");
        }
    }

    #[test]
    fn neutral_has_zero_chroma() {
        for v in [0.0_f32, 0.18, 1.0, 4.0] {
            let (_, chroma) = hue_and_chroma_ratio(v, v, v);
            assert_eq!(chroma, 0.0, "neutral {v} should have zero chroma");
        }
    }

    #[test]
    fn chroma_ratio_is_exposure_invariant() {
        // The reason for preferring the chroma ratio over HSL saturation:
        // brightening the scene must not change which pixels the hue
        // weights act on.
        let (h1, c1) = hue_and_chroma_ratio(0.4, 0.2, 0.1);
        let (h2, c2) = hue_and_chroma_ratio(1.6, 0.8, 0.4); // +2 EV
        assert!((h1 - h2).abs() < 1e-4, "hue moved: {h1} vs {h2}");
        assert!((c1 - c2).abs() < 1e-6, "chroma moved: {c1} vs {c2}");
    }

    #[test]
    fn hue_distance_wraps_around_the_circle() {
        assert!((hue_distance_deg(350.0, 0.0) - 10.0).abs() < 1e-4);
        assert!((hue_distance_deg(10.0, 350.0) - 20.0).abs() < 1e-4);
        assert!((hue_distance_deg(0.0, 180.0) - 180.0).abs() < 1e-4);
        assert!((hue_distance_deg(270.0, 30.0) - 120.0).abs() < 1e-4);
    }

    #[test]
    fn zero_weights_match_plain_luminance() {
        let img = Array3::<f32>::from_shape_fn((8, 8, 3), |(y, x, c)| {
            (y * 24 + x * 3 + c) as f32 / 192.0
        });
        let hsl = hsl_bw(img.view(), &HslWeightedParams::default()).unwrap();
        let plain = luminance_bw(img.view(), LuminanceStandard::Bt709).unwrap();
        for (&a, &b) in hsl.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn neutral_pixels_ignore_hue_weights() {
        // Zero chroma means zero modulation, whatever the weights are.
        let img = rgb_pixel(0.5, 0.5, 0.5);
        let params = HslWeightedParams::new([1.0; 8], LuminanceStandard::Bt709, 30.0);
        let out = hsl_bw(img.view(), &params).unwrap();
        assert!(
            (out[[0, 0, 0]] - 0.5).abs() < 1e-6,
            "got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn red_weight_doubles_a_saturated_red() {
        // Pure red: hue 0°, chroma 1, Gaussian peak 1 on the red band.
        // multiplier = +1 → output = base · (1 + 1·1) = 2·base.
        let img = rgb_pixel(1.0, 0.0, 0.0);
        let mut weights = [0.0_f32; 8];
        weights[0] = 1.0;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        let out = hsl_bw(img.view(), &params).unwrap();
        let expected = 2.0 * 0.2126;
        assert!(
            (out[[0, 0, 0]] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn output_is_clamped_at_zero() {
        // A weight below −1 would drive the multiplier negative enough to
        // flip the sign; luminance must not go negative.
        let img = rgb_pixel(1.0, 0.0, 0.0);
        let mut weights = [0.0_f32; 8];
        weights[0] = -3.0;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        let out = hsl_bw(img.view(), &params).unwrap();
        assert_eq!(out[[0, 0, 0]], 0.0);
    }

    #[test]
    fn band_influence_wraps_across_zero() {
        // Hue 350° is 10° from the red band centre, so a red weight must
        // reach it almost as strongly as it reaches hue 10°.
        let mut weights = [0.0_f32; 8];
        weights[0] = 1.0;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);

        // rgb(1, 0, 1/6) ≈ hue 350°; rgb(1, 1/6, 0) ≈ hue 10°.
        let below = hsl_bw(rgb_pixel(1.0, 0.0, 1.0 / 6.0).view(), &params).unwrap();
        let above = hsl_bw(rgb_pixel(1.0, 1.0 / 6.0, 0.0).view(), &params).unwrap();

        let ratio_below = below[[0, 0, 0]] / (0.2126 + 0.0722 / 6.0);
        let ratio_above = above[[0, 0, 0]] / (0.2126 + 0.7152 / 6.0);
        assert!(
            (ratio_below - ratio_above).abs() < 0.02,
            "wrap-around asymmetry: {ratio_below} vs {ratio_above}"
        );
        assert!(ratio_below > 1.8, "10° from centre should stay strong");
    }

    #[test]
    fn sigma_controls_band_overlap() {
        // A green pixel under a yellow-only weight: a wide sigma lets the
        // yellow band reach 120°, a narrow one does not.
        let mut weights = [0.0_f32; 8];
        weights[2] = 1.0; // yellow, centred 60°
        let green = rgb_pixel(0.0, 1.0, 0.0);

        let narrow = hsl_bw(
            green.view(),
            &HslWeightedParams::new(weights, LuminanceStandard::Bt709, 10.0),
        )
        .unwrap();
        let wide = hsl_bw(
            green.view(),
            &HslWeightedParams::new(weights, LuminanceStandard::Bt709, 60.0),
        )
        .unwrap();

        assert!((narrow[[0, 0, 0]] - 0.7152).abs() < 1e-4, "narrow leaked");
        assert!(wide[[0, 0, 0]] > 0.85, "wide sigma should reach green");
    }

    #[test]
    fn rejects_invalid_parameters() {
        let img = rgb_pixel(0.5, 0.2, 0.1);
        for bad_sigma in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let params = HslWeightedParams::new([0.0; 8], LuminanceStandard::Bt709, bad_sigma);
            assert!(matches!(
                hsl_bw(img.view(), &params).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        let mut weights = [0.0_f32; 8];
        weights[3] = f32::NAN;
        let params = HslWeightedParams::new(weights, LuminanceStandard::Bt709, 30.0);
        assert!(matches!(
            hsl_bw(img.view(), &params).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
    }

    #[test]
    fn accepts_non_contiguous_input() {
        let img = Array3::<f32>::from_shape_fn((6, 4, 3), |(y, x, c)| {
            ((y * 12 + x * 3 + c) % 7) as f32 / 7.0
        });
        let strided = img.slice(s![..;2, .., ..]);
        let params = HslWeightedParams::new([0.3; 8], LuminanceStandard::Bt709, 30.0);
        let out = hsl_bw(strided, &params).unwrap();
        assert_eq!(out.dim(), (3, 4, 1));
        assert_eq!(out, hsl_bw(strided.to_owned().view(), &params).unwrap());
    }
}
