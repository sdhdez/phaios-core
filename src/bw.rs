// SPDX-License-Identifier: GPL-3.0-or-later
//! Black-and-white conversion kernels.
//!
//! Three conversion methods for v0.1:
//!
//! 1. **Standard luminance** — weighted sum Y = w·RGB using ITU-R
//!    BT.601, BT.709 (default), or BT.2020 luminance coefficients.
//! 2. **Channel mixer** — arbitrary user weights (wR, wG, wB) in −2..+2,
//!    allowing infrared-style effects via negative values.
//! 3. **Coloured-filter simulation** — multiply RGB by a Wratten-style
//!    per-channel transmission vector, then collapse with a chosen
//!    luminance standard.
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
use pyo3::pyclass;

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

fn validate_rgb(img: ArrayView3<f32>) -> Result<(), PhaiosError> {
    if img.shape()[2] != 3 {
        return Err(PhaiosError::Shape(format!(
            "expected (H, W, 3) RGB array, got shape {:?}",
            img.shape()
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
/// Reference: ITU-R BT.709-6 (2015), Table 1.
///
/// # Errors
/// Returns [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn luminance_bw(
    img: ArrayView3<f32>,
    standard: LuminanceStandard,
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    let (h, w, _) = img.dim();
    let lw = standard.weights();
    let mut out = Array3::<f32>::zeros((h, w, 1));
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
/// Returns [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn channel_mixer_bw(
    img: ArrayView3<f32>,
    weights: [f32; 3],
) -> Result<Array3<f32>, PhaiosError> {
    validate_rgb(img)?;
    let (h, w, _) = img.dim();
    let [wr, wg, wb] = weights;
    let mut out = Array3::<f32>::zeros((h, w, 1));
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
/// Returns [`PhaiosError::Shape`] if the input is not `(H, W, 3)`.
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
    let mut out = Array3::<f32>::zeros((h, w, 1));
    ndarray::Zip::from(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .and(img.slice(s![.., .., 1]))
        .and(img.slice(s![.., .., 2]))
        .par_for_each(|y, &r, &g, &b| {
            *y = cw[0] * r + cw[1] * g + cw[2] * b;
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
}
