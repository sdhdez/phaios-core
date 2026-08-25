// SPDX-License-Identifier: GPL-3.0-or-later
#![deny(missing_docs)]
//! Numerical kernels for black-and-white RAW image processing.
//!
//! This crate is the reusable, GUI-free, I/O-free core for the phaios
//! pipeline. All public functions are pure: they take immutable inputs
//! and return new arrays. No globals, no thread-locals, no hidden state.
//!
//! The Python extension module is named `phaios_core`; import it with
//! `import phaios_core`.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

pub mod bw;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod encode;
pub mod error;
pub mod exposure;
pub mod film_grain;
pub mod geometry;
pub mod highlight_rolloff;
mod integral;
pub mod local_contrast;
pub mod quantize;
pub mod split_toning;
pub mod tone;
pub mod vignette;

#[cfg(feature = "cuda")]
mod gpu_py;

// ── Exposure binding ─────────────────────────────────────────────────────────

/// Apply exposure compensation in EV stops.
///
/// Computes ``out = img * 2**stops``. This is the first pipeline stage:
/// it operates on linear scene-referred data, where a stop is by
/// definition a factor of two.
///
/// Values are not clamped — highlights pushed above 1.0 stay there so
/// later tone stages can recover them.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// stops : float
///     Exposure adjustment in EV. Positive brightens, negative darkens,
///     ``0.0`` is the identity.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``stops`` is not finite.
#[pyfunction]
#[pyo3(name = "exposure")]
pub fn exposure_py(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    stops: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || exposure::exposure(view, stops))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Geometry bindings ────────────────────────────────────────────────────────

/// Crop to a rectangle.
///
/// A pure index copy — no pixel value is touched, so the result is
/// bit-identical on every backend. Geometry runs first in the pipeline:
/// the vignette centres on the frame it is given (which must be the
/// cropped frame) and film grain keys its noise to pixel coordinates
/// (which must be the final grid).
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : CropParams
///     The rectangle: ``x``, ``y`` top-left corner, ``width``,
///     ``height``. Must lie entirely within the frame.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(height, width, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If the rectangle exceeds the frame.
#[pyfunction]
pub fn crop(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, geometry::CropParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = *params;
    let result = py.detach(move || geometry::crop(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Apply one of the eight dihedral orientations (Exif tag 0x0112).
///
/// A pure index permutation — bit-identical on every backend. Rotations
/// are clockwise; the transposing variants swap width and height.
/// Orientation precedes crop in the pipeline, so crop rectangles are
/// expressed in the upright frame.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// orientation : Orientation
///     ``Orientation.Normal`` … ``Orientation.Rotate270``; the enum
///     values are the Exif orientation codes 1..=8.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)`` or ``(W, H, C)``, dtype ``float32``,
///     C-contiguous.
#[pyfunction]
pub fn orient(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    orientation: geometry::Orientation,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || geometry::orient(view, orientation))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Resample to a new size with a separable polynomial filter.
///
/// ``ResizeFilter.Area`` computes exact fractional pixel coverage — the
/// correct choice for downscaling; ``ResizeFilter.CatmullRom`` (Keys
/// 1981) is the photographic default for upscaling. Bit-exact across
/// backends. A same-size resize is the exact identity.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)``, dtype ``float32``, any layout.
/// params : ResizeParams
///     Target width, height and the filter.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(height, width, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If a target dimension is zero or the input is empty.
#[pyfunction]
pub fn resize(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, geometry::ResizeParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = *params;
    let result = py.detach(move || geometry::resize(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Rotate by a small angle (±45°, positive clockwise) and crop to the
/// largest inscribed rectangle.
///
/// 16-tap Catmull-Rom resampling; the only transcendentals (sin/cos of
/// the one angle) are computed on the host, so the kernel is bit-exact
/// across backends. ``degrees=0`` is the exact identity. Compose with
/// ``orient`` for quarter turns.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)``, dtype ``float32``, any layout.
/// params : StraightenParams
///     The angle in degrees.
///
/// Returns
/// -------
/// numpy.ndarray
///     The inscribed rectangle, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If the angle is not finite, exceeds ±45°, or leaves no whole
///     pixel inscribed.
#[pyfunction]
pub fn straighten(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, geometry::StraightenParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = *params;
    let result = py.detach(move || geometry::straighten(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── B&W bindings ─────────────────────────────────────────────────────────────

/// Convert a linear RGB image to greyscale using standard luminance weights.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory
///     layout, scene-referred linear sRGB values.
/// standard : LuminanceStandard, optional
///     Which ITU-R standard to use. Default: ``LuminanceStandard.Bt709``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``, linear luminance.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 3)``.
#[pyfunction]
#[pyo3(signature = (img, standard = bw::LuminanceStandard::Bt709))]
pub fn luminance_bw(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    standard: bw::LuminanceStandard,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || bw::luminance_bw(view, standard))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Convert a linear RGB image to greyscale using arbitrary channel weights.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory layout.
/// wr, wg, wb : float
///     Per-channel weights. Range −2..+2 is conventional; negative weights
///     produce infrared-like inversions. Weights need not sum to one.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 3)``.
#[pyfunction]
pub fn channel_mixer_bw(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    wr: f32,
    wg: f32,
    wb: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || bw::channel_mixer_bw(view, [wr, wg, wb]))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Convert a linear RGB image to greyscale using a Wratten-style filter.
///
/// Applies the filter's per-channel transmission vector then collapses
/// to luminance using ``standard``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory layout.
/// filter : ColorFilter, optional
///     Wratten-style preset. Default: ``ColorFilter.NoFilter``.
/// standard : LuminanceStandard, optional
///     Which ITU-R standard to use. Default: ``LuminanceStandard.Bt709``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 3)``.
#[pyfunction]
#[pyo3(signature = (img, filter = bw::ColorFilter::NoFilter, standard = bw::LuminanceStandard::Bt709))]
pub fn color_filter_bw(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    filter: bw::ColorFilter,
    standard: bw::LuminanceStandard,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || bw::color_filter_bw(view, filter, standard))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Convert a linear RGB image to greyscale with per-hue-band weighting.
///
/// Computes a base luminance, then scales it by
/// ``1 + Σ w_i · gaussian_i(hue) · chroma`` over eight hue bands
/// (red 0°, orange 30°, yellow 60°, green 120°, aqua 180°, blue 240°,
/// purple 270°, magenta 300°). The result is clamped at zero.
///
/// The saturation measure is the chroma ratio ``(max − min) / max``,
/// which is invariant under exposure changes — see the Rust docs for
/// why HSL saturation is not used.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, any layout.
/// params : HslWeightedParams
///     Eight band weights, luminance standard, and Gaussian width.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 3)``, if ``sigma_deg`` is not
///     finite and positive, or if any weight is not finite.
#[pyfunction]
pub fn hsl_bw(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, bw::HslWeightedParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || bw::hsl_bw(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Zone System binding ───────────────────────────────────────────────────────

/// Apply the Adams/Archer Zone System tone curve.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 1)``, dtype ``float32``, linear
///     luminance (output of a B&W conversion kernel). Any memory layout
///     is accepted; the returned array is always C-contiguous.
/// params : ZoneParams
///     Zone offsets mapping zone index 0..10 → stop offset.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``, tone-adjusted luminance.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 1)``.
#[pyfunction]
pub fn zone_system(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, tone::ZoneParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || tone::zone_system(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Apply a parametric slope/offset/power tone curve.
///
/// Computes ``out = max(img * slope + offset, 0) ** power``,
/// element-wise. This is the ASC Color Decision List primary
/// correction — "gain, lift, gamma" in photographic terms.
///
/// Monotonic for any positive ``slope`` and ``power``, so it cannot
/// invert tonal order. The clamp before the exponent means a negative
/// ``offset`` crushes to black rather than producing NaN.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : ToneCurveParams
///     Slope, offset and power. The identity is ``(1.0, 0.0, 1.0)``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If any parameter is not finite, or ``power`` is not positive.
#[pyfunction]
pub fn tone_curve(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, tone::ToneCurveParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || tone::tone_curve(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Highlight roll-off binding ────────────────────────────────────────────────

/// Roll highlights off into a shoulder instead of clipping them.
///
/// Below ``knee`` nothing changes. Between ``knee`` and ``white_point``
/// the transfer follows a quadratic Bézier that leaves the identity at
/// slope 1 and reaches exactly 1.0 at ``white_point`` with slope 0, so
/// neither end produces a visible edge. At and above ``white_point`` the
/// result is 1.0.
///
/// This is the stage that decides what becomes of the highlight headroom
/// every earlier stage preserved. The default ``RolloffParams()`` is a
/// hard clip at 1.0, reproducing ``numpy.clip(img, None, 1.0)`` exactly,
/// so adding the stage changes nothing until it is asked to.
///
/// Values below zero are left alone: clamping black is a separate
/// decision.
///
/// Order-sensitive: the last stage on linear scene-referred data,
/// immediately before ``encode_srgb``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : RolloffParams
///     Knee and white point. The default ``(1.0, 1.0)`` is a hard clip.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``knee`` is outside 0..=1, or ``white_point`` is not finite or
///     is below 1.0.
// Renamed to avoid clashing with the `highlight_rolloff` module; the
// Python name is restored by the attribute (CLAUDE.md §4).
#[pyfunction]
#[pyo3(name = "highlight_rolloff")]
pub fn highlight_rolloff_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, highlight_rolloff::RolloffParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || highlight_rolloff::highlight_rolloff(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Quantisation bindings ─────────────────────────────────────────────────────

/// Quantise display-referred float data to 8-bit integer codes.
///
/// Each sample is scaled to ``0..=255``, optionally perturbed by ±1 LSB
/// of triangular dither, and rounded. Values outside ``[0, 1]`` clamp to
/// the end codes; NaN maps to 0.
///
/// Eight bits is where dither earns its keep: without it a smooth sky
/// bands visibly, because the quantisation error of a smooth gradient is
/// itself smooth and collects into contour lines. Pass
/// ``Dither.Tpdf`` with a seed unless the image already carries grain,
/// which dithers it as a side effect.
///
/// Order-sensitive: terminal, and the input must already be
/// display-referred — ``encode_srgb`` has to have run. Quantising linear
/// data throws away most of the shadow range.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout. Expected in ``[0, 1]``.
/// params : QuantizeParams
///     Dither strategy and seed. Default: no dither.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``uint8``, C-contiguous.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
pub fn quantize_u8(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, quantize::QuantizeParams>>,
) -> PyResult<Py<numpy::PyArray3<u8>>> {
    let view = img.as_array();
    let owned = params.map_or_else(quantize::QuantizeParams::default, |p| p.clone());
    let result = py.detach(move || quantize::quantize_u8(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Quantise display-referred float data to 16-bit integer codes.
///
/// As ``quantize_u8`` but to ``0..=65535``. This is the archival
/// default: 16 bits leaves enough headroom that dither is a refinement
/// rather than a necessity, and enough precision that a consumer can
/// grade the file further without tearing it.
///
/// Order-sensitive: terminal, after ``encode_srgb``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout. Expected in ``[0, 1]``.
/// params : QuantizeParams
///     Dither strategy and seed. Default: no dither.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``uint16``, C-contiguous.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
pub fn quantize_u16(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, quantize::QuantizeParams>>,
) -> PyResult<Py<numpy::PyArray3<u16>>> {
    let view = img.as_array();
    let owned = params.map_or_else(quantize::QuantizeParams::default, |p| p.clone());
    let result = py.detach(move || quantize::quantize_u16(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Local contrast binding ────────────────────────────────────────────────────

/// Enhance local contrast using the He–Sun–Tang guided filter.
///
/// Computes ``output = L + strength · (L - guided_filter(L))``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 1)``, dtype ``float32``, any memory layout.
/// params : GuidedFilterParams
///     Filter radius and epsilon regularisation term.
/// strength : float
///     Detail amplification factor (0 = no change, 1 = standard unsharp
///     mask, > 1 = over-sharpening).
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 1)``.
#[pyfunction]
#[pyo3(name = "local_contrast")]
pub fn local_contrast_py(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, local_contrast::GuidedFilterParams>,
    strength: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result =
        py.detach(move || local_contrast::local_contrast(view, &params_owned, strength))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Film grain binding ────────────────────────────────────────────────────────

/// Add procedural film grain.
///
/// Computes ``out = max(L + intensity * 4*t*(1-t) * bandpass(noise), 0)``
/// with ``t = clip(L, 0, 1)``, so the grain peaks in the midtones and
/// vanishes at both ends of the range.
///
/// Deterministic by construction: each pixel's noise is a hash of
/// ``(seed, x, y)``, not a draw from a sequential generator, so the
/// output is bit-identical on any thread count and any platform.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 1)``, dtype ``float32``, any layout.
/// params : GrainParams
///     Intensity, grain size in pixels, and the explicit seed.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 1)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 1)``, if ``intensity`` is
///     negative or non-finite, or if ``size_pixels`` is not positive.
#[pyfunction]
#[pyo3(name = "film_grain")]
pub fn film_grain_py(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, film_grain::GrainParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || film_grain::film_grain(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Split-toning binding ──────────────────────────────────────────────────────

/// Tint shadows and highlights separately, returning linear sRGB.
///
/// **Changes the shape of the data**: takes ``(H, W, 1)`` monochrome
/// luminance and returns ``(H, W, 3)`` linear sRGB. It is the only
/// kernel that adds channels, which is why ``vignette``, ``tone_curve``
/// and ``encode_srgb`` all accept any channel count.
///
/// Works in OKLab (Ottosson 2020), so the tint adds chroma without
/// moving the lightness that the tone stages established. Only the
/// ``a`` and ``b`` components of each tint are used; the ``L``
/// component is ignored.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 1)``, dtype ``float32``, any layout.
/// params : SplitToningParams
///     Shadow and highlight tints as OKLab triples, plus pivot and
///     balance.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, 3)``, dtype ``float32``, C-contiguous, linear sRGB.
///
/// Raises
/// ------
/// ValueError
///     If ``img`` is not shape ``(H, W, 1)``, if ``pivot`` is outside
///     0..=1, if ``balance`` is outside -1..=1, or if a tint component
///     is not finite.
#[pyfunction]
#[pyo3(name = "split_toning")]
pub fn split_toning_py(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, split_toning::SplitToningParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || split_toning::split_toning(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Vignette binding ──────────────────────────────────────────────────────────

/// Apply a radial vignette.
///
/// Scales each pixel by ``1 - amount * falloff(distance)``, where the
/// distance is measured in normalised frame coordinates: the centre is
/// 0 and the corners are 1. Positive ``amount`` darkens the corners,
/// negative lightens them; the result is clamped at zero.
///
/// Because the coordinates are normalised, the result is
/// resolution-independent — a preview and the full-size frame get the
/// same picture from the same parameters.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout. Every channel of a pixel gets the
///     same factor, so the vignette darkens without tinting.
/// params : VignetteParams
///     Amount, feather and roundness.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If any parameter is not finite, or if ``feather`` or
///     ``roundness`` is outside 0..=1.
#[pyfunction]
#[pyo3(name = "vignette")]
pub fn vignette_py(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, vignette::VignetteParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || vignette::vignette(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── sRGB encode binding ───────────────────────────────────────────────────────

/// Apply the IEC 61966-2-1 sRGB transfer encoding.
///
/// This is always the last kernel in the pipeline. Converts scene-referred
/// linear f32 values to display-referred sRGB. Values are not clamped —
/// caller should clamp to [0, 1] beforehand if required.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)``, dtype ``float32``. Any channel
///     count and any memory layout are accepted; the returned array is
///     always C-contiguous.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, display-referred sRGB.
#[pyfunction]
pub fn encode_srgb(py: Python<'_>, img: PyReadonlyArray3<f32>) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || encode::encode_srgb(view))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Module entry point ────────────────────────────────────────────────────────

/// The `phaios_core` Python extension module.
///
/// Exposes the numerical kernels as Python-callable functions. All
/// arrays are `numpy.float32`, C-contiguous, shape `(H, W, C)`.
#[pymodule]
fn phaios_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The version this extension was compiled from, so a consumer can
    // introspect what it imported without importlib.metadata (which
    // reports the installed *distribution*, not necessarily this binary).
    // This is the raw Cargo/pyproject string, which CI keeps identical
    // between the two files; note that a pre-release such as "0.2.0-dev"
    // is normalised by maturin to "0.2.0.dev0" (PEP 440) on the wheel, so
    // the two spellings differ before a final release and agree after.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Enum types
    m.add_class::<bw::LuminanceStandard>()?;
    m.add_class::<bw::ColorFilter>()?;

    // Param structs
    m.add_class::<geometry::CropParams>()?;
    m.add_class::<geometry::Orientation>()?;
    m.add_class::<geometry::ResizeFilter>()?;
    m.add_class::<geometry::ResizeParams>()?;
    m.add_class::<geometry::StraightenParams>()?;
    m.add_class::<bw::HslWeightedParams>()?;
    m.add_class::<tone::ZoneParams>()?;
    m.add_class::<tone::ToneCurveParams>()?;
    m.add_class::<local_contrast::GuidedFilterParams>()?;
    m.add_class::<film_grain::GrainParams>()?;
    m.add_class::<split_toning::SplitToningParams>()?;
    m.add_class::<vignette::VignetteParams>()?;
    m.add_class::<highlight_rolloff::RolloffParams>()?;
    m.add_class::<quantize::Dither>()?;
    m.add_class::<quantize::QuantizeParams>()?;

    // Geometry
    m.add_function(wrap_pyfunction!(crop, m)?)?;
    m.add_function(wrap_pyfunction!(orient, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(straighten, m)?)?;

    // Exposure
    m.add_function(wrap_pyfunction!(exposure_py, m)?)?;

    // B&W kernels
    m.add_function(wrap_pyfunction!(luminance_bw, m)?)?;
    m.add_function(wrap_pyfunction!(channel_mixer_bw, m)?)?;
    m.add_function(wrap_pyfunction!(color_filter_bw, m)?)?;
    m.add_function(wrap_pyfunction!(hsl_bw, m)?)?;

    // Tone
    m.add_function(wrap_pyfunction!(zone_system, m)?)?;
    m.add_function(wrap_pyfunction!(tone_curve, m)?)?;
    m.add_function(wrap_pyfunction!(highlight_rolloff_fn, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_u8, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_u16, m)?)?;

    // Local contrast
    m.add_function(wrap_pyfunction!(local_contrast_py, m)?)?;

    // Finishing
    m.add_function(wrap_pyfunction!(film_grain_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_toning_py, m)?)?;
    m.add_function(wrap_pyfunction!(vignette_py, m)?)?;

    // sRGB encode
    m.add_function(wrap_pyfunction!(encode_srgb, m)?)?;

    // Optional GPU backend (only when built with --features cuda).
    #[cfg(feature = "cuda")]
    gpu_py::register(m.py(), m)?;

    Ok(())
}
