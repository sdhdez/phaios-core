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
//!
//! # Pipeline order
//!
//! ```text
//! orient -> hot_pixels -> denoise -> straighten -> crop -> resize
//! -> exposure -> B&W conversion -> zone_system -> blur -> glow
//! -> local_contrast -> sharpen -> shadow_rolloff -> tone_curve
//! -> film_grain -> split_toning -> vignette -> highlight_rolloff
//! -> encode_srgb -> quantize_u8 | quantize_u16
//! ```
//!
//! The channel count is 3 from the input to the B&W stage, 1 from there
//! through [`split_toning`], and 3 again after it. Every stage from
//! [`vignette`] onward accepts any channel count, so a pipeline that
//! skips split-toning stays at 1 and still runs. `hot_pixels`,
//! `denoise`, `blur`, `glow` and `sharpen` are optional.
//! [`lut::apply_lut`] may run at any point after the B&W stage, and
//! [`histogram`] is a reduction rather than a stage.
//!
//! # Kernel index
//!
//! | Module | Kernels |
//! |---|---|
//! | [`geometry`] | `orient`, `straighten`, `crop`, `resize` |
//! | [`hot_pixels`] | `hot_pixels` |
//! | [`denoise`] | `denoise` |
//! | [`exposure`] | `exposure` |
//! | [`bw`] | `luminance_bw`, `channel_mixer_bw`, `color_filter_bw`, `hsl_bw` |
//! | [`tone`] | `zone_system`, `tone_curve` |
//! | [`blur`] | `blur` |
//! | [`glow`] | `glow` |
//! | [`local_contrast`] | `local_contrast` |
//! | [`sharpen`] | `sharpen` |
//! | [`shadow_rolloff`] | `shadow_rolloff` |
//! | [`film_grain`] | `film_grain` |
//! | [`split_toning`] | `split_toning` |
//! | [`vignette`] | `vignette` |
//! | [`highlight_rolloff`] | `highlight_rolloff` |
//! | [`encode`] | `encode_srgb` |
//! | [`lut`] | `apply_lut` |
//! | [`histogram`] | `histogram` |
//! | [`quantize`] | `quantize_u8`, `quantize_u16` |
//!
//! Each has a CUDA twin in `cuda::kernels`, behind the `cuda` feature.
//! The CPU implementation is the specification.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

mod alloc;
pub mod blur;
pub mod bw;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod denoise;
pub mod encode;
pub mod error;
pub mod exposure;
pub mod film_grain;
pub mod geometry;
pub mod glow;
pub mod highlight_rolloff;
pub mod histogram;
pub mod hot_pixels;
mod integral;
pub mod local_contrast;
pub mod lut;
pub mod quantize;
pub mod shadow_rolloff;
pub mod sharpen;
pub mod split_toning;
pub mod tone;
pub mod vignette;

#[cfg(feature = "cuda")]
mod gpu_py;

// ── Exposure binding ─────────────────────────────────────────────────────────

/// Apply exposure compensation in EV stops.
///
/// Computes ``out = img * 2**stops``. This is the first tonal stage,
/// after geometry, ``hot_pixels`` and ``denoise``. It operates on linear
/// scene-referred data, where a stop is by definition a factor of two.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
///
/// Raises
/// ------
/// MemoryError
///     If the output exceeds the single-allocation limit.
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

/// Remove RAW sensor hot pixels with a conditional (switching) median.
///
/// Computes, per channel, ``m = median9(window)`` over the 3x3
/// neighbourhood clamped at the image border, then replaces the centre
/// sample with ``m`` when ``|p - m| > threshold + relative * abs(m)``,
/// and otherwise leaves it unchanged.
///
/// Right after ``orient``, before ``straighten``/``resize``: resampling
/// mixes a single bad sample into its neighbours, smearing a one-pixel
/// defect into a blob before it can be corrected.
///
/// Bit-exact across backends: the median step is a fixed comparator
/// network of ``min``/``max`` pairs only, with no arithmetic.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : HotPixelParams
///     Absolute and relative terms of the replace-vs-keep criterion. No
///     default: no finite ``threshold`` is an identity for arbitrary
///     input.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``threshold`` or ``relative`` is negative or not finite.
/// MemoryError
///     If the output exceeds the single-allocation limit.
#[pyfunction]
#[pyo3(name = "hot_pixels")]
pub fn hot_pixels_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: pyo3::PyRef<'_, hot_pixels::HotPixelParams>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let params_owned = params.clone();
    let result = py.detach(move || hot_pixels::hot_pixels(view, &params_owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Denoise with the guided filter — self-guided per channel, or
/// cross-guided from a shared luminance guide for RGB.
///
/// Computes, per channel, ``out = p - amount * (p - q)`` where ``q`` is
/// the guided filter's smoothed base term (``eps = noise_sigma**2``).
/// RGB input (``C == 3``) uses a cross-guided filter: edges come from
/// ``standard``'s luminance of the whole image, not each channel's own
/// signal. Every other channel count, including ``C == 1``, is
/// self-guided, computed directly on that channel.
///
/// ``amount = 0.0`` is the exact identity. On a single-channel image,
/// this kernel agrees with
/// ``local_contrast(img, GuidedFilterParams(radius, noise_sigma**2), -amount)``
/// within the guided filter's own bound (rtol 1e-4, atol 1e-6), not
/// bit-for-bit: both sum the same window statistics, but by different
/// routes.
///
/// Right after ``hot_pixels``, before ``straighten``/``resize`` and
/// before ``exposure``: resampling would mix ``noise_sigma``'s
/// per-pixel physical meaning across neighbours, and exposure would
/// couple it to the caller's stop choice; an uncorrected hot pixel would
/// read as structure the edge-aware filter protects instead of removes.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : DenoiseParams
///     Radius (at most ``MAX_RADIUS``), noise sigma, blend amount, and
///     the luminance standard used for the ``C == 3`` guide. Default:
///     amount 0, the identity.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``radius`` is above ``MAX_RADIUS``, ``noise_sigma`` is
///     negative or not finite, or ``amount`` is outside 0..=1 or not
///     finite.
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
#[pyfunction]
#[pyo3(name = "denoise", signature = (img, params = None))]
pub fn denoise_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, denoise::DenoiseParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let owned = params.map_or_else(denoise::DenoiseParams::default, |p| p.clone());
    let result = py.detach(move || denoise::denoise(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Resample to a new size with a separable polynomial filter.
///
/// ``ResizeFilter.Area`` computes exact fractional pixel coverage, the
/// correct choice for downscaling. ``ResizeFilter.CatmullRom`` (Keys
/// 1981) is the photographic default for upscaling.
/// ``ResizeFilter.Bilinear`` is the cheap linear alternative. Bit-exact
/// across backends. A same-size resize is the exact identity.
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
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
///     If ``img`` is not shape ``(H, W, 1)``, if a zone index is outside
///     0..=10, or if an offset is not finite.
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
// Renamed to avoid clashing with the `highlight_rolloff` module; the
// Python name is restored by the attribute.
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
///
/// Raises
/// ------
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
///
/// Raises
/// ------
/// MemoryError
///     If the output exceeds the single-allocation limit.
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

// ── Analysis bindings: histogram and LUT ──────────────────────────────────────

/// Count how the image's samples are distributed, per channel.
///
/// Returns a ``Histogram`` with ``(channels, bins)`` counts over
/// ``[min, max]``, plus separate tallies of the samples that fell below
/// the range, above it, or were NaN. Those three are kept apart from the
/// bins on purpose: folding out-of-range samples into the end bins is
/// why so many histogram displays show a spike at the right edge that
/// cannot be told apart from legitimately bright content.
///
/// **Call this after ``encode_srgb``** if the histogram is for a person
/// to look at. A linear scene-referred histogram is correct and
/// unreadable — 18% grey sits a fifth of the way up the axis. Call it on
/// linear data only when analysing headroom.
///
/// Deterministic at any thread count and on either backend: bin counts
/// are integers, and integer addition is associative.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : HistogramParams
///     Bin count and counted range. Default: 256 bins over ``[0, 1]``.
///
/// Returns
/// -------
/// Histogram
///     With ``counts()``, ``cdf()``, ``below()``, ``above()``,
///     ``non_finite()`` and ``total(channel)``.
///
/// Raises
/// ------
/// ValueError
///     If ``bins`` is below 2 or above 4194304; if the range is not
///     finite with ``max > min`` or spans more than float32 can
///     represent; or if the channel count and ``bins`` together would
///     need an accumulator above the backend's limit.
#[pyfunction]
#[pyo3(name = "histogram", signature = (img, params = None))]
pub fn histogram_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, histogram::HistogramParams>>,
) -> PyResult<histogram::Histogram> {
    let view = img.as_array();
    let owned = params.map_or_else(histogram::HistogramParams::default, |p| p.clone());
    Ok(py.detach(move || histogram::histogram(view, &owned))?)
}

/// Apply a 1-D lookup table as a tone transfer.
///
/// The table's entries are spread evenly across ``[params.min,
/// params.max]``; a sample between two entries is linearly interpolated,
/// and one outside the domain takes the nearest end entry — clamped, not
/// extrapolated. The same table is applied to every channel.
///
/// This one kernel covers the whole curve family. Sample a UI spline
/// into a table and it is a curves tool; pass a ``Histogram.cdf()`` row
/// and it is histogram equalisation; tabulate an H&D curve and it is
/// film emulation.
///
/// The table is **not** required to be monotone — non-monotone tables
/// are how solarisation is expressed, and it is the one thing
/// ``tone_curve`` and ``zone_system`` structurally cannot do.
///
/// Note that a table is *data*, not a parameter: a consumer promising
/// exact reproduction must store the whole table in its sidecar. See
/// ``docs/export.md``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// lut : numpy.ndarray
///     1-D ``float32`` table, at least 2 entries, all finite.
/// params : LutParams
///     The input range the table spans. Default: ``[0, 1]``.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If the table has fewer than 2 entries or a non-finite value, or
///     the domain is not finite with ``max > min`` or spans more than
///     float32 can represent.
/// MemoryError
///     If the output exceeds the single-allocation limit.
#[pyfunction]
#[pyo3(signature = (img, lut, params = None))]
pub fn apply_lut(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    lut: numpy::PyReadonlyArray1<f32>,
    params: Option<pyo3::PyRef<'_, lut::LutParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let table = lut.as_array();
    let owned = params.map_or_else(lut::LutParams::default, |p| p.clone());
    let result = py.detach(move || lut::apply_lut(view, table, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Shadow roll-off binding ───────────────────────────────────────────────────

/// Roll the deepest shadows off into black instead of holding full
/// contrast down to zero.
///
/// Above ``knee`` nothing changes. Below it the curve bends away,
/// reaching the origin with slope ``1 - strength``, so shadow separation
/// is compressed and tones run together as they approach black. The join
/// at the knee is C-1 in value and slope, so it does not show as a crease
/// in a gradient.
///
/// This is the **toe** of the characteristic curve.
/// ``highlight_rolloff`` is the shoulder, and ``tone_curve`` sets the
/// slope of the straight section between them; applied in that order
/// they compose the classic three-part film tone scale. For a *measured*
/// emulsion curve rather than a parametric one, tabulate the data and
/// use ``apply_lut``.
///
/// The default ``ShadowRolloffParams()`` has ``strength=0`` and is the
/// exact identity.
///
/// Order-sensitive: apply at the start of the tone stages, on linear
/// scene-referred data, before the contrast is set. In the canonical
/// order it runs before ``tone_curve``, ``film_grain`` and
/// ``split_toning``.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : ShadowRolloffParams
///     Knee and strength, both in 0..=1. Default: no compression.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``knee`` or ``strength`` is outside 0..=1, or is not finite.
/// MemoryError
///     If the output exceeds the single-allocation limit.
#[pyfunction]
#[pyo3(name = "shadow_rolloff", signature = (img, params = None))]
pub fn shadow_rolloff_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, shadow_rolloff::ShadowRolloffParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let owned = params.map_or_else(shadow_rolloff::ShadowRolloffParams::default, |p| p.clone());
    let result = py.detach(move || shadow_rolloff::shadow_rolloff(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Blur binding ──────────────────────────────────────────────────────────────

/// Blur an image with an isotropic Gaussian of standard deviation
/// sigma, in pixels.
///
/// sigma = 0.0 is the exact identity. Borders clamp, so a constant
/// image is preserved everywhere including its edges — and an impulse
/// near an edge loses the tail that falls outside.
///
/// Below sigma 6 this is a direct separable convolution; at or
/// above it, three box passes whose variances sum to sigma-squared,
/// which costs the same at any radius.
///
/// The result is in **pixels of the image as given**, so a blur on a
/// half-size preview is not the same picture as the same sigma on the
/// full frame. Scale sigma with the image when previewing.
///
/// Order-sensitive, and which way depends on what the blur is for: as a
/// capture-side effect (halation) it belongs early on linear data; as a
/// print-side one (diffusion) after the tone stages. Light adds
/// linearly, and only the linear frame gets that right.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape (H, W, C) for any channel count, dtype
///     float32, any memory layout.
/// params : BlurParams
///     Sigma in pixels and kernel shape. Default: sigma 0, the identity.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (H, W, C), dtype float32, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If sigma is negative, not finite, or above 4096 — a blur wider
///     than any frame this crate is built for, and the point past which
///     choosing the box widths stops being cheap.
/// MemoryError
///     If the output exceeds the single-allocation limit.
#[pyfunction]
#[pyo3(name = "blur", signature = (img, params = None))]
pub fn blur_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, blur::BlurParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let owned = params.map_or_else(blur::BlurParams::default, |p| p.clone());
    let result = py.detach(move || blur::blur(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Sharpen binding ─────────────────────────────────────────────────────────

/// Sharpen with a threshold-gated Gaussian unsharp mask.
///
/// Computes ``out = img + amount * soft_gate(detail, threshold) * detail``
/// where ``detail = img - blur(img, sigma)``. ``threshold`` gates the
/// residual, in units of ``detail`` itself, so flat or near-noise-level
/// regions are not amplified.
///
/// ``amount = 0.0`` or ``sigma = 0.0`` is the exact identity. The result
/// is never clamped: the overshoot and undershoot this produces at an
/// edge is unsharp masking's own ringing, not a defect this kernel
/// suppresses.
///
/// Recommended after ``local_contrast``, before ``film_grain`` —
/// sharpening amplifies noise, so it belongs before grain is added, not
/// after.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : SharpenParams
///     Amount, blur sigma in pixels, and detail threshold. Default:
///     amount 0, the identity.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``amount`` or ``threshold`` is negative or not finite, or
///     ``sigma`` is outside the blur's domain.
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
#[pyfunction]
#[pyo3(name = "sharpen", signature = (img, params = None))]
pub fn sharpen_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, sharpen::SharpenParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let owned = params.map_or_else(sharpen::SharpenParams::default, |p| p.clone());
    let result = py.detach(move || sharpen::sharpen(view, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Glow binding ──────────────────────────────────────────────────────────────

/// Spread light above ``threshold`` and add it back — halation,
/// diffusion and veiling glare, which are one operation at three sets of
/// parameters.
///
/// Computes ``out = in + amount * blur(max(in - threshold, 0), sigma)``.
///
/// What separates the three effects is the parameters and **where in the
/// pipeline the call sits**:
///
/// - **Veiling glare** — the lens. ``threshold=0`` so all light
///   scatters, a frame-spanning ``sigma``, applied earliest. Blacks lift
///   by an amount that depends on how bright the *whole frame* is, which
///   is precisely what a per-pixel tone curve cannot do.
/// - **Halation** — the emulsion. A high ``threshold``, moderate
///   ``sigma``, applied after ``exposure`` and *before* the tone stages,
///   because it happens at capture.
/// - **Diffusion** — the print. Mid ``threshold``, large ``sigma``,
///   applied after the tone stages.
///
/// The result may exceed 1.0, deliberately: headroom is carried to
/// ``highlight_rolloff`` rather than clamped here. ``amount = 0.0`` is
/// the exact identity.
///
/// Not an unsharp mask — ``amount`` must be non-negative. For detail
/// enhancement use ``local_contrast``, which is edge-aware and does not
/// halo.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)`` for any channel count, dtype
///     ``float32``, any memory layout.
/// params : GlowParams
///     Threshold, sigma in pixels, and amount. Default: amount 0.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``threshold`` or ``amount`` is negative or not finite, or
///     ``sigma`` is outside the blur's domain.
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
#[pyfunction]
#[pyo3(name = "glow", signature = (img, params = None))]
pub fn glow_fn(
    py: Python<'_>,
    img: PyReadonlyArray3<f32>,
    params: Option<pyo3::PyRef<'_, glow::GlowParams>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let owned = params.map_or_else(glow::GlowParams::default, |p| p.clone());
    let result = py.detach(move || glow::glow(view, &owned))?;
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
///     If ``img`` is not shape ``(H, W, 1)``, if ``eps`` is negative or
///     not finite, or if ``strength`` is not finite.
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
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
/// MemoryError
///     If the intermediates exceed the single-allocation limit.
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
/// kernel that adds channels, which is why ``shadow_rolloff``,
/// ``tone_curve``, ``vignette``, ``highlight_rolloff``, ``encode_srgb``
/// and the two ``quantize`` kernels all accept any channel count.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// 0 and the frame edges are 1. Coordinates are pixel centres, so the
/// largest distance any pixel reaches is just under 1. Positive
/// ``amount`` darkens the corners, negative lightens them; the result
/// is clamped at zero.
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
/// MemoryError
///     If the output exceeds the single-allocation limit.
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
/// This is the last kernel operating on linear data, and the only one
/// producing display-referred output. Only ``quantize_u8`` or
/// ``quantize_u16`` may follow it. Values are not clamped; the caller
/// should clamp to [0, 1] beforehand if required.
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
///
/// Raises
/// ------
/// MemoryError
///     If the output exceeds the single-allocation limit.
#[pyfunction]
pub fn encode_srgb(py: Python<'_>, img: PyReadonlyArray3<f32>) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let result = py.detach(move || encode::encode_srgb(view))?;
    Ok(result.into_pyarray(py).unbind())
}

// ── Module entry point ────────────────────────────────────────────────────────

/// The `phaios_core` Python extension module.
///
/// Exposes the numerical kernels as Python-callable functions. Image
/// arrays are `numpy.float32`, C-contiguous, shape `(H, W, C)`;
/// `quantize_u8` and `quantize_u16` return `uint8` and `uint16`, and
/// `apply_lut` takes a 1-D `float32` table.
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
    m.add_class::<hot_pixels::HotPixelParams>()?;
    m.add_class::<denoise::DenoiseParams>()?;
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
    m.add_class::<blur::BlurShape>()?;
    m.add_class::<blur::BlurParams>()?;
    m.add_class::<glow::GlowParams>()?;
    m.add_class::<highlight_rolloff::RolloffParams>()?;
    m.add_class::<shadow_rolloff::ShadowRolloffParams>()?;
    m.add_class::<quantize::Dither>()?;
    m.add_class::<quantize::QuantizeParams>()?;
    m.add_class::<histogram::HistogramParams>()?;
    m.add_class::<histogram::Histogram>()?;
    m.add_class::<lut::LutParams>()?;

    // Geometry
    m.add_function(wrap_pyfunction!(crop, m)?)?;
    m.add_function(wrap_pyfunction!(orient, m)?)?;
    m.add_function(wrap_pyfunction!(hot_pixels_fn, m)?)?;
    m.add_function(wrap_pyfunction!(denoise_fn, m)?)?;
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
    m.add_function(wrap_pyfunction!(blur_fn, m)?)?;
    m.add_function(wrap_pyfunction!(glow_fn, m)?)?;
    m.add_function(wrap_pyfunction!(highlight_rolloff_fn, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_rolloff_fn, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_u8, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_u16, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_fn, m)?)?;
    m.add_function(wrap_pyfunction!(apply_lut, m)?)?;

    // Local contrast
    m.add_function(wrap_pyfunction!(local_contrast_py, m)?)?;

    // Finishing
    m.add_class::<sharpen::SharpenParams>()?;
    m.add_function(wrap_pyfunction!(sharpen_fn, m)?)?;
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
