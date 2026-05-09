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
pub mod encode;
pub mod error;
pub mod exposure;
pub mod local_contrast;
pub mod tone;

// ── B&W bindings ─────────────────────────────────────────────────────────────

/// Convert a linear RGB image to greyscale using standard luminance weights.
///
/// Parameters
/// ----------
/// img : numpy.ndarray
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, C-contiguous,
///     scene-referred linear sRGB values.
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
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, C-contiguous.
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
///     Input array, shape ``(H, W, 3)``, dtype ``float32``, C-contiguous.
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

// ── Module entry point ────────────────────────────────────────────────────────

/// The `phaios_core` Python extension module.
///
/// Exposes the numerical kernels as Python-callable functions. All
/// arrays are `numpy.float32`, C-contiguous, shape `(H, W, C)`.
#[pymodule]
fn phaios_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Enum types
    m.add_class::<bw::LuminanceStandard>()?;
    m.add_class::<bw::ColorFilter>()?;

    // B&W kernels
    m.add_function(wrap_pyfunction!(luminance_bw, m)?)?;
    m.add_function(wrap_pyfunction!(channel_mixer_bw, m)?)?;
    m.add_function(wrap_pyfunction!(color_filter_bw, m)?)?;

    Ok(())
}
