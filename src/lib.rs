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

use pyo3::prelude::*;

mod bw;
mod encode;
mod error;
mod exposure;
mod local_contrast;
mod tone;

/// The `phaios_core` Python extension module.
///
/// Exposes the numerical kernels as Python-callable functions. All
/// arrays are `numpy.float32`, C-contiguous, shape `(H, W, C)`.
#[pymodule]
fn phaios_core(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
