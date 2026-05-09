// SPDX-License-Identifier: GPL-3.0-or-later
//! Error types for phaios-core.
//!
//! `PhaiosError` is the crate's single error type. All public functions
//! return `Result<_, PhaiosError>`. At the FFI boundary, `PhaiosError`
//! converts to `pyo3::PyErr` via the `From` impl below.

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use thiserror::Error;

/// Errors that can occur in phaios-core kernels.
#[derive(Debug, Error)]
pub enum PhaiosError {
    /// Input array has an unexpected shape.
    ///
    /// Carries a human-readable description of what was expected and
    /// what was received.
    #[error("shape error: {0}")]
    Shape(String),
}

impl From<PhaiosError> for PyErr {
    fn from(e: PhaiosError) -> PyErr {
        PyValueError::new_err(e.to_string())
    }
}
