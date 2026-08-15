// SPDX-License-Identifier: GPL-3.0-or-later
//! Error types for phaios-core.
//!
//! `PhaiosError` is the crate's single error type. All public functions
//! return `Result<_, PhaiosError>`. At the FFI boundary, `PhaiosError`
//! converts to `pyo3::PyErr` via the `From` impl below.

use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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

    /// A kernel parameter is outside its documented domain.
    ///
    /// Raised for values that have no meaningful interpretation — a zone
    /// index outside 0..=10, a negative regularisation term, a non-finite
    /// float — rather than for values that are merely extreme. Kernels do
    /// not silently clamp; see `docs/architecture.md`.
    #[error("parameter error: {0}")]
    Parameter(String),

    /// A compute backend failed or is unavailable.
    ///
    /// Raised when no CUDA device exists, the driver cannot be loaded,
    /// a device is below the supported compute capability, or a
    /// device-side operation fails. Maps to Python `RuntimeError`, not
    /// `ValueError`: a missing GPU is an environment condition, not a
    /// bad argument.
    #[error("backend error: {0}")]
    Backend(String),
}

impl From<PhaiosError> for PyErr {
    fn from(e: PhaiosError) -> PyErr {
        match e {
            PhaiosError::Shape(_) | PhaiosError::Parameter(_) => {
                PyValueError::new_err(e.to_string())
            }
            PhaiosError::Backend(_) => PyRuntimeError::new_err(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `From` impl matches exhaustively, so adding a variant without
    // deciding which Python exception it maps to will not compile. The
    // mapping itself cannot be asserted here: the crate is built with
    // pyo3's `extension-module` feature, so a test binary has no
    // interpreter to attach to.

    #[test]
    fn messages_name_the_kind_of_problem() {
        assert_eq!(
            PhaiosError::Shape("got (4, 4, 2)".into()).to_string(),
            "shape error: got (4, 4, 2)"
        );
        assert_eq!(
            PhaiosError::Parameter("eps must be >= 0".into()).to_string(),
            "parameter error: eps must be >= 0"
        );
    }
}
