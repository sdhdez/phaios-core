// SPDX-License-Identifier: GPL-3.0-or-later
//! The `phaios_core.gpu` Python submodule (built with `--features cuda`).
//!
//! Named `gpu`, not `cuda`: if a second backend ever lands, the Python
//! surface does not change. Registration includes the `sys.modules`
//! entry, without which `from phaios_core.gpu import exposure` fails
//! even though `phaios_core.gpu` is an attribute — see the PyO3 notes
//! in `docs/ffi.md`.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::cuda;

/// Information about one CUDA device.
///
/// Plain data: enumeration is free of side effects, and no method on
/// this class touches the device.
#[pyclass(name = "GpuInfo", frozen)]
pub struct GpuInfo {
    /// Device index, accepted by ``GpuContext(index=...)``.
    #[pyo3(get)]
    pub ordinal: usize,
    /// Marketing name, e.g. ``"NVIDIA GeForce RTX 5070 Ti"``.
    #[pyo3(get)]
    pub name: String,
    /// Compute capability ``(major, minor)``.
    #[pyo3(get)]
    pub compute_capability: (i32, i32),
    /// Whether this crate's kernels can run on it (requires ≥ 8.0).
    #[pyo3(get)]
    pub supported: bool,
}

#[pymethods]
impl GpuInfo {
    fn __repr__(&self) -> String {
        format!(
            "GpuInfo(ordinal={}, name={:?}, compute_capability=({}, {}), supported={})",
            self.ordinal,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.supported
        )
    }
}

/// An owned handle to one CUDA device.
///
/// Construct once, pass to every GPU kernel call. Dropping it releases
/// nothing shared: contexts are independent, and images produced by one
/// context are not valid with another.
#[pyclass(name = "GpuContext", frozen)]
pub struct GpuContext {
    pub(crate) inner: cuda::Context,
}

#[pymethods]
impl GpuContext {
    /// Open CUDA device ``index`` (default 0).
    ///
    /// Raises ``RuntimeError`` — never ``PanicException`` — if no device
    /// exists, the driver is missing, or the device is older than
    /// compute capability 8.0 (Ampere).
    #[new]
    #[pyo3(signature = (index = 0))]
    fn new(py: Python<'_>, index: usize) -> PyResult<Self> {
        // Context creation initialises the driver — worth detaching.
        let inner = py.detach(move || cuda::Context::new(index))?;
        Ok(Self { inner })
    }

    /// Information about the device this context owns.
    #[getter]
    fn info(&self) -> GpuInfo {
        let d = self.inner.info();
        GpuInfo {
            ordinal: d.ordinal,
            name: d.name.clone(),
            compute_capability: d.compute_capability,
            supported: d.supported,
        }
    }

    /// The backend fingerprint — the reproducibility key of
    /// ``docs/ffi.md`` §6. Record it wherever exact reproduction is
    /// promised.
    #[getter]
    fn fingerprint(&self) -> String {
        self.inner.fingerprint()
    }

    fn __repr__(&self) -> String {
        format!("GpuContext({})", self.inner.fingerprint())
    }
}

/// True if at least one supported CUDA device is present. Never raises.
#[pyfunction]
fn available() -> bool {
    cuda::available()
}

/// Enumerate CUDA devices. Never raises; an empty list means none.
#[pyfunction]
fn devices() -> Vec<GpuInfo> {
    cuda::devices()
        .into_iter()
        .map(|d| GpuInfo {
            ordinal: d.ordinal,
            name: d.name,
            compute_capability: d.compute_capability,
            supported: d.supported,
        })
        .collect()
}

/// Apply exposure compensation in EV stops on the GPU.
///
/// Bit-identical to ``phaios_core.exposure`` — the conformance suite
/// asserts equality with no tolerance. Accepts any input layout.
///
/// Parameters
/// ----------
/// ctx : GpuContext
///     The device to run on.
/// img : numpy.ndarray
///     Input array, shape ``(H, W, C)``, dtype ``float32``, any layout.
/// stops : float
///     Exposure adjustment in EV.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.
///
/// Raises
/// ------
/// ValueError
///     If ``stops`` is not finite (same message as the CPU kernel).
/// RuntimeError
///     If a device operation fails.
#[pyfunction]
fn exposure(
    py: Python<'_>,
    ctx: &GpuContext,
    img: PyReadonlyArray3<f32>,
    stops: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    let view = img.as_array();
    let inner = ctx.inner.clone();
    let result = py.detach(move || cuda::kernels::exposure(&inner, view, stops))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Register the `gpu` submodule on `phaios_core`.
pub(crate) fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let gpu = PyModule::new(py, "gpu")?;
    gpu.add_class::<GpuInfo>()?;
    gpu.add_class::<GpuContext>()?;
    gpu.add_function(wrap_pyfunction!(available, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(devices, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(exposure, &gpu)?)?;
    parent.add_submodule(&gpu)?;
    // Without this, `import phaios_core.gpu` / `from phaios_core.gpu
    // import ...` fail: add_submodule creates an attribute, not an
    // importable module.
    py.import("sys")?
        .getattr("modules")?
        .set_item("phaios_core.gpu", &gpu)?;
    Ok(())
}
