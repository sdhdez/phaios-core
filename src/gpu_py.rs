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

    /// Upload an image to this device, returning a ``GpuImage`` that
    /// stays resident until downloaded. Accepts any layout.
    fn upload(&self, py: Python<'_>, img: PyReadonlyArray3<f32>) -> PyResult<GpuImage> {
        let view = img.as_array();
        let inner = self.inner.clone();
        let device = py.detach(move || inner.upload(view))?;
        Ok(GpuImage { inner: device })
    }

    fn __repr__(&self) -> String {
        format!("GpuContext({})", self.inner.fingerprint())
    }
}

/// An image resident in GPU memory.
///
/// Produced by ``GpuContext.upload`` or returned by a GPU kernel; comes
/// back to numpy through ``download()``. Chaining kernels over
/// ``GpuImage`` values pays PCIe once at each end of the pipeline
/// instead of per stage. The image keeps its context alive, so dropping
/// the ``GpuContext`` first is safe.
#[pyclass(name = "GpuImage", frozen)]
pub struct GpuImage {
    pub(crate) inner: cuda::context::DeviceImage,
}

#[pymethods]
impl GpuImage {
    /// Shape as ``(H, W, C)``.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }

    /// The fingerprint of the context this image lives on.
    #[getter]
    fn fingerprint(&self) -> String {
        self.inner.context().fingerprint()
    }

    /// Download to a freshly allocated, C-contiguous float32 array.
    fn download(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f32>>> {
        let ctx = self.inner.context().clone();
        // Safety of the borrow across detach: DeviceImage is Sync.
        let result = py.detach(|| ctx.download(&self.inner))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn __repr__(&self) -> String {
        let (h, w, c) = self.inner.shape();
        format!("GpuImage(shape=({h}, {w}, {c}))")
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
/// asserts equality with no tolerance. Signature-compatible with the
/// CPU function: same arguments after the image, so a pipeline can
/// switch backends by switching what it passes.
///
/// Parameters
/// ----------
/// img : GpuImage
///     Device-resident input, from ``GpuContext.upload`` or a previous
///     kernel.
/// stops : float
///     Exposure adjustment in EV.
///
/// Returns
/// -------
/// GpuImage
///     Device-resident result; call ``download()`` to retrieve it.
///
/// Raises
/// ------
/// ValueError
///     If ``stops`` is not finite (same message as the CPU kernel).
/// RuntimeError
///     If a device operation fails.
#[pyfunction]
fn exposure(py: Python<'_>, img: &GpuImage, stops: f32) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::exposure_device(&img.inner, stops))?;
    Ok(GpuImage { inner: result })
}

/// Enhance local contrast using the guided filter, on the GPU.
///
/// Same algorithm as ``phaios_core.local_contrast`` with one documented
/// reformulation: the CPU's global f64 summed-area tables become
/// separable f32 box filters with compensated summation. Agreement with
/// the CPU is bounded at 1e-4 relative (asserted by the conformance
/// suite); output is bit-reproducible within this backend.
///
/// Parameters
/// ----------
/// img : GpuImage
///     Device-resident ``(H, W, 1)`` input.
/// params : GuidedFilterParams
///     The same parameter object the CPU kernel takes.
/// strength : float
///     Detail amplification factor.
///
/// Returns
/// -------
/// GpuImage
///     Device-resident ``(H, W, 1)`` result.
///
/// Raises
/// ------
/// ValueError
///     Same conditions and messages as the CPU kernel.
/// RuntimeError
///     If a device operation fails.
#[pyfunction]
fn local_contrast(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::local_contrast::GuidedFilterParams>,
    strength: f32,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result =
        py.detach(|| cuda::kernels::local_contrast_device(&img.inner, &params_owned, strength))?;
    Ok(GpuImage { inner: result })
}

/// IEC 61966-2-1 sRGB transfer on the GPU. Mirrors
/// ``phaios_core.encode_srgb``; agreement bounded by one ``powf``.
#[pyfunction]
fn encode_srgb(py: Python<'_>, img: &GpuImage) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::encode_srgb_device(&img.inner))?;
    Ok(GpuImage { inner: result })
}

/// ASC CDL tone curve on the GPU. Mirrors ``phaios_core.tone_curve``;
/// the ``power == 1`` and identity paths are bit-exact.
#[pyfunction]
fn tone_curve(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::tone::ToneCurveParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::tone_curve_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Radial vignette on the GPU. Mirrors ``phaios_core.vignette``;
/// bit-exact against the CPU.
#[pyfunction]
fn vignette(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::vignette::VignetteParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::vignette_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Standard-luminance B&W conversion on the GPU: ``(H, W, 3)`` in,
/// ``(H, W, 1)`` out. Mirrors ``phaios_core.luminance_bw``; bit-exact.
#[pyfunction]
#[pyo3(signature = (img, standard = crate::bw::LuminanceStandard::Bt709))]
fn luminance_bw(
    py: Python<'_>,
    img: &GpuImage,
    standard: crate::bw::LuminanceStandard,
) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::luminance_bw_device(&img.inner, standard))?;
    Ok(GpuImage { inner: result })
}

/// Register the `gpu` submodule on `phaios_core`.
pub(crate) fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let gpu = PyModule::new(py, "gpu")?;
    gpu.add_class::<GpuInfo>()?;
    gpu.add_class::<GpuContext>()?;
    gpu.add_class::<GpuImage>()?;
    gpu.add_function(wrap_pyfunction!(available, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(devices, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(exposure, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(local_contrast, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(encode_srgb, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(tone_curve, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(vignette, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(luminance_bw, &gpu)?)?;
    parent.add_submodule(&gpu)?;
    // Without this, `import phaios_core.gpu` / `from phaios_core.gpu
    // import ...` fail: add_submodule creates an attribute, not an
    // importable module.
    py.import("sys")?
        .getattr("modules")?
        .set_item("phaios_core.gpu", &gpu)?;
    Ok(())
}
