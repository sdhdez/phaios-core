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

/// Highlight roll-off on the GPU. Mirrors
/// ``phaios_core.highlight_rolloff``; bit-exact.
///
/// The default ``RolloffParams()`` is a hard clip at 1.0.
#[pyfunction]
fn highlight_rolloff(
    py: Python<'_>,
    img: &GpuImage,
    params: crate::highlight_rolloff::RolloffParams,
) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::highlight_rolloff_device(&img.inner, &params))?;
    Ok(GpuImage { inner: result })
}

/// Quantise a device image to 8-bit codes, returning a numpy array.
///
/// Terminal: quantisation is where the device-resident chain ends, so
/// this returns host data rather than another ``GpuImage``. Mirrors
/// ``phaios_core.quantize_u8``; bit-exact.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn quantize_u8(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::quantize::QuantizeParams>,
) -> PyResult<Py<PyArray3<u8>>> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::quantize_u8_device(&img.inner, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Quantise a device image to 16-bit codes, returning a numpy array.
///
/// Terminal, like ``quantize_u8``. Mirrors ``phaios_core.quantize_u16``;
/// bit-exact.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn quantize_u16(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::quantize::QuantizeParams>,
) -> PyResult<Py<PyArray3<u16>>> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::quantize_u16_device(&img.inner, &owned))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Per-channel histogram of a device image, returned on the host.
///
/// A reduction, so it returns a ``Histogram`` rather than a
/// ``GpuImage``: the counts are small and their destination is a display
/// or an auto-correction, both of which live on the host. Mirrors
/// ``phaios_core.histogram``; bit-identical, because bin counts are
/// integers and integer addition commutes.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn histogram(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::histogram::HistogramParams>,
) -> PyResult<crate::histogram::Histogram> {
    let owned = params.unwrap_or_default();
    Ok(py.detach(|| cuda::kernels::histogram_device(&img.inner, &owned))?)
}

/// Apply a 1-D lookup table on the GPU. Mirrors
/// ``phaios_core.apply_lut``; bit-exact.
#[pyfunction]
#[pyo3(signature = (img, lut, params = None))]
fn apply_lut(
    py: Python<'_>,
    img: &GpuImage,
    lut: numpy::PyReadonlyArray1<f32>,
    params: Option<crate::lut::LutParams>,
) -> PyResult<GpuImage> {
    let table = lut.as_array();
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::apply_lut_device(&img.inner, table, &owned))?;
    Ok(GpuImage { inner: result })
}

/// Shadow toe on the GPU. Mirrors ``phaios_core.shadow_rolloff``;
/// bit-exact.
///
/// The default ``ShadowRolloffParams()`` is the identity.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn shadow_rolloff(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::shadow_rolloff::ShadowRolloffParams>,
) -> PyResult<GpuImage> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::shadow_rolloff_device(&img.inner, &owned))?;
    Ok(GpuImage { inner: result })
}

/// Gaussian blur on the GPU. Mirrors ``phaios_core.blur``; agreement is
/// bounded rather than bit-exact, because the device accumulates each
/// separable pass in Kahan-compensated float32 where the host uses
/// float64 — a consumer card runs float64 at 1/64 rate.
///
/// ``sigma = 0.0`` is the exact identity on both backends.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn blur(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::blur::BlurParams>,
) -> PyResult<GpuImage> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::blur_device(&img.inner, &owned))?;
    Ok(GpuImage { inner: result })
}

/// Light scattering on the GPU — halation, diffusion and veiling glare.
/// Mirrors ``phaios_core.glow``; agreement is bounded, inherited from
/// the blur between the two element-wise halves.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn glow(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::glow::GlowParams>,
) -> PyResult<GpuImage> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::glow_device(&img.inner, &owned))?;
    Ok(GpuImage { inner: result })
}

/// Unsharp mask on the GPU — a threshold-gated Gaussian sharpen. Mirrors
/// ``phaios_core.sharpen``; agreement is bounded (blur's class), inherited
/// entirely from the blur beneath the pointwise gate-and-combine kernel,
/// which is itself bit-exact.
///
/// ``amount = 0.0`` or ``sigma = 0.0`` is the exact identity on both
/// backends.
#[pyfunction]
#[pyo3(signature = (img, params = None))]
fn sharpen(
    py: Python<'_>,
    img: &GpuImage,
    params: Option<crate::sharpen::SharpenParams>,
) -> PyResult<GpuImage> {
    let owned = params.unwrap_or_default();
    let result = py.detach(|| cuda::kernels::sharpen_device(&img.inner, &owned))?;
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

/// Arbitrary-weight channel mixer on the GPU. Mirrors
/// ``phaios_core.channel_mixer_bw``; bit-exact.
///
/// Collapses ``(H, W, 3)`` to ``(H, W, 1)``.
#[pyfunction]
fn channel_mixer_bw(
    py: Python<'_>,
    img: &GpuImage,
    wr: f32,
    wg: f32,
    wb: f32,
) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::channel_mixer_bw_device(&img.inner, [wr, wg, wb]))?;
    Ok(GpuImage { inner: result })
}

/// Wratten-style colour-filter conversion on the GPU. Mirrors
/// ``phaios_core.color_filter_bw``; bit-exact.
///
/// Collapses ``(H, W, 3)`` to ``(H, W, 1)``.
#[pyfunction]
#[pyo3(signature = (img, filter = crate::bw::ColorFilter::NoFilter, standard = crate::bw::LuminanceStandard::Bt709))]
fn color_filter_bw(
    py: Python<'_>,
    img: &GpuImage,
    filter: crate::bw::ColorFilter,
    standard: crate::bw::LuminanceStandard,
) -> PyResult<GpuImage> {
    let result =
        py.detach(|| cuda::kernels::color_filter_bw_device(&img.inner, filter, standard))?;
    Ok(GpuImage { inner: result })
}

/// HSL-weighted B&W conversion on the GPU. Mirrors
/// ``phaios_core.hsl_bw``; agreement bounded by ``expf``.
///
/// Collapses ``(H, W, 3)`` to ``(H, W, 1)``.
#[pyfunction]
fn hsl_bw(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::bw::HslWeightedParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::hsl_bw_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Zone System tone curve on the GPU. Mirrors
/// ``phaios_core.zone_system``; the dense in-order offset sum inherits
/// the CPU's ordered-reduction guarantee mechanically.
#[pyfunction]
fn zone_system(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::tone::ZoneParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::zone_system_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Split-toning on the GPU: ``(H, W, 1)`` in, ``(H, W, 3)`` out.
/// Mirrors ``phaios_core.split_toning``; agreement bounded by ``cbrtf``.
#[pyfunction]
fn split_toning(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::split_toning::SplitToningParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::split_toning_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Procedural film grain on the GPU. Mirrors
/// ``phaios_core.film_grain``: the splitmix64 hash is bit-exact against
/// the CPU (asserted over 2**20 coordinates); Box-Muller and the box
/// filters are bounded.
#[pyfunction]
fn film_grain(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::film_grain::GrainParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::film_grain_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Crop on the GPU. Mirrors ``phaios_core.crop``; bit-exact (a pure
/// index copy).
#[pyfunction]
fn crop(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::geometry::CropParams>,
) -> PyResult<GpuImage> {
    let params_owned = *params;
    let result = py.detach(|| cuda::kernels::crop_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Dihedral orientation on the GPU. Mirrors ``phaios_core.orient``;
/// bit-exact (a pure index permutation).
#[pyfunction]
fn orient(
    py: Python<'_>,
    img: &GpuImage,
    orientation: crate::geometry::Orientation,
) -> PyResult<GpuImage> {
    let result = py.detach(|| cuda::kernels::orient_device(&img.inner, orientation))?;
    Ok(GpuImage { inner: result })
}

/// Remove hot pixels with a conditional 3x3 median, on the GPU. Mirrors
/// ``phaios_core.hot_pixels``; bit-exact (a fixed comparator network of
/// ``min``/``max`` pairs with no arithmetic, the same class as ``crop``,
/// ``orient`` and ``vignette``).
#[pyfunction]
fn hot_pixels(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::hot_pixels::HotPixelParams>,
) -> PyResult<GpuImage> {
    let params_owned = params.clone();
    let result = py.detach(|| cuda::kernels::hot_pixels_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Resize on the GPU. Mirrors ``phaios_core.resize``; bit-exact.
#[pyfunction]
fn resize(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::geometry::ResizeParams>,
) -> PyResult<GpuImage> {
    let params_owned = *params;
    let result = py.detach(|| cuda::kernels::resize_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Straighten on the GPU. Mirrors ``phaios_core.straighten``; bit-exact.
#[pyfunction]
fn straighten(
    py: Python<'_>,
    img: &GpuImage,
    params: pyo3::PyRef<'_, crate::geometry::StraightenParams>,
) -> PyResult<GpuImage> {
    let params_owned = *params;
    let result = py.detach(|| cuda::kernels::straighten_device(&img.inner, &params_owned))?;
    Ok(GpuImage { inner: result })
}

/// Docstring for the Python-visible `phaios_core.gpu` module.
const GPU_MODULE_DOC: &str = "\
CUDA backend for phaios-core (built with `--features cuda`).

Every kernel in the parent module appears here a second time, operating
on a device-resident ``GpuImage`` instead of a numpy array, so a
pipeline uploads once and downloads once rather than round-tripping
per stage::

    ctx = gpu.GpuContext(0)
    img = ctx.upload(array)
    img = gpu.exposure(img, 0.5)
    img = gpu.luminance_bw(img)
    out = img.download()

The context is explicit: there is no global device state, and a context
the caller drops releases its device memory. Determinism is promised
per backend — bit-identical within a backend, bounded across — and
kernels free of transcendentals are bit-exact against the CPU as well.
See docs/ffi.md section 6.

Call ``available()`` before constructing a context; ``devices()`` lists
what is present.";

/// Register the `gpu` submodule on `phaios_core`.
pub(crate) fn register(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let gpu = PyModule::new(py, "gpu")?;
    // Without this the submodule's //! documentation never reaches Python
    // and `help(phaios_core.gpu)` shows nothing.
    gpu.add("__doc__", GPU_MODULE_DOC)?;
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
    gpu.add_function(wrap_pyfunction!(highlight_rolloff, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(shadow_rolloff, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(blur, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(glow, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(sharpen, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(quantize_u8, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(quantize_u16, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(histogram, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(apply_lut, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(luminance_bw, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(channel_mixer_bw, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(color_filter_bw, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(hsl_bw, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(zone_system, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(split_toning, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(film_grain, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(crop, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(orient, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(hot_pixels, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(resize, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(straighten, &gpu)?)?;
    parent.add_submodule(&gpu)?;
    // Without this, `import phaios_core.gpu` / `from phaios_core.gpu
    // import ...` fail: add_submodule creates an attribute, not an
    // importable module.
    py.import("sys")?
        .getattr("modules")?
        .set_item("phaios_core.gpu", &gpu)?;
    Ok(())
}
