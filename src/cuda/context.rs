// SPDX-License-Identifier: GPL-3.0-or-later
//! CUDA device context — the only file in the crate that names a
//! `cudarc` type.
//!
//! That confinement is deliberate: it is the exit strategy. If cudarc is
//! ever abandoned or breaks, this file is the entire replacement surface.
//!
//! # Driver-API entry points used (the exit-strategy ledger)
//!
//! Everything this crate does reaches the driver through the calls below,
//! over a C ABI NVIDIA guarantees backward-compatible. Replacing cudarc
//! means `dlopen("libcuda.so.1")` plus these, resolved with `dlsym` —
//! which is what cudarc itself does; it does *not* use `cuGetProcAddress`,
//! so the versioned names (`cuMemcpyHtoDAsync_v2` and friends) are
//! requested literally.
//!
//! | Entry point | Used for |
//! |---|---|
//! | `cuInit` | driver initialisation |
//! | `cuDeviceGetCount` | enumeration |
//! | `cuDeviceGet` | enumeration |
//! | `cuDeviceGetName` | device info |
//! | `cuDeviceGetAttribute` | compute capability, memory-pool support |
//! | `cuDevicePrimaryCtxRetain` | context creation |
//! | `cuDevicePrimaryCtxRelease_v2` | context teardown |
//! | `cuCtxGetCurrent`, `cuCtxSetCurrent` | binding the context to the calling thread, before every allocation, copy and launch |
//! | `cuModuleLoadData` | loading embedded PTX |
//! | `cuModuleGetFunction` | kernel lookup |
//! | `cuModuleUnload` | module teardown |
//! | `cuMemAllocAsync` | device buffers |
//! | `cuMemFreeAsync` | buffer release |
//! | `cuMemsetD8Async` | the one `alloc_zeros` (histogram bins) |
//! | `cuMemcpyHtoDAsync_v2` | upload |
//! | `cuMemcpyDtoHAsync_v2` | download |
//! | `cuMemcpyDtoDAsync_v2` | device-to-device copy (blur, sharpen, glow, zone, grain, denoise, elementwise) |
//! | `cuLaunchKernel` | dispatch |
//! | `cuStreamSynchronize` | completion |
//! | `cuEventCreate`, `cuEventDestroy_v2` | cudarc's own per-allocation bookkeeping, two events per buffer |
//!
//! Four things a reimplementation has to know, none of them obvious from
//! the cudarc call sites:
//!
//! - **The allocation path is stream-ordered, not classic.** cudarc takes
//!   `cuMemAllocAsync`/`cuMemFreeAsync` whenever the device reports
//!   `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`, and falls back to
//!   `cuMemAlloc_v2`/`cuMemFree_v2` (plus a stream sync before each free)
//!   when it does not — likewise `cuMemcpy*_v2` and `cuMemsetD8_v2` for
//!   the copies. Both paths are live; the async one is what runs on any
//!   modern card, including this project's. They are not interchangeable:
//!   the pool path returns memory to a pool rather than to the driver.
//! - **`cuMemAllocAsync` sets the version floor, at CUDA 11.2.** Every
//!   other entry point here is far older — the `_v2` suffixes are the
//!   CUDA 4.0 64-bit-pointer revision, and `cuDevicePrimaryCtx*` is CUDA
//!   7.0. The `cuda-12000` feature floor in `Cargo.toml` is about which
//!   generated declarations compile, not about what the driver must
//!   export.
//! - **No stream is ever created.** `default_stream()` hands back the
//!   null (legacy default) stream without a driver call, so there is no
//!   `cuStreamCreate`/`cuStreamDestroy_v2` here, and because the context
//!   stays in single-stream mode there is no `cuEventRecord` or
//!   `cuStreamWaitEvent` either — cudarc creates the events but never
//!   records them.
//! - **The event traffic is cudarc's, not this crate's.** A replacement
//!   would simply not allocate the two `CUevent`s per buffer.
//!
//! **Keep this table current**: one line per new call, checked in review.
//! It was last verified empirically, not by reading the call sites:
//! breakpoints on 34 candidate driver symbols under gdb, over
//! `examples/41_gpu_blur` (upload, blur, device-to-device copy, download)
//! and `examples/39_gpu_histogram_lut` (the `alloc_zeros` path). The 22
//! symbols above are exactly the ones that fired; `cuStreamCreate`,
//! `cuGetProcAddress`, `cuEventRecord`, `cuStreamWaitEvent` and every
//! synchronous `cuMem*` form fired zero times.
//!
//! # No hidden state
//!
//! There is no singleton, no `OnceLock`, no ambient context anywhere in
//! `src/cuda/`. A [`Context`] is constructed and owned by the caller and
//! passed in explicitly; a kernel's output depends on its arguments and
//! nothing else. The mutable state a device genuinely needs — the loaded
//! module cache — lives inside the context and cannot affect results.
//! The driver-library probe `driver_present` adds none either: it
//! answers one question per call and caches nothing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

use crate::error::PhaiosError;

/// Minimum supported compute capability (Ampere).
///
/// The embedded PTX is compiled at `compute_80`; PTX is
/// forward-compatible, so anything newer JITs it natively. Older cards
/// are rejected here with a clear message instead of a cryptic JIT
/// failure at first launch.
pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (8, 0);

/// Convert any cudarc driver error into the crate's error type.
///
/// Free function rather than a `From` impl so that no cudarc type leaks
/// into `error.rs`.
fn backend_err(what: &str, e: impl std::fmt::Display) -> PhaiosError {
    PhaiosError::Backend(format!("{what}: {e}"))
}

/// True when the NVIDIA driver library can be loaded at all.
///
/// This exists because cudarc's error type cannot express "no driver".
/// Under `dynamic-loading`, cudarc resolves `libcuda` lazily on the
/// first driver call, and when no candidate name loads it **panics**
/// (`panic_no_lib_found`, `cudarc/src/lib.rs:200`) rather than returning
/// an `Err`. PyO3 converts that unwind into `PanicException`, which
/// inherits from `BaseException` and so walks straight through a
/// consumer's `except Exception:` — the exact failure mode CLAUDE.md §2
/// rules out. Every public entry point that touches the driver
/// ([`devices`] and [`Context::new`]; every kernel needs a [`Context`])
/// is therefore gated on this first, so a machine with no driver gets
/// `false`, an empty list and [`PhaiosError::Backend`] instead of an
/// unwind.
///
/// `cudarc::driver::sys::is_culib_present` is cudarc's own fallible
/// probe over the same candidate-name list `culib()` searches, returning
/// `bool` where `culib()` panics — so this borrows cudarc's search
/// order rather than second-guessing it, and no `catch_unwind` or extra
/// dependency is needed.
///
/// **No state is introduced**: the probe opens a candidate library,
/// answers the question and drops the handle. It does not call `cuInit`,
/// caches nothing, and cannot affect any kernel's result. Each call
/// re-probes; on a machine that does have a driver the library is
/// already resident, so the repeat is a refcount bump.
fn driver_present() -> bool {
    // SAFETY: `is_culib_present` only attempts `dlopen` on a fixed list
    // of library names and drops each handle it obtains. It takes no
    // pointers, dereferences nothing, and initialises no driver state.
    unsafe { cudarc::driver::sys::is_culib_present() }
}

/// Information about one CUDA device, safe to expose to Python.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    /// Device ordinal, the index `Context::new` accepts.
    pub ordinal: usize,
    /// Marketing name, e.g. "NVIDIA GeForce RTX 5070 Ti".
    pub name: String,
    /// Compute capability as (major, minor).
    pub compute_capability: (i32, i32),
    /// Whether this crate's kernels can run on it (cc >= 8.0).
    pub supported: bool,
}

/// Enumerate CUDA devices. **Never fails and never panics**: any error —
/// no driver library at all, no device, broken installation — yields an
/// empty list, because "no GPU" is an ordinary state of the world, not an
/// exception.
///
/// The missing-library case is handled by `driver_present` before
/// cudarc is touched; the remaining cases are ordinary `Err`s.
#[must_use]
pub fn devices() -> Vec<DeviceInfo> {
    if !driver_present() {
        return Vec::new();
    }
    let Ok(count) = CudaContext::device_count() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ordinal in 0..count.max(0) as usize {
        // A device that errors mid-enumeration is skipped, not fatal.
        let Ok(ctx) = CudaContext::new(ordinal) else {
            continue;
        };
        let name = ctx.name().unwrap_or_else(|_| format!("device {ordinal}"));
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);
        out.push(DeviceInfo {
            ordinal,
            name,
            compute_capability: (major, minor),
            supported: (major, minor) >= MIN_COMPUTE_CAPABILITY,
        });
    }
    out
}

/// True if at least one supported CUDA device exists.
///
/// Never raises, including on a machine with no NVIDIA driver installed:
/// it delegates to [`devices`], which is gated on `driver_present`.
#[must_use]
pub fn available() -> bool {
    devices().iter().any(|d| d.supported)
}

/// An owned handle to one CUDA device.
///
/// Caller-constructed, caller-owned, explicitly passed to every GPU
/// kernel — see the module docs on hidden state. Cheap to clone (Arcs
/// inside); clones share the device, stream and module cache.
#[derive(Clone)]
pub struct Context {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    /// PTX modules already loaded on this context, keyed by the address
    /// of the embedded PTX source — one module per `.ptx` file. Keying by
    /// *kernel name* was the bug a previous audit caught; see `function`.
    /// Loading is idempotent and keyed content is `include_str!`-embedded,
    /// so this cache can only affect *speed*, never results.
    modules: Arc<Mutex<HashMap<usize, Arc<CudaModule>>>>,
    info: DeviceInfo,
}

impl Context {
    /// Open device `ordinal`.
    ///
    /// # Errors
    /// [`PhaiosError::Backend`] if the driver cannot be loaded, the
    /// ordinal does not exist, or the device's compute capability is
    /// below [`MIN_COMPUTE_CAPABILITY`]. Never panics: the
    /// no-driver-library case is caught by `driver_present` before
    /// cudarc is touched.
    pub fn new(ordinal: usize) -> Result<Self, PhaiosError> {
        if !driver_present() {
            return Err(PhaiosError::Backend(
                "CUDA driver unavailable: the NVIDIA driver library \
                 (libcuda) is not installed or not on the loader's search path"
                    .to_string(),
            ));
        }
        let count =
            CudaContext::device_count().map_err(|e| backend_err("CUDA driver unavailable", e))?;
        if ordinal >= count.max(0) as usize {
            return Err(PhaiosError::Backend(format!(
                "no CUDA device at index {ordinal}: {count} device(s) present"
            )));
        }
        let ctx =
            CudaContext::new(ordinal).map_err(|e| backend_err("cannot open CUDA device", e))?;

        let name = ctx.name().unwrap_or_else(|_| format!("device {ordinal}"));
        let major = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| backend_err("cannot query compute capability", e))?;
        let minor = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| backend_err("cannot query compute capability", e))?;
        if (major, minor) < MIN_COMPUTE_CAPABILITY {
            return Err(PhaiosError::Backend(format!(
                "device \"{name}\" has compute capability {major}.{minor}; \
                 this crate's kernels require >= {}.{} (Ampere)",
                MIN_COMPUTE_CAPABILITY.0, MIN_COMPUTE_CAPABILITY.1
            )));
        }

        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            modules: Arc::new(Mutex::new(HashMap::new())),
            info: DeviceInfo {
                ordinal,
                name,
                compute_capability: (major, minor),
                supported: true,
            },
        })
    }

    /// Information about the device this context owns.
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The backend fingerprint, the reproducibility key from
    /// `docs/ffi.md` §6:
    /// `cuda/<device>/cc<maj>.<min>/ptx-compute_80/nvcc-<maj>.<min>.<patch>`.
    ///
    /// Two machines with the same fingerprint produce bit-identical
    /// output for the same input, parameters and seed.
    ///
    /// The `nvcc-` segment is the toolkit that compiled the embedded PTX,
    /// captured by `build.rs` from the same `nvcc` the compile used. It
    /// is part of the key because it has to be: the PTX is regenerated at
    /// build time by whatever toolkit is present, and the bounded kernels
    /// (`hsl_bw`, `zone_system`, `local_contrast`, `denoise`, `film_grain`,
    /// `split_toning`, `tone_curve` at power != 1, `encode_srgb`, `blur`,
    /// `glow`, `sharpen`) inline libdevice bodies for `powf`, `expf`,
    /// `log2f` and `cbrtf` that change between toolkits. Without it, two
    /// builds from different toolkits reported the same fingerprint and
    /// could differ in the low bits — this project moved 13.3 -> 13.4 with
    /// the key unchanged. The bit-exact kernels use IEEE operations only
    /// and are unaffected either way.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        format!(
            "cuda/{}/cc{}.{}/ptx-compute_80/nvcc-{}",
            self.info.name,
            self.info.compute_capability.0,
            self.info.compute_capability.1,
            env!("PHAIOS_NVCC_VERSION"),
        )
    }

    /// Load (or fetch from cache) `kernel_name` from embedded PTX.
    pub(crate) fn function(
        &self,
        kernel_name: &'static str,
        ptx_src: &'static str,
    ) -> Result<CudaFunction, PhaiosError> {
        // Cache keyed by the PTX source's address (stable for 'static
        // data): one module per embedded PTX file. The audit caught the
        // previous keying by KERNEL NAME, which loaded the same module
        // once per kernel it contains (4x for the guided filter) and
        // would have silently returned the wrong module had two PTX
        // files ever shared a kernel name.
        let key = ptx_src.as_ptr() as usize;
        let module = {
            let mut cache = self
                .modules
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match cache.get(&key) {
                Some(m) => Arc::clone(m),
                None => {
                    let m = self
                        .ctx
                        .load_module(ptx_src.into())
                        .map_err(|e| backend_err("cannot load PTX module", e))?;
                    cache.insert(key, Arc::clone(&m));
                    m
                }
            }
        };
        module
            .load_function(kernel_name)
            .map_err(|e| backend_err("kernel not found in PTX", e))
    }
}

/// An image resident in device memory.
///
/// Created by [`Context::upload`] or returned by a device kernel; leaves
/// the device only through [`Context::download`]. Carries a clone of its
/// context (cheap: three `Arc`s), so it can never outlive the device
/// state it points into, and kernels never need a separate context
/// argument — which also makes cross-context mixing unrepresentable.
pub struct DeviceImage {
    pub(crate) ctx: Context,
    pub(crate) buf: cudarc::driver::CudaSlice<f32>,
    pub(crate) shape: (usize, usize, usize),
}

impl DeviceImage {
    /// Shape as `(H, W, C)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    /// The context this image lives on.
    #[must_use]
    pub fn context(&self) -> &Context {
        &self.ctx
    }
}

impl std::fmt::Debug for DeviceImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceImage")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl Context {
    /// Upload an image to the device. Accepts any layout, like every
    /// input in this crate; the staging copy is the same Zip-walk a CPU
    /// kernel would do.
    ///
    /// # Errors
    /// [`PhaiosError::Backend`] if allocation or the copy fails.
    pub fn upload(&self, img: ndarray::ArrayView3<f32>) -> Result<DeviceImage, PhaiosError> {
        let shape = img.dim();
        // `as_standard_layout` materialises a host copy of the *logical*
        // shape, so a zero-stride view reaches the same abort the CPU
        // kernels used to. Bound it first, with the same limit and the
        // same error, so the two surfaces refuse identically.
        crate::alloc::check_shape::<f32>(shape)?;
        let staged = img.as_standard_layout();
        let host = staged
            .as_slice()
            .expect("as_standard_layout output is contiguous by construction");
        let buf = self
            .stream
            .clone_htod(host)
            .map_err(|e| backend_err("upload failed", e))?;
        Ok(DeviceImage {
            ctx: self.clone(),
            buf,
            shape,
        })
    }

    /// Download an image from the device into a freshly allocated,
    /// C-contiguous array. Synchronises the stream, so on return the
    /// data is complete.
    ///
    /// # Errors
    /// [`PhaiosError::Backend`] if the copy fails.
    pub fn download(&self, img: &DeviceImage) -> Result<ndarray::Array3<f32>, PhaiosError> {
        let mut out = ndarray::Array3::<f32>::zeros(img.shape);
        if img.shape.0 * img.shape.1 * img.shape.2 > 0 {
            self.stream
                .memcpy_dtoh(
                    &img.buf,
                    out.as_slice_mut()
                        .expect("freshly allocated Array3 is contiguous"),
                )
                .map_err(|e| backend_err("download failed", e))?;
            self.stream
                .synchronize()
                .map_err(|e| backend_err("synchronize failed", e))?;
        }
        Ok(out)
    }

    /// Allocate an uninitialised device image of `shape`, for kernels
    /// that fully overwrite their output.
    pub(crate) fn alloc_image(
        &self,
        shape: (usize, usize, usize),
    ) -> Result<DeviceImage, PhaiosError> {
        let n = (shape.0 * shape.1 * shape.2).max(1);
        // Safety: callers write every element before it is read; empty
        // shapes allocate one element that is never read at all.
        let buf = unsafe { self.stream.alloc::<f32>(n) }
            .map_err(|e| backend_err("device allocation failed", e))?;
        Ok(DeviceImage {
            ctx: self.clone(),
            buf,
            shape,
        })
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}
