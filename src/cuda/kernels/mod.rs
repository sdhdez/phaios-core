// SPDX-License-Identifier: GPL-3.0-or-later
//! GPU kernels, one module per CPU kernel they mirror.
//!
//! Each module offers two forms:
//!
//! - `<name>_device(&DeviceImage, ...) -> DeviceImage` — the resident
//!   form: input and output stay on the GPU, so a pipeline pays PCIe
//!   once at each end instead of per stage.
//! - `<name>(&Context, ArrayView3, ...) -> Array3` — per-call offload:
//!   upload, run, download. Convenient for a single kernel; the
//!   transfers dominate for cheap ones.
//!
//! Both validate identically to the CPU kernel they mirror.

mod exposure;
mod local_contrast;

mod elementwise;

pub use elementwise::{
    encode_srgb, encode_srgb_device, luminance_bw, luminance_bw_device, tone_curve,
    tone_curve_device, vignette, vignette_device,
};
pub use exposure::{exposure, exposure_device};
pub use local_contrast::{local_contrast, local_contrast_device};

use crate::error::PhaiosError;

/// Map a cudarc driver error into the crate error type.
pub(crate) fn be(what: &'static str) -> impl Fn(cudarc::driver::DriverError) -> PhaiosError {
    move |e| PhaiosError::Backend(format!("{what}: {e}"))
}

/// 1-D launch geometry over `n` elements, 256 threads per block.
pub(crate) fn grid_1d(n: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig::for_num_elems(n as u32)
}
