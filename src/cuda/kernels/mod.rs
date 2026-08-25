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

mod elementwise;
mod exposure;
mod geometry;
mod grain;
mod hsl;
mod local_contrast;
mod split_toning;
mod zone;

pub use elementwise::{
    channel_mixer_bw, channel_mixer_bw_device, color_filter_bw, color_filter_bw_device,
    encode_srgb, encode_srgb_device, luminance_bw, luminance_bw_device, tone_curve,
    tone_curve_device, vignette, vignette_device,
};
pub use exposure::{exposure, exposure_device};
pub use geometry::{
    crop, crop_device, orient, orient_device, resize, resize_device, straighten, straighten_device,
};
pub use grain::{film_grain, film_grain_device, hash_grid};
pub use hsl::{hsl_bw, hsl_bw_device};
pub use local_contrast::{local_contrast, local_contrast_device};
pub use split_toning::{split_toning, split_toning_device};
pub use zone::{zone_system, zone_system_device};

use crate::error::PhaiosError;

/// Map a cudarc driver error into the crate error type.
pub(crate) fn be(what: &'static str) -> impl Fn(cudarc::driver::DriverError) -> PhaiosError {
    move |e| PhaiosError::Backend(format!("{what}: {e}"))
}

/// 1-D launch geometry over `n` elements.
///
/// Block size is whatever `cudarc::driver::LaunchConfig::for_num_elems`
/// picks (1024 threads at the time of writing) — deliberately not pinned
/// here, because the determinism contract is independent of launch
/// geometry: every kernel indexes by absolute element and none reduces
/// across threads, so a different block size cannot change a result.
pub(crate) fn grid_1d(n: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig::for_num_elems(n as u32)
}
