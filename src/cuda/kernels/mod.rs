// SPDX-License-Identifier: GPL-3.0-or-later
//! GPU kernels, grouped by the CPU kernels they mirror. A module can
//! hold several: `elementwise` holds eight, `geometry` four, `analysis`
//! two, `quantize` two.
//!
//! Each kernel offers two forms:
//!
//! - `<name>_device(&DeviceImage, ...) -> DeviceImage` — the resident
//!   form: input and output stay on the GPU, so a pipeline pays PCIe
//!   once at each end instead of per stage.
//! - `<name>(&Context, ArrayView3, ...) -> Array3` — per-call offload:
//!   upload, run, download. Convenient for a single kernel; the
//!   transfers dominate for cheap ones.
//!
//! Three terminal kernels are the exception to the resident form:
//! `histogram_device` returns a host `Histogram`, and
//! `quantize_u8_device` and `quantize_u16_device` return host
//! `Array3<u8>` and `Array3<u16>`. A reduction and a terminal stage
//! have nothing to chain into.
//!
//! Both forms validate identically to the CPU kernel they mirror.

mod analysis;
mod blur;
mod denoise;
mod elementwise;
mod exposure;
mod geometry;
mod glow;
mod grain;
mod hot_pixels;
mod hsl;
mod local_contrast;
mod quantize;
mod sharpen;
mod split_toning;
mod zone;

pub use analysis::{apply_lut, apply_lut_device, histogram, histogram_device};
pub use blur::{blur, blur_device};
pub use denoise::{denoise, denoise_device};
pub use elementwise::{
    channel_mixer_bw, channel_mixer_bw_device, color_filter_bw, color_filter_bw_device,
    encode_srgb, encode_srgb_device, highlight_rolloff, highlight_rolloff_device, luminance_bw,
    luminance_bw_device, shadow_rolloff, shadow_rolloff_device, tone_curve, tone_curve_device,
    vignette, vignette_device,
};
pub use exposure::{exposure, exposure_device};
pub use geometry::{
    crop, crop_device, orient, orient_device, resize, resize_device, straighten, straighten_device,
};
pub use glow::{glow, glow_device};
pub use grain::{film_grain, film_grain_device, hash_grid};
pub use hot_pixels::{hot_pixels, hot_pixels_device};
pub use hsl::{hsl_bw, hsl_bw_device};
pub use local_contrast::{local_contrast, local_contrast_device};
pub use quantize::{quantize_u8, quantize_u8_device, quantize_u16, quantize_u16_device};
pub use sharpen::{sharpen, sharpen_device};
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
/// geometry: every kernel launched through this helper indexes by
/// absolute element and none reduces across threads, so a different
/// block size cannot change a result. The histogram kernels do reduce
/// across threads, through atomics, and use their own launch geometry.
/// Their accumulation is over integers, so it is order-independent
/// anyway.
pub(crate) fn grid_1d(n: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig::for_num_elems(n as u32)
}
