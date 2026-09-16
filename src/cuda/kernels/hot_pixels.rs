// SPDX-License-Identifier: GPL-3.0-or-later
//! Hot-pixel removal on the CUDA backend: one pointwise-with-neighbourhood
//! kernel, bit-exact against the CPU.
//!
//! Mirrors [`crate::hot_pixels::hot_pixels`] exactly: the clamped 3x3
//! gather and the 19-comparator sorting network are transcribed verbatim
//! into `src/cuda/ptx/hot_pixels.cu`, pair by pair -- see that file's
//! module comment for the operation-by-operation correspondence. No
//! transcendentals and no accumulation, so with `-fmad=false` (build.rs)
//! every operation is correctly rounded on both sides and the whole
//! kernel is **bit-exact** -- the same class as `crop`, `orient` and
//! `vignette` (`docs/ffi.md` section 6).

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::hot_pixels::HotPixelParams;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hot_pixels.ptx"));

/// Remove hot pixels with a conditional 3x3 median, device-resident.
/// Mirrors [`crate::hot_pixels::hot_pixels`]; **bit-exact**.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn hot_pixels_device(
    img: &DeviceImage,
    params: &HotPixelParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::hot_pixels::validate(params)?;

    let (h, w, c) = img.shape();
    let ctx = img.context().clone();
    let mut out = ctx.alloc_image((h, w, c))?;
    let n = h * w * c;
    if n == 0 {
        return Ok(out);
    }

    let (h_i, w_i, c_i) = (h as i32, w as i32, c as i32);
    let n_ll = n as i64;
    let threshold = params.threshold;
    let relative = params.relative;

    let func = ctx.function("hot_pixels_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&threshold)
        .arg(&relative)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&c_i)
        .arg(&n_ll);
    // Safety: signature matches the .cu; both buffers hold n elements and
    // every gathered index is clamped into [0, h-1] x [0, w-1] before use.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;

    Ok(out)
}

/// Per-call offload form of [`hot_pixels_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn hot_pixels(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &HotPixelParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::hot_pixels::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&hot_pixels_device(&device, params)?)
}
