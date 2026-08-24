// SPDX-License-Identifier: GPL-3.0-or-later
//! Crop and orientation on the CUDA backend — bit-exact by
//! construction, being pure index permutations.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::geometry::{CropParams, Orientation};

/// PTX for the geometry kernels, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/geometry.ptx"));

/// Crop to a rectangle, device-resident. Mirrors
/// [`crate::geometry::crop`]; **bit-exact**.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn crop_device(img: &DeviceImage, params: &CropParams) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    crate::geometry::validate_crop(&[h, w, c], params)?;

    let ctx = img.context().clone();
    let (out_h, out_w) = (params.height as usize, params.width as usize);
    let mut out = ctx.alloc_image((out_h, out_w, c))?;
    let n = out_h * out_w * c;
    if n == 0 {
        return Ok(out);
    }

    let (in_w, c_i) = (w as i32, c as i32);
    let (x0, y0) = (params.x as i32, params.y as i32);
    let (oh, ow) = (out_h as i32, out_w as i32);
    let func = ctx.function("crop_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&in_w)
        .arg(&c_i)
        .arg(&x0)
        .arg(&y0)
        .arg(&oh)
        .arg(&ow);
    // Safety: signature matches the .cu; validated rectangle keeps every
    // source index in bounds; the kernel bounds-checks the output.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Apply a dihedral orientation, device-resident. Mirrors
/// [`crate::geometry::orient`]; **bit-exact**. Uses the same
/// `Orientation::flags()` decomposition as the CPU, so the two backends
/// share one definition.
///
/// # Errors
/// [`PhaiosError::Backend`] on device failure (the kernel itself is
/// infallible, like its CPU mirror).
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn orient_device(
    img: &DeviceImage,
    orientation: Orientation,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    let (transpose, flip_y, flip_x) = orientation.flags();
    let (out_h, out_w) = if transpose { (w, h) } else { (h, w) };

    let ctx = img.context().clone();
    let mut out = ctx.alloc_image((out_h, out_w, c))?;
    let n = out_h * out_w * c;
    if n == 0 {
        return Ok(out);
    }

    let (h_i, w_i, c_i) = (h as i32, w as i32, c as i32);
    let (t_i, fy_i, fx_i) = (i32::from(transpose), i32::from(flip_y), i32::from(flip_x));
    let func = ctx.function("orient_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&c_i)
        .arg(&t_i)
        .arg(&fy_i)
        .arg(&fx_i);
    // Safety: signature matches the .cu; the mapped source coordinates
    // stay in bounds for every output index the kernel accepts.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`crop_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn crop(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &CropParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::geometry::validate_crop(img.shape(), params)?;
    let device = ctx.upload(img)?;
    ctx.download(&crop_device(&device, params)?)
}

/// Per-call offload form of [`orient_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn orient(
    ctx: &Context,
    img: ArrayView3<f32>,
    orientation: Orientation,
) -> Result<Array3<f32>, PhaiosError> {
    let device = ctx.upload(img)?;
    ctx.download(&orient_device(&device, orientation)?)
}
