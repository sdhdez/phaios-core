// SPDX-License-Identifier: GPL-3.0-or-later
//! Geometry on the CUDA backend: crop, orient, resize and straighten.
//!
//! All four are bit-exact against the CPU. `crop` and `orient` because
//! they are pure index permutations, `resize` and `straighten` because
//! their filters are polynomial.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::geometry::{CropParams, Orientation, ResizeFilter, ResizeParams, StraightenParams};

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

/// Resample to a new size, device-resident. Mirrors
/// [`crate::geometry::resize`]; **bit-exact** — the filters are
/// polynomial and the device kernel transcribes the CPU tap loop
/// operation for operation.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn resize_device(img: &DeviceImage, params: &ResizeParams) -> Result<DeviceImage, PhaiosError> {
    let (in_h, in_w, c) = img.shape();
    crate::geometry::validate_resize(&[in_h, in_w, c], params)?;
    let (out_w, out_h) = (params.width as usize, params.height as usize);
    // Two distinct overflow guards, both on caller-controlled sizes.
    //
    // Element counts: grid_1d takes a u32, and a target large enough to
    // overflow it would silently launch too few blocks and return
    // uninitialised memory.
    for n in [in_h * out_w * c, out_h * out_w * c] {
        if n > u32::MAX as usize {
            return Err(PhaiosError::Parameter(format!(
                "resize target {}x{} exceeds the CUDA backend's element limit",
                params.width, params.height
            )));
        }
    }
    // Individual extents: resample_kernel takes them as `int`, so a
    // dimension of 2^31 or more wraps negative on narrowing, the kernel's
    // element count goes negative, every thread returns early, and the
    // caller receives an allocated-but-never-written buffer. The element
    // guard above does not cover this — a 2^31 x 1 target passes it.
    for extent in [in_h, in_w, out_h, out_w, c] {
        if extent > i32::MAX as usize {
            return Err(PhaiosError::Parameter(format!(
                "resize extent {extent} exceeds the CUDA backend's per-axis limit of {}",
                i32::MAX
            )));
        }
    }
    let ctx = img.context().clone();
    let filter = params.filter as i32;

    let func = ctx.function("resample_kernel", PTX)?;
    let pass = |input: &cudarc::driver::CudaSlice<f32>,
                output: &mut cudarc::driver::CudaSlice<f32>,
                rows: usize,
                in_len: usize,
                out_len: usize,
                axis: i32|
     -> Result<(), PhaiosError> {
        // Same host arithmetic the CPU kernel uses for scale/support.
        let scale = in_len as f32 / out_len as f32;
        let support = crate::geometry::filter_support(params.filter, scale);
        let area_minify = i32::from(params.filter == ResizeFilter::Area && scale > 1.0);
        let (rows_i, in_i, out_i, c_i) = (rows as i32, in_len as i32, out_len as i32, c as i32);
        let n = rows * out_len * c;
        // Nothing to write — a zero-channel or zero-extent image. Launching
        // with gridDim.x == 0 is a driver error, and the CPU returns an
        // empty array here (docs/ffi.md §1).
        if n == 0 {
            return Ok(());
        }
        let mut launch = ctx.stream.launch_builder(&func);
        launch
            .arg(input)
            .arg(output)
            .arg(&rows_i)
            .arg(&in_i)
            .arg(&out_i)
            .arg(&c_i)
            .arg(&scale)
            .arg(&support)
            .arg(&filter)
            .arg(&area_minify)
            .arg(&axis);
        // Safety: signature matches the .cu; buffers sized rows·len·c;
        // tap indices clamp; the kernel bounds-checks the output.
        unsafe { launch.launch(grid_1d(n)) }.map_err(be("resample launch failed"))?;
        Ok(())
    };

    // Horizontal: (in_h, in_w, c) -> (in_h, out_w, c).
    let mut mid: cudarc::driver::CudaSlice<f32> =
        unsafe { ctx.stream.alloc((in_h * out_w * c).max(1)) }
            .map_err(be("device allocation failed"))?;
    pass(&img.buf, &mut mid, in_h, in_w, out_w, 0)?;

    // Vertical: (in_h, out_w, c) -> (out_h, out_w, c); rows = out_w is
    // both the column count and the row pitch of the C-contiguous mid.
    let mut out = ctx.alloc_image((out_h, out_w, c))?;
    pass(&mid, &mut out.buf, out_w, in_h, out_h, 1)?;
    Ok(out)
}

/// Rotate by a small angle and crop to the inscribed rectangle,
/// device-resident. Mirrors [`crate::geometry::straighten`];
/// **bit-exact** — sin/cos come from the same host computation
/// (`geometry::straighten_geometry`) and the 16-tap
/// Catmull-Rom accumulates in the CPU's exact order.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn straighten_device(
    img: &DeviceImage,
    params: &StraightenParams,
) -> Result<DeviceImage, PhaiosError> {
    let (in_h, in_w, c) = img.shape();
    let (out_h, out_w, sin_a, cos_a) =
        crate::geometry::straighten_geometry(in_h, in_w, params.degrees)?;

    let ctx = img.context().clone();
    let mut out = ctx.alloc_image((out_h, out_w, c))?;
    let n = out_h * out_w * c;
    if n == 0 {
        return Ok(out);
    }
    let (ih, iw, c_i) = (in_h as i32, in_w as i32, c as i32);
    let (oh, ow) = (out_h as i32, out_w as i32);
    let func = ctx.function("straighten_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&ih)
        .arg(&iw)
        .arg(&c_i)
        .arg(&oh)
        .arg(&ow)
        .arg(&sin_a)
        .arg(&cos_a);
    // Safety: signature matches the .cu; taps clamp to the frame; the
    // kernel bounds-checks the output.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("straighten launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`resize_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn resize(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &ResizeParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::geometry::validate_resize(img.shape(), params)?;
    let device = ctx.upload(img)?;
    ctx.download(&resize_device(&device, params)?)
}

/// Per-call offload form of [`straighten_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn straighten(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &StraightenParams,
) -> Result<Array3<f32>, PhaiosError> {
    let device = ctx.upload(img)?;
    ctx.download(&straighten_device(&device, params)?)
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
