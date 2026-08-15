// SPDX-License-Identifier: GPL-3.0-or-later
//! Exposure compensation on the CUDA backend.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;

/// PTX for `exposure_kernel`, compiled by build.rs at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/exposure.ptx"));

/// Apply exposure compensation of `stops` EV, device-resident.
///
/// Bit-identical to [`crate::exposure::exposure`]: the host computes
/// `2^stops` exactly as the CPU kernel does, and the device performs one
/// correctly-rounded IEEE-754 multiply per element. The conformance
/// suite asserts equality with `assert_eq!`, not a tolerance.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn exposure_device(img: &DeviceImage, stops: f32) -> Result<DeviceImage, PhaiosError> {
    crate::exposure::validate(stops)?;

    let ctx = img.context().clone();
    let shape = img.shape();
    let n = shape.0 * shape.1 * shape.2;
    let mut out = ctx.alloc_image(shape)?;
    if n == 0 {
        return Ok(out);
    }

    let gain = 2.0_f32.powf(stops);
    let n_ll = n as i64;
    let func = ctx.function("exposure_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch.arg(&img.buf).arg(&mut out.buf).arg(&gain).arg(&n_ll);
    // Safety: signature matches the .cu declaration; both buffers hold
    // exactly `n` elements and the kernel bounds-checks against `n`.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form: upload, run, download. Accepts any layout.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn exposure(
    ctx: &Context,
    img: ArrayView3<f32>,
    stops: f32,
) -> Result<Array3<f32>, PhaiosError> {
    crate::exposure::validate(stops)?;
    let device = ctx.upload(img)?;
    let result = exposure_device(&device, stops)?;
    ctx.download(&result)
}
