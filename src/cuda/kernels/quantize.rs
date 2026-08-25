// SPDX-License-Identifier: GPL-3.0-or-later
//! Dithered quantisation on the device.
//!
//! The only kernels whose output is not an image. Quantisation is
//! terminal — the codes it produces go into a file, not into another
//! kernel — so there is no device-resident form to chain: these take a
//! [`DeviceImage`] and return a host array, and that download *is* the
//! end of the pipeline.
//!
//! **Bit-exact** against the CPU kernels. The dither is exact 64-bit
//! integer arithmetic (the same splitmix64 the grain kernel uses) and
//! the rounding is `floor(v + 0.5)`, built from an exact operation and a
//! correctly-rounded one rather than a library rounding routine.

use cudarc::driver::{CudaSlice, PushKernelArg};
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::quantize::{Dither, QuantizeParams};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/quantize.ptx"));

/// Quantise a device-resident image to 8-bit codes, returning a host
/// array. Mirrors [`crate::quantize::quantize_u8`]; bit-exact.
///
/// # Errors
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns the quantised codes; ignoring them wastes work"]
pub fn quantize_u8_device(
    img: &DeviceImage,
    params: &QuantizeParams,
) -> Result<Array3<u8>, PhaiosError> {
    let (h, w, c) = img.shape();
    let n = h * w * c;
    let mut out = Array3::<u8>::zeros((h, w, c));
    if n == 0 {
        return Ok(out);
    }

    let ctx = img.context().clone();
    let mut d_out: CudaSlice<u8> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;

    let (w_i, c_i) = (w as i32, c as i32);
    let dithered = i32::from(params.dither == Dither::Tpdf);
    let seed = params.seed;
    let n_ll = n as i64;

    let func = ctx.function("quantize_u8_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut d_out)
        .arg(&w_i)
        .arg(&c_i)
        .arg(&seed)
        .arg(&dithered)
        .arg(&n_ll);
    // Safety: signature matches the .cu; the input holds n floats and the
    // output n bytes, and the kernel bounds-checks against n.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;

    let host = ctx
        .stream
        .clone_dtoh(&d_out)
        .map_err(be("download failed"))?;
    ctx.stream.synchronize().map_err(be("synchronize failed"))?;
    out.as_slice_mut()
        .expect("freshly allocated Array3 is contiguous")
        .copy_from_slice(&host);
    Ok(out)
}

/// Quantise a device-resident image to 16-bit codes, returning a host
/// array. Mirrors [`crate::quantize::quantize_u16`]; bit-exact.
///
/// # Errors
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns the quantised codes; ignoring them wastes work"]
pub fn quantize_u16_device(
    img: &DeviceImage,
    params: &QuantizeParams,
) -> Result<Array3<u16>, PhaiosError> {
    let (h, w, c) = img.shape();
    let n = h * w * c;
    let mut out = Array3::<u16>::zeros((h, w, c));
    if n == 0 {
        return Ok(out);
    }

    let ctx = img.context().clone();
    let mut d_out: CudaSlice<u16> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;

    let (w_i, c_i) = (w as i32, c as i32);
    let dithered = i32::from(params.dither == Dither::Tpdf);
    let seed = params.seed;
    let n_ll = n as i64;

    let func = ctx.function("quantize_u16_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut d_out)
        .arg(&w_i)
        .arg(&c_i)
        .arg(&seed)
        .arg(&dithered)
        .arg(&n_ll);
    // Safety: signature matches the .cu; the input holds n floats and the
    // output n u16s, and the kernel bounds-checks against n.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;

    let host = ctx
        .stream
        .clone_dtoh(&d_out)
        .map_err(be("download failed"))?;
    ctx.stream.synchronize().map_err(be("synchronize failed"))?;
    out.as_slice_mut()
        .expect("freshly allocated Array3 is contiguous")
        .copy_from_slice(&host);
    Ok(out)
}

/// Per-call offload form of [`quantize_u8_device`].
#[must_use = "kernel returns the quantised codes; ignoring them wastes work"]
pub fn quantize_u8(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &QuantizeParams,
) -> Result<Array3<u8>, PhaiosError> {
    let device = ctx.upload(img)?;
    quantize_u8_device(&device, params)
}

/// Per-call offload form of [`quantize_u16_device`].
#[must_use = "kernel returns the quantised codes; ignoring them wastes work"]
pub fn quantize_u16(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &QuantizeParams,
) -> Result<Array3<u16>, PhaiosError> {
    let device = ctx.upload(img)?;
    quantize_u16_device(&device, params)
}
