// SPDX-License-Identifier: GPL-3.0-or-later
//! Unsharp masking on the device: one pointwise kernel after the shared
//! device blur.
//!
//! [`crate::glow`]'s element-wise work brackets its blur -- a weight
//! kernel before, an add kernel after. `sharpen`'s blur runs on `img`
//! directly, so the subtract/gate/combine work all happens after it and
//! fits **one** pointwise launch, not two. The blur itself is
//! [`super::blur_device`], so there is exactly one Gaussian
//! implementation on the device and this module cannot drift from it.
//!
//! Bounded, not bit-exact -- inherited entirely from the blur. The
//! pointwise kernel is bit-exact; see `src/cuda/ptx/sharpen.cu`.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, blur_device, grid_1d};
use crate::blur::{BlurParams, BlurShape};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::sharpen::SharpenParams;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/sharpen.ptx"));

/// Threshold-gated Gaussian unsharp mask, device-resident. Mirrors
/// [`crate::sharpen::sharpen`].
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn sharpen_device(
    img: &DeviceImage,
    params: &SharpenParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::sharpen::validate(params)?;

    let ctx = img.context().clone();
    let (h, w, c) = img.shape();
    let n = h * w * c;

    if params.amount == 0.0 || params.sigma == 0.0 || n == 0 {
        // Identity, mirroring the CPU's fresh-copy fast path. Kept
        // despite `blur_device` already special-casing `sigma == 0.0`
        // on its own: it skips an unneeded launch and keeps this entry
        // point bit-exact without relying on the callee's fast path.
        let mut out = ctx.alloc_image((h, w, c))?;
        if n > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let n_ll = n as i64;

    // The blur `detail` is measured against -- the crate's one device
    // Gaussian, reused exactly as glow_device reuses it.
    let blurred = blur_device(img, &BlurParams::new(params.sigma, BlurShape::Gaussian))?;

    // Subtract, gate and combine -- one pointwise kernel, since nothing
    // here brackets the blur the way glow's weight kernel does.
    let mut out = ctx.alloc_image((h, w, c))?;
    {
        let amount = params.amount;
        let threshold = params.threshold;
        let func = ctx.function("sharpen_apply_kernel", PTX)?;
        let mut launch = ctx.stream.launch_builder(&func);
        launch
            .arg(&img.buf)
            .arg(&blurred.buf)
            .arg(&mut out.buf)
            .arg(&amount)
            .arg(&threshold)
            .arg(&n_ll);
        // Safety: signature matches the .cu; all three buffers hold n
        // elements and the kernel bounds-checks against n.
        unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    }

    Ok(out)
}

/// Per-call offload form of [`sharpen_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn sharpen(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &SharpenParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::sharpen::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&sharpen_device(&device, params)?)
}
