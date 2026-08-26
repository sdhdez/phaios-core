// SPDX-License-Identifier: GPL-3.0-or-later
//! Light scattering on the device: halation, diffusion and veiling
//! glare, which are one operation at three sets of parameters.
//!
//! Composed rather than fused. The weight and the add are element-wise
//! kernels here; the spread between them is [`super::blur_device`], so
//! there is exactly one Gaussian implementation on the device and this
//! module cannot drift from it.
//!
//! Bounded, not bit-exact — inherited entirely from the blur. The two
//! element-wise halves are exact.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, blur_device, grid_1d};
use crate::blur::{BlurParams, BlurShape};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::glow::GlowParams;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/glow.ptx"));

/// Spread light above the threshold and add it back, device-resident.
/// Mirrors [`crate::glow::glow`].
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn glow_device(img: &DeviceImage, params: &GlowParams) -> Result<DeviceImage, PhaiosError> {
    crate::glow::validate(params)?;

    let ctx = img.context().clone();
    let (h, w, c) = img.shape();
    let n = h * w * c;

    if params.amount == 0.0 || n == 0 {
        // Identity, mirroring the CPU's fresh-copy fast path.
        let mut out = ctx.alloc_image((h, w, c))?;
        if n > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let n_ll = n as i64;

    // 1. The light that scatters.
    let mut weight = ctx.alloc_image((h, w, c))?;
    {
        let threshold = params.threshold;
        let func = ctx.function("glow_weight_kernel", PTX)?;
        let mut launch = ctx.stream.launch_builder(&func);
        launch
            .arg(&img.buf)
            .arg(&mut weight.buf)
            .arg(&threshold)
            .arg(&n_ll);
        // Safety: signature matches the .cu; both buffers hold n elements
        // and the kernel bounds-checks against n.
        unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    }

    // 2. Spread it — the crate's one device Gaussian.
    let spread = blur_device(&weight, &BlurParams::new(params.sigma, BlurShape::Gaussian))?;
    drop(weight);

    // 3. Add it back.
    let mut out = ctx.alloc_image((h, w, c))?;
    {
        let amount = params.amount;
        let func = ctx.function("glow_add_kernel", PTX)?;
        let mut launch = ctx.stream.launch_builder(&func);
        launch
            .arg(&img.buf)
            .arg(&spread.buf)
            .arg(&mut out.buf)
            .arg(&amount)
            .arg(&n_ll);
        // Safety: signature matches the .cu; all three buffers hold n
        // elements and the kernel bounds-checks against n.
        unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    }

    Ok(out)
}

/// Per-call offload form of [`glow_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn glow(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &GlowParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::glow::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&glow_device(&device, params)?)
}
