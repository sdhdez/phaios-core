// SPDX-License-Identifier: GPL-3.0-or-later
//! Separable Gaussian blur on the device.
//!
//! Two paths, chosen by the same threshold the CPU uses: direct
//! convolution below σ = [`crate::blur::BOX_CROSSOVER_SIGMA`], three box
//! passes at or above it. The kernel *weights* and the box *widths* are
//! both computed by the shared host code in [`crate::blur`], so the two
//! backends can never disagree about which filter they are applying —
//! only about the order in which they sum it.
//!
//! **Bounded, not bit-exact.** The CPU accumulates in f64; a consumer
//! card runs f64 at 1/64 rate, so the device accumulates in f32 with
//! Kahan compensation instead. Same trade as `local_contrast`, same
//! reasoning, and the bound is committed in `docs/ffi.md` §6.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::blur::{BOX_CROSSOVER_SIGMA, BlurParams, BlurShape};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/blur.ptx"));

/// Run one separable pass over `src` into a fresh image, along `axis`.
fn pass(
    ctx: &Context,
    src: &DeviceImage,
    axis: i32,
    kind: &Pass,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = src.shape();
    let mut out = ctx.alloc_image((h, w, c))?;
    let n = h * w * c;
    if n == 0 {
        return Ok(out);
    }

    // axis 0 filters along width, axis 1 along height.
    let (rows, len) = if axis == 0 { (h, w) } else { (w, h) };
    let (rows_i, len_i, c_i) = (rows as i32, len as i32, c as i32);

    match kind {
        Pass::Conv(weights) => {
            let taps = i32::try_from(weights.len()).map_err(|_| {
                PhaiosError::Parameter(format!("blur kernel has {} taps", weights.len()))
            })?;
            let d_w = ctx
                .stream
                .clone_htod(weights)
                .map_err(be("weight upload failed"))?;
            let n_ll = n as i64;
            let func = ctx.function("blur_conv_kernel", PTX)?;
            let mut launch = ctx.stream.launch_builder(&func);
            launch
                .arg(&src.buf)
                .arg(&mut out.buf)
                .arg(&d_w)
                .arg(&taps)
                .arg(&rows_i)
                .arg(&len_i)
                .arg(&c_i)
                .arg(&axis)
                .arg(&n_ll);
            // Safety: signature matches the .cu; both images hold n
            // elements, the weight buffer holds `taps`, and every tap
            // index is clamped into [0, len-1] before use.
            unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
        }
        Pass::Box(radius) => {
            let r = i32::try_from(*radius)
                .map_err(|_| PhaiosError::Parameter(format!("box radius {radius} too large")))?;
            // One thread per (row, channel) lane, not per element: the
            // sliding sum is serial along a lane.
            let lanes = rows * c;
            let lanes_ll = lanes as i64;
            let func = ctx.function("blur_box_kernel", PTX)?;
            let mut launch = ctx.stream.launch_builder(&func);
            launch
                .arg(&src.buf)
                .arg(&mut out.buf)
                .arg(&r)
                .arg(&rows_i)
                .arg(&len_i)
                .arg(&c_i)
                .arg(&axis)
                .arg(&lanes_ll);
            // Safety: signature matches the .cu; each lane writes exactly
            // `len` elements of the output, all within its own row, and
            // every read index is clamped into [0, len-1].
            unsafe { launch.launch(grid_1d(lanes)) }.map_err(be("kernel launch failed"))?;
        }
    }
    Ok(out)
}

enum Pass {
    Conv(Vec<f32>),
    Box(usize),
}

/// Gaussian blur, device-resident. Mirrors [`crate::blur::blur`];
/// agreement is bounded, not bit-exact — see the module documentation.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn blur_device(img: &DeviceImage, params: &BlurParams) -> Result<DeviceImage, PhaiosError> {
    crate::blur::validate(params)?;

    let (h, w, c) = img.shape();
    if params.sigma == 0.0 || h == 0 || w == 0 || c == 0 {
        // Identity, mirroring the CPU's fresh-copy fast path.
        let ctx = img.context().clone();
        let mut out = ctx.alloc_image((h, w, c))?;
        let n = h * w * c;
        if n > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let ctx = img.context().clone();
    match params.shape {
        BlurShape::Gaussian => {
            if params.sigma < BOX_CROSSOVER_SIGMA {
                let weights = crate::blur::gaussian_weights(params.sigma);
                let mid = pass(&ctx, img, 0, &Pass::Conv(weights.clone()))?;
                pass(&ctx, &mid, 1, &Pass::Conv(weights))
            } else {
                let widths = crate::blur::box_widths(params.sigma);
                let mut cur: Option<DeviceImage> = None;
                for wdt in widths {
                    let r = (wdt - 1) / 2;
                    let src = cur.as_ref().unwrap_or(img);
                    let mid = pass(&ctx, src, 0, &Pass::Box(r))?;
                    cur = Some(pass(&ctx, &mid, 1, &Pass::Box(r))?);
                }
                Ok(cur.expect("three box passes always run at least once"))
            }
        }
    }
}

/// Per-call offload form of [`blur_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn blur(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &BlurParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::blur::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&blur_device(&device, params)?)
}
