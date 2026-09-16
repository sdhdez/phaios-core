// SPDX-License-Identifier: GPL-3.0-or-later
//! Split-toning on the CUDA backend.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::split_toning::SplitToningParams;

/// PTX for `split_toning_kernel`, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/split_toning.ptx"));

/// Split-toning, device-resident: `(H, W, 1)` → `(H, W, 3)` — the
/// shape-restoring kernel, on the GPU as on the CPU.
///
/// Mirrors [`crate::split_toning::split_toning`]. The crossfade window
/// and the neutral row-sum are computed by the same host code the CPU
/// kernel uses (`crossfade_edges`, `NEUTRAL_ROW_SUM`), so `cbrtf` is
/// the only implementation-defined operation in the chain; agreement
/// is bounded (rtol 1e-5, atol 1e-7).
///
/// # Errors
/// Same [`PhaiosError::Shape`] / [`PhaiosError::Parameter`] as the CPU
/// kernel; plus [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn split_toning_device(
    img: &DeviceImage,
    params: &SplitToningParams,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    crate::split_toning::validate(&[h, w, c], params)?;

    let ctx = img.context().clone();
    let npix = h * w;
    let mut out = ctx.alloc_image((h, w, 3))?;
    if npix == 0 {
        return Ok(out);
    }

    let (edge0, edge1) = crate::split_toning::crossfade_edges(params);
    let row_sum = crate::split_toning::NEUTRAL_ROW_SUM;
    let [_, shadow_a, shadow_b] = params.shadow_oklab;
    let [_, highlight_a, highlight_b] = params.highlight_oklab;
    let n_ll = npix as i64;

    let func = ctx.function("split_toning_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&row_sum)
        .arg(&edge0)
        .arg(&edge1)
        .arg(&shadow_a)
        .arg(&shadow_b)
        .arg(&highlight_a)
        .arg(&highlight_b)
        .arg(&n_ll);
    // Safety: signature matches the .cu; input npix, output 3·npix,
    // bounds-checked against npix.
    unsafe { launch.launch(grid_1d(npix)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`split_toning_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn split_toning(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &SplitToningParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::split_toning::validate(img.shape(), params)?;
    let device = ctx.upload(img)?;
    ctx.download(&split_toning_device(&device, params)?)
}
