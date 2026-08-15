// SPDX-License-Identifier: GPL-3.0-or-later
//! HSL-weighted B&W conversion on the CUDA backend.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::bw::HslWeightedParams;
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;

/// PTX for `hsl_bw_kernel`, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hsl_bw.ptx"));

/// HSL-weighted conversion, device-resident: `(H, W, 3)` → `(H, W, 1)`.
///
/// Mirrors [`crate::bw::hsl_bw`] line for line — hexagonal hue, chroma
/// ratio, circular band distance, fixed-order eight-term Gaussian sum.
/// `expf` is the one implementation-defined operation; agreement is
/// bounded (rtol 1e-5, atol 1e-7) and everything else matches exactly.
///
/// # Errors
/// Same [`PhaiosError::Shape`] / [`PhaiosError::Parameter`] as the CPU
/// kernel; plus [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn hsl_bw_device(
    img: &DeviceImage,
    params: &HslWeightedParams,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    if c != 3 {
        return Err(PhaiosError::Shape(format!(
            "expected (H, W, 3) RGB array, got shape [{h}, {w}, {c}]"
        )));
    }
    crate::bw::validate_hsl(params)?;

    let ctx = img.context().clone();
    let npix = h * w;
    let mut out = ctx.alloc_image((h, w, 1))?;
    if npix == 0 {
        return Ok(out);
    }

    let [wr, wg, wb] = params.standard.weights();
    let hw = params.hue_weights;
    let two_sigma_sq = 2.0 * params.sigma_deg * params.sigma_deg;
    let n_ll = npix as i64;

    let func = ctx.function("hsl_bw_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&wr)
        .arg(&wg)
        .arg(&wb)
        .arg(&hw[0])
        .arg(&hw[1])
        .arg(&hw[2])
        .arg(&hw[3])
        .arg(&hw[4])
        .arg(&hw[5])
        .arg(&hw[6])
        .arg(&hw[7])
        .arg(&two_sigma_sq)
        .arg(&n_ll);
    // Safety: signature matches the .cu; input 3·npix, output npix,
    // bounds-checked against npix.
    unsafe { launch.launch(grid_1d(npix)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`hsl_bw_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn hsl_bw(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &HslWeightedParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::bw::validate_rgb(img)?;
    crate::bw::validate_hsl(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&hsl_bw_device(&device, params)?)
}
