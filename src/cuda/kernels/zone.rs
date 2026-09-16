// SPDX-License-Identifier: GPL-3.0-or-later
//! Adams/Archer Zone System on the CUDA backend.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::tone::ZoneParams;

/// PTX for `zone_system_kernel`, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/zone_system.ptx"));

/// Zone System tone curve, device-resident.
///
/// Mirrors [`crate::tone::zone_system`]. The offsets cross to the
/// device as a dense 11-entry array in zone order; adding an exact
/// `+0.0` term is the IEEE-754 identity, so the dense ascending loop
/// computes bit-for-bit the same sum as the CPU's sparse sorted
/// iteration — the ordered-reduction rule ports mechanically.
/// `log2f`/`expf`/`powf` are implementation-defined; agreement is
/// bounded (rtol 1e-5, atol 1e-7). The empty-offsets identity path is
/// a device copy, exactly as the CPU assigns its input through.
///
/// # Errors
/// Same [`PhaiosError::Shape`] / [`PhaiosError::Parameter`] as the CPU
/// kernel; plus [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn zone_system_device(
    img: &DeviceImage,
    params: &ZoneParams,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    crate::tone::validate_zones(&[h, w, c], params)?;

    let ctx = img.context().clone();

    if params.is_identity() {
        // Mirror the CPU identity path with a device copy.
        let mut out = ctx.alloc_image((h, w, c))?;
        if h * w > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let n = h * w;
    let mut out = ctx.alloc_image((h, w, 1))?;
    if n == 0 {
        return Ok(out);
    }

    let dense = params.dense_offsets();
    let d_offsets = ctx
        .stream
        .clone_htod(&dense[..])
        .map_err(be("offsets upload failed"))?;
    let n_ll = n as i64;

    let func = ctx.function("zone_system_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&d_offsets)
        .arg(&n_ll);
    // Safety: signature matches the .cu; buffers hold n (offsets 11);
    // the kernel bounds-checks against n and reads offsets 0..=10.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`zone_system_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn zone_system(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &ZoneParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::tone::validate_zones(img.shape(), params)?;
    let device = ctx.upload(img)?;
    ctx.download(&zone_system_device(&device, params)?)
}
