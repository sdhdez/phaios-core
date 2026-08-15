// SPDX-License-Identifier: GPL-3.0-or-later
//! Exposure compensation on the CUDA backend.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use ndarray::{Array3, ArrayView3};

use crate::cuda::context::Context;
use crate::error::PhaiosError;

/// PTX for `exposure_kernel`, compiled by build.rs at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/exposure.ptx"));

/// Apply exposure compensation of `stops` EV on the GPU.
///
/// Bit-identical to [`crate::exposure::exposure`]: the host computes
/// `2^stops` exactly as the CPU kernel does, and the device performs one
/// correctly-rounded IEEE-754 multiply per element — the same operation
/// the CPU performs. The conformance suite asserts equality with
/// `assert_eq!`, not a tolerance.
///
/// Per-call offload (upload, compute, download); the device-resident
/// API arrives in a later stage. Accepts any input layout, like every
/// kernel in this crate.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `stops` is not finite — the identical
///   check, message included, as the CPU kernel.
/// - [`PhaiosError::Backend`] if a device operation fails.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn exposure(
    ctx: &Context,
    img: ArrayView3<f32>,
    stops: f32,
) -> Result<Array3<f32>, PhaiosError> {
    // Same validation, same error text, as the CPU path.
    crate::exposure::validate(stops)?;

    let dim = img.dim();
    let n = img.len();
    let mut out = Array3::<f32>::zeros(dim);
    if n == 0 {
        return Ok(out);
    }

    // Stage into contiguous memory. For C-contiguous input this borrows;
    // for strided/Fortran input it copies — the same Zip-walk any other
    // kernel would do.
    let staged = img.as_standard_layout();
    let host_in = staged
        .as_slice()
        .expect("as_standard_layout output is contiguous by construction");

    let gain = 2.0_f32.powf(stops);
    let func = ctx.function("exposure_kernel", PTX)?;

    let d_in = ctx
        .stream
        .clone_htod(host_in)
        .map_err(|e| PhaiosError::Backend(format!("upload failed: {e}")))?;
    // Safety: uninitialised device memory; the kernel writes every
    // element `< n` and the buffer holds exactly `n`, so all of it is
    // written before the download reads it.
    let mut d_out = unsafe { ctx.stream.alloc::<f32>(n) }
        .map_err(|e| PhaiosError::Backend(format!("device allocation failed: {e}")))?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_ll = n as i64;
    let mut launch = ctx.stream.launch_builder(&func);
    launch.arg(&d_in).arg(&mut d_out).arg(&gain).arg(&n_ll);
    // Safety: the kernel signature is (const float*, float*, float,
    // long long), matched by the four args above; both buffers hold
    // exactly `n` elements and the kernel bounds-checks against `n`.
    unsafe { launch.launch(cfg) }
        .map_err(|e| PhaiosError::Backend(format!("kernel launch failed: {e}")))?;

    // Download straight into the output array — no intermediate Vec,
    // no second host-side copy. The dtoh copy is stream-async, so the
    // synchronize below is what makes the host data valid to read.
    ctx.stream
        .memcpy_dtoh(
            &d_out,
            out.as_slice_mut()
                .expect("freshly allocated Array3 is contiguous"),
        )
        .map_err(|e| PhaiosError::Backend(format!("download failed: {e}")))?;
    ctx.stream
        .synchronize()
        .map_err(|e| PhaiosError::Backend(format!("synchronize failed: {e}")))?;
    Ok(out)
}
