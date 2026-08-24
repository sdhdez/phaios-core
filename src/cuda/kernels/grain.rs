// SPDX-License-Identifier: GPL-3.0-or-later
//! Procedural film grain on the CUDA backend.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::film_grain::GrainParams;

/// PTX for the grain kernels, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/film_grain.ptx"));

/// 2-D launch geometry, as in the guided filter.
fn grid_2d(h: usize, w: usize) -> LaunchConfig {
    const B: u32 = 16;
    LaunchConfig {
        grid_dim: ((w as u32).div_ceil(B), (h as u32).div_ceil(B), 1),
        block_dim: (B, B, 1),
        shared_mem_bytes: 0,
    }
}

/// Add procedural film grain, device-resident.
///
/// Mirrors [`crate::film_grain::film_grain`]. The integer half —
/// `splitmix64` over `(seed, x, y)` — is native 64-bit arithmetic on
/// the device and **bit-exact** against the CPU, asserted over 2²⁰
/// coordinates by the conformance suite. The Box–Muller half (`logf`,
/// `sqrtf`, `cosf`) is implementation-defined and cannot be; the
/// band-pass reuses the separable Kahan-compensated box filters proven
/// in `local_contrast`, and the radii and analytic normalisation come
/// from the same host function the CPU kernel calls
/// (`film_grain::bandpass_geometry`). Agreement with the CPU
/// oracle is bounded; within-backend output is bit-reproducible.
///
/// # Errors
/// Same [`PhaiosError::Shape`] / [`PhaiosError::Parameter`] as the CPU
/// kernel; plus [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn film_grain_device(
    img: &DeviceImage,
    params: &GrainParams,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    crate::film_grain::validate(&[h, w, c], params)?;

    let ctx = img.context().clone();

    // Identity path, mirroring the CPU (intensity == 0 or empty image).
    if params.intensity == 0.0 || h == 0 || w == 0 {
        let mut out = ctx.alloc_image((h, w, c))?;
        if h * w > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let n = h * w;
    let (inner, outer, normalisation) = crate::film_grain::bandpass_geometry(params.size_pixels);
    let (r_in, r_out) = (inner as i32, outer as i32);
    let (h_i, w_i) = (h as i32, w as i32);
    let cfg = grid_2d(h, w);

    // Safety (all three): uninitialised buffers, fully written by the
    // producing kernel before any consumer reads them; the stream
    // serialises the passes.
    let mut noise: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut hsum_in: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut hsum_out: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut out = ctx.alloc_image((h, w, 1))?;

    let seed = params.seed;
    let f = ctx.function("grain_noise", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch.arg(&mut noise).arg(&seed).arg(&h_i).arg(&w_i);
    // Safety: signatures match the .cu declarations; all buffers hold
    // h*w elements and every kernel bounds-checks on (h, w).
    unsafe { launch.launch(cfg) }.map_err(be("grain_noise launch failed"))?;

    let f = ctx.function("grain_h", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&noise)
        .arg(&mut hsum_in)
        .arg(&mut hsum_out)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r_in)
        .arg(&r_out);
    unsafe { launch.launch(cfg) }.map_err(be("grain_h launch failed"))?;

    let intensity = params.intensity;
    let f = ctx.function("grain_final", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&hsum_in)
        .arg(&hsum_out)
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r_in)
        .arg(&r_out)
        .arg(&intensity)
        .arg(&normalisation);
    unsafe { launch.launch(cfg) }.map_err(be("grain_final launch failed"))?;

    Ok(out)
}

/// Per-call offload form of [`film_grain_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn film_grain(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &GrainParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::film_grain::validate(img.shape(), params)?;
    let device = ctx.upload(img)?;
    ctx.download(&film_grain_device(&device, params)?)
}

/// Raw pixel hashes for a `h × w` grid — the conformance oracle's view
/// of the device-side `pixel_hash`. Not part of the image pipeline.
#[doc(hidden)]
pub fn hash_grid(ctx: &Context, seed: u64, h: usize, w: usize) -> Result<Vec<u64>, PhaiosError> {
    let n = h * w;
    let mut d_out: CudaSlice<u64> =
        unsafe { ctx.stream.alloc(n.max(1)) }.map_err(be("device allocation failed"))?;
    let (w_i, n_ll) = (w as i32, n as i64);
    let f = ctx.function("grain_hash_kernel", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch.arg(&mut d_out).arg(&seed).arg(&w_i).arg(&n_ll);
    // Safety: signature matches the .cu; buffer holds n; bounds-checked.
    unsafe { launch.launch(grid_1d(n.max(1))) }.map_err(be("grain_hash launch failed"))?;
    let host = ctx
        .stream
        .clone_dtoh(&d_out)
        .map_err(be("download failed"))?;
    ctx.stream.synchronize().map_err(be("synchronize failed"))?;
    Ok(host)
}
