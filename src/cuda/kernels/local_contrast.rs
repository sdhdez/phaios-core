// SPDX-License-Identifier: GPL-3.0-or-later
//! Guided-filter local contrast on the CUDA backend.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{Array3, ArrayView3};

use super::be;
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::local_contrast::GuidedFilterParams;

/// PTX for the four guided-filter kernels, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/local_contrast.ptx"));

/// 2-D launch geometry: 16×16 threads per block over the image plane.
fn grid_2d(h: usize, w: usize) -> LaunchConfig {
    const B: u32 = 16;
    LaunchConfig {
        grid_dim: ((w as u32).div_ceil(B), (h as u32).div_ceil(B), 1),
        block_dim: (B, B, 1),
        shared_mem_bytes: 0,
    }
}

/// Enhance local contrast, device-resident.
///
/// Same algorithm as [`crate::local_contrast::local_contrast`], with one
/// documented reformulation (`docs/ffi.md` §6): the four global **f64**
/// summed-area tables become separable **f32** box filters with
/// Kahan-compensated accumulation, because each window sum then adds at
/// most `2r+1` values per pass instead of feeding on a
/// 24-million-element prefix sum. Every output element is produced by
/// one thread accumulating in a fixed sequential order, so the result
/// is bit-reproducible on any launch geometry. Agreement with the CPU
/// oracle is asserted at rtol 1e-4 / atol 1e-6.
///
/// # Errors
/// Same [`PhaiosError::Shape`] / [`PhaiosError::Parameter`] as the CPU
/// kernel; plus [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn local_contrast_device(
    img: &DeviceImage,
    params: &GuidedFilterParams,
    strength: f32,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    crate::local_contrast::validate(&[h, w, c], params, strength)?;

    let ctx = img.context().clone();
    let n = h * w;
    let mut out = ctx.alloc_image((h, w, 1))?;
    if n == 0 {
        return Ok(out);
    }

    // Radius beyond the image is legal (windows clamp), and clamping it
    // here also keeps the i32 kernel parameter in range.
    let r = params.radius.min(h.max(w) as u32) as i32;
    let (h_i, w_i) = (h as i32, w as i32);
    let cfg = grid_2d(h, w);

    // Safety (all four): uninitialised device buffers, each fully
    // written by the kernel that produces it before any kernel reads
    // it; the stream serialises the passes.
    // f64: these carry L and L² partial sums, whose magnitudes and whose
    // cancelling difference an f32 buffer cannot hold. See the .cu.
    let mut buf_1: CudaSlice<f64> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut buf_2: CudaSlice<f64> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut buf_a: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut buf_b: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    // The a/b window sums stay f32 — they feed no cancelling difference —
    // so they need their own buffers rather than reusing the f64 pair.
    let mut buf_c: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut buf_d: CudaSlice<f32> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;

    // Pass 1: row-window sums of L and L².
    let f = ctx.function("box_h_l_l2", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&img.buf)
        .arg(&mut buf_1)
        .arg(&mut buf_2)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r);
    // Safety: signatures match the .cu declarations; every buffer holds
    // exactly h*w elements and the kernels bounds-check on (h, w).
    unsafe { launch.launch(cfg) }.map_err(be("box_h_l_l2 launch failed"))?;

    // Pass 2: column windows -> means -> a, b.
    let f = ctx.function("coeff_ab", PTX)?;
    let eps = params.eps;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&buf_1)
        .arg(&buf_2)
        .arg(&mut buf_a)
        .arg(&mut buf_b)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r)
        .arg(&eps);
    unsafe { launch.launch(cfg) }.map_err(be("coeff_ab launch failed"))?;

    // Pass 3: row-window sums of a and b.
    let f = ctx.function("box_h_ab", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&buf_a)
        .arg(&buf_b)
        .arg(&mut buf_c)
        .arg(&mut buf_d)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r);
    unsafe { launch.launch(cfg) }.map_err(be("box_h_ab launch failed"))?;

    // Pass 4: column windows -> mean_a, mean_b -> q -> output.
    let f = ctx.function("final_out", PTX)?;
    let mut launch = ctx.stream.launch_builder(&f);
    launch
        .arg(&buf_c)
        .arg(&buf_d)
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&r)
        .arg(&strength);
    unsafe { launch.launch(cfg) }.map_err(be("final_out launch failed"))?;

    Ok(out)
}

/// Per-call offload form of [`local_contrast_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn local_contrast(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &GuidedFilterParams,
    strength: f32,
) -> Result<Array3<f32>, PhaiosError> {
    crate::local_contrast::validate(img.shape(), params, strength)?;
    let device = ctx.upload(img)?;
    ctx.download(&local_contrast_device(&device, params, strength)?)
}
