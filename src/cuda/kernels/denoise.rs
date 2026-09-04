// SPDX-License-Identifier: GPL-3.0-or-later
//! Guided-filter denoise on the CUDA backend. Mirrors
//! [`crate::denoise::denoise`]: self-guided per channel (`C == 1`, and
//! every other channel count), cross-guided from a shared luminance
//! guide at `C == 3`.
//!
//! **Self-guided, `C == 1`.** Zero new device kernels: a direct call to
//! [`local_contrast_device`] with `strength = -amount`. The same
//! exact-negation argument [`crate::denoise`]'s module documentation
//! gives for the CPU kernel applies bit for bit here too, since both
//! device kernels round the very same sequence of operations —
//! [`denoise_device_c1_is_bit_exact_with_local_contrast_device`] in
//! `tests/cuda_conformance.rs` pins it empirically.
//!
//! **Self-guided, every other `C`.** `local_contrast_device` takes a
//! whole `(H, W, 1)` [`DeviceImage`], but a channel of an interleaved
//! `(H, W, C)` buffer is not one — and cudarc's device-to-device copy is
//! linear only (checked against
//! `cudarc::driver::result::memcpy_dtod_sync`'s signature: no strided or
//! 2-D form), so there is no way to carve one out on the device without
//! a new kernel. No such kernel is authorised for this rare path (`C`
//! this crate's own pipeline never actually produces — every real image
//! is `C == 1` or `C == 3`), so this branch downloads the image once,
//! slices each channel host-side exactly as
//! [`crate::denoise::self_guided`] does, re-uploads it as its own
//! `(H, W, 1)` image and calls `local_contrast_device` on it — the
//! result is byte-for-byte what that function produces per channel, by
//! construction, just not fully device-resident internally. `C == 1`
//! and `C == 3` never take this branch.
//!
//! **Cross-guided, `C == 3`.** Three new kernels
//! (`src/cuda/ptx/denoise.cu`): `box_h_cross` and `coeff_ab_cross`
//! generalise `box_h_l_l2`/`coeff_ab` from one array to a guide/channel
//! pair; `final_out_cross` generalises `final_out` the same way. Reused
//! unchanged: `luminance_bw_kernel` (the guide), `box_h_l_l2` (the
//! guide's shared row sums, called once), `box_h_ab` (the a/b row sums,
//! called once per channel). Total per call: 1 (luminance) + 1
//! (shared `box_h_l_l2`) + 3 × (`box_h_cross` + `coeff_ab_cross` +
//! `box_h_ab` + `final_out_cross`) = **14 launches**, against 4 for
//! `local_contrast` alone.
//!
//! Agreement class for both dispatched paths: `local_contrast`'s own,
//! **(rtol 1e-4, atol 1e-6)** — confirmed empirically for `C == 1`
//! (inherited automatically, since it is the same device kernel) and
//! for `C == 3` (a new cancelling subtraction, `cov(I, p_c)`, that the
//! self-guided path never exercises) by the sweeps in
//! `tests/cuda_conformance.rs`.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{Array3, ArrayView3};

use super::{be, local_contrast_device, luminance_bw_device};
use crate::bw::LuminanceStandard;
use crate::cuda::context::{Context, DeviceImage};
use crate::denoise::DenoiseParams;
use crate::error::PhaiosError;
use crate::local_contrast::GuidedFilterParams;

/// PTX for the three new cross-guided kernels, compiled at `compute_80`.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/denoise.ptx"));

/// PTX for the two reused `local_contrast` kernels (`box_h_l_l2`,
/// `box_h_ab`). A second `include_str!` of the same source file rather
/// than a shared constant: `src/cuda/kernels/local_contrast.rs` is
/// off-limits for this kernel (its own `PTX` constant is private to
/// that module), so this is the only way to reuse those two kernels
/// without editing it. `Context::function` caches modules by the
/// embedded string's address, not its content, so this loads the same
/// PTX text into a second cache entry rather than colliding with
/// `local_contrast.rs`'s own — extra JIT work the first time a context
/// uses both, never a correctness issue.
const PTX_LOCAL_CONTRAST: &str = include_str!(concat!(env!("OUT_DIR"), "/local_contrast.ptx"));

/// 2-D launch geometry: 16×16 threads per block over the image plane.
/// Duplicated from `local_contrast.rs` (which does not expose it)
/// rather than shared — the box-sum kernels this file launches,
/// reused and new alike, share its indexing exactly.
fn grid_2d(h: usize, w: usize) -> LaunchConfig {
    const B: u32 = 16;
    LaunchConfig {
        grid_dim: ((w as u32).div_ceil(B), (h as u32).div_ceil(B), 1),
        block_dim: (B, B, 1),
        shared_mem_bytes: 0,
    }
}

/// Denoise with the He–Sun–Tang guided filter, device-resident. Mirrors
/// [`crate::denoise::denoise`]; see the module documentation for the
/// three dispatch branches.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn denoise_device(
    img: &DeviceImage,
    params: &DenoiseParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::denoise::validate(params)?;

    let (h, w, c) = img.shape();
    let ctx = img.context().clone();
    let n = h * w * c;

    if n == 0 || params.amount == 0.0 {
        // Identity fast path -- bit-exact device copy, mirrors
        // `sharpen_device`'s own `amount == 0.0` fast path. Correct even
        // though the CPU kernel takes no such shortcut: the CPU's
        // combine reduces to an exact identity at `amount = 0.0`
        // through ordinary IEEE-754 cancellation
        // (`p + (-0.0)*(p-q) == p`, see the module doc's exact-negation
        // argument), so a plain copy here produces the same bits by a
        // different, cheaper route.
        let mut out = ctx.alloc_image((h, w, c))?;
        if n > 0 {
            ctx.stream
                .memcpy_dtod(&img.buf, &mut out.buf)
                .map_err(be("device copy failed"))?;
        }
        return Ok(out);
    }

    let eps = params.noise_sigma * params.noise_sigma;
    let neg_amount = -params.amount;

    if c == 3 {
        return cross_guided_device(img, params.radius, eps, neg_amount, params.standard);
    }

    let gf = GuidedFilterParams::new(params.radius, eps);
    if c == 1 {
        return local_contrast_device(img, &gf, neg_amount);
    }

    // Every other channel count: self-guided per channel, via a host
    // round trip -- see the module documentation for why.
    let host = ctx.download(img)?;
    let mut out_host = crate::alloc::zeros3::<f32>((h, w, c))?;
    for ch in 0..c {
        let channel = host.slice(ndarray::s![.., .., ch..ch + 1]);
        let channel_device = ctx.upload(channel)?;
        let filtered = local_contrast_device(&channel_device, &gf, neg_amount)?;
        let filtered_host = ctx.download(&filtered)?;
        out_host
            .slice_mut(ndarray::s![.., .., ch..ch + 1])
            .assign(&filtered_host);
    }
    ctx.upload(out_host.view())
}

/// Cross-guided path (`C == 3`), device-resident: the 14-launch sequence
/// the module documentation describes. `img` is known to be `(H, W, 3)`
/// and `radius`/`eps`/`neg_amount` already derived — see
/// [`denoise_device`].
fn cross_guided_device(
    img: &DeviceImage,
    radius: u32,
    eps: f32,
    neg_amount: f32,
    standard: LuminanceStandard,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, _) = img.shape();
    let ctx = img.context().clone();
    let n = h * w;
    let mut out = ctx.alloc_image((h, w, 3))?;

    // Radius beyond the image is legal (windows clamp), and clamping it
    // here also keeps the i32 kernel parameter in range -- mirrors
    // `local_contrast_device`'s own clamp exactly.
    let r = radius.min(h.max(w) as u32) as i32;
    let (h_i, w_i, nc_i) = (h as i32, w as i32, 3_i32);
    let cfg = grid_2d(h, w);

    // Launch 1: the shared guide.
    let guide = luminance_bw_device(img, standard)?;

    // Launch 2: shared row sums of I, I^2 -- box_h_l_l2, reused
    // unchanged, queried once per channel below.
    let mut hsum_i: CudaSlice<f64> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    let mut hsum_i2: CudaSlice<f64> =
        unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
    {
        let f = ctx.function("box_h_l_l2", PTX_LOCAL_CONTRAST)?;
        let mut launch = ctx.stream.launch_builder(&f);
        launch
            .arg(&guide.buf)
            .arg(&mut hsum_i)
            .arg(&mut hsum_i2)
            .arg(&h_i)
            .arg(&w_i)
            .arg(&r);
        // Safety: signature matches box_h_l_l2's .cu declaration; every
        // buffer holds exactly h*w elements, bounds-checked on (h, w).
        unsafe { launch.launch(cfg) }.map_err(be("box_h_l_l2 launch failed"))?;
    }

    for ch in 0..3_i32 {
        // Launches 3, 7, 11: box_h_cross -- row sums of p_c, I*p_c, read
        // straight out of the interleaved input buffer.
        let mut hsum_p: CudaSlice<f64> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        let mut hsum_ip: CudaSlice<f64> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        {
            let f = ctx.function("box_h_cross", PTX)?;
            let mut launch = ctx.stream.launch_builder(&f);
            launch
                .arg(&guide.buf)
                .arg(&img.buf)
                .arg(&ch)
                .arg(&nc_i)
                .arg(&mut hsum_p)
                .arg(&mut hsum_ip)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&r);
            // Safety: signature matches box_h_cross's .cu declaration;
            // `img.buf` holds 3*h*w elements and every read is
            // `channel + i*n_channels` for `i` bounds-checked on `w`.
            unsafe { launch.launch(cfg) }.map_err(be("box_h_cross launch failed"))?;
        }

        // Launches 4, 8, 12: coeff_ab_cross -- a_c, b_c.
        let mut a_arr: CudaSlice<f32> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        let mut b_arr: CudaSlice<f32> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        {
            let f = ctx.function("coeff_ab_cross", PTX)?;
            let mut launch = ctx.stream.launch_builder(&f);
            launch
                .arg(&hsum_i)
                .arg(&hsum_i2)
                .arg(&hsum_p)
                .arg(&hsum_ip)
                .arg(&mut a_arr)
                .arg(&mut b_arr)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&r)
                .arg(&eps);
            // Safety: signature matches coeff_ab_cross's .cu
            // declaration; every buffer holds h*w elements.
            unsafe { launch.launch(cfg) }.map_err(be("coeff_ab_cross launch failed"))?;
        }

        // Launches 5, 9, 13: box_h_ab -- row sums of a_c, b_c, reused
        // unchanged.
        let mut hsum_a: CudaSlice<f32> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        let mut hsum_b: CudaSlice<f32> =
            unsafe { ctx.stream.alloc(n) }.map_err(be("device allocation failed"))?;
        {
            let f = ctx.function("box_h_ab", PTX_LOCAL_CONTRAST)?;
            let mut launch = ctx.stream.launch_builder(&f);
            launch
                .arg(&a_arr)
                .arg(&b_arr)
                .arg(&mut hsum_a)
                .arg(&mut hsum_b)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&r);
            // Safety: signature matches box_h_ab's .cu declaration;
            // every buffer holds h*w elements.
            unsafe { launch.launch(cfg) }.map_err(be("box_h_ab launch failed"))?;
        }

        // Launches 6, 10, 14: final_out_cross -- q_c, then out_c,
        // written straight into the interleaved output buffer.
        {
            let f = ctx.function("final_out_cross", PTX)?;
            let mut launch = ctx.stream.launch_builder(&f);
            launch
                .arg(&hsum_a)
                .arg(&hsum_b)
                .arg(&guide.buf)
                .arg(&img.buf)
                .arg(&mut out.buf)
                .arg(&ch)
                .arg(&nc_i)
                .arg(&h_i)
                .arg(&w_i)
                .arg(&r)
                .arg(&neg_amount);
            // Safety: signature matches final_out_cross's .cu
            // declaration; `out.buf` holds 3*h*w elements and every
            // write is `channel + pix*n_channels` for `pix` bounds
            // checked on (h, w).
            unsafe { launch.launch(cfg) }.map_err(be("final_out_cross launch failed"))?;
        }
    }

    Ok(out)
}

/// Per-call offload form of [`denoise_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn denoise(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &DenoiseParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::denoise::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&denoise_device(&device, params)?)
}
