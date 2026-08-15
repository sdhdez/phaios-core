// SPDX-License-Identifier: GPL-3.0-or-later
//! The element-wise finishing kernels: `encode_srgb`, `tone_curve`,
//! `vignette`, and the (H, W, 3) → (H, W, 1) `luminance_bw`.
//!
//! With FMA contraction disabled at PTX compile time (`-fmad=false` in
//! build.rs) and every operation involved correctly rounded (mul, add,
//! div, sqrt, min/max), `vignette`, `luminance_bw` and `tone_curve`'s
//! `power == 1` path are **bit-exact** against their CPU kernels — the
//! conformance suite asserts them with `assert_eq!`. `encode_srgb` and
//! the general `tone_curve` path each contain one `powf`, the only
//! operation IEEE-754 leaves implementation-defined, and are asserted
//! against a committed tolerance instead.

use cudarc::driver::PushKernelArg;
use ndarray::{Array3, ArrayView3};

use super::{be, grid_1d};
use crate::bw::LuminanceStandard;
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::tone::ToneCurveParams;
use crate::vignette::VignetteParams;

const PTX_ENCODE: &str = include_str!(concat!(env!("OUT_DIR"), "/encode_srgb.ptx"));
const PTX_TONE: &str = include_str!(concat!(env!("OUT_DIR"), "/tone_curve.ptx"));
const PTX_VIGNETTE: &str = include_str!(concat!(env!("OUT_DIR"), "/vignette.ptx"));
const PTX_LUMINANCE: &str = include_str!(concat!(env!("OUT_DIR"), "/luminance_bw.ptx"));

/// Copy a device image (used by identity fast paths, so they mirror the
/// CPU's `out.assign(&img)` exactly: fresh output, same bytes).
fn copy_device(img: &DeviceImage) -> Result<DeviceImage, PhaiosError> {
    let ctx = img.context().clone();
    let mut out = ctx.alloc_image(img.shape())?;
    let n = {
        let (h, w, c) = img.shape();
        h * w * c
    };
    if n > 0 {
        ctx.stream
            .memcpy_dtod(&img.buf, &mut out.buf)
            .map_err(be("device copy failed"))?;
    }
    Ok(out)
}

// ── encode_srgb ──────────────────────────────────────────────────────────────

/// IEC 61966-2-1 sRGB transfer, device-resident. Mirrors
/// [`crate::encode::encode_srgb`]; agreement bounded by one `powf`.
///
/// # Errors
/// [`PhaiosError::Backend`] on device failure (the CPU kernel is
/// infallible for every 3-D input, and so is this one).
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn encode_srgb_device(img: &DeviceImage) -> Result<DeviceImage, PhaiosError> {
    let ctx = img.context().clone();
    let shape = img.shape();
    let n = shape.0 * shape.1 * shape.2;
    let mut out = ctx.alloc_image(shape)?;
    if n == 0 {
        return Ok(out);
    }
    // Same f32 constant the CPU uses in `x.powf(1.0 / 2.4)`.
    let inv_gamma = 1.0_f32 / 2.4_f32;
    let n_ll = n as i64;
    let func = ctx.function("encode_srgb_kernel", PTX_ENCODE)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&inv_gamma)
        .arg(&n_ll);
    // Safety: signature matches the .cu; buffers hold n; bounds-checked.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`encode_srgb_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn encode_srgb(ctx: &Context, img: ArrayView3<f32>) -> Result<Array3<f32>, PhaiosError> {
    let device = ctx.upload(img)?;
    ctx.download(&encode_srgb_device(&device)?)
}

// ── tone_curve ───────────────────────────────────────────────────────────────

/// ASC CDL slope/offset/power, device-resident. Mirrors
/// [`crate::tone::tone_curve`], including the identity fast path (a
/// device copy) and the `power == 1` path that skips `powf` — both of
/// which are bit-exact; the general path is bounded by one `powf`.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn tone_curve_device(
    img: &DeviceImage,
    params: &ToneCurveParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::tone::validate_tone_curve(params)?;

    let ToneCurveParams {
        slope,
        offset,
        power,
    } = *params;
    if slope == 1.0 && offset == 0.0 && power == 1.0 {
        return copy_device(img);
    }

    let ctx = img.context().clone();
    let shape = img.shape();
    let n = shape.0 * shape.1 * shape.2;
    let mut out = ctx.alloc_image(shape)?;
    if n == 0 {
        return Ok(out);
    }
    let unit_power = i32::from(power == 1.0);
    let n_ll = n as i64;
    let func = ctx.function("tone_curve_kernel", PTX_TONE)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&slope)
        .arg(&offset)
        .arg(&power)
        .arg(&unit_power)
        .arg(&n_ll);
    // Safety: signature matches the .cu; buffers hold n; bounds-checked.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`tone_curve_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn tone_curve(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &ToneCurveParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::tone::validate_tone_curve(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&tone_curve_device(&device, params)?)
}

// ── vignette ─────────────────────────────────────────────────────────────────

/// Radial vignette, device-resident. Mirrors
/// [`crate::vignette::vignette`] line for line; **bit-exact** against
/// the CPU (every operation involved is correctly rounded).
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn vignette_device(
    img: &DeviceImage,
    params: &VignetteParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::vignette::validate(params)?;

    let (h, w, c) = img.shape();
    // Identity fast path, mirroring the CPU exactly.
    if params.amount == 0.0 || h == 0 || w == 0 {
        return copy_device(img);
    }

    let ctx = img.context().clone();
    let n = h * w * c;
    let mut out = ctx.alloc_image((h, w, c))?;
    let (h_i, w_i, c_i) = (h as i32, w as i32, c as i32);
    let inner = 1.0 - params.feather;
    let amount = params.amount;
    let roundness = params.roundness;
    let func = ctx.function("vignette_kernel", PTX_VIGNETTE)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&h_i)
        .arg(&w_i)
        .arg(&c_i)
        .arg(&amount)
        .arg(&inner)
        .arg(&roundness);
    // Safety: signature matches the .cu; buffers hold n; bounds-checked.
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`vignette_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn vignette(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &VignetteParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::vignette::validate(params)?;
    let device = ctx.upload(img)?;
    ctx.download(&vignette_device(&device, params)?)
}

// ── luminance_bw ─────────────────────────────────────────────────────────────

/// Standard-luminance B&W conversion, device-resident: `(H, W, 3)` in,
/// `(H, W, 1)` out — the first shape-changing kernel on the backend.
/// Mirrors [`crate::bw::luminance_bw`]; **bit-exact** (left-to-right
/// dot product, no FMA contraction).
///
/// # Errors
/// Same [`PhaiosError::Shape`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn luminance_bw_device(
    img: &DeviceImage,
    standard: LuminanceStandard,
) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    if c != 3 {
        return Err(PhaiosError::Shape(format!(
            "expected (H, W, 3) RGB array, got shape [{h}, {w}, {c}]"
        )));
    }
    let ctx = img.context().clone();
    let npix = h * w;
    let mut out = ctx.alloc_image((h, w, 1))?;
    if npix == 0 {
        return Ok(out);
    }
    let [wr, wg, wb] = standard.weights();
    let n_ll = npix as i64;
    let func = ctx.function("luminance_bw_kernel", PTX_LUMINANCE)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&wr)
        .arg(&wg)
        .arg(&wb)
        .arg(&n_ll);
    // Safety: signature matches the .cu; input holds 3·npix elements,
    // output npix; the kernel bounds-checks against npix.
    unsafe { launch.launch(grid_1d(npix)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Shared launcher: the (H, W, 3) → (H, W, 1) dot product all three
/// classic B&W methods compile to. One PTX serves them all.
fn dot3_device(img: &DeviceImage, weights: [f32; 3]) -> Result<DeviceImage, PhaiosError> {
    let (h, w, c) = img.shape();
    if c != 3 {
        return Err(PhaiosError::Shape(format!(
            "expected (H, W, 3) RGB array, got shape [{h}, {w}, {c}]"
        )));
    }
    let ctx = img.context().clone();
    let npix = h * w;
    let mut out = ctx.alloc_image((h, w, 1))?;
    if npix == 0 {
        return Ok(out);
    }
    let [wr, wg, wb] = weights;
    let n_ll = npix as i64;
    let func = ctx.function("luminance_bw_kernel", PTX_LUMINANCE)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&wr)
        .arg(&wg)
        .arg(&wb)
        .arg(&n_ll);
    // Safety: signature matches the .cu; input holds 3·npix elements,
    // output npix; the kernel bounds-checks against npix.
    unsafe { launch.launch(grid_1d(npix)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Arbitrary-weight channel mixer, device-resident. Mirrors
/// [`crate::bw::channel_mixer_bw`]; **bit-exact** (same dot product as
/// [`luminance_bw_device`], caller-supplied weights).
///
/// # Errors
/// Same [`PhaiosError::Shape`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn channel_mixer_bw_device(
    img: &DeviceImage,
    weights: [f32; 3],
) -> Result<DeviceImage, PhaiosError> {
    dot3_device(img, weights)
}

/// Per-call offload form of [`channel_mixer_bw_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn channel_mixer_bw(
    ctx: &Context,
    img: ArrayView3<f32>,
    weights: [f32; 3],
) -> Result<Array3<f32>, PhaiosError> {
    crate::bw::validate_rgb(img)?;
    let device = ctx.upload(img)?;
    ctx.download(&channel_mixer_bw_device(&device, weights)?)
}

/// Wratten-style colour-filter conversion, device-resident. Mirrors
/// [`crate::bw::color_filter_bw`]; **bit-exact**. The combined weights
/// `tᵢ·wᵢ` are computed on the host exactly as the CPU kernel does.
///
/// # Errors
/// Same [`PhaiosError::Shape`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn color_filter_bw_device(
    img: &DeviceImage,
    filter: crate::bw::ColorFilter,
    standard: LuminanceStandard,
) -> Result<DeviceImage, PhaiosError> {
    let t = filter.transmission();
    let lw = standard.weights();
    dot3_device(img, [t[0] * lw[0], t[1] * lw[1], t[2] * lw[2]])
}

/// Per-call offload form of [`color_filter_bw_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn color_filter_bw(
    ctx: &Context,
    img: ArrayView3<f32>,
    filter: crate::bw::ColorFilter,
    standard: LuminanceStandard,
) -> Result<Array3<f32>, PhaiosError> {
    crate::bw::validate_rgb(img)?;
    let device = ctx.upload(img)?;
    ctx.download(&color_filter_bw_device(&device, filter, standard)?)
}

/// Per-call offload form of [`luminance_bw_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn luminance_bw(
    ctx: &Context,
    img: ArrayView3<f32>,
    standard: LuminanceStandard,
) -> Result<Array3<f32>, PhaiosError> {
    crate::bw::validate_rgb(img)?;
    let device = ctx.upload(img)?;
    ctx.download(&luminance_bw_device(&device, standard)?)
}
