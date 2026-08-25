// SPDX-License-Identifier: GPL-3.0-or-later
//! The analysis pair on the device: `histogram` and `apply_lut`.
//!
//! `histogram` is the crate's first device kernel that reduces rather
//! than maps. It is also, unusually, deterministic for free: the only
//! arithmetic on pixel values is the bin assignment, and everything
//! after that is integer counting, whose result cannot depend on the
//! order in which the atomics complete.
//!
//! Both kernels are **bit-exact** against their CPU counterparts.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ndarray::{Array3, ArrayView1, ArrayView3};

use super::{be, grid_1d};
use crate::cuda::context::{Context, DeviceImage};
use crate::error::PhaiosError;
use crate::histogram::{Histogram, HistogramParams};
use crate::lut::LutParams;

const PTX_HISTOGRAM: &str = include_str!(concat!(env!("OUT_DIR"), "/histogram.ptx"));
const PTX_LUT: &str = include_str!(concat!(env!("OUT_DIR"), "/lut.ptx"));

/// Counters that fit in shared memory, at 4 bytes each. 48 KB is the
/// per-block shared allocation every supported architecture provides
/// without opting into the larger carve-out.
const MAX_SHARED_SLOTS: usize = 48 * 1024 / 4;

/// Blocks to launch for the reduction. Fixed rather than derived from
/// the element count so each block's 32-bit private counters stay far
/// below wrapping: at this grid size a 24 MP three-channel frame gives
/// roughly 70 000 samples per block.
const HIST_BLOCKS: u32 = 1024;
const HIST_THREADS: u32 = 256;

/// Per-channel histogram of a device-resident image, returned on the
/// host. Mirrors [`crate::histogram::histogram`]; bit-identical.
///
/// A reduction, so there is no device-resident form to chain: the counts
/// are small and their destination is a caller's display or an
/// auto-correction, both of which live on the host.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "the histogram is the result; ignoring it wastes a full pass"]
pub fn histogram_device(
    img: &DeviceImage,
    params: &HistogramParams,
) -> Result<Histogram, PhaiosError> {
    // Shape-aware: bounds `channels x bins`, which also keeps `bins`
    // below i32::MAX so the narrowing below cannot go negative and send
    // the device kernel writing outside its counter buffer.
    crate::histogram::validate_shape(img.shape(), params)?;

    let (h, w, c) = img.shape();
    let bins = params.bins as usize;
    let slots = c * (bins + 3);
    let n = h * w * c;

    if n == 0 || c == 0 {
        return Ok(crate::histogram::assemble(
            vec![0_u64; slots],
            c,
            bins,
            params.min,
            params.max,
        ));
    }

    let ctx = img.context().clone();
    // Zeroed rather than uninitialised: the kernel accumulates into this
    // buffer, it does not overwrite it.
    let mut d_counts: CudaSlice<u64> = ctx
        .stream
        .alloc_zeros(slots)
        .map_err(be("device allocation failed"))?;

    let (c_i, bins_i) = (c as i32, bins as i32);
    let (min, max) = (params.min, params.max);
    let n_ll = n as i64;

    // Privatise per block when the table fits; fall back to global
    // atomics for the large-bin analysis case.
    let use_shared = slots <= MAX_SHARED_SLOTS;
    let name = if use_shared {
        "histogram_shared_kernel"
    } else {
        "histogram_global_kernel"
    };
    let func = ctx.function(name, PTX_HISTOGRAM)?;

    let blocks = HIST_BLOCKS.min(n.div_ceil(HIST_THREADS as usize).max(1) as u32);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (HIST_THREADS, 1, 1),
        shared_mem_bytes: if use_shared {
            u32::try_from(slots * std::mem::size_of::<u32>()).map_err(|_| {
                PhaiosError::Parameter(format!(
                    "histogram needs {slots} counters, more than the backend can privatise"
                ))
            })?
        } else {
            0
        },
    };

    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut d_counts)
        .arg(&c_i)
        .arg(&bins_i)
        .arg(&min)
        .arg(&max)
        .arg(&n_ll);
    // Safety: signature matches the .cu; the input holds n floats, the
    // output holds `slots` counters, the kernel indexes both within those
    // bounds, and the shared allocation is exactly what it reads.
    unsafe { launch.launch(cfg) }.map_err(be("kernel launch failed"))?;

    let host = ctx
        .stream
        .clone_dtoh(&d_counts)
        .map_err(be("download failed"))?;
    ctx.stream.synchronize().map_err(be("synchronize failed"))?;

    Ok(crate::histogram::assemble(
        host, c, bins, params.min, params.max,
    ))
}

/// Per-call offload form of [`histogram_device`].
#[must_use = "the histogram is the result; ignoring it wastes a full pass"]
pub fn histogram(
    ctx: &Context,
    img: ArrayView3<f32>,
    params: &HistogramParams,
) -> Result<Histogram, PhaiosError> {
    crate::histogram::validate_shape(img.dim(), params)?;
    let device = ctx.upload(img)?;
    histogram_device(&device, params)
}

/// Apply a 1-D lookup table to a device-resident image. Mirrors
/// [`crate::lut::apply_lut`]; **bit-exact**.
///
/// # Errors
/// Same [`PhaiosError::Parameter`] as the CPU kernel; plus
/// [`PhaiosError::Backend`] on device failure.
#[must_use = "kernel returns a new image; ignoring it wastes work"]
pub fn apply_lut_device(
    img: &DeviceImage,
    lut: ArrayView1<f32>,
    params: &LutParams,
) -> Result<DeviceImage, PhaiosError> {
    crate::lut::validate(lut, params)?;

    let ctx = img.context().clone();
    let shape = img.shape();
    let n = shape.0 * shape.1 * shape.2;
    let mut out = ctx.alloc_image(shape)?;
    if n == 0 {
        return Ok(out);
    }

    // The table has to be contiguous to upload; a caller passing a
    // strided view of one should not be refused.
    // `as_standard_layout`, not `to_owned`: `to_owned` preserves the
    // source's memory order, so a reversed view stays negative-stride and
    // `as_slice` returns None a second time — the review found that the
    // `expect` then fired as a PanicException across the FFI, on caller
    // layout, which CLAUDE.md §2 forbids by name. `Context::upload` uses
    // `as_standard_layout` for exactly this reason. Reversed tables are
    // not exotic: `np.flip(cdf)` is the documented way to build a
    // histogram-matching transfer.
    let compact = lut.as_standard_layout();
    let table: &[f32] = compact
        .as_slice()
        .expect("as_standard_layout guarantees a contiguous C-order array");
    let d_lut = ctx
        .stream
        .clone_htod(table)
        .map_err(be("lut upload failed"))?;

    let lut_len = i32::try_from(table.len()).map_err(|_| {
        PhaiosError::Parameter(format!(
            "lut has {} entries, more than the backend can index",
            table.len()
        ))
    })?;
    let LutParams { min, max } = *params;
    let n_ll = n as i64;

    let func = ctx.function("lut_kernel", PTX_LUT)?;
    let mut launch = ctx.stream.launch_builder(&func);
    launch
        .arg(&img.buf)
        .arg(&mut out.buf)
        .arg(&d_lut)
        .arg(&lut_len)
        .arg(&min)
        .arg(&max)
        .arg(&n_ll);
    // Safety: signature matches the .cu; both image buffers hold n
    // elements and the table holds `lut_len`, the kernel bounds-checks
    // its index against n and clamps every table index to [0, lut_len-1].
    unsafe { launch.launch(grid_1d(n)) }.map_err(be("kernel launch failed"))?;
    Ok(out)
}

/// Per-call offload form of [`apply_lut_device`].
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn apply_lut(
    ctx: &Context,
    img: ArrayView3<f32>,
    lut: ArrayView1<f32>,
    params: &LutParams,
) -> Result<Array3<f32>, PhaiosError> {
    crate::lut::validate(lut, params)?;
    let device = ctx.upload(img)?;
    ctx.download(&apply_lut_device(&device, lut, params)?)
}
