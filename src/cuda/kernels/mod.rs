// SPDX-License-Identifier: GPL-3.0-or-later
//! GPU kernels, grouped by the CPU kernels they mirror. A module can
//! hold several: `elementwise` holds eight, `geometry` four, `analysis`
//! two, `quantize` two.
//!
//! Each kernel offers two forms:
//!
//! - `<name>_device(&DeviceImage, ...) -> DeviceImage` — the resident
//!   form: input and output stay on the GPU, so a pipeline pays PCIe
//!   once at each end instead of per stage.
//! - `<name>(&Context, ArrayView3, ...) -> Array3` — per-call offload:
//!   upload, run, download. Convenient for a single kernel; the
//!   transfers dominate for cheap ones.
//!
//! Three terminal kernels are the exception to the resident form:
//! `histogram_device` returns a host `Histogram`, and
//! `quantize_u8_device` and `quantize_u16_device` return host
//! `Array3<u8>` and `Array3<u16>`. A reduction and a terminal stage
//! have nothing to chain into.
//!
//! Both forms validate identically to the CPU kernel they mirror.

mod analysis;
mod blur;
mod denoise;
mod elementwise;
mod exposure;
mod geometry;
mod glow;
mod grain;
mod hot_pixels;
mod hsl;
mod local_contrast;
mod quantize;
mod sharpen;
mod split_toning;
mod zone;

pub use analysis::{apply_lut, apply_lut_device, histogram, histogram_device};
pub use blur::{blur, blur_device};
pub use denoise::{denoise, denoise_device};
pub use elementwise::{
    channel_mixer_bw, channel_mixer_bw_device, color_filter_bw, color_filter_bw_device,
    encode_srgb, encode_srgb_device, highlight_rolloff, highlight_rolloff_device, luminance_bw,
    luminance_bw_device, shadow_rolloff, shadow_rolloff_device, tone_curve, tone_curve_device,
    vignette, vignette_device,
};
pub use exposure::{exposure, exposure_device};
pub use geometry::{
    crop, crop_device, orient, orient_device, resize, resize_device, straighten, straighten_device,
};
pub use glow::{glow, glow_device};
pub use grain::{film_grain, film_grain_device, hash_grid};
pub use hot_pixels::{hot_pixels, hot_pixels_device};
pub use hsl::{hsl_bw, hsl_bw_device};
pub use local_contrast::{local_contrast, local_contrast_device};
pub use quantize::{quantize_u8, quantize_u8_device, quantize_u16, quantize_u16_device};
pub use sharpen::{sharpen, sharpen_device};
pub use split_toning::{split_toning, split_toning_device};
pub use zone::{zone_system, zone_system_device};

use crate::error::PhaiosError;

/// Map a cudarc driver error into the crate error type.
pub(crate) fn be(what: &'static str) -> impl Fn(cudarc::driver::DriverError) -> PhaiosError {
    move |e| PhaiosError::Backend(format!("{what}: {e}"))
}

/// 1-D launch geometry over `n` elements.
///
/// Block size is whatever `cudarc::driver::LaunchConfig::for_num_elems`
/// picks (1024 threads at the time of writing) — deliberately not pinned
/// here, because the determinism contract is independent of launch
/// geometry: every kernel launched through this helper indexes by
/// absolute element and none reduces across threads, so a different
/// block size cannot change a result. The histogram kernels do reduce
/// across threads, through atomics, and use their own launch geometry.
/// Their accumulation is over integers, so it is order-independent
/// anyway.
pub(crate) fn grid_1d(n: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig::for_num_elems(n as u32)
}

/// The PTX each `.cu` file compiles to carries the signature of the
/// precision mode `build.rs`'s `PRECISION` table gives it. This pins the
/// table: switching a file's mode by mistake fails here, even where the
/// conformance tests' inputs (dyadic values in `[0, 1)`, no subnormals)
/// would not notice the difference.
#[cfg(test)]
mod precision_tests {
    macro_rules! ptx {
        ($name:literal) => {
            (
                $name,
                include_str!(concat!(env!("OUT_DIR"), "/", $name, ".ptx")),
            )
        };
    }

    /// Compiled strict: IEEE division and square root, no contraction.
    const STRICT: &[(&str, &str)] = &[
        ptx!("blur"),
        ptx!("exposure"),
        ptx!("film_grain"),
        ptx!("geometry"),
        ptx!("glow"),
        ptx!("highlight_rolloff"),
        ptx!("histogram"),
        ptx!("hot_pixels"),
        ptx!("luminance_bw"),
        ptx!("lut"),
        ptx!("quantize"),
        ptx!("shadow_rolloff"),
        ptx!("sharpen"),
        ptx!("split_toning"),
        ptx!("vignette"),
    ];
    /// Compiled with FMA contraction, IEEE otherwise.
    const FMA: &[(&str, &str)] = &[ptx!("denoise"), ptx!("local_contrast")];
    /// Compiled with `--use_fast_math`.
    const FAST: &[(&str, &str)] = &[
        ptx!("encode_srgb"),
        ptx!("hsl_bw"),
        ptx!("tone_curve"),
        ptx!("zone_system"),
    ];

    /// The opcode of each PTX instruction line, e.g. `add.rn.f32`.
    fn opcodes(ptx: &str) -> impl Iterator<Item = &str> {
        ptx.lines().filter_map(|line| {
            let line = line.trim_start();
            let line = if line.starts_with('@') {
                line.split_once(char::is_whitespace)?.1.trim_start()
            } else {
                line
            };
            let op = line.split(|c: char| c.is_whitespace() || c == ';').next()?;
            (!op.is_empty() && !op.starts_with(['.', '/', '$', '{', '}']) && op.contains('.'))
                .then_some(op)
        })
    }

    /// A float add, subtract or multiply with no rounding modifier: the
    /// form the driver's JIT may fuse into an FMA.
    fn unpinned(op: &str) -> bool {
        let float = op.ends_with(".f32") || op.ends_with(".f64");
        let arith = ["add.", "sub.", "mul."].iter().any(|p| op.starts_with(p));
        let rounded = [".rn", ".rz", ".rm", ".rp"].iter().any(|r| op.contains(r));
        float && arith && !rounded
    }

    #[test]
    fn strict_files_pin_every_float_operation() {
        for (name, ptx) in STRICT {
            let loose: Vec<_> = opcodes(ptx).filter(|op| unpinned(op)).collect();
            assert!(loose.is_empty(), "{name}.ptx: unpinned {loose:?}");
            // split_toning inlines libdevice cbrtf, whose body uses
            // approximate instructions; every other strict file uses none.
            if *name != "split_toning" {
                let approx: Vec<_> = opcodes(ptx)
                    .filter(|op| op.contains(".approx") || op.contains(".ftz"))
                    .collect();
                assert!(approx.is_empty(), "{name}.ptx: {approx:?}");
            }
        }
    }

    #[test]
    fn fma_files_contract_but_stay_ieee() {
        for (name, ptx) in FMA {
            assert!(
                opcodes(ptx).any(|op| op.starts_with("fma.rn.f")),
                "{name}.ptx: no FMA"
            );
            let approx: Vec<_> = opcodes(ptx)
                .filter(|op| op.contains(".approx") || op.contains(".ftz"))
                .collect();
            assert!(approx.is_empty(), "{name}.ptx: {approx:?}");
        }
    }

    /// libdevice brings `.approx.ftz` instructions into strict and FMA
    /// builds too, so the signature of fast math is flush-to-zero on the
    /// file's own arithmetic, and no plain correctly-rounded f32 add or
    /// multiply (`tone_curve`'s pinned step is `.rn.ftz`).
    #[test]
    fn fast_files_are_compiled_with_fast_math() {
        for (name, ptx) in FAST {
            assert!(
                opcodes(ptx).any(|op| op.contains(".ftz")
                    && !op.contains(".approx")
                    && !op.starts_with("setp")),
                "{name}.ptx: not compiled with fast math"
            );
            let ieee: Vec<_> = opcodes(ptx)
                .filter(|op| *op == "add.rn.f32" || *op == "mul.rn.f32")
                .collect();
            assert!(ieee.is_empty(), "{name}.ptx: compiled strict? {ieee:?}");
        }
    }

    /// tone_curve's affine step is pinned in the source (`__fmul_rn`,
    /// `__fadd_rn`): fused into one FMA it kept a residue where the CPU
    /// cancels to zero, which powf at power < 1 lifted by up to 8 codes.
    #[test]
    fn tone_curve_pins_its_affine_step() {
        let (_, ptx) = FAST.iter().find(|(n, _)| *n == "tone_curve").unwrap();
        let ops: Vec<_> = opcodes(ptx).collect();
        assert!(ops.contains(&"mul.rn.ftz.f32"), "{ops:?}");
        assert!(ops.contains(&"add.rn.ftz.f32"), "{ops:?}");
    }

    #[test]
    fn every_cu_file_is_covered() {
        let covered = STRICT.len() + FMA.len() + FAST.len();
        let files = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda/ptx"))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .and_then(|x| x.to_str())
                    == Some("cu")
            })
            .count();
        assert_eq!(covered, files, "a .cu file is missing from these lists");
    }
}
