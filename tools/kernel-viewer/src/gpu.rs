// SPDX-License-Identifier: GPL-3.0-or-later
//! CUDA backend facade (compiled only with `--features cuda`).
//!
//! Mirrors the CPU dispatch chains with the device-resident kernel
//! forms: one upload, prep + kernel on-device, one download.

use ndarray::{Array3, ArrayView3};
use phaios_core::cuda;
use phaios_core::error::PhaiosError;

use crate::kernels::{AllParams, KernelId};

pub struct Gpu {
    ctx: cuda::Context,
    pub fingerprint: String,
}

impl Gpu {
    /// Open device 0 if a supported one exists; warm it up with a tiny
    /// call so the first displayed timing excludes PTX module load.
    pub fn try_new() -> Option<Gpu> {
        if !cuda::available() {
            return None;
        }
        let ctx = cuda::Context::new(0).ok()?;
        let warm = ndarray::Array3::<f32>::from_elem((8, 8, 1), 0.5);
        let _ = cuda::kernels::exposure(&ctx, warm.view(), 0.0).ok()?;
        let fingerprint = ctx.fingerprint();
        Some(Gpu { ctx, fingerprint })
    }

    /// Run `id` on the device: upload once, chain resident, download once.
    pub fn run(
        &self,
        id: KernelId,
        src_rgb: ArrayView3<f32>,
        p: &AllParams,
    ) -> Result<Array3<f32>, PhaiosError> {
        use cuda::kernels as k;
        use phaios_core::bw::LuminanceStandard;

        let d = self.ctx.upload(src_rgb)?;
        // Prep stage for luminance-input kernels, on-device (fixed Bt709,
        // matching the CPU prep in app.rs).
        let luma = |d: &cuda::DeviceImage| k::luminance_bw_device(d, LuminanceStandard::Bt709);

        let (src_h, src_w, _) = src_rgb.dim();
        let out = match id {
            KernelId::Crop => k::crop_device(&d, &p.crop_params(src_h, src_w))?,
            KernelId::Orient => k::orient_device(&d, p.orientation)?,
            KernelId::Straighten => k::straighten_device(
                &d,
                &phaios_core::geometry::StraightenParams::new(p.straighten_deg),
            )?,
            KernelId::Resize => k::resize_device(&d, &p.resize_params(src_h, src_w))?,
            KernelId::Exposure => k::exposure_device(&d, p.exposure_stops)?,
            KernelId::LuminanceBw => k::luminance_bw_device(&d, p.standard)?,
            KernelId::ChannelMixerBw => k::channel_mixer_bw_device(&d, p.mixer_weights)?,
            KernelId::ColorFilterBw => k::color_filter_bw_device(&d, p.filter, p.standard)?,
            KernelId::HslBw => k::hsl_bw_device(&d, &p.hsl_params())?,
            KernelId::ZoneSystem => k::zone_system_device(&luma(&d)?, &p.zone_params())?,
            KernelId::ToneCurve => k::tone_curve_device(&d, &p.tone_curve_params())?,
            KernelId::LocalContrast => {
                k::local_contrast_device(&luma(&d)?, &p.guided_params(), p.lc_strength)?
            }
            KernelId::FilmGrain => k::film_grain_device(&luma(&d)?, &p.grain_params())?,
            KernelId::SplitToning => k::split_toning_device(&luma(&d)?, &p.split_params())?,
            KernelId::Vignette => k::vignette_device(&d, &p.vignette_params())?,
            KernelId::EncodeSrgb => k::encode_srgb_device(&d)?,
        };
        self.ctx.download(&out)
    }
}
