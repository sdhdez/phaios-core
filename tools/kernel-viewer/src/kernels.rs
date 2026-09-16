// SPDX-License-Identifier: GPL-3.0-or-later
//! The kernel registry: identity, input kind, CPU dispatch, and the
//! parameter state that drives the UI.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use ndarray::{Array3, ArrayView3};
use phaios_core::bw::{ColorFilter, HslWeightedParams, LuminanceStandard};
use phaios_core::error::PhaiosError;
use phaios_core::film_grain::GrainParams;
use phaios_core::geometry::{
    CropParams, Orientation, ResizeFilter, ResizeParams, StraightenParams,
};
use phaios_core::local_contrast::GuidedFilterParams;
use phaios_core::split_toning::SplitToningParams;
use phaios_core::tone::{ToneCurveParams, ZoneParams};
use phaios_core::vignette::VignetteParams;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KernelId {
    Crop,
    Orient,
    Straighten,
    Resize,
    Exposure,
    LuminanceBw,
    ChannelMixerBw,
    ColorFilterBw,
    HslBw,
    ZoneSystem,
    ToneCurve,
    LocalContrast,
    FilmGrain,
    SplitToning,
    Vignette,
    EncodeSrgb,
}

impl KernelId {
    pub const ALL: [KernelId; 16] = [
        KernelId::Crop,
        KernelId::Orient,
        KernelId::Straighten,
        KernelId::Resize,
        KernelId::Exposure,
        KernelId::LuminanceBw,
        KernelId::ChannelMixerBw,
        KernelId::ColorFilterBw,
        KernelId::HslBw,
        KernelId::ZoneSystem,
        KernelId::ToneCurve,
        KernelId::LocalContrast,
        KernelId::FilmGrain,
        KernelId::SplitToning,
        KernelId::Vignette,
        KernelId::EncodeSrgb,
    ];

    pub fn name(self) -> &'static str {
        match self {
            KernelId::Crop => "crop",
            KernelId::Orient => "orient",
            KernelId::Straighten => "straighten",
            KernelId::Resize => "resize",
            KernelId::Exposure => "exposure",
            KernelId::LuminanceBw => "luminance_bw",
            KernelId::ChannelMixerBw => "channel_mixer_bw",
            KernelId::ColorFilterBw => "color_filter_bw",
            KernelId::HslBw => "hsl_bw",
            KernelId::ZoneSystem => "zone_system",
            KernelId::ToneCurve => "tone_curve",
            KernelId::LocalContrast => "local_contrast",
            KernelId::FilmGrain => "film_grain",
            KernelId::SplitToning => "split_toning",
            KernelId::Vignette => "vignette",
            KernelId::EncodeSrgb => "encode_srgb",
        }
    }

    pub fn from_name(name: &str) -> Option<KernelId> {
        KernelId::ALL.iter().copied().find(|k| k.name() == name)
    }

    /// Only the encode kernel's output is already display-referred.
    pub fn already_encoded(self) -> bool {
        self == KernelId::EncodeSrgb
    }

    /// Run this kernel on the CPU. `rgb` and `luma` are the two prepped
    /// source variants; `input()` decides which is consumed.
    pub fn run_cpu(
        self,
        rgb: ArrayView3<f32>,
        luma: ArrayView3<f32>,
        p: &AllParams,
    ) -> Result<Array3<f32>, PhaiosError> {
        match self {
            KernelId::Crop => {
                phaios_core::geometry::crop(rgb, &p.crop_params(rgb.dim().0, rgb.dim().1))
            }
            KernelId::Orient => phaios_core::geometry::orient(rgb, p.orientation),
            KernelId::Straighten => {
                phaios_core::geometry::straighten(rgb, &StraightenParams::new(p.straighten_deg))
            }
            KernelId::Resize => {
                phaios_core::geometry::resize(rgb, &p.resize_params(rgb.dim().0, rgb.dim().1))
            }
            KernelId::Exposure => phaios_core::exposure::exposure(rgb, p.exposure_stops),
            KernelId::LuminanceBw => phaios_core::bw::luminance_bw(rgb, p.standard),
            KernelId::ChannelMixerBw => phaios_core::bw::channel_mixer_bw(rgb, p.mixer_weights),
            KernelId::ColorFilterBw => phaios_core::bw::color_filter_bw(rgb, p.filter, p.standard),
            KernelId::HslBw => phaios_core::bw::hsl_bw(rgb, &p.hsl_params()),
            KernelId::ZoneSystem => phaios_core::tone::zone_system(luma, &p.zone_params()),
            KernelId::ToneCurve => phaios_core::tone::tone_curve(rgb, &p.tone_curve_params()),
            KernelId::LocalContrast => {
                phaios_core::local_contrast::local_contrast(luma, &p.guided_params(), p.lc_strength)
            }
            KernelId::FilmGrain => phaios_core::film_grain::film_grain(luma, &p.grain_params()),
            KernelId::SplitToning => {
                phaios_core::split_toning::split_toning(luma, &p.split_params())
            }
            KernelId::Vignette => phaios_core::vignette::vignette(rgb, &p.vignette_params()),
            KernelId::EncodeSrgb => phaios_core::encode::encode_srgb(rgb),
        }
    }

    /// Hash only the parameters this kernel actually consumes, so edits
    /// to another kernel's sliders don't force a recompute.
    pub fn hash_params(self, p: &AllParams, h: &mut impl Hasher) {
        fn f(v: f32, h: &mut impl Hasher) {
            v.to_bits().hash(h);
        }
        match self {
            KernelId::Crop => p.crop_frac.iter().for_each(|&v| f(v, h)),
            KernelId::Orient => (p.orientation as u16).hash(h),
            KernelId::Straighten => f(p.straighten_deg, h),
            KernelId::Resize => {
                f(p.resize_scale, h);
                p.resize_filter.hash(h);
            }
            KernelId::Exposure => f(p.exposure_stops, h),
            KernelId::LuminanceBw => p.standard.hash(h),
            KernelId::ChannelMixerBw => p.mixer_weights.iter().for_each(|&w| f(w, h)),
            KernelId::ColorFilterBw => {
                p.filter.hash(h);
                p.standard.hash(h);
            }
            KernelId::HslBw => {
                p.hsl_weights.iter().for_each(|&w| f(w, h));
                f(p.hsl_sigma, h);
                p.standard.hash(h);
            }
            KernelId::ZoneSystem => p.zone_offsets.iter().for_each(|&z| f(z, h)),
            KernelId::ToneCurve => {
                f(p.tc_slope, h);
                f(p.tc_offset, h);
                f(p.tc_power, h);
            }
            KernelId::LocalContrast => {
                p.lc_radius.hash(h);
                f(p.lc_eps, h);
                f(p.lc_strength, h);
            }
            KernelId::FilmGrain => {
                f(p.grain_intensity, h);
                f(p.grain_size, h);
                p.grain_seed.hash(h);
            }
            KernelId::SplitToning => {
                f(p.st_shadow_a, h);
                f(p.st_shadow_b, h);
                f(p.st_highlight_a, h);
                f(p.st_highlight_b, h);
                f(p.st_pivot, h);
                f(p.st_balance, h);
            }
            KernelId::Vignette => {
                f(p.vg_amount, h);
                f(p.vg_feather, h);
                f(p.vg_roundness, h);
            }
            KernelId::EncodeSrgb => {}
        }
    }
}

/// Every kernel's parameters, held simultaneously so switching kernels
/// preserves edits. Defaults are each kernel's neutral.
pub struct AllParams {
    /// Crop rectangle as fractions of the frame: [x, y, width, height].
    pub crop_frac: [f32; 4],
    pub orientation: Orientation,
    pub straighten_deg: f32,
    pub resize_scale: f32,
    pub resize_filter: ResizeFilter,
    pub exposure_stops: f32,
    pub standard: LuminanceStandard,
    pub mixer_weights: [f32; 3],
    pub filter: ColorFilter,
    pub hsl_weights: [f32; 8],
    pub hsl_sigma: f32,
    pub zone_offsets: [f32; 11],
    pub tc_slope: f32,
    pub tc_offset: f32,
    pub tc_power: f32,
    pub lc_radius: u32,
    pub lc_eps: f32,
    pub lc_strength: f32,
    pub grain_intensity: f32,
    pub grain_size: f32,
    pub grain_seed: u64,
    pub st_shadow_a: f32,
    pub st_shadow_b: f32,
    pub st_highlight_a: f32,
    pub st_highlight_b: f32,
    pub st_pivot: f32,
    pub st_balance: f32,
    pub vg_amount: f32,
    pub vg_feather: f32,
    pub vg_roundness: f32,
}

impl Default for AllParams {
    fn default() -> Self {
        Self {
            crop_frac: [0.1, 0.1, 0.8, 0.8],
            orientation: Orientation::Rotate90,
            straighten_deg: 2.0,
            resize_scale: 0.5,
            resize_filter: ResizeFilter::Area,
            exposure_stops: 0.0,
            standard: LuminanceStandard::Bt709,
            mixer_weights: [0.21, 0.72, 0.07],
            filter: ColorFilter::Red25A,
            hsl_weights: [0.0; 8],
            hsl_sigma: 30.0,
            zone_offsets: [0.0; 11],
            tc_slope: 1.0,
            tc_offset: 0.0,
            tc_power: 1.0,
            lc_radius: 8,
            lc_eps: 0.01,
            lc_strength: 0.5,
            grain_intensity: 0.15,
            grain_size: 1.5,
            grain_seed: 20_260_815,
            st_shadow_a: -0.02,
            st_shadow_b: -0.04,
            st_highlight_a: 0.03,
            st_highlight_b: 0.03,
            st_pivot: 0.5,
            st_balance: 0.0,
            vg_amount: 0.35,
            vg_feather: 0.8,
            vg_roundness: 0.0,
        }
    }
}

impl AllParams {
    /// Pixel crop rectangle from the fractional sliders, clamped valid
    /// for the current frame.
    pub fn crop_params(&self, h: usize, w: usize) -> CropParams {
        let (h, w) = (h as f32, w as f32);
        let x = (self.crop_frac[0] * w).clamp(0.0, w - 1.0) as u32;
        let y = (self.crop_frac[1] * h).clamp(0.0, h - 1.0) as u32;
        let cw = ((self.crop_frac[2] * w) as u32).clamp(1, w as u32 - x);
        let ch = ((self.crop_frac[3] * h) as u32).clamp(1, h as u32 - y);
        CropParams::new(x, y, cw, ch)
    }

    pub fn resize_params(&self, h: usize, w: usize) -> ResizeParams {
        let tw = ((w as f32 * self.resize_scale) as u32).max(1);
        let th = ((h as f32 * self.resize_scale) as u32).max(1);
        ResizeParams::new(tw, th, self.resize_filter)
    }

    pub fn hsl_params(&self) -> HslWeightedParams {
        HslWeightedParams::new(self.hsl_weights, self.standard, self.hsl_sigma)
    }
    pub fn zone_params(&self) -> ZoneParams {
        let map: HashMap<i32, f32> = self
            .zone_offsets
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v != 0.0)
            .map(|(z, &v)| (z as i32, v))
            .collect();
        ZoneParams::new(map)
    }
    pub fn tone_curve_params(&self) -> ToneCurveParams {
        ToneCurveParams::new(self.tc_slope, self.tc_offset, self.tc_power)
    }
    pub fn guided_params(&self) -> GuidedFilterParams {
        GuidedFilterParams::new(self.lc_radius, self.lc_eps)
    }
    pub fn grain_params(&self) -> GrainParams {
        GrainParams::new(self.grain_intensity, self.grain_size, self.grain_seed)
    }
    pub fn split_params(&self) -> SplitToningParams {
        SplitToningParams::new(
            [0.0, self.st_shadow_a, self.st_shadow_b],
            [0.0, self.st_highlight_a, self.st_highlight_b],
            self.st_pivot,
            self.st_balance,
        )
    }
    pub fn vignette_params(&self) -> VignetteParams {
        VignetteParams::new(self.vg_amount, self.vg_feather, self.vg_roundness)
    }
}
