// SPDX-License-Identifier: GPL-3.0-or-later
//! Widgets: the kernel selector, per-kernel parameter controls, and the
//! A/B split painting.

use egui::{Color32, Rect, Sense, Ui, pos2, vec2};

use crate::kernels::{AllParams, KernelId};

// ── The three helpers backing every control block ────────────────────────────

fn slider(ui: &mut Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.add(egui::Slider::new(v, range).text(label));
}

fn slider_log(ui: &mut Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>) {
    ui.add(egui::Slider::new(v, range).logarithmic(true).text(label));
}

fn combo<T: PartialEq + Copy>(ui: &mut Ui, label: &str, v: &mut T, options: &[(T, &str)]) {
    let current = options
        .iter()
        .find(|(val, _)| *val == *v)
        .map(|(_, name)| *name)
        .unwrap_or("?");
    egui::ComboBox::from_label(label)
        .selected_text(current)
        .show_ui(ui, |ui| {
            for (val, name) in options {
                ui.selectable_value(v, *val, *name);
            }
        });
}

fn standard_combo(ui: &mut Ui, v: &mut phaios_core::bw::LuminanceStandard) {
    use phaios_core::bw::LuminanceStandard as L;
    combo(
        ui,
        "standard",
        v,
        &[
            (L::Bt601, "BT.601"),
            (L::Bt709, "BT.709"),
            (L::Bt2020, "BT.2020"),
        ],
    );
}

// ── Per-kernel controls ──────────────────────────────────────────────────────

pub fn kernel_controls(ui: &mut Ui, id: KernelId, p: &mut AllParams) {
    match id {
        KernelId::Crop => {
            slider(ui, "x", &mut p.crop_frac[0], 0.0..=0.9);
            slider(ui, "y", &mut p.crop_frac[1], 0.0..=0.9);
            slider(ui, "width", &mut p.crop_frac[2], 0.05..=1.0);
            slider(ui, "height", &mut p.crop_frac[3], 0.05..=1.0);
            ui.label("(fractions of the frame; clamped to stay inside)");
        }
        KernelId::Orient => {
            use phaios_core::geometry::Orientation as O;
            combo(
                ui,
                "orientation",
                &mut p.orientation,
                &[
                    (O::Normal, "normal"),
                    (O::FlipHorizontal, "flip H"),
                    (O::Rotate180, "rotate 180"),
                    (O::FlipVertical, "flip V"),
                    (O::Transpose, "transpose"),
                    (O::Rotate90, "rotate 90 CW"),
                    (O::Transverse, "transverse"),
                    (O::Rotate270, "rotate 270 CW"),
                ],
            );
        }
        KernelId::Straighten => slider(ui, "degrees", &mut p.straighten_deg, -45.0..=45.0),
        KernelId::Resize => {
            slider(ui, "scale", &mut p.resize_scale, 0.1..=2.0);
            use phaios_core::geometry::ResizeFilter as F;
            combo(
                ui,
                "filter",
                &mut p.resize_filter,
                &[
                    (F::Area, "area"),
                    (F::Bilinear, "bilinear"),
                    (F::CatmullRom, "Catmull-Rom"),
                ],
            );
        }
        KernelId::Exposure => slider(ui, "stops (EV)", &mut p.exposure_stops, -5.0..=5.0),
        KernelId::LuminanceBw => standard_combo(ui, &mut p.standard),
        KernelId::ChannelMixerBw => {
            slider(ui, "red", &mut p.mixer_weights[0], -2.0..=2.0);
            slider(ui, "green", &mut p.mixer_weights[1], -2.0..=2.0);
            slider(ui, "blue", &mut p.mixer_weights[2], -2.0..=2.0);
        }
        KernelId::ColorFilterBw => {
            use phaios_core::bw::ColorFilter as F;
            combo(
                ui,
                "filter",
                &mut p.filter,
                &[
                    (F::NoFilter, "none"),
                    (F::Yellow8K2, "Yellow #8 K2"),
                    (F::Orange21, "Orange #21"),
                    (F::Red25A, "Red #25 A"),
                    (F::Green11X1, "Green #11 X1"),
                    (F::Blue47C5, "Blue #47 C5"),
                ],
            );
            standard_combo(ui, &mut p.standard);
        }
        KernelId::HslBw => {
            for (i, name) in phaios_core::bw::HUE_BAND_NAMES.iter().enumerate() {
                slider(ui, name, &mut p.hsl_weights[i], -1.0..=1.0);
            }
            slider(ui, "sigma (deg)", &mut p.hsl_sigma, 5.0..=90.0);
            standard_combo(ui, &mut p.standard);
        }
        KernelId::ZoneSystem => {
            const ROMAN: [&str; 11] = [
                "Zone 0",
                "Zone I",
                "Zone II",
                "Zone III",
                "Zone IV",
                "Zone V",
                "Zone VI",
                "Zone VII",
                "Zone VIII",
                "Zone IX",
                "Zone X",
            ];
            for (i, name) in ROMAN.iter().enumerate() {
                slider(ui, name, &mut p.zone_offsets[i], -3.0..=3.0);
            }
        }
        KernelId::ToneCurve => {
            slider(ui, "slope (gain)", &mut p.tc_slope, 0.1..=3.0);
            slider(ui, "offset (lift)", &mut p.tc_offset, -0.5..=0.5);
            slider(ui, "power (gamma)", &mut p.tc_power, 0.1..=4.0);
        }
        KernelId::LocalContrast => {
            ui.add(egui::Slider::new(&mut p.lc_radius, 0..=64).text("radius (px)"));
            slider_log(ui, "eps", &mut p.lc_eps, 1e-4..=1.0);
            slider(ui, "strength", &mut p.lc_strength, 0.0..=2.0);
        }
        KernelId::FilmGrain => {
            slider(ui, "intensity", &mut p.grain_intensity, 0.0..=1.0);
            slider(ui, "size (px)", &mut p.grain_size, 0.5..=8.0);
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut p.grain_seed).prefix("seed "));
                if ui.button("reroll").clicked() {
                    p.grain_seed = phaios_core::film_grain::splitmix64(p.grain_seed);
                }
            });
        }
        KernelId::SplitToning => {
            slider(ui, "shadow a", &mut p.st_shadow_a, -0.15..=0.15);
            slider(ui, "shadow b", &mut p.st_shadow_b, -0.15..=0.15);
            slider(ui, "highlight a", &mut p.st_highlight_a, -0.15..=0.15);
            slider(ui, "highlight b", &mut p.st_highlight_b, -0.15..=0.15);
            slider(ui, "pivot", &mut p.st_pivot, 0.0..=1.0);
            slider(ui, "balance", &mut p.st_balance, -1.0..=1.0);
        }
        KernelId::Vignette => {
            slider(ui, "amount", &mut p.vg_amount, -1.0..=1.0);
            slider(ui, "feather", &mut p.vg_feather, 0.0..=1.0);
            slider(ui, "roundness", &mut p.vg_roundness, 0.0..=1.0);
        }
        KernelId::EncodeSrgb => {
            ui.label(
                "No parameters. This kernel's output is display-referred, \
                 so the viewer skips its usual display encode here — \
                 everything else you see goes through encode_srgb once.",
            );
        }
    }
}

// ── Image painting ───────────────────────────────────────────────────────────

/// Fit `tex_size` into the available space, aspect preserved.
fn fitted_rect(ui: &Ui, tex_size: [usize; 2]) -> Rect {
    let avail = ui.available_size();
    let (w, h) = (tex_size[0] as f32, tex_size[1] as f32);
    let s = (avail.x / w).min(avail.y / h).min(4.0);
    let size = vec2(w * s, h * s);
    let min = ui.min_rect().min + vec2((avail.x - size.x) * 0.5, 0.0);
    Rect::from_min_size(min, size)
}

/// Paint a single texture, aspect-fitted.
pub fn paint_single(ui: &mut Ui, tex: &egui::TextureHandle) {
    let rect = fitted_rect(ui, tex.size());
    let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
    ui.allocate_rect(rect, Sense::hover());
    ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
}

/// Paint the CPU|GPU split: both textures at the full rect with full UV,
/// revealed by clip rects, so the halves stay pixel-aligned. Returns the
/// updated split position.
pub fn paint_split(
    ui: &mut Ui,
    cpu: &egui::TextureHandle,
    gpu: &egui::TextureHandle,
    mut split: f32,
    gpu_label: &str,
) -> f32 {
    let rect = fitted_rect(ui, cpu.size());
    let resp = ui.allocate_rect(rect, Sense::click_and_drag());
    let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

    let sx = rect.left() + split * rect.width();
    ui.painter()
        .with_clip_rect(Rect::from_min_max(rect.min, pos2(sx, rect.bottom())))
        .image(cpu.id(), rect, uv, Color32::WHITE);
    ui.painter()
        .with_clip_rect(Rect::from_min_max(pos2(sx, rect.top()), rect.max))
        .image(gpu.id(), rect, uv, Color32::WHITE);

    // Grabber.
    ui.painter()
        .vline(sx, rect.y_range(), egui::Stroke::new(2.0, Color32::WHITE));
    ui.painter().circle_filled(
        pos2(sx, rect.center().y),
        6.0,
        Color32::from_white_alpha(200),
    );
    if resp.dragged()
        && let Some(pos) = resp.interact_pointer_pos()
    {
        split = ((pos.x - rect.left()) / rect.width()).clamp(0.05, 0.95);
    }

    // Corner labels.
    let font = egui::FontId::monospace(12.0);
    ui.painter().text(
        rect.left_top() + vec2(6.0, 6.0),
        egui::Align2::LEFT_TOP,
        "CPU",
        font.clone(),
        Color32::WHITE,
    );
    ui.painter().text(
        rect.right_top() + vec2(-6.0, 6.0),
        egui::Align2::RIGHT_TOP,
        gpu_label,
        font,
        Color32::WHITE,
    );
    split
}
