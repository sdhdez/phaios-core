// SPDX-License-Identifier: GPL-3.0-or-later
//! The viewer application: state, hash-gated recompute, key handling.

use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use ndarray::Array3;

use crate::kernels::{AllParams, KernelId};
use crate::{color, raw, scenes, ui};

/// CPU backend fingerprint, per docs/ffi.md §6.
const CPU_FINGERPRINT: &str = concat!("cpu/", env!("KV_TARGET"));

/// Longest edge of the interactive working copy (`--full` lifts it).
const MAX_EDGE: usize = 1600;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum SceneId {
    Photo,
    Macbeth,
    File(PathBuf),
}

pub struct ViewerApp {
    // Source state.
    scene: SceneId,
    scene_generation: u64,
    full_res: bool,
    src_rgb: Array3<f32>,
    src_luma: Array3<f32>,
    load_error: Option<String>,

    // Kernel state.
    kernel: KernelId,
    params: AllParams,

    // Backend state.
    #[cfg(feature = "cuda")]
    gpu: Option<crate::gpu::Gpu>,
    use_gpu: bool,
    ab_split: bool,
    show_diff: bool,
    split_pos: f32,

    // Render cache.
    last_hash: u64,
    tex_cpu: Option<egui::TextureHandle>,
    tex_gpu: Option<egui::TextureHandle>,
    cpu_ms: f32,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    gpu_ms: f32,
    kernel_error: Option<String>,
}

impl ViewerApp {
    pub fn new(kernel: KernelId, image: Option<PathBuf>, use_gpu: bool, full_res: bool) -> Self {
        // Default photo: the maintainer's test DNG when present.
        let default_dng = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/R0000096.DNG"
        ));
        let scene = match image {
            Some(p) => SceneId::File(p),
            None if default_dng.is_file() => SceneId::File(default_dng),
            None => SceneId::Photo,
        };

        let mut app = Self {
            scene,
            scene_generation: 0,
            full_res,
            src_rgb: Array3::zeros((1, 1, 3)),
            src_luma: Array3::zeros((1, 1, 1)),
            load_error: None,
            kernel,
            params: AllParams::default(),
            #[cfg(feature = "cuda")]
            gpu: crate::gpu::Gpu::try_new(),
            use_gpu,
            ab_split: false,
            show_diff: false,
            split_pos: 0.5,
            last_hash: 0,
            tex_cpu: None,
            tex_gpu: None,
            cpu_ms: 0.0,
            gpu_ms: 0.0,
            kernel_error: None,
        };
        app.load_scene();
        app
    }

    fn gpu_present(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn load_scene(&mut self) {
        self.load_error = None;
        let full = match &self.scene {
            SceneId::Photo => scenes::synthetic_photo(),
            SceneId::Macbeth => scenes::macbeth(),
            SceneId::File(path) => {
                let loaded = if raw::is_raw_extension(path) {
                    raw::load_dng(path)
                } else {
                    std::fs::read(path)
                        .map_err(|e| format!("cannot read {}: {e}", path.display()))
                        .and_then(|bytes| color::load_linear(&bytes))
                };
                match loaded {
                    Ok(img) => img,
                    Err(e) => {
                        self.load_error = Some(e);
                        scenes::synthetic_photo()
                    }
                }
            }
        };
        self.src_rgb = if self.full_res {
            full
        } else {
            scenes::downscale_max_edge(&full, MAX_EDGE)
        };
        // Prep stage for luminance-input kernels (fixed Bt709; the
        // luminance *kernel entry* has its own standard control, but the
        // prep is not a parameter — documented in the README).
        self.src_luma = phaios_core::bw::luminance_bw(
            self.src_rgb.view(),
            phaios_core::bw::LuminanceStandard::Bt709,
        )
        .expect("source is (H, W, 3) by construction");
        self.scene_generation += 1;
    }

    fn state_hash(&self) -> u64 {
        let mut h = std::hash::DefaultHasher::new();
        self.kernel.hash(&mut h);
        self.kernel.hash_params(&self.params, &mut h);
        self.scene_generation.hash(&mut h);
        self.use_gpu.hash(&mut h);
        self.ab_split.hash(&mut h);
        self.show_diff.hash(&mut h);
        h.finish()
    }

    fn recompute(&mut self, ctx: &egui::Context) {
        self.kernel_error = None;
        let want_gpu = self.gpu_present() && (self.use_gpu || self.ab_split || self.show_diff);
        let want_cpu = !self.use_gpu || self.ab_split || self.show_diff;

        let mut cpu_out: Option<Array3<f32>> = None;
        if want_cpu {
            let t = Instant::now();
            match self
                .kernel
                .run_cpu(self.src_rgb.view(), self.src_luma.view(), &self.params)
            {
                Ok(out) => {
                    self.cpu_ms = t.elapsed().as_secs_f32() * 1000.0;
                    cpu_out = Some(out);
                }
                Err(e) => self.kernel_error = Some(e.to_string()),
            }
        }

        #[allow(unused_mut)]
        let mut gpu_out: Option<Array3<f32>> = None;
        #[cfg(feature = "cuda")]
        if want_gpu && let Some(gpu) = &self.gpu {
            let t = Instant::now();
            match gpu.run(self.kernel, self.src_rgb.view(), &self.params) {
                Ok(out) => {
                    self.gpu_ms = t.elapsed().as_secs_f32() * 1000.0;
                    gpu_out = Some(out);
                }
                Err(e) => self.kernel_error = Some(e.to_string()),
            }
        }
        let _ = want_gpu;

        // Diff mode replaces both textures with one amplified difference.
        if self.show_diff
            && let (Some(c), Some(g)) = (&cpu_out, &gpu_out)
        {
            let diff = color::diff_image(c.view(), g.view(), 64.0);
            let img = color::to_display(diff.view(), false);
            self.set_tex(ctx, true, img.clone());
            self.set_tex(ctx, false, img);
            return;
        }

        let encoded = self.kernel.already_encoded();
        if let Some(c) = &cpu_out {
            let img = color::to_display(c.view(), encoded);
            self.set_tex(ctx, true, img);
        }
        if let Some(g) = &gpu_out {
            let img = color::to_display(g.view(), encoded);
            self.set_tex(ctx, false, img);
        } else if self.use_gpu && !self.gpu_present() {
            self.use_gpu = false; // degrade gracefully, note in status bar
        }
    }

    fn set_tex(&mut self, ctx: &egui::Context, cpu: bool, img: egui::ColorImage) {
        let opts = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Linear,
            ..Default::default()
        };
        let slot = if cpu {
            &mut self.tex_cpu
        } else {
            &mut self.tex_gpu
        };
        match slot {
            Some(handle) => handle.set(img, opts),
            None => {
                *slot = Some(ctx.load_texture(if cpu { "cpu" } else { "gpu" }, img, opts));
            }
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            let idx = KernelId::ALL
                .iter()
                .position(|&k| k == self.kernel)
                .unwrap();
            if i.key_pressed(egui::Key::CloseBracket) {
                self.kernel = KernelId::ALL[(idx + 1) % KernelId::ALL.len()];
            }
            if i.key_pressed(egui::Key::OpenBracket) {
                self.kernel = KernelId::ALL[(idx + KernelId::ALL.len() - 1) % KernelId::ALL.len()];
            }
            if i.key_pressed(egui::Key::G) && self.gpu_present() {
                self.use_gpu = !self.use_gpu;
            }
            if i.key_pressed(egui::Key::D) && self.gpu_present() {
                self.show_diff = !self.show_diff;
            }
            if i.key_pressed(egui::Key::R) && self.kernel == KernelId::FilmGrain {
                self.params.grain_seed =
                    phaios_core::film_grain::splitmix64(self.params.grain_seed);
            }
            // Dropped files (egui 0.36: handles with path()).
            if let Some(file) = i.raw.dropped_files.first() {
                self.scene = SceneId::File(file.path().to_path_buf());
            }
        });
    }

    /// CLI `--scene` override, applied before the first frame.
    pub fn force_scene(&mut self, name: &str) {
        self.scene = match name {
            "macbeth" => SceneId::Macbeth,
            _ => SceneId::Photo,
        };
        self.load_scene();
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let ctx = &ctx;
        let scene_before = self.scene.clone();
        self.handle_keys(ctx);
        if scene_before != self.scene {
            self.load_scene();
        }

        // Side panel: selector + controls.
        egui::Panel::left("controls")
            .min_size(280.0)
            .show(root, |panel| {
                egui::ComboBox::from_label("kernel")
                    .selected_text(self.kernel.name())
                    .show_ui(panel, |ui| {
                        for k in KernelId::ALL {
                            ui.selectable_value(&mut self.kernel, k, k.name());
                        }
                    });
                panel.separator();
                ui::kernel_controls(panel, self.kernel, &mut self.params);
                panel.separator();

                // Scene selection.
                let mut scene_choice = match &self.scene {
                    SceneId::Photo => 0_usize,
                    SceneId::Macbeth => 1,
                    SceneId::File(_) => 2,
                };
                let before = scene_choice;
                egui::ComboBox::from_label("scene")
                    .selected_text(["synthetic photo", "macbeth", "file"][scene_choice])
                    .show_ui(panel, |ui| {
                        ui.selectable_value(&mut scene_choice, 0, "synthetic photo");
                        ui.selectable_value(&mut scene_choice, 1, "macbeth");
                    });
                if scene_choice != before {
                    self.scene = match scene_choice {
                        0 => SceneId::Photo,
                        _ => SceneId::Macbeth,
                    };
                    self.load_scene();
                }
                panel.label("(drop a JPEG/PNG/DNG to load it)");

                // Backend controls.
                if self.gpu_present() {
                    panel.separator();
                    panel.checkbox(&mut self.use_gpu, "GPU backend (G)");
                    panel.checkbox(&mut self.ab_split, "A/B split CPU|GPU");
                    panel.checkbox(&mut self.show_diff, "difference ×64 (D)");
                }

                if let Some(e) = &self.load_error {
                    panel.separator();
                    panel.colored_label(egui::Color32::YELLOW, format!("load: {e}"));
                }
                if let Some(e) = &self.kernel_error {
                    panel.separator();
                    panel.colored_label(egui::Color32::RED, e);
                }
            });

        // Status bar.
        egui::Panel::bottom("status").show(root, |bar| {
            bar.horizontal(|bar| {
                let (h, w, _) = self.src_rgb.dim();
                bar.monospace(format!("{w}×{h}"));
                bar.separator();
                bar.monospace(self.kernel.name());
                bar.separator();
                bar.monospace(format!("cpu {:.1} ms · {CPU_FINGERPRINT}", self.cpu_ms));
                #[cfg(feature = "cuda")]
                if let Some(gpu) = &self.gpu {
                    bar.separator();
                    bar.monospace(format!("gpu {:.1} ms · {}", self.gpu_ms, gpu.fingerprint));
                }
                #[cfg(feature = "cuda")]
                if self.gpu.is_none() {
                    bar.separator();
                    bar.monospace("no CUDA device — CPU only");
                }
            });
        });

        // Recompute only when the state actually changed.
        let h = self.state_hash();
        if h != self.last_hash {
            self.recompute(ctx);
            self.last_hash = h;
        }

        // Central image.
        egui::CentralPanel::default().show(root, |panel| {
            let split_mode = self.ab_split && !self.show_diff && self.gpu_present();
            match (split_mode, &self.tex_cpu, &self.tex_gpu) {
                (true, Some(cpu), Some(gpu)) => {
                    let label = {
                        #[cfg(feature = "cuda")]
                        {
                            self.gpu.as_ref().map(|g| g.fingerprint.clone())
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            None::<String>
                        }
                    }
                    .unwrap_or_else(|| "GPU".into());
                    self.split_pos = ui::paint_split(panel, cpu, gpu, self.split_pos, &label);
                }
                (_, _, Some(gpu)) if self.use_gpu => ui::paint_single(panel, gpu),
                (_, Some(cpu), _) => ui::paint_single(panel, cpu),
                _ => {
                    panel.label("no image");
                }
            }
        });
    }
}
