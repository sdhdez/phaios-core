// SPDX-License-Identifier: GPL-3.0-or-later
//! kernel-viewer — interactive parameter explorer for phaios-core.
//!
//! The interactive counterpart of the crate's `examples/` directory, in
//! the spirit of the classic OpenGL sample programs: open a kernel on an
//! image, drag the sliders, watch it act.
//!
//! ```text
//! kernel-viewer [KERNEL] [IMAGE] [--scene photo|macbeth] [--gpu] [--full]
//! ```

mod app;
mod color;
#[cfg(feature = "cuda")]
mod gpu;
mod kernels;
mod raw;
mod scenes;
mod ui;

use kernels::KernelId;

fn usage() -> String {
    let names: Vec<&str> = KernelId::ALL.iter().map(|k| k.name()).collect();
    format!(
        "usage: kernel-viewer [KERNEL] [IMAGE] [--scene photo|macbeth] [--gpu] [--full]\n\
         \n\
         KERNEL: {}\n\
         IMAGE:  a JPEG/PNG/DNG to load (default: testdata/R0000096.DNG if\n\
         present, else the synthetic scene)\n\
         --gpu   start on the CUDA backend (needs --features cuda + a device)\n\
         --full  disable the 1600-px working-copy cap (full-resolution renders)\n\
         \n\
         keys: [ ] cycle kernel · G toggle backend · D difference ×64 ·\n\
         R reroll grain seed · drop a file to load it",
        names.join(" ")
    )
}

fn main() -> eframe::Result {
    let mut kernel = KernelId::Exposure;
    let mut image: Option<std::path::PathBuf> = None;
    let mut use_gpu = false;
    let mut full = false;
    let mut force_scene: Option<&str> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            "--gpu" => use_gpu = true,
            "--full" => full = true,
            "--scene" => match it.next().map(String::as_str) {
                Some("photo") => force_scene = Some("photo"),
                Some("macbeth") => force_scene = Some("macbeth"),
                other => {
                    eprintln!("--scene needs photo|macbeth, got {other:?}\n\n{}", usage());
                    std::process::exit(2);
                }
            },
            name => {
                if let Some(k) = KernelId::from_name(name) {
                    kernel = k;
                } else if std::path::Path::new(name).is_file() {
                    image = Some(name.into());
                } else {
                    eprintln!("unknown kernel or missing file: {name}\n\n{}", usage());
                    std::process::exit(2);
                }
            }
        }
    }

    let mut app = app::ViewerApp::new(kernel, image, use_gpu, full);
    if let Some(scene) = force_scene {
        app.force_scene(scene);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("phaios kernel-viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "phaios kernel-viewer",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
}
