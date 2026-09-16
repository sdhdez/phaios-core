// SPDX-License-Identifier: GPL-3.0-or-later
//! DNG / RAW loading via rawler — the viewer's own decode, because
//! phaios-core deliberately never opens RAW files (its CLAUDE.md §1) and
//! this tool sits in the consumer seat, like the desktop app does with
//! rawpy/libraw.
//!
//! Viewer-grade development only: rawler's own pipeline (rescale,
//! demosaic, crop, white balance, colour calibration) **minus the final
//! sRGB gamma step**, so the result is the linear scene-referred data
//! the kernels contract for. Highlight reconstruction, CA and lens
//! corrections are the desktop's job, not this tool's.

use ndarray::Array3;
use rawler::decoders::Orientation;
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};

/// Apply the camera's recorded orientation via the core's geometry
/// kernel — the same code every consumer uses, so a sidecar recording
/// an orientation reproduces identically everywhere. Only the enum
/// mapping (rawler's naming → Exif codes) lives here.
fn apply_orientation(img: Array3<f32>, o: Orientation) -> Array3<f32> {
    let exif = match o {
        Orientation::Normal | Orientation::Unknown => 1_u16,
        Orientation::HorizontalFlip => 2,
        Orientation::Rotate180 => 3,
        Orientation::VerticalFlip => 4,
        Orientation::Transpose => 5,
        Orientation::Rotate90 => 6,
        Orientation::Transverse => 7,
        Orientation::Rotate270 => 8,
    };
    let core = phaios_core::geometry::Orientation::from_exif(exif)
        .expect("exif codes 1..=8 by construction");
    phaios_core::geometry::orient(img.view(), core).expect("orient is infallible")
}

/// Decode and develop a RAW file to linear (H, W, 3) f32.
pub fn load_dng(path: &std::path::Path) -> Result<Array3<f32>, String> {
    let raw = rawler::decode_file(path).map_err(|e| format!("RAW decode failed: {e}"))?;
    let orientation = raw.orientation;

    // rawler's default pipeline, with the terminal SRgb gamma removed:
    // the kernels want linear, and display encoding is color.rs's job.
    let dev = RawDevelop {
        steps: vec![
            ProcessingStep::Rescale,
            ProcessingStep::Demosaic,
            ProcessingStep::CropActiveArea,
            ProcessingStep::WhiteBalance,
            ProcessingStep::Calibrate,
            ProcessingStep::CropDefault,
        ],
    };
    let intermediate = dev
        .develop_intermediate(&raw)
        .map_err(|e| format!("RAW develop failed: {e}"))?;

    match intermediate {
        Intermediate::ThreeColor(pixels) => {
            let dim = pixels.dim();
            let (w, h) = (dim.w, dim.h);
            let data = pixels.into_inner();
            let mut out = Array3::<f32>::zeros((h, w, 3));
            for (i, px) in data.iter().enumerate() {
                let (y, x) = (i / w, i % w);
                for c in 0..3 {
                    out[[y, x, c]] = px[c];
                }
            }
            Ok(apply_orientation(out, orientation))
        }
        Intermediate::Monochrome(pixels) => {
            let dim = pixels.dim();
            let (w, h) = (dim.w, dim.h);
            let data = pixels.into_inner();
            let mut out = Array3::<f32>::zeros((h, w, 3));
            for (i, &v) in data.iter().enumerate() {
                let (y, x) = (i / w, i % w);
                for c in 0..3 {
                    out[[y, x, c]] = v;
                }
            }
            Ok(apply_orientation(out, orientation))
        }
        _ => Err("unsupported RAW colour layout (4-colour)".into()),
    }
}

/// True if the extension looks like a RAW file rawler should try.
pub fn is_raw_extension(path: &std::path::Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("dng" | "raf" | "nef" | "cr2" | "cr3" | "arw" | "orf" | "rw2" | "pef")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode the maintainer's test DNG when present (testdata/ is
    /// gitignored, so this skips on machines without the file).
    #[test]
    fn decodes_the_test_dng() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/R0000096.DNG"
        ));
        if !path.is_file() {
            eprintln!("skip: no test DNG");
            return;
        }
        let img = load_dng(path).expect("decode");
        let (h, w, c) = img.dim();
        assert_eq!(c, 3);
        // GR IIIx active area near 6000×4000; this file is EXIF
        // orientation 8 (Rotate 270 CW), so corrected output is PORTRAIT.
        assert!(
            h > 5900 && w > 3900 && h > w,
            "orientation not applied: got {w}x{h}, expected portrait"
        );
        let mean = img.iter().sum::<f32>() / img.len() as f32;
        assert!(
            mean > 0.005 && mean < 0.9,
            "implausible mean {mean} — decode or scaling broken"
        );
        let max = img.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max > 0.05, "image appears black (max {max})");
        assert!(img.iter().all(|v| v.is_finite()), "non-finite pixels");
    }
}
