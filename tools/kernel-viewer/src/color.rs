// SPDX-License-Identifier: GPL-3.0-or-later
//! Colour management: the one place linear ↔ display conversion happens.
//!
//! The contract (mirrors phaios-core's own):
//! - Everything handed to a kernel is LINEAR scene-referred f32.
//! - Everything shown on screen goes clamp(0,1) → `encode_srgb` → u8.
//! - The `encode_srgb` *kernel view* is already display-referred and
//!   must not be encoded twice (`already_encoded`).

use ndarray::{Array3, ArrayView3};

/// Inverse of the IEC 61966-2-1 sRGB transfer — the decode phaios-core
/// deliberately does not ship (a decoded file is the consumer's job).
#[inline]
pub fn srgb_decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Load a JPEG/PNG from bytes into linear (H, W, 3) f32, honouring the
/// EXIF orientation recorded by the camera.
pub fn load_linear(bytes: &[u8]) -> Result<Array3<f32>, String> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("cannot read image: {e}"))?
        .into_decoder()
        .map_err(|e| format!("cannot decode image: {e}"))?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut dynimg = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| format!("cannot decode image: {e}"))?;
    dynimg.apply_orientation(orientation);
    let img = dynimg.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = Array3::<f32>::zeros((h, w, 3));
    for (x, y, p) in img.enumerate_pixels() {
        for c in 0..3 {
            out[[y as usize, x as usize, c]] = srgb_decode(p.0[c] as f32 / 255.0);
        }
    }
    Ok(out)
}

/// Linear kernel output → egui image for display.
///
/// `already_encoded` is true only for the `encode_srgb` kernel view,
/// whose output is display-referred by definition.
pub fn to_display(linear: ArrayView3<f32>, already_encoded: bool) -> egui::ColorImage {
    let (h, w, c) = linear.dim();
    let quant = |v: f32| -> u8 {
        let clamped = v.clamp(0.0, 1.0);
        let encoded = if already_encoded {
            clamped
        } else {
            // phaios-core's own terminal transfer, element-wise.
            if clamped <= 0.0031308 {
                12.92 * clamped
            } else {
                1.055 * clamped.powf(1.0 / 2.4) - 0.055
            }
        };
        (encoded * 255.0 + 0.5) as u8
    };

    if c == 1 {
        let gray: Vec<u8> = linear.iter().map(|&v| quant(v)).collect();
        egui::ColorImage::from_gray([w, h], &gray)
    } else {
        let mut rgb = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    rgb.push(quant(linear[[y, x, ch]]));
                }
            }
        }
        egui::ColorImage::from_rgb([w, h], &rgb)
    }
}

/// Amplified absolute difference of two linear images: `|a − b| × gain`.
///
/// Black means the backends conform; anything visible is divergence of
/// at least `1/gain` in linear units.
pub fn diff_image(a: ArrayView3<f32>, b: ArrayView3<f32>, gain: f32) -> Array3<f32> {
    let mut out = Array3::<f32>::zeros(a.dim());
    ndarray::Zip::from(&mut out)
        .and(a)
        .and(b)
        .for_each(|o, &x, &y| {
            *o = ((x - y).abs() * gain).clamp(0.0, 1.0);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_inverts_the_crate_encode() {
        // decode(encode(x)) ≈ x across the range, including both branches.
        for i in 0..=1000 {
            let x = i as f32 / 1000.0;
            let img = Array3::from_elem((1, 1, 1), x);
            let encoded = phaios_core::encode::encode_srgb(img.view()).unwrap()[[0, 0, 0]];
            let back = srgb_decode(encoded);
            assert!((back - x).abs() < 1e-5, "round trip failed at {x}: {back}");
        }
    }

    #[test]
    fn decode_branch_point() {
        // 0.04045 display maps to 0.0031308 linear (the encode threshold).
        assert!((srgb_decode(0.04045) - 0.0031308).abs() < 1e-6);
    }
}
