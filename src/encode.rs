// SPDX-License-Identifier: GPL-3.0-or-later
//! sRGB transfer encoding (terminal pipeline stage).
//!
//! Applies the IEC 61966-2-1 piecewise transfer function to convert
//! scene-referred linear f32 values to display-referred sRGB. This
//! must be the **last** kernel in the pipeline; all other kernels
//! operate on linear data.
//!
//! The transfer function is:
//!
//! ```text
//! f(x) = 12.92 · x                       if x ≤ 0.0031308
//! f(x) = 1.055 · x^(1/2.4) − 0.055      if x > 0.0031308
//! ```
//!
//! Values are **not** clamped by this kernel — pass values in [0, 1]
//! if downstream code requires display-referred values in that range.
//!
//! Reference: IEC 61966-2-1:1999, "Multimedia systems and equipment —
//! Colour measurement and management — Part 2-1: Colour management —
//! Default RGB colour space — sRGB."

use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;

use crate::error::PhaiosError;

/// sRGB threshold between the linear and power segments.
const THRESHOLD: f32 = 0.0031308;

/// Apply the IEC 61966-2-1 sRGB piecewise transfer to a single value.
#[inline]
fn encode_pixel(x: f32) -> f32 {
    if x <= THRESHOLD {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Apply the IEC 61966-2-1 sRGB transfer to a linear image.
///
/// Element-wise mapping: `encode_pixel(x)` is applied to every value.
/// Order-sensitive: this is always the last stage of the pipeline.
///
/// Input shape: any `(H, W, C)` — linear scene-referred f32 values.
/// Output shape: `(H, W, C)` — display-referred sRGB f32 values.
///
/// Values are **not** clamped. Inputs outside [0, 1] produce outputs
/// outside the standard display range; the caller is responsible for
/// clamping if required.
///
/// Reference: IEC 61966-2-1:1999.
///
/// # Errors
/// Returns [`PhaiosError::Shape`] if the input does not have 3 dimensions.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn encode_srgb(img: ArrayView3<f32>) -> Result<Array3<f32>, PhaiosError> {
    if img.ndim() != 3 {
        return Err(PhaiosError::Shape(format!(
            "encode_srgb expects a 3-D array, got {} dimensions",
            img.ndim()
        )));
    }
    let shape = img.dim();
    let in_slice = img
        .as_slice()
        .expect("encode_srgb: input must be C-contiguous");
    let mut out_data: Vec<f32> = vec![0.0_f32; in_slice.len()];
    out_data
        .par_iter_mut()
        .zip(in_slice.par_iter())
        .for_each(|(o, &v)| {
            *o = encode_pixel(v);
        });
    Array3::from_shape_vec(shape, out_data)
        .map_err(|e| PhaiosError::Shape(format!("encode_srgb shape error (internal): {e}")))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_encode_zero_is_zero() {
        assert_eq!(encode_pixel(0.0), 0.0);
    }

    #[test]
    fn srgb_encode_one_is_one() {
        let v = encode_pixel(1.0);
        assert!((v - 1.0).abs() < 1e-5, "encode(1.0) = {v}");
    }

    #[test]
    fn srgb_encode_monotonic() {
        let n = 10_000_usize;
        let mut prev = encode_pixel(0.0);
        for i in 1..=n {
            let x = i as f32 / n as f32;
            let y = encode_pixel(x);
            assert!(
                y >= prev,
                "encode not monotonic at x={x}: encode({})={prev} > encode({x})={y}",
                (i - 1) as f32 / n as f32
            );
            prev = y;
        }
    }

    #[test]
    fn srgb_c1_continuous_at_threshold() {
        // Both branches must agree at the threshold to within 1e-5.
        let linear = 12.92 * THRESHOLD;
        let power = 1.055 * THRESHOLD.powf(1.0 / 2.4) - 0.055;
        assert!(
            (linear - power).abs() < 1e-5,
            "C¹ discontinuity: linear={linear}, power={power}, diff={}",
            (linear - power).abs()
        );
    }
}
