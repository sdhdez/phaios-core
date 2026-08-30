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
//! The two segments meet continuously at the threshold but their slopes
//! do not: 12.920 below against 12.703 above. The transfer is C⁰ there
//! and **not** C¹, so anything assuming differentiability across the
//! join — a smooth inverse, a gradient, a spline fitted through it —
//! has to treat the two segments separately.
//!
//! Values are **not** clamped by this kernel — pass values in [0, 1]
//! if downstream code requires display-referred values in that range.
//!
//! Reference: IEC 61966-2-1:1999, "Multimedia systems and equipment —
//! Colour measurement and management — Part 2-1: Colour management —
//! Default RGB colour space — sRGB."

use ndarray::{Array3, ArrayView3};

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
/// Input shape: any `(H, W, C)` — linear scene-referred f32 values. Any
/// memory layout is accepted (strided views and Fortran-order arrays
/// included); the output is always freshly allocated and C-contiguous.
/// Output shape: `(H, W, C)` — display-referred sRGB f32 values.
///
/// Unlike the other kernels this one places no constraint on the channel
/// count: the transfer is a scalar function applied element-wise, so it
/// is equally valid on `(H, W, 1)` luminance and `(H, W, 3)` RGB.
///
/// Values are **not** clamped. Inputs outside [0, 1] produce outputs
/// outside the standard display range; the caller is responsible for
/// clamping if required. Negative inputs take the linear branch and stay
/// negative (no NaN).
///
/// Reference: IEC 61966-2-1:1999.
///
/// # Errors
/// Currently infallible — every 3-D `f32` input is valid. The `Result`
/// is retained for API stability and for future variants that may
/// validate the channel count.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn encode_srgb(img: ArrayView3<f32>) -> Result<Array3<f32>, PhaiosError> {
    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;
    ndarray::Zip::from(&mut out).and(img).par_for_each(|o, &v| {
        *o = encode_pixel(v);
    });
    Ok(out)
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
    fn accepts_non_contiguous_input() {
        // Regression: `as_slice().expect(...)` panicked on strided views;
        // across the FFI that surfaced as a `PanicException`.
        let img =
            Array3::<f32>::from_shape_fn((6, 4, 3), |(y, x, c)| (y * 12 + x * 3 + c) as f32 / 72.0);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        assert!(
            !strided.is_standard_layout(),
            "test setup: view should not be C-contiguous"
        );

        let out = encode_srgb(strided).unwrap();
        assert_eq!(out.dim(), (3, 4, 3));
        assert!(out.is_standard_layout(), "output must be C-contiguous");
        assert_eq!(out, encode_srgb(strided.to_owned().view()).unwrap());
    }

    #[test]
    fn negative_input_stays_negative() {
        // Negative values take the linear branch — `powf` on a negative
        // base would produce NaN.
        let img = Array3::<f32>::from_elem((2, 2, 1), -0.05_f32);
        let out = encode_srgb(img.view()).unwrap();
        for &v in out.iter() {
            assert!(v.is_finite(), "negative input produced {v}");
            assert!((v - 12.92 * -0.05).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn srgb_c0_continuous_at_threshold() {
        // The two branches must meet in *value* at the threshold. This
        // exercises the kernel itself rather than re-deriving the formula:
        // `encode_pixel` takes the linear branch at exactly THRESHOLD and
        // the power branch just above it.
        //
        // Note this is C⁰ only. The IEC constants are not C¹: the slope
        // is 12.920 below the threshold and 12.703 above it (1.7% jump).
        // Exact C¹ would need a threshold of 0.0030407 instead.
        let below = encode_pixel(THRESHOLD);
        let above = encode_pixel(THRESHOLD * (1.0 + 1e-5));
        assert!(
            (above - below).abs() < 1e-5,
            "C⁰ discontinuity at threshold: below={below}, above={above}, diff={}",
            (above - below).abs()
        );
    }
}
