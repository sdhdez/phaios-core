// SPDX-License-Identifier: GPL-3.0-or-later
//! Exposure compensation.
//!
//! Multiplies every value by `2^stops`. This opens the *look* pipeline —
//! the geometry kernels (`orient`, `straighten`, `crop`, `resize`) run
//! before it, on the delivered scene-referred linear f32 data — and it is
//! the only kernel meaningful on both RGB and luminance input.
//!
//! In linear scene-referred data a stop *is* a factor of two — that is
//! what makes the operation a single multiply. Applying it after a
//! transfer function would not be exposure; it would be an arbitrary
//! curve. Hence the pipeline position: before B&W conversion, and long
//! before [`crate::encode`].
//!
//! Values are not clamped. Highlights pushed above 1.0 stay above 1.0,
//! which is what lets a later stage pull them back down — clipping here
//! would destroy information the tone stages need.
//!
//! Reference: the stop as a doubling of luminous exposure is standard
//! photographic practice; see Ansel Adams, *The Negative*, Little, Brown
//! (1948), chapter 4, on exposure and the density scale.

use ndarray::{Array3, ArrayView3};

use crate::error::PhaiosError;

/// Validate `stops`. Shared verbatim by the CPU kernel below and the
/// CUDA kernel in `src/cuda/` (feature-gated), so both backends reject
/// exactly the same inputs with exactly the same message.
pub(crate) fn validate(stops: f32) -> Result<(), PhaiosError> {
    if !stops.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "stops is {stops}, expected a finite number of EV"
        )));
    }
    Ok(())
}

/// Apply exposure compensation of `stops` EV.
///
/// Computes `out = in · 2^stops`. Positive values brighten, negative
/// values darken; `0.0` is the identity.
///
/// Input shape: `(H, W, C)` for any channel count, in any memory
/// layout. Output shape: `(H, W, C)`, freshly allocated and
/// C-contiguous.
///
/// Order-sensitive: this opens the look pipeline, operating on linear
/// scene-referred data — after the geometry kernels, before B&W
/// conversion. Applying it after a transfer function would not be
/// exposure. See the module documentation.
///
/// Values above 1.0 are preserved, not clipped: highlight headroom is
/// the caller's to spend downstream.
///
/// # Errors
/// Returns [`PhaiosError::Parameter`] if `stops` is not finite.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn exposure(img: ArrayView3<f32>, stops: f32) -> Result<Array3<f32>, PhaiosError> {
    validate(stops)?;

    // One multiply per pixel: 2^stops is constant across the image.
    let gain = 2.0_f32.powf(stops);
    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;
    ndarray::Zip::from(&mut out).and(img).par_for_each(|o, &v| {
        *o = v * gain;
    });
    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn one_stop_doubles() {
        let img = array![[[0.18_f32, 0.5, 1.0]]];
        let out = exposure(img.view(), 1.0).unwrap();
        assert!((out[[0, 0, 0]] - 0.36).abs() < 1e-6);
        assert!((out[[0, 0, 1]] - 1.0).abs() < 1e-6);
        assert!((out[[0, 0, 2]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn minus_one_stop_halves() {
        let img = array![[[0.36_f32]]];
        let out = exposure(img.view(), -1.0).unwrap();
        assert!((out[[0, 0, 0]] - 0.18).abs() < 1e-6);
    }

    #[test]
    fn zero_stops_is_identity() {
        let img = Array3::<f32>::from_shape_fn((4, 4, 3), |(y, x, c)| (y + x + c) as f32 / 10.0);
        let out = exposure(img.view(), 0.0).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn highlights_are_not_clipped() {
        // Pushing above 1.0 must survive: later tone stages need the
        // headroom, and clipping here would be irreversible.
        let img = array![[[0.9_f32]]];
        let out = exposure(img.view(), 2.0).unwrap();
        assert!(
            (out[[0, 0, 0]] - 3.6).abs() < 1e-5,
            "got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn stops_compose_additively() {
        // +1 then +2 must equal +3, to f32 precision.
        let img = Array3::<f32>::from_elem((8, 8, 1), 0.1_f32);
        let stepwise = exposure(exposure(img.view(), 1.0).unwrap().view(), 2.0).unwrap();
        let direct = exposure(img.view(), 3.0).unwrap();
        for (&a, &b) in stepwise.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((6, 4, 3), |(y, x, c)| (y * 12 + x * 3 + c) as f32);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        assert!(!strided.is_standard_layout());
        let out = exposure(strided, 1.0).unwrap();
        assert_eq!(out.dim(), (3, 4, 3));
        assert!(out.is_standard_layout());
        assert_eq!(out, exposure(strided.to_owned().view(), 1.0).unwrap());

        // Luminance input is equally valid.
        assert!(exposure(Array3::<f32>::zeros((2, 2, 1)).view(), 0.5).is_ok());
    }

    #[test]
    fn rejects_non_finite_stops() {
        let img = array![[[0.5_f32]]];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                exposure(img.view(), bad).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
    }
}
