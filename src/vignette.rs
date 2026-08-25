// SPDX-License-Identifier: GPL-3.0-or-later
//! Radial vignette — corner darkening or lightening.
//!
//! A finishing stage: it multiplies each pixel by a factor that depends
//! only on where the pixel sits relative to the frame centre.
//!
//! ```text
//! out = in · (1 − amount · falloff(distance))
//! ```
//!
//! Real lenses vignette because off-axis illumination falls as cos⁴ of
//! the field angle (Ray, *Applied Photographic Optics*, 3rd ed., Focal
//! Press 2002, §14). This kernel does not model that: a physical
//! correction has to know the lens, the aperture and the focal length,
//! and belongs to the RAW decoder. What is implemented here is the
//! darkroom gesture — burning the edges to hold a viewer's eye in the
//! frame — with a shape the photographer chooses directly.
//!
//! Distance is measured in normalised frame coordinates, so the result
//! is resolution-independent: the same parameters give the same picture
//! whether applied to a full frame or to a preview one eighth the size.
//! A consumer rendering a preview and then the full image gets a
//! matching vignette without rescaling anything.

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`vignette`].
///
/// ```python
/// # Classic subtle corner burn
/// params = phaios_core.VignetteParams(amount=0.35, feather=0.6, roundness=0.0)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct VignetteParams {
    /// Strength at the corners. **Positive darkens, negative lightens.**
    ///
    /// −1..+1 is the useful range: at `+1.0` the corners reach black, at
    /// `−1.0` they are doubled. Values beyond that are allowed; the
    /// result is clamped at zero so the image never goes negative.
    #[pyo3(get, set)]
    pub amount: f32,
    /// Width of the transition, 0..=1.
    ///
    /// `1.0` spreads the falloff from the centre all the way to the
    /// corners — the gentlest, most natural-looking option. Smaller
    /// values push the transition outwards, concentrating it near the
    /// corners; `0.0` is a hard edge with no gradient at all.
    #[pyo3(get, set)]
    pub feather: f32,
    /// Corner shape, 0..=1. `0.0` is a circle, `1.0` follows the frame.
    ///
    /// A circular vignette darkens the middle of each edge as well as
    /// the corners; at `1.0` the iso-lines are rectangles parallel to
    /// the frame, so only the border darkens, evenly.
    #[pyo3(get, set)]
    pub roundness: f32,
}

#[pymethods]
impl VignetteParams {
    /// Create new ``VignetteParams``.
    #[new]
    #[pyo3(signature = (amount = 0.0, feather = 0.5, roundness = 0.0))]
    pub fn new(amount: f32, feather: f32, roundness: f32) -> Self {
        Self {
            amount,
            feather,
            roundness,
        }
    }

    /// Two ``VignetteParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.amount.to_bits() == other.amount.to_bits()
            && self.feather.to_bits() == other.feather.to_bits()
            && self.roundness.to_bits() == other.roundness.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "VignetteParams(amount={}, feather={}, roundness={})",
            self.amount, self.feather, self.roundness
        )
    }
}

impl Default for VignetteParams {
    fn default() -> Self {
        Self {
            amount: 0.0,
            feather: 0.5,
            roundness: 0.0,
        }
    }
}

// ── Falloff ───────────────────────────────────────────────────────────────────

/// Hermite smoothstep: 0 below `edge0`, 1 above `edge1`, S-curve between.
///
/// `3t² − 2t³` is the cubic with zero derivative at both ends, so the
/// vignette meets the untouched centre and the fully-applied corner
/// without a visible seam. A linear ramp would leave a mach band at
/// each end.
///
/// Reference: Ebert et al., *Texturing & Modeling: A Procedural
/// Approach*, 3rd ed., Morgan Kaufmann (2003), §2.3.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        // Degenerate window: fall back to a hard step at the edge.
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate vignette parameters. Shared by CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate(params: &VignetteParams) -> Result<(), PhaiosError> {
    if !params.amount.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "amount is {}, expected a finite value",
            params.amount
        )));
    }
    if !params.feather.is_finite() || !(0.0..=1.0).contains(&params.feather) {
        return Err(PhaiosError::Parameter(format!(
            "feather is {}, expected a value in 0..=1",
            params.feather
        )));
    }
    if !params.roundness.is_finite() || !(0.0..=1.0).contains(&params.roundness) {
        return Err(PhaiosError::Parameter(format!(
            "roundness is {}, expected a value in 0..=1",
            params.roundness
        )));
    }
    Ok(())
}

/// Apply a radial vignette.
///
/// For each pixel, normalised coordinates `(nx, ny)` are formed with the
/// frame centre at the origin and the corners at `(±1, ±1)`. Two
/// distance measures are blended by `roundness`:
///
/// - the Euclidean distance `√(nx² + ny²) / √2` — a circle, which
///   reaches 1 only at the corners;
/// - the Chebyshev distance `max(|nx|, |ny|)` — a rectangle following
///   the frame, which reaches 1 along the whole border.
///
/// The blended distance is passed through a smoothstep from
/// `1 − feather` to `1`, and the pixel is scaled by
/// `1 − amount · falloff`, clamped at zero.
///
/// Because the coordinates are normalised, output is resolution
/// independent: the same parameters produce the same picture on a
/// preview and on the full-size frame.
///
/// Input shape: `(H, W, C)`, any channel count, any layout. All channels
/// of a pixel receive the same factor, so this is equally correct on
/// luminance and on a split-toned three-channel image. Output:
/// `(H, W, C)`, C-contiguous.
///
/// Order-sensitive: apply after tone stages. Vignetting before them
/// makes the tone curve act on the darkened corners, which is a
/// different picture and rarely the intended one.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if any field is not finite, or if
///   `feather` or `roundness` is outside 0..=1.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn vignette(img: ArrayView3<f32>, params: &VignetteParams) -> Result<Array3<f32>, PhaiosError> {
    validate(params)?;

    let (h, w, _) = img.dim();
    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;

    // amount = 0 is the identity, and it is the default: skip the work.
    if params.amount == 0.0 || h == 0 || w == 0 {
        out.assign(&img);
        return Ok(out);
    }

    let VignetteParams {
        amount,
        feather,
        roundness,
    } = *params;
    let inner = 1.0 - feather;

    // Pixel centres map to (-1, 1): with h == 1 the single row sits on
    // the centre line rather than at an edge.
    let half_h = h as f32 / 2.0;
    let half_w = w as f32 / 2.0;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    ndarray::Zip::indexed(&mut out)
        .and(img)
        .par_for_each(|(y, x, _), o, &v| {
            let ny = (y as f32 + 0.5 - half_h) / half_h;
            let nx = (x as f32 + 0.5 - half_w) / half_w;

            let circular = (nx * nx + ny * ny).sqrt() * inv_sqrt2;
            let rectangular = nx.abs().max(ny.abs());
            let distance = circular + (rectangular - circular) * roundness;

            let falloff = smoothstep(inner, 1.0, distance);
            *o = (v * (1.0 - amount * falloff)).max(0.0);
        });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_elem((h, w, c), 1.0_f32)
    }

    #[test]
    fn zero_amount_is_identity() {
        let img = Array3::<f32>::from_shape_fn((16, 24, 3), |(y, x, c)| (y + x + c) as f32);
        let out = vignette(img.view(), &VignetteParams::default()).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn positive_amount_darkens_corners_not_the_centre() {
        let img = flat(64, 64, 1);
        let out = vignette(img.view(), &VignetteParams::new(0.5, 1.0, 0.0)).unwrap();

        let centre = out[[32, 32, 0]];
        let corner = out[[0, 0, 0]];
        assert!(
            (centre - 1.0).abs() < 1e-3,
            "centre should be untouched, got {centre}"
        );
        assert!(corner < 0.6, "corner should darken, got {corner}");
    }

    #[test]
    fn negative_amount_lightens_corners() {
        let img = flat(64, 64, 1);
        let out = vignette(img.view(), &VignetteParams::new(-0.5, 1.0, 0.0)).unwrap();
        assert!(out[[0, 0, 0]] > 1.4, "corner should lighten");
        assert!((out[[32, 32, 0]] - 1.0).abs() < 1e-3, "centre untouched");
    }

    #[test]
    fn output_is_never_negative() {
        // amount beyond the useful range must not flip the image.
        let img = flat(32, 32, 1);
        let out = vignette(img.view(), &VignetteParams::new(4.0, 1.0, 0.0)).unwrap();
        for &v in out.iter() {
            assert!(v >= 0.0, "got {v}");
        }
        assert_eq!(out[[0, 0, 0]], 0.0, "corner should clamp to black");
    }

    #[test]
    fn result_is_symmetric_about_both_axes() {
        let img = flat(32, 48, 1);
        let out = vignette(img.view(), &VignetteParams::new(0.7, 0.8, 0.3)).unwrap();
        for y in 0..32 {
            for x in 0..48 {
                let mirrored_x = out[[y, 47 - x, 0]];
                let mirrored_y = out[[31 - y, x, 0]];
                assert!(
                    (out[[y, x, 0]] - mirrored_x).abs() < 1e-6,
                    "not symmetric horizontally at ({y}, {x})"
                );
                assert!(
                    (out[[y, x, 0]] - mirrored_y).abs() < 1e-6,
                    "not symmetric vertically at ({y}, {x})"
                );
            }
        }
    }

    #[test]
    fn roundness_changes_which_regions_darken() {
        // A circular vignette darkens the middle of each edge; a
        // rectangular one leaves it much closer to the border value.
        let img = flat(64, 64, 1);
        let circular = vignette(img.view(), &VignetteParams::new(0.8, 1.0, 0.0)).unwrap();
        let rectangular = vignette(img.view(), &VignetteParams::new(0.8, 1.0, 1.0)).unwrap();

        let edge_mid_circular = circular[[32, 0, 0]];
        let edge_mid_rect = rectangular[[32, 0, 0]];
        assert!(
            edge_mid_rect < edge_mid_circular,
            "rectangular should darken the edge midpoint more: {edge_mid_rect} vs {edge_mid_circular}"
        );

        // Corners agree: both distance measures reach 1 there.
        assert!((circular[[0, 0, 0]] - rectangular[[0, 0, 0]]).abs() < 0.05);
    }

    #[test]
    fn is_resolution_independent() {
        // The same parameters on a preview and on the full frame must
        // give the same picture, so a consumer can render either.
        let params = VignetteParams::new(0.6, 0.7, 0.2);
        let small = vignette(flat(64, 64, 1).view(), &params).unwrap();
        let large = vignette(flat(512, 512, 1).view(), &params).unwrap();

        for (sy, sx) in [(0, 0), (16, 16), (32, 32), (63, 0)] {
            let (ly, lx) = (sy * 8 + 4, sx * 8 + 4);
            let a = small[[sy, sx, 0]];
            let b = large[[ly, lx, 0]];
            assert!(
                (a - b).abs() < 0.02,
                "preview and full frame disagree at ({sy}, {sx}): {a} vs {b}"
            );
        }
    }

    #[test]
    fn feather_controls_the_transition_width() {
        let img = flat(64, 64, 1);
        let wide = vignette(img.view(), &VignetteParams::new(0.8, 1.0, 0.0)).unwrap();
        let narrow = vignette(img.view(), &VignetteParams::new(0.8, 0.2, 0.0)).unwrap();

        // Halfway out, a wide feather is already darkening; a narrow one
        // has not started.
        let mid_wide = wide[[32, 48, 0]];
        let mid_narrow = narrow[[32, 48, 0]];
        assert!(
            mid_wide < mid_narrow,
            "wide feather should reach further in: {mid_wide} vs {mid_narrow}"
        );
        assert!(
            (mid_narrow - 1.0).abs() < 1e-4,
            "narrow should be untouched"
        );
    }

    #[test]
    fn all_channels_of_a_pixel_share_the_factor() {
        // Otherwise the vignette would tint, not darken.
        let img = Array3::from_shape_fn((16, 16, 3), |(_, _, c)| 0.2 + c as f32 * 0.3);
        let out = vignette(img.view(), &VignetteParams::new(0.5, 1.0, 0.0)).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let ratios: Vec<f32> = (0..3).map(|c| out[[y, x, c]] / img[[y, x, c]]).collect();
                assert!(
                    (ratios[0] - ratios[1]).abs() < 1e-5 && (ratios[1] - ratios[2]).abs() < 1e-5,
                    "channel factors differ at ({y}, {x}): {ratios:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        let img = flat(8, 8, 1);
        for bad in [-0.1_f32, 1.1, f32::NAN] {
            assert!(matches!(
                vignette(img.view(), &VignetteParams::new(0.5, bad, 0.0)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
            assert!(matches!(
                vignette(img.view(), &VignetteParams::new(0.5, 0.5, bad)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        assert!(matches!(
            vignette(img.view(), &VignetteParams::new(f32::NAN, 0.5, 0.0)).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
    }

    #[test]
    fn accepts_any_layout() {
        let img = Array3::<f32>::from_shape_fn((8, 6, 1), |(y, x, _)| (y * 6 + x) as f32 / 48.0);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let params = VignetteParams::new(0.5, 0.8, 0.0);
        let out = vignette(strided, &params).unwrap();
        assert_eq!(out.dim(), (4, 6, 1));
        assert!(out.is_standard_layout());
        assert_eq!(out, vignette(strided.to_owned().view(), &params).unwrap());
    }

    #[test]
    fn handles_degenerate_shapes() {
        assert!(vignette(flat(1, 1, 1).view(), &VignetteParams::new(0.5, 0.5, 0.0)).is_ok());
        assert!(vignette(flat(1, 64, 1).view(), &VignetteParams::new(0.5, 0.5, 0.0)).is_ok());
        let empty = Array3::<f32>::zeros((0, 8, 1));
        assert_eq!(
            vignette(empty.view(), &VignetteParams::new(0.5, 0.5, 0.0))
                .unwrap()
                .dim(),
            (0, 8, 1)
        );
    }
}
