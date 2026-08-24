// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact geometry: crop and the eight dihedral orientations.
//!
//! These are the operations that must be **bit-exact** and **owned by
//! the core**, because their parameters live in consumers' sidecar
//! files: if two front ends implemented the same crop differently, the
//! same sidecar would render different images and the reproducibility
//! promise would break silently.
//!
//! Both kernels are pure index permutations — no arithmetic on pixel
//! values at all — so they are bit-identical across every backend,
//! unconditionally.
//!
//! **Pipeline position: geometry runs first**, before exposure and the
//! look pipeline. Two kernels make the order load-bearing:
//! [`crate::vignette`] centres on the frame it is given, which must be
//! the *cropped* frame; and [`crate::film_grain`] keys its noise to
//! pixel coordinates, which must be the final grid. Orientation before
//! crop, so crop rectangles are expressed in the upright image.
//!
//! Reference for the orientation encoding: JEITA CP-3451 (Exif 2.3),
//! tag 0x0112 — the enum discriminants are the Exif values 1..=8.

use ndarray::{Array3, ArrayView3, s};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Orientation ───────────────────────────────────────────────────────────────

/// One of the eight dihedral transforms of the frame, encoded as its
/// Exif orientation value (JEITA CP-3451, tag 0x0112).
///
/// Rotations are **clockwise**, matching the Exif reading ("Rotate 90
/// CW" is what a viewer must apply to display the frame upright).
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Orientation {
    /// No transform (Exif 1).
    #[default]
    Normal = 1,
    /// Mirror left-right (Exif 2).
    FlipHorizontal = 2,
    /// Rotate 180° (Exif 3).
    Rotate180 = 3,
    /// Mirror top-bottom (Exif 4).
    FlipVertical = 4,
    /// Transpose: mirror about the main diagonal (Exif 5).
    Transpose = 5,
    /// Rotate 90° clockwise (Exif 6).
    Rotate90 = 6,
    /// Transverse: mirror about the anti-diagonal (Exif 7).
    Transverse = 7,
    /// Rotate 270° clockwise (Exif 8).
    Rotate270 = 8,
}

impl Orientation {
    /// From a raw Exif orientation value, if valid (1..=8).
    #[must_use]
    pub fn from_exif(value: u16) -> Option<Orientation> {
        Some(match value {
            1 => Orientation::Normal,
            2 => Orientation::FlipHorizontal,
            3 => Orientation::Rotate180,
            4 => Orientation::FlipVertical,
            5 => Orientation::Transpose,
            6 => Orientation::Rotate90,
            7 => Orientation::Transverse,
            8 => Orientation::Rotate270,
            _ => return None,
        })
    }

    /// The orientation that undoes this one.
    ///
    /// Six of the eight are involutions; the two quarter-turns invert
    /// each other.
    #[must_use]
    pub fn inverse(self) -> Orientation {
        match self {
            Orientation::Rotate90 => Orientation::Rotate270,
            Orientation::Rotate270 => Orientation::Rotate90,
            other => other,
        }
    }

    /// Whether this transform swaps width and height.
    #[must_use]
    pub fn transposes(self) -> bool {
        matches!(
            self,
            Orientation::Transpose
                | Orientation::Rotate90
                | Orientation::Transverse
                | Orientation::Rotate270
        )
    }

    /// Decompose into (transpose, flip_y, flip_x) applied in that order
    /// to *source* coordinates — the shared definition both backends
    /// implement, so they cannot disagree.
    pub(crate) fn flags(self) -> (bool, bool, bool) {
        match self {
            Orientation::Normal => (false, false, false),
            Orientation::FlipHorizontal => (false, false, true),
            Orientation::Rotate180 => (false, true, true),
            Orientation::FlipVertical => (false, true, false),
            Orientation::Transpose => (true, false, false),
            Orientation::Rotate90 => (true, false, true),
            Orientation::Transverse => (true, true, true),
            Orientation::Rotate270 => (true, true, false),
        }
    }
}

// ── Crop parameters ───────────────────────────────────────────────────────────

/// A crop rectangle, in pixels of the (already oriented) input frame.
///
/// `x`, `y` locate the top-left corner; the rectangle must lie entirely
/// within the image.
///
/// ```python
/// params = phaios_core.CropParams(x=100, y=50, width=3000, height=2000)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct CropParams {
    /// Left edge, pixels from the left of the frame.
    #[pyo3(get, set)]
    pub x: u32,
    /// Top edge, pixels from the top of the frame.
    #[pyo3(get, set)]
    pub y: u32,
    /// Width of the result in pixels.
    #[pyo3(get, set)]
    pub width: u32,
    /// Height of the result in pixels.
    #[pyo3(get, set)]
    pub height: u32,
}

#[pymethods]
impl CropParams {
    /// Create new ``CropParams``.
    #[new]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Two ``CropParams`` are equal when all four fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.x == other.x
            && self.y == other.y
            && self.width == other.width
            && self.height == other.height
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "CropParams(x={}, y={}, width={}, height={})",
            self.x, self.y, self.width, self.height
        )
    }
}

/// Validate that the rectangle lies within `shape`. Shared verbatim by
/// the CPU kernel and the CUDA kernel so both backends reject exactly
/// the same inputs with exactly the same messages.
pub(crate) fn validate_crop(shape: &[usize], params: &CropParams) -> Result<(), PhaiosError> {
    let (h, w) = (shape[0] as u64, shape[1] as u64);
    let right = params.x as u64 + params.width as u64;
    let bottom = params.y as u64 + params.height as u64;
    if right > w || bottom > h {
        return Err(PhaiosError::Parameter(format!(
            "crop rectangle {}x{}+{}+{} exceeds the {w}x{h} frame",
            params.width, params.height, params.x, params.y
        )));
    }
    Ok(())
}

// ── Kernels ──────────────────────────────────────────────────────────────────

/// Crop to a rectangle.
///
/// A pure index copy: no pixel value is touched, so the result is
/// bit-identical on every backend. Zero-size rectangles are legal and
/// return an empty array, consistent with the crate's zero-size policy.
///
/// Input shape: `(H, W, C)`, any channel count, any layout.
/// Output shape: `(height, width, C)`, freshly allocated, C-contiguous.
///
/// Order-sensitive: geometry runs **first** in the pipeline — see the
/// module documentation for why vignette and grain make this
/// load-bearing.
///
/// # Errors
/// [`PhaiosError::Parameter`] if the rectangle does not lie entirely
/// within the frame.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn crop(img: ArrayView3<f32>, params: &CropParams) -> Result<Array3<f32>, PhaiosError> {
    validate_crop(img.shape(), params)?;
    let (x, y) = (params.x as usize, params.y as usize);
    let (w, h) = (params.width as usize, params.height as usize);
    let c = img.dim().2;

    let mut out = Array3::<f32>::zeros((h, w, c));
    if h > 0 && w > 0 {
        out.assign(&img.slice(s![y..y + h, x..x + w, ..]));
    }
    Ok(out)
}

/// Apply one of the eight dihedral orientations.
///
/// A pure index permutation: no pixel value is touched, so the result
/// is bit-identical on every backend. The four transposing variants
/// swap the output's width and height.
///
/// Input shape: `(H, W, C)`, any channel count, any layout.
/// Output shape: `(H, W, C)` or `(W, H, C)`, freshly allocated,
/// C-contiguous.
///
/// Order-sensitive: orientation precedes crop, so crop rectangles are
/// expressed in the upright frame.
///
/// # Errors
/// Currently infallible — every input and orientation is valid. The
/// `Result` is kept for signature consistency across kernels.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn orient(img: ArrayView3<f32>, orientation: Orientation) -> Result<Array3<f32>, PhaiosError> {
    let (h, w, c) = img.dim();
    let (transpose, flip_y, flip_x) = orientation.flags();

    let (oh, ow) = if transpose { (w, h) } else { (h, w) };
    let mut out = Array3::<f32>::zeros((oh, ow, c));
    if oh == 0 || ow == 0 {
        return Ok(out);
    }

    // Build the source view whose element (oy, ox, c) is exactly the
    // output element, then let ndarray do one layout-normalising copy.
    let mut view = img;
    if transpose {
        view = view.permuted_axes([1, 0, 2]);
    }
    if flip_y {
        view = view.slice_move(s![..;-1, .., ..]);
    }
    if flip_x {
        view = view.slice_move(s![.., ..;-1, ..]);
    }
    out.assign(&view);
    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| (y * w * c + x * c + ch) as f32)
    }

    #[test]
    fn crop_extracts_the_exact_rectangle() {
        let img = numbered(6, 8, 3);
        let out = crop(img.view(), &CropParams::new(2, 1, 4, 3)).unwrap();
        assert_eq!(out.dim(), (3, 4, 3));
        for y in 0..3 {
            for x in 0..4 {
                for ch in 0..3 {
                    assert_eq!(out[[y, x, ch]], img[[y + 1, x + 2, ch]]);
                }
            }
        }
    }

    #[test]
    fn crop_full_frame_is_identity() {
        let img = numbered(5, 7, 1);
        let out = crop(img.view(), &CropParams::new(0, 0, 7, 5)).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn crop_zero_size_is_legal() {
        let img = numbered(4, 4, 3);
        let out = crop(img.view(), &CropParams::new(2, 2, 0, 0)).unwrap();
        assert_eq!(out.dim(), (0, 0, 3));
    }

    #[test]
    fn crop_rejects_rectangles_outside_the_frame() {
        let img = numbered(4, 4, 1);
        for bad in [
            CropParams::new(0, 0, 5, 4),
            CropParams::new(0, 0, 4, 5),
            CropParams::new(1, 0, 4, 4),
            CropParams::new(4, 4, 1, 1),
            // Near-overflow coordinates must not wrap.
            CropParams::new(u32::MAX, 0, 2, 2),
        ] {
            assert!(
                matches!(
                    crop(img.view(), &bad).unwrap_err(),
                    PhaiosError::Parameter(_)
                ),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn crop_composes() {
        // crop(crop(img, a), b) == crop(img, a ∘ b) — the property that
        // makes interactive crop refinement safe.
        let img = numbered(16, 16, 3);
        let outer = crop(img.view(), &CropParams::new(2, 3, 10, 9)).unwrap();
        let inner = crop(outer.view(), &CropParams::new(1, 2, 5, 4)).unwrap();
        let direct = crop(img.view(), &CropParams::new(3, 5, 5, 4)).unwrap();
        assert_eq!(inner, direct);
    }

    #[test]
    fn orient_dimensions_and_corner_tracking() {
        // Track the top-left corner pixel through every orientation; its
        // destination uniquely identifies each of the eight transforms.
        let img = numbered(2, 3, 1);
        let tl = img[[0, 0, 0]];
        type Case = (Orientation, (usize, usize, usize), (usize, usize));
        let cases: [Case; 8] = [
            (Orientation::Normal, (2, 3, 1), (0, 0)),
            (Orientation::FlipHorizontal, (2, 3, 1), (0, 2)),
            (Orientation::Rotate180, (2, 3, 1), (1, 2)),
            (Orientation::FlipVertical, (2, 3, 1), (1, 0)),
            (Orientation::Transpose, (3, 2, 1), (0, 0)),
            (Orientation::Rotate90, (3, 2, 1), (0, 1)),
            (Orientation::Transverse, (3, 2, 1), (2, 1)),
            (Orientation::Rotate270, (3, 2, 1), (2, 0)),
        ];
        for (o, dim, (ty, tx)) in cases {
            let out = orient(img.view(), o).unwrap();
            assert_eq!(out.dim(), dim, "{o:?} output shape");
            assert_eq!(
                out[[ty, tx, 0]],
                tl,
                "{o:?}: top-left pixel landed in the wrong place"
            );
            assert!(out.is_standard_layout());
        }
    }

    #[test]
    fn every_orientation_undoes_through_its_inverse() {
        let img = numbered(5, 7, 3);
        for value in 1..=8_u16 {
            let o = Orientation::from_exif(value).unwrap();
            let there = orient(img.view(), o).unwrap();
            let back = orient(there.view(), o.inverse()).unwrap();
            assert_eq!(back, img, "{o:?} ∘ inverse ≠ identity");
        }
    }

    #[test]
    fn rotate90_four_times_is_identity() {
        let img = numbered(4, 6, 2);
        let mut x = img.clone();
        for _ in 0..4 {
            x = orient(x.view(), Orientation::Rotate90).unwrap();
        }
        assert_eq!(x, img);
    }

    #[test]
    fn from_exif_round_trips_and_rejects() {
        for v in 1..=8_u16 {
            assert_eq!(Orientation::from_exif(v).unwrap() as u16, v);
        }
        assert!(Orientation::from_exif(0).is_none());
        assert!(Orientation::from_exif(9).is_none());
    }

    #[test]
    fn accepts_any_layout() {
        let img = numbered(8, 6, 3);
        let strided = img.slice(s![..;2, ..;-1, ..]);
        let params = CropParams::new(1, 1, 3, 2);

        let a = crop(strided, &params).unwrap();
        let b = crop(strided.to_owned().view(), &params).unwrap();
        assert_eq!(a, b);

        let a = orient(strided, Orientation::Rotate90).unwrap();
        let b = orient(strided.to_owned().view(), Orientation::Rotate90).unwrap();
        assert_eq!(a, b);
        assert!(a.is_standard_layout());
    }
}
