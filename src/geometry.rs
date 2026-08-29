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
//! Reference for the orientation encoding: JEITA CP-3451C / CIPA DC-008-2012 (Exif 2.3),
//! tag 0x0112 — the enum discriminants are the Exif values 1..=8.
//!
//! [`resize`] and [`straighten`] resample, but with **polynomial
//! filters only** (area coverage, triangle, Catmull-Rom) and the one
//! transcendental — `sin`/`cos` of the straighten angle — evaluated
//! once on the host and passed to both backends as identical scalars.
//! Every per-pixel operation is a correctly-rounded mul/add/div/floor
//! in a fixed accumulation order, so both kernels are **bit-exact
//! across backends**, like `crop` and `orient`.
//!
//! Resampling references: the pixel-centre alignment convention and
//! Catmull-Rom kernel follow Keys, "Cubic convolution interpolation
//! for digital image processing", *IEEE Trans. ASSP* 29(6), 1981
//! (a = −0.5); the area filter computes exact fractional pixel
//! coverage, equivalent to integrating a box over the source grid.

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

/// A crop rectangle, in pixels of the frame it is applied to — after
/// `orient` and `straighten` in the standard geometry order, i.e. the
/// upright, levelled frame.
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

    let mut out = crate::alloc::zeros3::<f32>((h, w, c))?;
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
    let mut out = crate::alloc::zeros3::<f32>((oh, ow, c))?;
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

// ── Resize ───────────────────────────────────────────────────────────────────

/// Resampling filter for [`resize`].
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum ResizeFilter {
    /// Exact fractional pixel coverage — the correct filter for
    /// **downscaling** (true area averaging at any ratio). When
    /// upscaling it degenerates to a half-pixel box: one source tap for
    /// most outputs, the average of the two neighbours at exact
    /// half-pixel ties. Pick [`ResizeFilter::CatmullRom`] for
    /// upscaling instead.
    #[default]
    Area = 0,
    /// Triangle filter (bilinear). Cheap, slightly soft.
    Bilinear = 1,
    /// Catmull-Rom cubic (Keys 1981, a = −0.5) — the photographic
    /// default for **upscaling**: sharper than bilinear without the
    /// haloes of stronger sharpening kernels.
    CatmullRom = 2,
}

/// Parameters for [`resize`].
///
/// ```python
/// params = phaios_core.ResizeParams(2048, 1365, phaios_core.ResizeFilter.Area)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct ResizeParams {
    /// Target width in pixels.
    #[pyo3(get, set)]
    pub width: u32,
    /// Target height in pixels.
    #[pyo3(get, set)]
    pub height: u32,
    /// The resampling filter.
    #[pyo3(get, set)]
    pub filter: ResizeFilter,
}

#[pymethods]
impl ResizeParams {
    /// Create new ``ResizeParams``.
    #[new]
    #[pyo3(signature = (width, height, filter = ResizeFilter::Area))]
    pub fn new(width: u32, height: u32, filter: ResizeFilter) -> Self {
        Self {
            width,
            height,
            filter,
        }
    }

    /// Two ``ResizeParams`` are equal when all fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.filter == other.filter
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "ResizeParams(width={}, height={}, filter={:?})",
            self.width, self.height, self.filter
        )
    }
}

/// The filter profile at distance `t` (in *output-scaled* source
/// pixels). One definition; the CUDA kernel transcribes it operation
/// for operation, and the conformance suite holds the two to
/// `assert_eq!`.
#[inline]
pub(crate) fn filter_eval(filter: ResizeFilter, t: f32) -> f32 {
    let t = t.abs();
    match filter {
        // Area is handled by exact coverage in `axis_weights`, not here;
        // this arm is the box profile used when upscaling.
        ResizeFilter::Area => {
            if t <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        ResizeFilter::Bilinear => (1.0 - t).max(0.0),
        ResizeFilter::CatmullRom => {
            if t <= 1.0 {
                ((1.5 * t - 2.5) * t) * t + 1.0
            } else if t < 2.0 {
                ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
            } else {
                0.0
            }
        }
    }
}

/// Filter radius in source pixels for a given axis scale
/// (`scale = in_len / out_len`); minification widens the support.
#[inline]
pub(crate) fn filter_support(filter: ResizeFilter, scale: f32) -> f32 {
    let base = match filter {
        ResizeFilter::Area => 0.5,
        ResizeFilter::Bilinear => 1.0,
        ResizeFilter::CatmullRom => 2.0,
    };
    base * scale.max(1.0)
}

/// The tap range and weight function for output index `i` on one axis.
///
/// Centre alignment: source centre `c = (i + 0.5)·scale − 0.5`. For
/// [`ResizeFilter::Area`] when minifying, the weight of source pixel
/// `k` is its exact overlap with the output pixel's source footprint
/// `[c − scale/2, c + scale/2]`; otherwise it is the filter profile at
/// `(k − c) / max(scale, 1)`. Weights are accumulated and normalised
/// left to right — the fixed order both backends share.
#[inline]
pub(crate) fn axis_taps(scale: f32, i: usize) -> (f32, f32) {
    let c = (i as f32 + 0.5) * scale - 0.5;
    (c, scale.max(1.0))
}

/// Resample to a new size with a separable filter.
///
/// Two passes (horizontal, then vertical), each accumulating its taps
/// left-to-right/top-to-bottom and normalising by the weight sum, so
/// the result is bit-reproducible and — the filters being polynomial —
/// **bit-exact across backends**. Tap coordinates are clamped to the
/// frame (replicate borders), consistent with the crate's other
/// windowed kernels.
///
/// A constant image is preserved to ~1 ULP (the weighted sum and the
/// weight sum round separately before the normalising division), and a
/// same-size resize with any filter is the **exact** identity (centre
/// alignment puts a unit weight on the source pixel).
///
/// Order-sensitive: the **last** geometry stage (after `orient`,
/// `straighten` and `crop`) or export preparation — resampling after
/// grain would change the grain's size on screen.
///
/// Input shape: `(H, W, C)`, any channel count, any layout.
/// Output shape: `(height, width, C)`, C-contiguous.
///
/// # Errors
/// [`PhaiosError::Parameter`] if either target dimension is zero.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn resize(img: ArrayView3<f32>, params: &ResizeParams) -> Result<Array3<f32>, PhaiosError> {
    validate_resize(img.shape(), params)?;
    let (in_h, in_w, c) = img.dim();
    let (out_w, out_h) = (params.width as usize, params.height as usize);

    // Horizontal pass: (in_h, in_w, c) -> (in_h, out_w, c).
    let scale_x = in_w as f32 / out_w as f32;
    let mut mid = crate::alloc::zeros3::<f32>((in_h, out_w, c))?;
    resample_axis1(img, mid.view_mut(), scale_x, params.filter);

    // Vertical pass: transpose H<->W views so the same routine walks
    // the other axis with identical arithmetic.
    let scale_y = in_h as f32 / out_h as f32;
    let mut out = crate::alloc::zeros3::<f32>((out_h, out_w, c))?;
    resample_axis1(
        mid.view().permuted_axes([1, 0, 2]),
        out.view_mut().permuted_axes([1, 0, 2]),
        scale_y,
        params.filter,
    );
    Ok(out)
}

/// Validate resize parameters and input shape. Shared verbatim by the
/// CPU kernel and the CUDA kernel so both backends reject exactly the
/// same inputs with exactly the same messages.
pub(crate) fn validate_resize(shape: &[usize], params: &ResizeParams) -> Result<(), PhaiosError> {
    if params.width == 0 || params.height == 0 {
        return Err(PhaiosError::Parameter(format!(
            "resize target {}x{} has a zero dimension",
            params.width, params.height
        )));
    }
    if shape[0] == 0 || shape[1] == 0 {
        return Err(PhaiosError::Parameter(format!(
            "cannot resize an empty {}x{} image",
            shape[1], shape[0]
        )));
    }
    Ok(())
}

/// Resample along axis 1 of `src` into `dst` (axis 0 and 2 unchanged).
///
/// The per-output-pixel tap loop is the reference the CUDA kernel
/// mirrors: same centre, same weights, same left-to-right accumulation.
fn resample_axis1(
    src: ArrayView3<f32>,
    mut dst: ndarray::ArrayViewMut3<f32>,
    scale: f32,
    filter: ResizeFilter,
) {
    let (rows, in_len, c) = src.dim();
    let out_len = dst.dim().1;
    let support = filter_support(filter, scale);
    let area_minify = filter == ResizeFilter::Area && scale > 1.0;

    ndarray::Zip::indexed(dst.rows_mut()).par_for_each(|(row, i), mut out_px| {
        let (centre, denom) = axis_taps(scale, i);
        let k0 = (centre - support).floor() as i64;
        let k1 = (centre + support).ceil() as i64;

        for ch in 0..c {
            out_px[ch] = 0.0;
        }
        let mut wsum = 0.0_f32;
        for k in k0..=k1 {
            let w = if area_minify {
                // Exact fractional coverage of source pixel k by the
                // output footprint [centre - scale/2, centre + scale/2].
                let lo = (k as f32 - 0.5).max(centre - scale * 0.5);
                let hi = (k as f32 + 0.5).min(centre + scale * 0.5);
                (hi - lo).max(0.0)
            } else {
                filter_eval(filter, (k as f32 - centre) / denom)
            };
            if w != 0.0 {
                let kc = k.clamp(0, in_len as i64 - 1) as usize;
                for ch in 0..c {
                    out_px[ch] += w * src[[row, kc, ch]];
                }
                wsum += w;
            }
        }
        if wsum != 0.0 {
            for ch in 0..c {
                out_px[ch] /= wsum;
            }
        } else {
            // Unreachable for the three shipped filters: Area and Bilinear
            // have non-negative weights with a positive centre tap, and
            // Catmull-Rom's taps sum to the (positive) scale factor. The
            // branch exists so the two backends cannot drift — the CUDA
            // kernel writes 0.0 here, and leaving an un-normalised
            // accumulator instead would be a divergence on the one path
            // no test can construct.
            for ch in 0..c {
                out_px[ch] = 0.0;
            }
        }
    });
    let _ = rows;
    let _ = out_len;
}

// ── Straighten ───────────────────────────────────────────────────────────────

/// Parameters for [`straighten`].
///
/// ```python
/// params = phaios_core.StraightenParams(degrees=-1.8)  # level a tilted horizon
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct StraightenParams {
    /// Rotation in degrees, **positive clockwise**, limited to ±45°
    /// (compose with [`orient`] for quarter turns).
    #[pyo3(get, set)]
    pub degrees: f32,
}

#[pymethods]
impl StraightenParams {
    /// Create new ``StraightenParams``.
    #[new]
    pub fn new(degrees: f32) -> Self {
        Self { degrees }
    }

    /// Two ``StraightenParams`` are equal when the angle bits match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.degrees.to_bits() == other.degrees.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!("StraightenParams(degrees={})", self.degrees)
    }
}

/// Everything the straighten kernels need, computed **once on the
/// host** and shared verbatim by both backends: the output dimensions
/// (largest axis-aligned rectangle inscribed in the rotated frame, by
/// the standard max-area construction) and the f32 sin/cos — the only
/// transcendentals in the whole operation.
///
/// # Errors
/// [`PhaiosError::Parameter`] if `degrees` is not finite, exceeds ±45°,
/// or leaves no whole pixel inscribed.
pub(crate) fn straighten_geometry(
    in_h: usize,
    in_w: usize,
    degrees: f32,
) -> Result<(usize, usize, f32, f32), PhaiosError> {
    if !degrees.is_finite() || degrees.abs() > 45.0 {
        return Err(PhaiosError::Parameter(format!(
            "degrees is {degrees}, expected a finite angle in -45..=45 (compose with orient() for quarter turns)"
        )));
    }
    let radians = (degrees as f64).to_radians();
    let (sin_a, cos_a) = (radians.sin().abs(), radians.cos().abs());

    // Largest axis-aligned rectangle inscribed in the rotated w x h
    // rectangle (max-area construction; aspect may change slightly).
    let (w, h) = (in_w as f64, in_h as f64);
    let (out_w, out_h) = if in_h == 0 || in_w == 0 {
        (0.0, 0.0)
    } else {
        let (short, long) = if w <= h { (w, h) } else { (h, w) };
        if short <= 2.0 * sin_a * cos_a * long {
            // Half-constrained: two inscribed corners touch the short
            // sides' midlines. The LONG axis of the result follows the
            // long axis of the input: landscape wr = x/sin_a (wide),
            // portrait hr = x/sin_a (tall). The review workflow caught
            // this tuple swapped — producing portrait crops from
            // landscape frames — a bug invisible to cross-backend
            // conformance because both backends share this function.
            let half = 0.5 * short;
            if w <= h {
                (half / cos_a, half / sin_a)
            } else {
                (half / sin_a, half / cos_a)
            }
        } else {
            let cos_2a = cos_a * cos_a - sin_a * sin_a;
            (
                (w * cos_a - h * sin_a) / cos_2a,
                (h * cos_a - w * sin_a) / cos_2a,
            )
        }
    };
    let (out_w, out_h) = (out_w.floor() as usize, out_h.floor() as usize);
    if out_w == 0 || out_h == 0 {
        return Err(PhaiosError::Parameter(format!(
            "straighten by {degrees} deg leaves no whole pixel of the {in_w}x{in_h} frame"
        )));
    }

    let r = (degrees as f64).to_radians();
    Ok((out_h, out_w, r.sin() as f32, r.cos() as f32))
}

/// Rotate by a small angle and crop to the largest inscribed rectangle.
///
/// Positive degrees rotate the image **clockwise** (consistent with
/// [`Orientation::Rotate90`]); ±45° is the limit — compose with
/// [`orient`] for anything larger. The output is the largest
/// axis-aligned rectangle inscribed in the rotated frame (max-area
/// construction), so every output pixel's *sample point* lies inside
/// the source. The cubic's ±2-pixel support can still reach
/// frame-edge pixels near the inscribed boundary, where taps clamp to
/// the border (replicate) — standard resampling practice, confined to
/// the outermost ~2-pixel band of the result.
///
/// Order-sensitive: between `orient` and `crop`, so crop rectangles
/// are expressed in the levelled frame.
///
/// Resampling is 16-tap Catmull-Rom (Keys 1981, a = −0.5), evaluated
/// in a fixed 4×4 order. With sin/cos computed once on the host, every
/// per-pixel operation is polynomial, so the kernel is **bit-exact
/// across backends**.
///
/// `degrees == 0` is the exact identity (the cubic collapses to a unit
/// tap on the source pixel).
///
/// Input shape: `(H, W, C)`, any channel count, any layout.
/// Output shape: the inscribed rectangle, C-contiguous.
///
/// # Errors
/// [`PhaiosError::Parameter`] if `degrees` is not finite, exceeds
/// ±45°, or leaves no whole pixel inscribed.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn straighten(
    img: ArrayView3<f32>,
    params: &StraightenParams,
) -> Result<Array3<f32>, PhaiosError> {
    let (in_h, in_w, c) = img.dim();
    let (out_h, out_w, sin_a, cos_a) = straighten_geometry(in_h, in_w, params.degrees)?;

    let mut out = crate::alloc::zeros3::<f32>((out_h, out_w, c))?;
    let (cx_out, cy_out) = (out_w as f32 * 0.5, out_h as f32 * 0.5);
    let (cx_in, cy_in) = (in_w as f32 * 0.5, in_h as f32 * 0.5);

    ndarray::Zip::indexed(out.rows_mut()).par_for_each(|(oy, ox), mut px| {
        // Inverse mapping: the output pixel centre, rotated back into
        // the source frame. Clockwise image rotation means the sampling
        // grid rotates counter-clockwise.
        let dx = ox as f32 + 0.5 - cx_out;
        let dy = oy as f32 + 0.5 - cy_out;
        let sx = cos_a * dx + sin_a * dy + cx_in - 0.5;
        let sy = -sin_a * dx + cos_a * dy + cy_in - 0.5;

        let fx = sx.floor();
        let fy = sy.floor();
        let tx = sx - fx;
        let ty = sy - fy;
        let ix = fx as i64;
        let iy = fy as i64;

        // Catmull-Rom weights for the four taps on each axis, evaluated
        // in fixed order; both backends share this exact sequence.
        let wx = catmull_weights(tx);
        let wy = catmull_weights(ty);

        for ch in 0..c {
            let mut acc = 0.0_f32;
            for (j, wyj) in wy.iter().enumerate() {
                let yj = (iy - 1 + j as i64).clamp(0, in_h as i64 - 1) as usize;
                let mut row_acc = 0.0_f32;
                for (i, wxi) in wx.iter().enumerate() {
                    let xi = (ix - 1 + i as i64).clamp(0, in_w as i64 - 1) as usize;
                    row_acc += wxi * img[[yj, xi, ch]];
                }
                acc += wyj * row_acc;
            }
            px[ch] = acc;
        }
    });
    Ok(out)
}

/// The four Catmull-Rom tap weights for fractional position `t` ∈ [0, 1).
///
/// Weights sum to exactly the polynomial identity (1 at t = 0), and the
/// evaluation order is part of the cross-backend contract.
#[inline]
pub(crate) fn catmull_weights(t: f32) -> [f32; 4] {
    [
        filter_eval(ResizeFilter::CatmullRom, t + 1.0),
        filter_eval(ResizeFilter::CatmullRom, t),
        filter_eval(ResizeFilter::CatmullRom, 1.0 - t),
        filter_eval(ResizeFilter::CatmullRom, 2.0 - t),
    ]
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

    // ── resize ───────────────────────────────────────────────────────────────

    #[test]
    fn resize_same_size_is_identity_for_every_filter() {
        // Centre alignment puts c = i exactly, so the filters collapse
        // to a unit tap: bit-exact identity, not approximate.
        let img = numbered(7, 9, 3);
        for filter in [
            ResizeFilter::Area,
            ResizeFilter::Bilinear,
            ResizeFilter::CatmullRom,
        ] {
            let out = resize(img.view(), &ResizeParams::new(9, 7, filter)).unwrap();
            assert_eq!(out, img, "{filter:?} same-size resize must be identity");
        }
    }

    #[test]
    fn resize_constant_image_stays_constant_to_a_ulp() {
        // The weighted sum and weight sum round separately before the
        // normalising division, so a flat image is preserved to ~1 ULP
        // (not the bit — the same wobble on every backend).
        let img = Array3::from_elem((37, 53, 1), 0.42_f32);
        for filter in [
            ResizeFilter::Area,
            ResizeFilter::Bilinear,
            ResizeFilter::CatmullRom,
        ] {
            for (w, h) in [(17_u32, 11_u32), (105, 71)] {
                let out = resize(img.view(), &ResizeParams::new(w, h, filter)).unwrap();
                for &v in out.iter() {
                    assert!(
                        (v - 0.42).abs() < 1e-6,
                        "{filter:?} {w}x{h} broke a constant image: {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn area_downscale_by_two_is_the_block_mean() {
        let img = numbered(4, 4, 1);
        let out = resize(img.view(), &ResizeParams::new(2, 2, ResizeFilter::Area)).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let mean = (img[[2 * y, 2 * x, 0]]
                    + img[[2 * y, 2 * x + 1, 0]]
                    + img[[2 * y + 1, 2 * x, 0]]
                    + img[[2 * y + 1, 2 * x + 1, 0]])
                    / 4.0;
                assert!(
                    (out[[y, x, 0]] - mean).abs() < 1e-5,
                    "2x2 block mean mismatch at ({y}, {x})"
                );
            }
        }
    }

    #[test]
    fn resize_preserves_mean_when_downscaling() {
        let img =
            Array3::from_shape_fn((64, 96, 1), |(y, x, _)| ((y * 96 + x) % 251) as f32 / 251.0);
        let out = resize(img.view(), &ResizeParams::new(31, 21, ResizeFilter::Area)).unwrap();
        let mean_in = img.iter().sum::<f32>() / img.len() as f32;
        let mean_out = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            (mean_in - mean_out).abs() < 5e-3,
            "area downscale should preserve the mean: {mean_in} vs {mean_out}"
        );
    }

    #[test]
    fn catmull_upscale_interpolates_a_linear_ramp_exactly_inside() {
        // Keys' a = -0.5 cubic is third-order accurate: it reproduces
        // polynomials up to degree 2 exactly (a cubic it does not) —
        // a linear ramp upscales to a linear ramp (interior pixels).
        let img = Array3::from_shape_fn((1, 8, 1), |(_, x, _)| x as f32);
        let out = resize(
            img.view(),
            &ResizeParams::new(16, 1, ResizeFilter::CatmullRom),
        )
        .unwrap();
        for x in 3..13 {
            let expected = (x as f32 + 0.5) * 0.5 - 0.5;
            assert!(
                (out[[0, x, 0]] - expected).abs() < 1e-4,
                "ramp broken at {x}: {} vs {expected}",
                out[[0, x, 0]]
            );
        }
    }

    #[test]
    fn catmull_rom_taps_are_the_keys_cubic_with_negative_lobes() {
        // Audit finding F18: making `ResizeFilter::CatmullRom` evaluate
        // the Bilinear profile left the whole suite green, including the
        // ramp test above — a triangle filter reproduces a linear ramp
        // exactly too, so that test cannot tell the two apart. Nothing
        // pinned the cubic itself.
        //
        // Keys' a = -0.5 kernel is negative on 1 < |t| < 2; a triangle
        // (or any non-negative kernel) cannot produce that sign. The
        // taps below are dyadic rationals — every mul/add in the Horner
        // form is exact in f32 — so they are compared bit-for-bit, which
        // also fixes the polynomial's coefficients against a typo.
        //
        // Reference: Robert G. Keys, "Cubic Convolution Interpolation
        // for Digital Image Processing", IEEE Transactions on Acoustics,
        // Speech, and Signal Processing 29(6), 1981, eq. (4) with
        // a = -0.5.
        for (t, expected) in [
            (0.5_f32, [-0.0625_f32, 0.5625, 0.5625, -0.0625]),
            (0.25, [-0.0703125, 0.8671875, 0.2265625, -0.0234375]),
        ] {
            let taps = catmull_weights(t);
            assert_eq!(taps, expected, "Keys taps wrong at t = {t}");
            // Partition of unity: an interpolating kernel must leave a
            // constant image constant before any normalisation.
            assert_eq!(
                taps[0] + taps[1] + taps[2] + taps[3],
                1.0,
                "Keys taps must sum to 1 at t = {t}"
            );
            // The discriminating property: outer lobes are negative.
            assert!(
                taps[0] < 0.0 && taps[3] < 0.0,
                "outer taps must be negative at t = {t}, got {taps:?}"
            );
        }

        // And the profile itself, at the two distances that separate the
        // filters: inside the first lobe Catmull-Rom is above the
        // triangle, outside it the triangle is already zero.
        assert_eq!(filter_eval(ResizeFilter::CatmullRom, 0.5), 0.5625);
        assert_eq!(filter_eval(ResizeFilter::Bilinear, 0.5), 0.5);
        assert_eq!(filter_eval(ResizeFilter::CatmullRom, 1.5), -0.0625);
        assert_eq!(filter_eval(ResizeFilter::Bilinear, 1.5), 0.0);
        // Support must cover the second lobe, or the negative taps are
        // never reached and the kernel degenerates back to a triangle.
        assert_eq!(filter_support(ResizeFilter::CatmullRom, 1.0), 2.0);
    }

    #[test]
    fn catmull_upscale_overshoots_a_step_edge_where_bilinear_cannot() {
        // The visible consequence of the negative lobes, and the second
        // half of the F18 guard: on a 0 -> 1 step, Catmull-Rom rings
        // outside the input range. Bilinear's weights are non-negative
        // and normalised, so every output is a convex combination of its
        // taps and provably stays within [0, 1] — it cannot fake this.
        //
        // Measured on this 8 -> 32 upscale: Catmull-Rom reaches
        // -0.0732 and 1.0732, roughly 7% of the step height, which is
        // the ringing a photographer trades for the extra acutance.
        let img = Array3::from_shape_fn((1, 8, 1), |(_, x, _)| if x < 4 { 0.0 } else { 1.0 });

        let cr = resize(
            img.view(),
            &ResizeParams::new(32, 1, ResizeFilter::CatmullRom),
        )
        .unwrap();
        let (cr_min, cr_max) = cr
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        assert!(
            cr_min < -0.03,
            "Catmull-Rom must undershoot a step edge, min was {cr_min}"
        );
        assert!(
            cr_max > 1.03,
            "Catmull-Rom must overshoot a step edge, max was {cr_max}"
        );

        let bl = resize(
            img.view(),
            &ResizeParams::new(32, 1, ResizeFilter::Bilinear),
        )
        .unwrap();
        let (bl_min, bl_max) = bl
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        assert!(
            bl_min >= 0.0 && bl_max <= 1.0,
            "Bilinear has non-negative weights and must not ring: [{bl_min}, {bl_max}]"
        );
    }

    #[test]
    fn resize_rejects_zero_targets_and_empty_input() {
        let img = numbered(4, 4, 1);
        for (w, h) in [(0_u32, 4_u32), (4, 0)] {
            assert!(matches!(
                resize(img.view(), &ResizeParams::new(w, h, ResizeFilter::Area)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        // The empty-input rejection lives in the SHARED validate, so both
        // backends refuse identically (the review caught it duplicated).
        let empty = Array3::<f32>::zeros((0, 4, 1));
        assert!(matches!(
            resize(empty.view(), &ResizeParams::new(4, 4, ResizeFilter::Area)).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
    }

    #[test]
    fn an_oversized_resize_target_is_refused_rather_than_aborting() {
        // `resize` is the only kernel whose output size comes purely from
        // caller parameters: a 2x2 input may ask for 200000x200000, and
        // `validate_resize` bounds only *zero* dimensions. `alloc::zeros3`
        // is therefore the sole guard between a caller and a request the
        // allocator cannot meet — and a failed `Vec` allocation *aborts*
        // through `handle_alloc_error`, which no Python `except` can catch
        // (CLAUDE.md section 2).
        //
        // That makes this the one abort the oversized-input sweep in
        // tests/ffi.py cannot reach: there the oversize is in the input
        // view, here it is in the parameters. Both passes allocate, so
        // both are covered, and both sizes are refused on the shape alone
        // without materialising anything.
        let img = Array3::<f32>::zeros((2, 2, 3));

        // The output buffer: 200000 x 200000 x 3 f32 is 480 GB. The
        // horizontal pass ahead of it is only (2, 200000, 3) = 4.8 MB, so
        // it is the *output* allocation being refused here, not scratch.
        let err = resize(
            img.view(),
            &ResizeParams::new(200_000, 200_000, ResizeFilter::Area),
        )
        .unwrap_err();
        assert!(
            matches!(err, PhaiosError::Allocation(_)),
            "a 480 GB target must be refused, got {err:?}"
        );
        assert!(
            err.to_string().contains("above the"),
            "the error should name the limit it exceeded: {err}"
        );

        // The scratch buffer, which is guarded separately from the output.
        // A tall, narrow input asked to widen: the horizontal pass allocates
        // (in_h, out_w, c) = (1000000, 1000000, 3) = 12 TB, while the output
        // it feeds is only (1, 1000000, 3) = 12 MB and would pass — so this
        // reaches the scratch guard specifically.
        //
        // The size is chosen to be refused by the *allocator* as well as by
        // the guard. An earlier version used 9.6 GB, which clears the 8 GiB
        // cap but which Linux would happily grant on any ordinary machine:
        // had the guard regressed, the test would have allocated it and
        // paged it in rather than failing. A request larger than RAM plus
        // swap is refused outright, so the failure stays inside this
        // process. The tall input costs 24 MB of real storage.
        let tall = Array3::<f32>::zeros((1_000_000, 2, 3));
        let err = resize(
            tall.view(),
            &ResizeParams::new(1_000_000, 1, ResizeFilter::Area),
        )
        .unwrap_err();
        assert!(
            matches!(err, PhaiosError::Allocation(_)),
            "a 12 TB scratch buffer must be refused, got {err:?}"
        );

        // The guard must be nowhere near an ordinary enlargement.
        let ok = resize(img.view(), &ResizeParams::new(64, 48, ResizeFilter::Area)).unwrap();
        assert_eq!(ok.dim(), (48, 64, 3));
    }

    // ── straighten ───────────────────────────────────────────────────────────

    #[test]
    fn straighten_zero_degrees_is_identity() {
        let img = numbered(9, 13, 3);
        let out = straighten(img.view(), &StraightenParams::new(0.0)).unwrap();
        assert_eq!(out, img, "0 degrees must be the exact identity");
    }

    #[test]
    fn straighten_constant_image_stays_constant() {
        // Catmull-Rom weights sum to 1, so a flat image survives the
        // resampling exactly wherever all taps are interior.
        let img = Array3::from_elem((64, 96, 1), 0.6_f32);
        let out = straighten(img.view(), &StraightenParams::new(7.3)).unwrap();
        let (h, w, _) = out.dim();
        assert!(h < 64 && w < 96, "inscribed crop must shrink the frame");
        for &v in out.iter() {
            assert!((v - 0.6).abs() < 1e-5, "flat image broken: {v}");
        }
    }

    #[test]
    fn straighten_direction_is_clockwise() {
        // A bright column right of centre must move DOWN under positive
        // (clockwise) rotation — pin the direction, not just the shape.
        let mut img = Array3::<f32>::zeros((101, 101, 1));
        for y in 0..101 {
            img[[y, 85, 0]] = 1.0;
        }
        let out = straighten(img.view(), &StraightenParams::new(10.0)).unwrap();
        let (h, w, _) = out.dim();
        // Find the brightest pixel in the top and bottom quarters.
        let brightest_x = |rows: std::ops::Range<usize>| -> f32 {
            let mut best = (0.0_f32, 0_usize);
            for y in rows {
                for x in 0..w {
                    if out[[y, x, 0]] > best.0 {
                        best = (out[[y, x, 0]], x);
                    }
                }
            }
            best.1 as f32
        };
        let top_x = brightest_x(0..h / 4);
        let bottom_x = brightest_x(3 * h / 4..h);
        assert!(
            top_x > bottom_x + 2.0,
            "clockwise rotation should tilt a right-of-centre column \
             top-rightward: top x {top_x}, bottom x {bottom_x}"
        );
    }

    #[test]
    fn straighten_interior_is_free_of_border_influence() {
        // Mark the border with a sentinel. The cubic support may reach
        // it in the outermost ~2-pixel band (documented); the INTERIOR
        // must be entirely free of it.
        let mut img = Array3::from_elem((80, 120, 1), 0.5_f32);
        for x in 0..120 {
            img[[0, x, 0]] = 100.0;
            img[[79, x, 0]] = 100.0;
        }
        for y in 0..80 {
            img[[y, 0, 0]] = 100.0;
            img[[y, 119, 0]] = 100.0;
        }
        let out = straighten(img.view(), &StraightenParams::new(5.0)).unwrap();
        let (h, w, _) = out.dim();
        let mut max_interior = 0.0_f32;
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                max_interior = max_interior.max(out[[y, x, 0]]);
            }
        }
        assert!(
            max_interior < 10.0,
            "border sentinel leaked into the interior: max {max_interior}"
        );
    }

    #[test]
    fn straighten_inscribed_rect_is_valid_in_both_branches() {
        // The half-constrained branch (large angle relative to aspect)
        // had its (wr, hr) tuple swapped — caught by review, invisible
        // to cross-backend tests since both backends share the host
        // geometry. Validity check: the four corners of the inscribed
        // rectangle, rotated forward, must lie inside the source frame,
        // for every branch and both orientations.
        for (h, w) in [(257_usize, 389_usize), (389, 257), (100, 1000), (1000, 100)] {
            for degrees in [2.0_f32, 10.0, 30.0, 44.0, -30.0] {
                let (oh, ow, sin_a, cos_a) = straighten_geometry(h, w, degrees).unwrap();
                assert!(oh <= h && ow <= w, "{h}x{w} @ {degrees}: grew");
                // Landscape stays landscape, portrait stays portrait.
                if w > 2 * h {
                    assert!(ow > oh, "{h}x{w} @ {degrees}: aspect flipped");
                }
                if h > 2 * w {
                    assert!(oh > ow, "{h}x{w} @ {degrees}: aspect flipped");
                }
                // Forward-map the output corners into the source frame.
                let (cx_o, cy_o) = (ow as f32 * 0.5, oh as f32 * 0.5);
                let (cx_i, cy_i) = (w as f32 * 0.5, h as f32 * 0.5);
                for (ox, oy) in [
                    (0.0, 0.0),
                    (ow as f32, 0.0),
                    (0.0, oh as f32),
                    (ow as f32, oh as f32),
                ] {
                    let dx = ox - cx_o;
                    let dy = oy - cy_o;
                    let sx = cos_a * dx + sin_a * dy + cx_i;
                    let sy = -sin_a * dx + cos_a * dy + cy_i;
                    assert!(
                        (-0.51..=w as f32 + 0.51).contains(&sx)
                            && (-0.51..=h as f32 + 0.51).contains(&sy),
                        "{h}x{w} @ {degrees}: corner ({ox}, {oy}) maps to \
                         ({sx}, {sy}) outside the frame"
                    );
                }
            }
        }
    }

    #[test]
    fn straighten_rejects_out_of_domain_angles() {
        let img = numbered(8, 8, 1);
        for bad in [46.0_f32, -50.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                straighten(img.view(), &StraightenParams::new(bad)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        // 45 degrees exactly is legal.
        assert!(straighten(img.view(), &StraightenParams::new(45.0)).is_ok());
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

        let rp = ResizeParams::new(5, 3, ResizeFilter::CatmullRom);
        let a = resize(strided, &rp).unwrap();
        let b = resize(strided.to_owned().view(), &rp).unwrap();
        assert_eq!(a, b, "resize must be layout-agnostic");

        let sp = StraightenParams::new(6.0);
        let a = straighten(strided, &sp).unwrap();
        let b = straighten(strided.to_owned().view(), &sp).unwrap();
        assert_eq!(a, b, "straighten must be layout-agnostic");
    }
}
