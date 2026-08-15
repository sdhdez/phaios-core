// SPDX-License-Identifier: GPL-3.0-or-later
//! Split-toning — tinting shadows and highlights different colours.
//!
//! The darkroom operation this imitates is selenium or sepia toning,
//! where the toner reaches some densities more than others. Here the
//! shadow tint and the highlight tint are chosen independently and
//! crossfaded across the tonal range.
//!
//! **This kernel changes the shape of the data.** It takes `(H, W, 1)`
//! monochrome luminance and returns `(H, W, 3)` linear sRGB — the only
//! kernel in the crate that adds channels. Everything downstream of it
//! ([`crate::vignette`], [`crate::tone`]'s parametric curve,
//! [`crate::encode`]) accepts any channel count for exactly this reason.
//!
//! # Why OKLab
//!
//! Tinting means adding chroma without disturbing the lightness the tone
//! stages just established. That requires a space whose lightness axis is
//! actually independent of its colour axes. In linear sRGB, adding a
//! constant to the red channel makes the pixel both redder *and*
//! brighter. In CIELAB the axes are separable in principle, but its
//! blue-hue non-uniformity bends a constant-chroma sweep visibly towards
//! purple.
//!
//! OKLab was fitted to fix exactly that, and its transform is cheap: two
//! 3×3 matrices with a cube root between them.
//!
//! Reference: Björn Ottosson, "A perceptual color space for image
//! processing" (2020), <https://bottosson.github.io/posts/oklab/>.

use ndarray::{Array3, ArrayView3, s};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── OKLab conversion ──────────────────────────────────────────────────────────

/// Half-width of the shadow/highlight crossover, in OKLab lightness.
///
/// The mix reaches pure shadow tint a quarter of the range below the
/// pivot and pure highlight tint a quarter above it. Fixed rather than
/// exposed: a fourth control would let the user reproduce the pivot's
/// job with a different combination of numbers.
const CROSSOVER_HALF_WIDTH: f32 = 0.25;

/// Convert linear sRGB to OKLab.
///
/// Reference: Ottosson (2020), the `linear_srgb_to_oklab` reference
/// implementation.
#[must_use]
#[inline]
pub fn linear_srgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = 0.412_221_47 * r + 0.536_332_54 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;

    // Signed cube root: scene-referred data can be slightly negative
    // after white balance, and `cbrt` is defined there — unlike `powf`.
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Convert OKLab to linear sRGB.
///
/// Reference: Ottosson (2020), the `oklab_to_linear_srgb` reference
/// implementation.
#[must_use]
#[inline]
pub fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let [big_l, a, b] = lab;

    let l_ = big_l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = big_l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = big_l - 0.089_484_18 * a - 1.291_485_5 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// OKLab lightness of a neutral whose linear value is `y`.
///
/// For a neutral the three cone responses are all equal to `y`, so the
/// forward transform collapses to a single cube root scaled by the sum
/// of the lightness row. Using this instead of the full matrix saves two
/// cube roots per pixel, which is most of the kernel's arithmetic;
/// `neutral_shortcut_matches_the_full_transform` pins the two together.
#[inline]
fn neutral_oklab_lightness(y: f32) -> f32 {
    y.cbrt() * NEUTRAL_ROW_SUM
}

/// Hermite smoothstep, as in [`crate::vignette`].
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`split_toning`].
///
/// ```python
/// # Cool shadows, warm highlights — the classic cross-process look.
/// params = phaios_core.SplitToningParams(
///     shadow_oklab=[0.0, -0.02, -0.06],
///     highlight_oklab=[0.0, 0.03, 0.05],
///     pivot=0.5,
///     balance=0.0,
/// )
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct SplitToningParams {
    /// Shadow tint as an OKLab triple `[L, a, b]`.
    ///
    /// **Only `a` and `b` are used.** The lightness component is ignored
    /// deliberately: toning must not move the tonal rendering that the
    /// zone system and the tone curve just established. The field takes
    /// a full triple so a colour picked in an OKLab picker can be passed
    /// through unmodified.
    ///
    /// Typical chroma magnitudes are small — 0.02 is a clear tint, 0.1 is
    /// heavy-handed.
    #[pyo3(get, set)]
    pub shadow_oklab: [f32; 3],
    /// Highlight tint as an OKLab triple `[L, a, b]`. As above, only `a`
    /// and `b` are used.
    #[pyo3(get, set)]
    pub highlight_oklab: [f32; 3],
    /// Lightness at which the two tints mix equally, 0..=1, in OKLab
    /// lightness (not linear luminance — OKLab L of middle grey is about
    /// 0.57, not 0.18).
    #[pyo3(get, set)]
    pub pivot: f32,
    /// Shifts the crossover, −1..=1. Positive favours the highlight
    /// tint by moving the crossover down; negative favours the shadow
    /// tint. `0.0` leaves the pivot where it is.
    #[pyo3(get, set)]
    pub balance: f32,
}

#[pymethods]
impl SplitToningParams {
    /// Create new ``SplitToningParams``. Defaults are the identity
    /// (no chroma in either tint).
    #[new]
    #[pyo3(signature = (
        shadow_oklab = [0.0, 0.0, 0.0],
        highlight_oklab = [0.0, 0.0, 0.0],
        pivot = 0.5,
        balance = 0.0,
    ))]
    pub fn new(
        shadow_oklab: [f32; 3],
        highlight_oklab: [f32; 3],
        pivot: f32,
        balance: f32,
    ) -> Self {
        Self {
            shadow_oklab,
            highlight_oklab,
            pivot,
            balance,
        }
    }

    /// Two ``SplitToningParams`` are equal when all fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        let same = |a: &[f32; 3], b: &[f32; 3]| {
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        };
        same(&self.shadow_oklab, &other.shadow_oklab)
            && same(&self.highlight_oklab, &other.highlight_oklab)
            && self.pivot.to_bits() == other.pivot.to_bits()
            && self.balance.to_bits() == other.balance.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "SplitToningParams(shadow_oklab={:?}, highlight_oklab={:?}, pivot={}, balance={})",
            self.shadow_oklab, self.highlight_oklab, self.pivot, self.balance
        )
    }
}

impl Default for SplitToningParams {
    fn default() -> Self {
        Self {
            shadow_oklab: [0.0; 3],
            highlight_oklab: [0.0; 3],
            pivot: 0.5,
            balance: 0.0,
        }
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate shape and parameters. Shared verbatim by the CPU kernel and
/// the CUDA kernel so both backends reject exactly the same inputs with
/// exactly the same messages.
pub(crate) fn validate(shape: &[usize], params: &SplitToningParams) -> Result<(), PhaiosError> {
    if shape[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "split_toning expects (H, W, 1) luminance input, got shape {shape:?}"
        )));
    }
    if !params.pivot.is_finite() || !(0.0..=1.0).contains(&params.pivot) {
        return Err(PhaiosError::Parameter(format!(
            "pivot is {}, expected a value in 0..=1",
            params.pivot
        )));
    }
    if !params.balance.is_finite() || !(-1.0..=1.0).contains(&params.balance) {
        return Err(PhaiosError::Parameter(format!(
            "balance is {}, expected a value in -1..=1",
            params.balance
        )));
    }
    for (name, tint) in [
        ("shadow_oklab", &params.shadow_oklab),
        ("highlight_oklab", &params.highlight_oklab),
    ] {
        if let Some(bad) = tint.iter().find(|v| !v.is_finite()) {
            return Err(PhaiosError::Parameter(format!(
                "{name} contains {bad}, expected finite values"
            )));
        }
    }
    Ok(())
}

/// The smoothstep window `(edge0, edge1)` implied by pivot and balance —
/// one definition for both backends.
pub(crate) fn crossfade_edges(params: &SplitToningParams) -> (f32, f32) {
    let pivot = (params.pivot - params.balance * 0.5).clamp(0.0, 1.0);
    (pivot - CROSSOVER_HALF_WIDTH, pivot + CROSSOVER_HALF_WIDTH)
}

/// The neutral-lightness row sum, shared with the CUDA kernel so the
/// host passes the exact f32 constant the CPU folds at compile time.
pub(crate) const NEUTRAL_ROW_SUM: f32 = 0.210_454_26 + 0.793_617_8 - 0.004_072_047;

/// Tint shadows and highlights separately, returning linear sRGB.
///
/// For each luminance sample:
/// 1. Take its OKLab lightness `L`.
/// 2. Crossfade the two tints with a smoothstep centred on the pivot
///    (shifted by `balance`), giving chroma `(a, b)`.
/// 3. Convert `(L, a, b)` back to linear sRGB.
///
/// Lightness is carried through untouched, so the tonal rendering
/// established upstream survives: only chroma is added.
///
/// # Behaviour at black
///
/// OKLab lightness 0 with non-zero chroma is not a colour: nothing is
/// both black and tinted. Asked for one, the inverse transform returns
/// the nearest thing it can, which is slightly outside the sRGB cube —
/// mostly a small *negative* blue. The kernel does not clamp it, in
/// keeping with the rest of the crate; the values are small and the
/// negative sign means a clamping consumer sees black rather than a
/// lifted shadow.
///
/// Measured worst-case channel magnitude at `L = 0`, tint applied to
/// both `a` and `b`:
///
/// | tint | worst channel |
/// |------|---------------|
/// | 0.02 | 3.6e−5 |
/// | 0.05 | 5.6e−4 |
/// | 0.10 | 4.5e−3 |
/// | 0.20 | 3.6e−2 |
///
/// The cubing in the inverse transform is what keeps this bounded: the
/// error grows as the cube of the tint, so it is negligible over the
/// range anyone tones in and only becomes visible if a caller asks for
/// a tint far past "heavy-handed".
///
/// Input shape: `(H, W, 1)` — monochrome luminance, any layout.
/// Output shape: `(H, W, 3)` — linear sRGB, C-contiguous.
///
/// Order-sensitive: this is a finishing stage. It must follow the B&W
/// conversion (it needs single-channel input) and precede
/// [`crate::encode::encode_srgb`] (it produces linear values).
///
/// Reference: Björn Ottosson, "A perceptual color space for image
/// processing" (2020).
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
/// - [`PhaiosError::Parameter`] if `pivot` is outside 0..=1, `balance`
///   is outside −1..=1, or any tint component is not finite.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn split_toning(
    img: ArrayView3<f32>,
    params: &SplitToningParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate(img.shape(), params)?;

    let (h, w, _) = img.dim();
    let mut out = Array3::<f32>::zeros((h, w, 3));

    let [_, shadow_a, shadow_b] = params.shadow_oklab;
    let [_, highlight_a, highlight_b] = params.highlight_oklab;
    // Positive balance moves the crossover down, so more of the image
    // reads as "highlight" and takes the highlight tint.
    let (edge0, edge1) = crossfade_edges(params);

    // One lane per pixel: `rows_mut` hands the closure the whole
    // three-element channel axis, so the converted triple is written in
    // one place rather than through three separate passes.
    ndarray::Zip::from(out.rows_mut())
        .and(img.slice(s![.., .., 0]))
        .par_for_each(|mut pixel, &y| {
            let lightness = neutral_oklab_lightness(y);
            let mix = smoothstep(edge0, edge1, lightness);
            let a = shadow_a + (highlight_a - shadow_a) * mix;
            let b = shadow_b + (highlight_b - shadow_b) * mix;

            let rgb = oklab_to_linear_srgb([lightness, a, b]);
            pixel[0] = rgb[0];
            pixel[1] = rgb[1];
            pixel[2] = rgb[2];
        });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grey(values: &[f32]) -> Array3<f32> {
        Array3::from_shape_vec((1, values.len(), 1), values.to_vec()).unwrap()
    }

    #[test]
    fn oklab_round_trips() {
        // Both directions must compose to the identity, or every tint is
        // applied on top of a systematic error.
        let cases = [
            [0.0_f32, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [0.9, 0.2, 0.05],
            [0.05, 0.4, 0.8],
            [4.0, 2.0, 1.0], // scene-referred highlight
        ];
        for [r, g, b] in cases {
            let lab = linear_srgb_to_oklab(r, g, b);
            let back = oklab_to_linear_srgb(lab);
            for (original, returned) in [r, g, b].iter().zip(back.iter()) {
                assert!(
                    (original - returned).abs() < 1e-5 * original.abs().max(1.0),
                    "round trip failed for ({r}, {g}, {b}): got {back:?}"
                );
            }
        }
    }

    #[test]
    fn neutrals_have_no_chroma() {
        // The defining property the kernel leans on: a grey converts to
        // (L, 0, 0), so any chroma in the output came from the tint.
        for y in [0.0_f32, 0.02, 0.18, 0.5, 1.0, 3.0] {
            let [_, a, b] = linear_srgb_to_oklab(y, y, y);
            assert!(a.abs() < 1e-6, "grey {y} has a = {a}");
            assert!(b.abs() < 1e-6, "grey {y} has b = {b}");
        }
    }

    #[test]
    fn neutral_shortcut_matches_the_full_transform() {
        // The kernel skips two cube roots by exploiting l == m == s for
        // neutrals. If that ever stops holding, this catches it.
        for y in [0.0_f32, 0.001, 0.18, 0.5, 1.0, 2.5, 10.0] {
            let full = linear_srgb_to_oklab(y, y, y)[0];
            let shortcut = neutral_oklab_lightness(y);
            assert!(
                (full - shortcut).abs() < 1e-6 * full.abs().max(1.0),
                "shortcut disagrees at {y}: {shortcut} vs {full}"
            );
        }
    }

    #[test]
    fn oklab_lightness_of_middle_grey() {
        // Sanity anchor against the published space: OKLab L of 18% grey
        // is about 0.57, which is why the pivot is documented in OKLab
        // lightness rather than linear luminance.
        let l = neutral_oklab_lightness(0.18);
        assert!((l - 0.5647).abs() < 0.002, "got {l}");
    }

    #[test]
    fn zero_tint_produces_a_neutral_image() {
        let img = grey(&[0.0, 0.05, 0.18, 0.5, 1.0]);
        let out = split_toning(img.view(), &SplitToningParams::default()).unwrap();
        assert_eq!(out.dim(), (1, 5, 3));
        for x in 0..5 {
            let (r, g, b) = (out[[0, x, 0]], out[[0, x, 1]], out[[0, x, 2]]);
            let y = img[[0, x, 0]];
            for (channel, v) in [("r", r), ("g", g), ("b", b)] {
                assert!(
                    (v - y).abs() < 1e-5,
                    "untinted {channel} at {y} became {v} — should stay neutral"
                );
            }
        }
    }

    #[test]
    fn shadows_and_highlights_get_their_own_tints() {
        // Warm shadows (positive a, positive b), cool highlights.
        let params = SplitToningParams::new([0.0, 0.05, 0.05], [0.0, -0.05, -0.05], 0.5, 0.0);
        let img = grey(&[0.005, 0.9]); // deep shadow, bright highlight
        let out = split_toning(img.view(), &params).unwrap();

        let shadow = [out[[0, 0, 0]], out[[0, 0, 1]], out[[0, 0, 2]]];
        let highlight = [out[[0, 1, 0]], out[[0, 1, 1]], out[[0, 1, 2]]];

        assert!(
            shadow[0] > shadow[2],
            "shadow should be warm (r > b): {shadow:?}"
        );
        assert!(
            highlight[2] > highlight[0],
            "highlight should be cool (b > r): {highlight:?}"
        );
    }

    #[test]
    fn lightness_survives_toning() {
        // The point of working in OKLab: adding chroma must not move the
        // tonal rendering the previous stages established.
        let img = grey(&[0.02, 0.1, 0.18, 0.4, 0.8]);
        let params = SplitToningParams::new([0.0, 0.06, -0.04], [0.0, -0.05, 0.06], 0.5, 0.0);
        let out = split_toning(img.view(), &params).unwrap();

        for x in 0..5 {
            let lab = linear_srgb_to_oklab(out[[0, x, 0]], out[[0, x, 1]], out[[0, x, 2]]);
            let expected = neutral_oklab_lightness(img[[0, x, 0]]);
            assert!(
                (lab[0] - expected).abs() < 1e-4,
                "lightness moved at x={x}: {} vs {expected}",
                lab[0]
            );
        }
    }

    #[test]
    fn black_stays_essentially_black() {
        // Lightness 0 with chroma is not a colour, so the inverse
        // transform lands just outside the sRGB cube. The kernel does not
        // clamp; what matters is that the excursion stays tiny and that
        // the visible channel goes negative rather than positive, so a
        // clamping consumer sees black instead of a lifted shadow.
        //
        // The bound tracks the cube of the tint: 5.6e-4 at 0.05, 4.5e-3
        // at 0.1. See the table on `split_toning`.
        for (tint, bound) in [(0.02_f32, 4.0e-5_f32), (0.05, 6.0e-4), (0.1, 5.0e-3)] {
            let params = SplitToningParams::new([0.0, tint, tint], [0.0, -tint, -tint], 0.5, 0.0);
            let out = split_toning(grey(&[0.0]).view(), &params).unwrap();
            for c in 0..3 {
                assert!(
                    out[[0, 0, c]].abs() < bound,
                    "tint {tint}: channel {c} reached {} (bound {bound})",
                    out[[0, 0, c]]
                );
            }
            assert!(
                out[[0, 0, 2]] <= 0.0,
                "tint {tint}: blue should err negative, got {}",
                out[[0, 0, 2]]
            );
        }
    }

    #[test]
    fn balance_shifts_which_tint_dominates() {
        let mid = grey(&[0.18]);
        let warm_shadow_cool_highlight =
            |balance| SplitToningParams::new([0.0, 0.06, 0.0], [0.0, -0.06, 0.0], 0.5, balance);

        let toward_shadow = split_toning(mid.view(), &warm_shadow_cool_highlight(-1.0)).unwrap();
        let neutral_balance = split_toning(mid.view(), &warm_shadow_cool_highlight(0.0)).unwrap();
        let toward_highlight = split_toning(mid.view(), &warm_shadow_cool_highlight(1.0)).unwrap();

        let redness = |a: &Array3<f32>| a[[0, 0, 0]] - a[[0, 0, 2]];
        assert!(
            redness(&toward_shadow) > redness(&neutral_balance),
            "negative balance should favour the shadow tint"
        );
        assert!(
            redness(&toward_highlight) < redness(&neutral_balance),
            "positive balance should favour the highlight tint"
        );
    }

    #[test]
    fn pivot_moves_the_crossover() {
        let img = grey(&[0.25]);
        let params =
            |pivot| SplitToningParams::new([0.0, 0.06, 0.0], [0.0, -0.06, 0.0], pivot, 0.0);
        // With a low pivot this sample reads as a highlight; with a high
        // one it reads as a shadow.
        let low = split_toning(img.view(), &params(0.1)).unwrap();
        let high = split_toning(img.view(), &params(0.95)).unwrap();
        assert!(
            low[[0, 0, 0]] < low[[0, 0, 2]],
            "low pivot → highlight tint"
        );
        assert!(
            high[[0, 0, 0]] > high[[0, 0, 2]],
            "high pivot → shadow tint"
        );
    }

    #[test]
    fn rejects_invalid_parameters() {
        let img = grey(&[0.5]);
        for bad_pivot in [-0.1_f32, 1.1, f32::NAN] {
            let p = SplitToningParams::new([0.0; 3], [0.0; 3], bad_pivot, 0.0);
            assert!(matches!(
                split_toning(img.view(), &p).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        for bad_balance in [-1.5_f32, 1.5, f32::INFINITY] {
            let p = SplitToningParams::new([0.0; 3], [0.0; 3], 0.5, bad_balance);
            assert!(matches!(
                split_toning(img.view(), &p).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        let p = SplitToningParams::new([0.0, f32::NAN, 0.0], [0.0; 3], 0.5, 0.0);
        assert!(matches!(
            split_toning(img.view(), &p).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
    }

    #[test]
    fn shape_error_on_rgb_input() {
        let img = Array3::<f32>::zeros((4, 4, 3));
        assert!(matches!(
            split_toning(img.view(), &SplitToningParams::default()).unwrap_err(),
            PhaiosError::Shape(_)
        ));
    }

    #[test]
    fn accepts_any_layout() {
        let img = Array3::<f32>::from_shape_fn((8, 4, 1), |(y, x, _)| (y * 4 + x) as f32 / 32.0);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let params = SplitToningParams::new([0.0, 0.03, -0.02], [0.0, -0.03, 0.02], 0.5, 0.0);
        let out = split_toning(strided, &params).unwrap();
        assert_eq!(out.dim(), (4, 4, 3));
        assert!(out.is_standard_layout());
        assert_eq!(
            out,
            split_toning(strided.to_owned().view(), &params).unwrap()
        );
    }
}
