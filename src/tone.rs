// SPDX-License-Identifier: GPL-3.0-or-later
//! Adams/Archer Zone System tone curve.
//!
//! Implements the eleven-zone system (0..=10) with the Gaussian-blending
//! modernisation from Phil Davis (1999). Each zone is one stop apart;
//! Zone V = middle grey at 18% reflectance.
//!
//! Given user-supplied offsets `{zone_index: stops}`, the output
//! luminance at each pixel is:
//!
//! ```text
//! zone_pos = 5 + log₂(max(L, ε) / 0.18)
//! total_offset = Σ_z  offset[z] · exp(-(zone_pos - z)² / (2·σ²))
//! L_out = L · 2^total_offset
//! ```
//!
//! where σ = 0.8 zones and ε = 1e-10 (guards against log₂(0)).
//!
//! References:
//! - Ansel Adams, *The Negative*, Little, Brown (1948), chapter 5.
//! - Phil Davis, *Beyond the Zone System*, 4th ed., Focal Press (1999).

use std::collections::HashMap;

use ndarray::{Array3, ArrayView3};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameter type ────────────────────────────────────────────────────────────

/// Zone System tone offsets.
///
/// Maps zone index (0..=10) to a stop offset in −3..+3. Zones not
/// present in the map are treated as having a 0-stop offset.
///
/// ```python
/// # Lift Zone V by half a stop, deepen Zone III by 0.3 stops
/// params = phaios_core.ZoneParams({5: 0.5, 3: -0.3})
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Default)]
pub struct ZoneParams {
    offsets: HashMap<i32, f32>,
}

#[pymethods]
impl ZoneParams {
    /// Create a new ``ZoneParams`` from a dict mapping zone index to stop offset.
    ///
    /// The map is not validated here — [`zone_system`] rejects zone
    /// indices outside 0..=10 and non-finite offsets when it runs. The
    /// constructor stays infallible because its signature shipped in
    /// v0.1 and the public API is stable (CLAUDE.md §2).
    #[new]
    pub fn new(offsets: HashMap<i32, f32>) -> Self {
        Self { offsets }
    }

    /// The zone-index → stop-offset map, as a dict.
    ///
    /// Returns a copy: mutating it does not change the ``ZoneParams``.
    /// Consumers need this to serialise settings — the desktop app's
    /// sidecar and preset files round-trip through it.
    #[getter]
    pub fn offsets(&self) -> HashMap<i32, f32> {
        self.offsets.clone()
    }

    /// Two ``ZoneParams`` are equal when they hold the same offsets.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.offsets == other.offsets
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        let mut pairs: Vec<(i32, f32)> = self.offsets.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort_by_key(|(k, _)| *k);
        let inner: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        format!("ZoneParams({{{}}})", inner.join(", "))
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Middle grey reference: 18% reflectance.
const MIDDLE_GREY: f32 = 0.18;

/// Gaussian σ in zone-position space (Davis 1999).
const SIGMA: f32 = 0.8;

/// 2·σ² denominator — precomputed.
const TWO_SIGMA_SQ: f32 = 2.0 * SIGMA * SIGMA;

/// Guard against log₂(0).
const L_EPSILON: f32 = 1e-10;

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate shape and zone offsets. Shared by CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate_zones(shape: &[usize], params: &ZoneParams) -> Result<(), PhaiosError> {
    if shape[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "zone_system expects (H, W, 1) luminance input, got shape {shape:?}"
        )));
    }
    for (&zone, &offset) in &params.offsets {
        if !(0..=10).contains(&zone) {
            return Err(PhaiosError::Parameter(format!(
                "zone index {zone} is outside the eleven zones 0..=10"
            )));
        }
        if !offset.is_finite() {
            return Err(PhaiosError::Parameter(format!(
                "offset for zone {zone} is {offset}, expected a finite number of stops"
            )));
        }
    }
    Ok(())
}

impl ZoneParams {
    /// Whether the map is empty (the identity configuration).
    pub(crate) fn is_identity(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The offsets as a dense 11-entry array in zone order, absent zones
    /// as 0.0. Adding an exact +0.0 term is the IEEE-754 identity, so a
    /// dense ascending iteration computes bit-for-bit the same sum as
    /// the sparse sorted one — this is how the CUDA kernel inherits the
    /// ordered-reduction guarantee mechanically.
    #[cfg(feature = "cuda")]
    pub(crate) fn dense_offsets(&self) -> [f32; 11] {
        let mut dense = [0.0_f32; 11];
        for (&z, &off) in &self.offsets {
            if (0..=10).contains(&z) {
                dense[z as usize] = off;
            }
        }
        dense
    }
}

/// Apply the Adams/Archer Zone System tone curve.
///
/// Eleven zones (0..=10), each one stop apart; Zone V = middle grey
/// at 18% reflectance. Offsets are blended via a Gaussian in
/// zone-position space (σ = 0.8 zones, Davis 1999).
///
/// Input shape: `(H, W, 1)` — linear luminance. Any memory layout is
/// accepted (strided views and Fortran-order arrays included); the
/// output is always freshly allocated and C-contiguous.
/// Output shape: `(H, W, 1)` — tone-adjusted luminance.
///
/// Reference: Ansel Adams, *The Negative*, Little, Brown (1948), ch. 5;
/// modernised in Davis, *Beyond the Zone System*, Focal Press (1999).
///
/// Offsets are **not** clamped: an offset of +50 stops really does
/// multiply by 2^50. Only values with no meaningful interpretation are
/// rejected — see Errors.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
/// - [`PhaiosError::Parameter`] if a zone index is outside 0..=10, or an
///   offset is not finite. A zone index of, say, 99 used to be accepted
///   and then contribute nothing (its Gaussian is zero everywhere in
///   range), silently swallowing what is almost always a caller bug.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn zone_system(img: ArrayView3<f32>, params: &ZoneParams) -> Result<Array3<f32>, PhaiosError> {
    validate_zones(img.shape(), params)?;

    // Sort by zone index. `HashMap` iteration order depends on the
    // per-instance `RandomState` seed, and f32 addition is not
    // associative, so an unsorted Gaussian sum would make the output
    // depend on that seed — different bytes for the same input on every
    // process. CLAUDE.md §2: determinism.
    let mut offsets: Vec<(f32, f32)> = params
        .offsets
        .iter()
        .map(|(&z, &off)| (z as f32, off))
        .collect();
    offsets.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let (h, w, _) = img.dim();
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;

    // No offsets → identity. Hoisted out of the pixel loop.
    if params.is_identity() {
        out.assign(&img);
        return Ok(out);
    }

    ndarray::Zip::from(&mut out).and(img).par_for_each(|o, &l| {
        let l_pos = l.max(L_EPSILON);
        let zone_pos = 5.0 + (l_pos / MIDDLE_GREY).log2();
        let total: f32 = offsets
            .iter()
            .map(|(z, off)| {
                let d = zone_pos - z;
                off * (-d * d / TWO_SIGMA_SQ).exp()
            })
            .sum();
        *o = l * 2.0_f32.powf(total);
    });

    Ok(out)
}

// ── Parametric tone curve (ASC CDL) ───────────────────────────────────────────

/// Parameters for [`tone_curve`], in ASC CDL terms.
///
/// The three knobs a colourist expects, under their photographic names:
///
/// | Field | Also called | Effect |
/// |-------|-------------|--------|
/// | `slope` | gain | scales the whole range about black |
/// | `offset` | lift | shifts the whole range, black included |
/// | `power` | gamma | bends the midtones, leaving 0 and 1 fixed |
///
/// The identity is `(1.0, 0.0, 1.0)`.
///
/// ```python
/// # Lift the blacks slightly and open up the midtones
/// params = phaios_core.ToneCurveParams(slope=1.0, offset=0.02, power=0.85)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ToneCurveParams {
    /// Multiplier applied before the offset. 1.0 is neutral.
    #[pyo3(get, set)]
    pub slope: f32,
    /// Added after the slope. 0.0 is neutral. Positive values lift black.
    #[pyo3(get, set)]
    pub offset: f32,
    /// Exponent applied last. 1.0 is neutral; below 1 brightens the
    /// midtones, above 1 darkens them. Must be positive.
    #[pyo3(get, set)]
    pub power: f32,
}

#[pymethods]
impl ToneCurveParams {
    /// Create new ``ToneCurveParams``. Defaults are the identity.
    #[new]
    #[pyo3(signature = (slope = 1.0, offset = 0.0, power = 1.0))]
    pub fn new(slope: f32, offset: f32, power: f32) -> Self {
        Self {
            slope,
            offset,
            power,
        }
    }

    /// Two ``ToneCurveParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.slope.to_bits() == other.slope.to_bits()
            && self.offset.to_bits() == other.offset.to_bits()
            && self.power.to_bits() == other.power.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "ToneCurveParams(slope={}, offset={}, power={})",
            self.slope, self.offset, self.power
        )
    }
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            slope: 1.0,
            offset: 0.0,
            power: 1.0,
        }
    }
}

/// Validate tone-curve parameters. Shared by CPU and CUDA backends so
/// both reject the same inputs with the same messages.
pub(crate) fn validate_tone_curve(params: &ToneCurveParams) -> Result<(), PhaiosError> {
    if !params.slope.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "slope is {}, expected a finite value",
            params.slope
        )));
    }
    if !params.offset.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "offset is {}, expected a finite value",
            params.offset
        )));
    }
    if !params.power.is_finite() || params.power <= 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "power is {}, expected a finite value > 0",
            params.power
        )));
    }
    Ok(())
}

/// Apply a parametric slope/offset/power tone curve.
///
/// Computes `out = max(in · slope + offset, 0)^power`, element-wise.
///
/// This is the ASC Color Decision List primary correction, chosen over a
/// spline for three reasons: it is a published interchange standard, so
/// a grade means the same thing in other tools; three numbers are enough
/// for the "lift/gamma/gain" adjustment photographers already know; and
/// it is monotonic for any positive `slope` and `power`, so it cannot
/// invert tonal order the way a mis-shaped spline can.
///
/// The clamp before the exponent is required, not cosmetic: a negative
/// base raised to a fractional power has no real value. Clamping there
/// means a negative `offset` crushes to black rather than producing
/// NaN. It also means the curve is *not* invertible below the clamp
/// point — information pushed below zero is gone.
///
/// Unlike [`zone_system`], this kernel places no constraint on the
/// channel count: it is a scalar function applied element-wise, and it
/// sits after split-toning in the pipeline, where the data has become
/// three-channel again.
///
/// Input shape: `(H, W, C)`, any layout. Output: `(H, W, C)`,
/// C-contiguous.
///
/// Reference: American Society of Cinematographers Technology Committee,
/// "ASC Color Decision List (ASC CDL) Transfer Functions and Interchange
/// Syntax", version 1.2 (2009), §2.1.
///
/// # Errors
/// Returns [`PhaiosError::Parameter`] if any field is not finite, or if
/// `power` is not strictly positive.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn tone_curve(
    img: ArrayView3<f32>,
    params: &ToneCurveParams,
) -> Result<Array3<f32>, PhaiosError> {
    validate_tone_curve(params)?;

    let ToneCurveParams {
        slope,
        offset,
        power,
    } = *params;
    let is_identity = slope == 1.0 && offset == 0.0 && power == 1.0;

    let mut out = crate::alloc::zeros3::<f32>(img.dim())?;
    if is_identity {
        out.assign(&img);
        return Ok(out);
    }

    // powf is the dominant cost; skip it entirely when the exponent is 1.
    let unit_power = power == 1.0;
    ndarray::Zip::from(&mut out).and(img).par_for_each(|o, &v| {
        let t = (v * slope + offset).max(0.0);
        *o = if unit_power { t } else { t.powf(power) };
    });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn no_op_params_are_identity() {
        let img = array![[[0.18_f32]], [[0.36_f32]]];
        let params = ZoneParams::default();
        let out = zone_system(img.view(), &params).unwrap();
        for (&expected, &got) in img.iter().zip(out.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn zone_v_plus_one_stop_doubles_middle_grey() {
        // Middle grey at Zone V → +1 stop should approximately double the value.
        // The Gaussian blend means the multiplier is slightly less than 2.0
        // (Gaussian peak is 1.0 at zone_pos == 5, so total_offset ≈ 1.0).
        let img = array![[[MIDDLE_GREY]]];
        let mut offsets = HashMap::new();
        offsets.insert(5_i32, 1.0_f32);
        let params = ZoneParams::new(offsets);
        let out = zone_system(img.view(), &params).unwrap();
        let ratio = out[[0, 0, 0]] / MIDDLE_GREY;
        // At zone_pos = 5 (exactly Zone V), Gaussian peak = exp(0) = 1.0.
        // So output = 0.18 * 2^1 = 0.36 exactly.
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "expected ~2×, got ratio {ratio:.4} (value {})",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn shape_error_on_rgb_input() {
        let img = Array3::<f32>::zeros((4, 4, 3));
        assert!(zone_system(img.view(), &ZoneParams::default()).is_err());
    }

    // ── Parametric tone curve ────────────────────────────────────────────────

    #[test]
    fn tone_curve_default_is_identity() {
        let img = Array3::<f32>::from_shape_fn((8, 8, 3), |(y, x, c)| (y * 24 + x * 3 + c) as f32);
        let out = tone_curve(img.view(), &ToneCurveParams::default()).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn tone_curve_slope_is_gain() {
        let img = array![[[0.2_f32, 0.5, 0.8]]];
        let out = tone_curve(img.view(), &ToneCurveParams::new(2.0, 0.0, 1.0)).unwrap();
        for (i, expected) in [0.4_f32, 1.0, 1.6].iter().enumerate() {
            assert!((out[[0, 0, i]] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn tone_curve_offset_is_lift() {
        // Offset moves black, which is exactly what distinguishes it from
        // slope: a pure gain leaves 0 at 0.
        let img = array![[[0.0_f32, 0.5]]];
        let out = tone_curve(img.view(), &ToneCurveParams::new(1.0, 0.1, 1.0)).unwrap();
        assert!((out[[0, 0, 0]] - 0.1).abs() < 1e-6);
        assert!((out[[0, 0, 1]] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn tone_curve_power_fixes_zero_and_one() {
        // Gamma bends the middle while pinning both ends.
        let img = array![[[0.0_f32, 0.25, 1.0]]];
        let out = tone_curve(img.view(), &ToneCurveParams::new(1.0, 0.0, 0.5)).unwrap();
        assert!((out[[0, 0, 0]] - 0.0).abs() < 1e-6, "0 must stay 0");
        assert!((out[[0, 0, 2]] - 1.0).abs() < 1e-6, "1 must stay 1");
        assert!(
            (out[[0, 0, 1]] - 0.5).abs() < 1e-6,
            "0.25^0.5 = 0.5, got {}",
            out[[0, 0, 1]]
        );
    }

    #[test]
    fn tone_curve_is_monotonic() {
        let n = 4096;
        let img = Array3::from_shape_fn((1, n, 1), |(_, x, _)| x as f32 / n as f32 * 2.0);
        let out = tone_curve(img.view(), &ToneCurveParams::new(1.3, -0.05, 1.7)).unwrap();
        let mut prev = f32::NEG_INFINITY;
        for &v in out.iter() {
            assert!(v >= prev, "tone curve not monotonic: {prev} then {v}");
            prev = v;
        }
    }

    #[test]
    fn tone_curve_clamps_before_the_exponent() {
        // A negative base under a fractional exponent has no real value;
        // the clamp must turn it into black, not NaN.
        let img = array![[[0.0_f32, 0.01, 0.5]]];
        let out = tone_curve(img.view(), &ToneCurveParams::new(1.0, -0.2, 0.5)).unwrap();
        for &v in out.iter() {
            assert!(v.is_finite(), "produced {v}");
            assert!(v >= 0.0);
        }
        assert_eq!(out[[0, 0, 0]], 0.0);
        assert_eq!(out[[0, 0, 1]], 0.0);
    }

    #[test]
    fn tone_curve_rejects_invalid_parameters() {
        let img = array![[[0.5_f32]]];
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    tone_curve(img.view(), &ToneCurveParams::new(1.0, 0.0, bad)).unwrap_err(),
                    PhaiosError::Parameter(_)
                ),
                "power {bad} should be rejected"
            );
        }
        assert!(matches!(
            tone_curve(img.view(), &ToneCurveParams::new(f32::NAN, 0.0, 1.0)).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
        assert!(matches!(
            tone_curve(img.view(), &ToneCurveParams::new(1.0, f32::INFINITY, 1.0)).unwrap_err(),
            PhaiosError::Parameter(_)
        ));
    }

    #[test]
    fn tone_curve_accepts_any_layout_and_channel_count() {
        let img = Array3::<f32>::from_shape_fn((6, 4, 3), |(y, x, c)| {
            ((y * 12 + x * 3 + c) % 11) as f32 / 11.0
        });
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let params = ToneCurveParams::new(1.2, 0.01, 0.9);
        let out = tone_curve(strided, &params).unwrap();
        assert_eq!(out.dim(), (3, 4, 3));
        assert!(out.is_standard_layout());
        assert_eq!(out, tone_curve(strided.to_owned().view(), &params).unwrap());

        assert!(tone_curve(Array3::<f32>::zeros((2, 2, 1)).view(), &params).is_ok());
    }

    #[test]
    fn tone_curve_params_round_trip() {
        let p = ToneCurveParams::new(1.1, 0.02, 0.8);
        assert!(p.__eq__(&ToneCurveParams::new(1.1, 0.02, 0.8)));
        assert!(!p.__eq__(&ToneCurveParams::default()));
    }

    #[test]
    fn rejects_zone_index_outside_the_eleven_zones() {
        let img = array![[[0.18_f32]]];
        for bad in [-1_i32, 11, 99] {
            let mut offsets = HashMap::new();
            offsets.insert(bad, 1.0_f32);
            let err = zone_system(img.view(), &ZoneParams::new(offsets)).unwrap_err();
            assert!(
                matches!(err, PhaiosError::Parameter(_)),
                "zone {bad} should be a parameter error, got {err:?}"
            );
        }
        // The valid range is accepted.
        for good in 0..=10 {
            let mut offsets = HashMap::new();
            offsets.insert(good, 0.5_f32);
            assert!(zone_system(img.view(), &ZoneParams::new(offsets)).is_ok());
        }
    }

    #[test]
    fn rejects_non_finite_offset() {
        let img = array![[[0.18_f32]]];
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut offsets = HashMap::new();
            offsets.insert(5_i32, bad);
            assert!(matches!(
                zone_system(img.view(), &ZoneParams::new(offsets)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
    }

    #[test]
    fn extreme_offsets_are_applied_not_clamped() {
        // Documented behaviour: offsets are a number of stops, and the
        // kernel does not clamp them. ±3 is a recommendation, not a limit.
        let img = array![[[MIDDLE_GREY]]];
        let mut offsets = HashMap::new();
        offsets.insert(5_i32, 10.0_f32);
        let out = zone_system(img.view(), &ZoneParams::new(offsets)).unwrap();
        let expected = MIDDLE_GREY * 2.0_f32.powi(10);
        assert!(
            (out[[0, 0, 0]] - expected).abs() < expected * 1e-5,
            "expected {expected}, got {}",
            out[[0, 0, 0]]
        );
    }

    #[test]
    fn offsets_round_trip_through_the_getter() {
        let mut map = HashMap::new();
        map.insert(3_i32, -0.4_f32);
        map.insert(7_i32, 1.25_f32);
        let params = ZoneParams::new(map.clone());
        assert_eq!(params.offsets(), map);
        assert!(params.__eq__(&ZoneParams::new(map)));
        assert!(!params.__eq__(&ZoneParams::default()));
    }

    #[test]
    fn output_is_independent_of_hashmap_iteration_order() {
        // `RandomState` seeds every HashMap instance separately, so two maps
        // holding the same keys iterate in different orders — within one
        // process as well as across processes. f32 addition is not
        // associative, so summing the Gaussian terms in hash order makes the
        // output depend on that seed. Measured before the fix: eight
        // processes produced eight different images (12.8% of pixels off by
        // 1 ULP; 23 in 65536 differed by one code at 16-bit).
        let img = Array3::<f32>::from_shape_fn((64, 64, 1), |(y, x, _)| {
            0.001 + (y * 64 + x) as f32 / 2048.0
        });
        let pairs: [(i32, f32); 11] = [
            (0, -0.7),
            (1, 0.3),
            (2, -0.2),
            (3, 0.9),
            (4, -0.4),
            (5, 0.6),
            (6, -0.15),
            (7, 0.45),
            (8, -0.8),
            (9, 0.25),
            (10, 0.55),
        ];

        let reference = zone_system(
            img.view(),
            &ZoneParams::new(pairs.iter().copied().collect()),
        )
        .unwrap();

        for round in 0..32 {
            let mut map = HashMap::new();
            if round % 2 == 0 {
                for &(z, o) in pairs.iter() {
                    map.insert(z, o);
                }
            } else {
                for &(z, o) in pairs.iter().rev() {
                    map.insert(z, o);
                }
            }
            let out = zone_system(img.view(), &ZoneParams::new(map)).unwrap();
            assert_eq!(
                out, reference,
                "round {round}: output depends on HashMap iteration order"
            );
        }
    }

    #[test]
    fn accepts_non_contiguous_input() {
        // Regression: the kernel used to call `as_slice().expect(...)`,
        // which panicked on any strided view. Reaching Python that panic
        // became `pyo3_runtime.PanicException`, which does not inherit
        // from `Exception` and so escapes ordinary caller error handling.
        let img =
            Array3::<f32>::from_shape_fn((8, 4, 1), |(y, x, _)| (y * 4 + x) as f32 / 32.0 + 0.01);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        assert!(
            !strided.is_standard_layout(),
            "test setup: view should not be C-contiguous"
        );

        let mut offsets = HashMap::new();
        offsets.insert(5_i32, 0.5_f32);
        let params = ZoneParams::new(offsets);

        let out = zone_system(strided, &params).unwrap();
        assert_eq!(out.dim(), (4, 4, 1));
        assert!(out.is_standard_layout(), "output must be C-contiguous");

        // Must agree bit-for-bit with the same data laid out contiguously.
        let expected = zone_system(strided.to_owned().view(), &params).unwrap();
        assert_eq!(out, expected);
    }

    #[test]
    fn identity_path_accepts_non_contiguous_input() {
        // The empty-offsets fast path takes a different code branch.
        let img = Array3::<f32>::from_shape_fn((6, 3, 1), |(y, x, _)| (y + x) as f32);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let out = zone_system(strided, &ZoneParams::default()).unwrap();
        assert_eq!(out, strided.to_owned());
    }
}
