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
    if img.shape()[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "zone_system expects (H, W, 1) luminance input, got shape {:?}",
            img.shape()
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
    let mut out = Array3::<f32>::zeros((h, w, 1));

    // No offsets → identity. Hoisted out of the pixel loop.
    if offsets.is_empty() {
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
