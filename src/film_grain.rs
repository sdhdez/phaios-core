// SPDX-License-Identifier: GPL-3.0-or-later
//! Procedural film grain.
//!
//! Adds band-limited noise whose amplitude peaks in the midtones, in the
//! manner of developed silver-halide grain:
//!
//! ```text
//! out = L + intensity · 4·L·(1−L) · bandpass(noise, size)
//! ```
//!
//! # Determinism
//!
//! The noise is **not** drawn from a sequential generator. Each pixel's
//! value is a hash of `(seed, x, y)`, so it depends on nothing but its
//! own coordinates:
//!
//! - identical output for one thread or thirty-two, since no pixel waits
//!   on another's RNG state;
//! - identical output across platforms, since integer hashing has no
//!   floating-point tolerance;
//! - the grain at a given pixel does not move when unrelated parameters
//!   change.
//!
//! A per-tile `SeedableRng` would also be reproducible, but only as long
//! as the tiling never changes — the tile size would silently become
//! part of the output contract. Hashing coordinates has no such hidden
//! parameter, and it is why this kernel needs no RNG dependency.
//!
//! The hash is splitmix64, the finalizer from Sebastiano Vigna's
//! SplittableRandom / xoshiro family: four multiply-xor-shift rounds
//! that pass BigCrush as a bit mixer. Gaussians come from the
//! Box–Muller transform (G. E. P. Box and Mervin E. Muller, "A Note on
//! the Generation of Random Normal Deviates", *Annals of Mathematical
//! Statistics* 29(2), 1958, pp. 610–611).
//!
//! # Why band-pass rather than white noise
//!
//! White noise added per pixel looks like sensor noise, not grain: it
//! has no scale, so it disappears when the image is downsampled and
//! turns to mush when it is enlarged. Real grain has a characteristic
//! size. Filtering the noise to a band around `size_pixels` gives it
//! one.

use ndarray::{Array3, ArrayView3, s};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;
use crate::integral::{sat, window_sum};

// ── Noise generation ──────────────────────────────────────────────────────────

/// splitmix64 finalizer — the bit mixer behind `SplittableRandom`.
///
/// Reference: Guy L. Steele Jr., Doug Lea and Christine H. Flood,
/// "Fast splittable pseudorandom number generators", *OOPSLA '14*,
/// ACM SIGPLAN Notices 49(10), pp. 453–472, DOI 10.1145/2660193.2660195.
///
/// The specific finalizer constants below are the widely-used variant
/// popularised by Sebastiano Vigna's public-domain `splitmix64.c`
/// (<https://prng.di.unimi.it/splitmix64.c>) rather than the exact
/// constants in the OOPSLA paper. An earlier version of this comment
/// credited the whole construction to Vigna's 2017 *Journal of
/// Computational and Applied Mathematics* paper, which is about the
/// xorshift family and does not describe splitmix64 at all.
#[inline]
#[doc(hidden)]
pub fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash a pixel coordinate and a seed into 64 well-mixed bits.
///
/// The coordinates go through their own mixing round before meeting the
/// seed: without it, `(x, y)` and `(y, x)` would collide, and the grain
/// would show a visible diagonal symmetry.
#[inline]
#[doc(hidden)]
pub fn pixel_hash(seed: u64, x: u64, y: u64) -> u64 {
    let key = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ y.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    splitmix64(seed ^ splitmix64(key))
}

/// A standard normal deviate from 64 random bits, via Box–Muller.
///
/// Uses the top 24 bits and the next 24 for the two uniforms — 24 bits
/// is the full precision of an `f32` mantissa, so nothing is discarded.
/// The `+ 0.5` offset keeps `u1` strictly positive, since `ln(0)` is
/// −∞.
#[inline]
fn standard_normal(bits: u64) -> f32 {
    const SCALE: f32 = 1.0 / 16_777_216.0; // 2^-24
    let u1 = ((bits >> 40) as f32 + 0.5) * SCALE;
    let u2 = (((bits >> 16) & 0xFF_FFFF) as f32) * SCALE;
    let radius = (-2.0 * u1.ln()).sqrt();
    radius * (std::f32::consts::TAU * u2).cos()
}

// ── Parameter type ────────────────────────────────────────────────────────────

/// Parameters for [`film_grain`].
///
/// ```python
/// params = phaios_core.GrainParams(intensity=0.25, size_pixels=1.5, seed=20260815)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GrainParams {
    /// Grain amplitude at the midtones, 0..=1 for the useful range.
    ///
    /// The band-passed noise is normalised to unit variance, so this is
    /// the **standard deviation** of the grain in linear units where the
    /// `4·L·(1−L)` envelope peaks, at L = 0.5. Individual pixels reach
    /// further, as Gaussian tails do. Not clamped, but values above 1
    /// swamp the image.
    ///
    /// The measured spread falls slightly below `intensity` in dark
    /// tones, where the output clamp at zero truncates the lower tail.
    #[pyo3(get, set)]
    pub intensity: f32,
    /// Characteristic grain size in pixels; 0.5..=4.0 is typical.
    ///
    /// This is a *scale*, not a radius: the noise is band-passed around
    /// it, so raising it makes the clumps coarser rather than merely
    /// blurrier. Sizes below 1 pixel cannot be resolved and behave as 1.
    #[pyo3(get, set)]
    pub size_pixels: f32,
    /// Explicit RNG seed. Same seed, same parameters, same input →
    /// bit-identical output, on any thread count and any platform.
    #[pyo3(get, set)]
    pub seed: u64,
}

#[pymethods]
impl GrainParams {
    /// Create new ``GrainParams``.
    #[new]
    #[pyo3(signature = (intensity = 0.0, size_pixels = 1.0, seed = 0))]
    pub fn new(intensity: f32, size_pixels: f32, seed: u64) -> Self {
        Self {
            intensity,
            size_pixels,
            seed,
        }
    }

    /// Two ``GrainParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.intensity.to_bits() == other.intensity.to_bits()
            && self.size_pixels.to_bits() == other.size_pixels.to_bits()
            && self.seed == other.seed
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "GrainParams(intensity={}, size_pixels={}, seed={})",
            self.intensity, self.size_pixels, self.seed
        )
    }
}

impl Default for GrainParams {
    fn default() -> Self {
        Self {
            intensity: 0.0,
            size_pixels: 1.0,
            seed: 0,
        }
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Validate shape and parameters. Shared verbatim by the CPU kernel and
/// the CUDA kernel so both backends reject exactly the same inputs with
/// exactly the same messages.
pub(crate) fn validate(shape: &[usize], params: &GrainParams) -> Result<(), PhaiosError> {
    if shape[2] != 1 {
        return Err(PhaiosError::Shape(format!(
            "film_grain expects (H, W, 1) luminance input, got shape {shape:?}"
        )));
    }
    if !params.intensity.is_finite() || params.intensity < 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "intensity is {}, expected a finite value >= 0",
            params.intensity
        )));
    }
    if !params.size_pixels.is_finite() || params.size_pixels <= 0.0 {
        return Err(PhaiosError::Parameter(format!(
            "size_pixels is {}, expected a finite value > 0",
            params.size_pixels
        )));
    }
    Ok(())
}

/// The band-pass radii and analytic normalisation for `size_pixels`,
/// shared with the CUDA backend so both compute the identical f32
/// constant from the identical f64 arithmetic.
pub(crate) fn bandpass_geometry(size_pixels: f32) -> (usize, usize, f32) {
    let outer = size_pixels.max(1.0).round() as usize;
    let inner = ((size_pixels * 0.5).floor() as usize).min(outer - 1);
    let n_inner = ((2 * inner + 1) * (2 * inner + 1)) as f64;
    let n_outer = ((2 * outer + 1) * (2 * outer + 1)) as f64;
    let variance = (1.0 / n_inner - 1.0 / n_outer).max(f64::MIN_POSITIVE);
    let normalisation = (1.0 / variance.sqrt()) as f32;
    (inner, outer, normalisation)
}

/// Add procedural film grain.
///
/// ```text
/// out = max(L + intensity · 4·t·(1−t) · bandpass(noise), 0)   where t = clamp(L, 0, 1)
/// ```
///
/// The band-pass is a difference of two box filters over the same white
/// noise field, with radii `floor(size/2)` and `round(size)`, which
/// concentrates the noise energy around the requested scale. The inner
/// radius is capped one below the outer: equal radii would make the
/// difference identically zero. It is normalised to unit
/// variance analytically — for nested windows of `n₁` and `n₂` pixels
/// the difference has variance `1/n₁ − 1/n₂` — so `intensity` means the
/// same thing at every grain size, with no extra pass over the image and
/// no order-dependent reduction.
///
/// The `4·t·(1−t)` envelope peaks at mid-grey and vanishes at both ends:
/// film has no grain where no silver was developed, and none where the
/// emulsion is fully saturated. `t` is clamped before the envelope, so
/// scene-referred highlights above 1.0 get no grain rather than negative
/// grain.
///
/// Boundary windows are clamped to the image, so grain variance rises
/// slightly in the outermost `size` pixels — the analytic normalisation
/// uses the interior window area.
///
/// Input shape: `(H, W, 1)`, any layout. Output: `(H, W, 1)`,
/// C-contiguous.
///
/// Order-sensitive: a finishing stage, applied after tone work. Grain
/// added before a tone curve would be reshaped by it, and the envelope
/// would no longer sit on the midtones the viewer sees.
///
/// # Errors
/// - [`PhaiosError::Shape`] if the input is not `(H, W, 1)`.
/// - [`PhaiosError::Parameter`] if `intensity` is negative or
///   non-finite, or `size_pixels` is not finite and positive.
#[must_use = "kernel returns a new array; ignoring it wastes work"]
pub fn film_grain(img: ArrayView3<f32>, params: &GrainParams) -> Result<Array3<f32>, PhaiosError> {
    validate(img.shape(), params)?;

    let (h, w, _) = img.dim();
    let mut out = crate::alloc::zeros3::<f32>((h, w, 1))?;

    if params.intensity == 0.0 || h == 0 || w == 0 {
        out.assign(&img);
        return Ok(out);
    }

    // White noise, keyed by coordinate: reproducible regardless of how
    // the work is scheduled.
    let seed = params.seed;
    let mut noise = crate::alloc::zeros2::<f32>((h, w))?;
    ndarray::Zip::indexed(&mut noise).par_for_each(|(y, x), n| {
        *n = standard_normal(pixel_hash(seed, x as u64, y as u64));
    });

    // Both box filters read the same table.
    let table = sat(noise.view(), |v| v as f64)?;
    drop(noise);

    // The two window radii must never coincide: equal radii make the
    // difference identically zero, i.e. no grain at all. Rounding both
    // did exactly that at size_pixels = 1.0, the commonest setting.
    // Flooring the inner radius sends size 1 to (0, 1) — the raw noise
    // minus its 3×3 mean, the finest grain that a pixel grid can carry —
    // and the `outer - 1` cap keeps them apart at every other size.
    let (inner, outer, normalisation) = bandpass_geometry(params.size_pixels);

    let intensity = params.intensity;
    ndarray::Zip::indexed(out.slice_mut(s![.., .., 0]))
        .and(img.slice(s![.., .., 0]))
        .par_for_each(|(y, x), o, &l| {
            let (sum_inner, area_inner) = window_sum(&table, y, x, inner, h, w);
            let (sum_outer, area_outer) = window_sum(&table, y, x, outer, h, w);
            let bandpass = (sum_inner / area_inner - sum_outer / area_outer) as f32;

            // Clamped before the envelope: above 1.0 the parabola would
            // go negative and invert the grain.
            let t = l.clamp(0.0, 1.0);
            let envelope = 4.0 * t * (1.0 - t);

            *o = (l + intensity * envelope * normalisation * bandpass).max(0.0);
        });

    Ok(out)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(h: usize, w: usize, value: f32) -> Array3<f32> {
        Array3::from_elem((h, w, 1), value)
    }

    fn mean_and_std(values: impl Iterator<Item = f32> + Clone) -> (f32, f32) {
        let v: Vec<f32> = values.collect();
        let n = v.len() as f32;
        let mean = v.iter().sum::<f32>() / n;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        (mean, var.sqrt())
    }

    #[test]
    fn standard_normal_is_standard() {
        // A wrong Box–Muller constant would still look like noise; only
        // the moments catch it.
        let samples: Vec<f32> = (0..100_000)
            .map(|i| standard_normal(pixel_hash(1, i % 317, i / 317)))
            .collect();
        let (mean, std) = mean_and_std(samples.iter().copied());
        assert!(mean.abs() < 0.02, "mean {mean} should be ~0");
        assert!((std - 1.0).abs() < 0.02, "std {std} should be ~1");
    }

    #[test]
    fn noise_is_not_symmetric_in_x_and_y() {
        // Regression: a naive `x ^ y` key makes (x, y) and (y, x) collide,
        // which shows up as a diagonal mirror line through the grain.
        let mut collisions = 0;
        for x in 0..64_u64 {
            for y in 0..64_u64 {
                if x != y && pixel_hash(7, x, y) == pixel_hash(7, y, x) {
                    collisions += 1;
                }
            }
        }
        assert_eq!(collisions, 0, "transposed coordinates collide");
    }

    #[test]
    fn different_seeds_give_different_noise() {
        let a = standard_normal(pixel_hash(1, 10, 20));
        let b = standard_normal(pixel_hash(2, 10, 20));
        assert!((a - b).abs() > 1e-6, "seed had no effect: {a} vs {b}");
    }

    #[test]
    fn zero_intensity_is_identity() {
        let img =
            Array3::<f32>::from_shape_fn((16, 16, 1), |(y, x, _)| (y * 16 + x) as f32 / 256.0);
        let out = film_grain(img.view(), &GrainParams::new(0.0, 2.0, 42)).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn same_seed_gives_identical_output() {
        let img = flat(48, 48, 0.5);
        let params = GrainParams::new(0.3, 2.0, 12345);
        let first = film_grain(img.view(), &params).unwrap();
        for _ in 0..8 {
            assert_eq!(
                film_grain(img.view(), &params).unwrap(),
                first,
                "grain must be bit-identical for the same seed"
            );
        }
    }

    #[test]
    fn different_seeds_give_different_grain() {
        let img = flat(32, 32, 0.5);
        let a = film_grain(img.view(), &GrainParams::new(0.3, 2.0, 1)).unwrap();
        let b = film_grain(img.view(), &GrainParams::new(0.3, 2.0, 2)).unwrap();
        assert_ne!(a, b, "different seeds must give different grain");
    }

    #[test]
    fn grain_is_independent_of_the_thread_count() {
        // The property that motivates hashing coordinates instead of
        // running a sequential generator per tile.
        let img = flat(96, 96, 0.5);
        let params = GrainParams::new(0.4, 2.0, 99);

        let reference = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| film_grain(img.view(), &params).unwrap());

        for threads in [2, 3, 8] {
            let out = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| film_grain(img.view(), &params).unwrap());
            assert_eq!(out, reference, "output changed with {threads} threads");
        }
    }

    #[test]
    fn envelope_silences_grain_at_black_and_white() {
        // 4·t·(1−t) is zero at both ends: no grain where no silver was
        // developed, none where the emulsion saturated.
        for value in [0.0_f32, 1.0] {
            let img = flat(32, 32, value);
            let out = film_grain(img.view(), &GrainParams::new(1.0, 2.0, 5)).unwrap();
            for &v in out.iter() {
                assert!(
                    (v - value).abs() < 1e-6,
                    "grain appeared at L = {value}: got {v}"
                );
            }
        }
    }

    #[test]
    fn highlights_above_one_get_no_grain() {
        // Without clamping t, the parabola goes negative above 1.0 and
        // the grain inverts instead of fading out.
        let img = flat(32, 32, 2.5);
        let out = film_grain(img.view(), &GrainParams::new(1.0, 2.0, 5)).unwrap();
        for &v in out.iter() {
            assert!((v - 2.5).abs() < 1e-5, "grain at L = 2.5: got {v}");
        }
    }

    #[test]
    fn grain_peaks_in_the_midtones() {
        let params = GrainParams::new(0.3, 2.0, 7);
        let deviation = |level: f32| {
            let img = flat(64, 64, level);
            let out = film_grain(img.view(), &params).unwrap();
            mean_and_std(out.iter().map(|&v| v - level)).1
        };
        let mid = deviation(0.5);
        let shadow = deviation(0.1);
        let highlight = deviation(0.9);
        assert!(mid > shadow * 2.0, "midtone {mid} vs shadow {shadow}");
        assert!(
            mid > highlight * 2.0,
            "midtone {mid} vs highlight {highlight}"
        );
    }

    #[test]
    fn intensity_scales_the_deviation_linearly() {
        let img = flat(64, 64, 0.5);
        let spread = |intensity| {
            let out = film_grain(img.view(), &GrainParams::new(intensity, 2.0, 3)).unwrap();
            mean_and_std(out.iter().map(|&v| v - 0.5)).1
        };
        let (weak, strong) = (spread(0.1), spread(0.2));
        assert!(
            (strong / weak - 2.0).abs() < 0.05,
            "doubling intensity should double the spread: {weak} then {strong}"
        );
    }

    #[test]
    fn every_valid_size_actually_produces_grain() {
        // Regression: rounding both radii collapsed them to the same
        // value at size 1.0 — the commonest setting — and the difference
        // of two identical box filters is exactly zero, so the kernel
        // silently did nothing.
        let img = flat(64, 64, 0.5);
        for size in [0.5_f32, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0] {
            let out = film_grain(img.view(), &GrainParams::new(0.3, size, 17)).unwrap();
            let spread = mean_and_std(out.iter().map(|&v| v - 0.5)).1;
            assert!(
                spread > 0.01,
                "size {size} produced no grain (spread {spread})"
            );
        }
    }

    #[test]
    fn normalisation_keeps_intensity_comparable_across_sizes() {
        // Without the analytic normalisation, coarser grain would come
        // out far weaker, and the intensity slider would mean something
        // different at every size.
        let img = flat(128, 128, 0.5);
        let spread = |size| {
            let out = film_grain(img.view(), &GrainParams::new(0.2, size, 11)).unwrap();
            mean_and_std(out.iter().map(|&v| v - 0.5)).1
        };
        let fine = spread(1.0);
        let coarse = spread(4.0);
        assert!(
            (coarse / fine - 1.0).abs() < 0.5,
            "spread should be within 50% across sizes: fine {fine}, coarse {coarse}"
        );
    }

    #[test]
    fn larger_size_produces_coarser_structure() {
        // Neighbour correlation is what "size" means: at size 1 adjacent
        // pixels are nearly independent, at size 6 they move together.
        let img = flat(128, 128, 0.5);
        let correlation = |size| {
            let out = film_grain(img.view(), &GrainParams::new(0.3, size, 21)).unwrap();
            let dev: Vec<f32> = out.iter().map(|&v| v - 0.5).collect();
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for y in 0..128 {
                for x in 0..127 {
                    let a = dev[y * 128 + x] as f64;
                    let b = dev[y * 128 + x + 1] as f64;
                    num += a * b;
                    den += a * a;
                }
            }
            num / den
        };
        let fine = correlation(1.0);
        let coarse = correlation(6.0);
        assert!(
            coarse > fine + 0.2,
            "coarse grain should be more correlated: {fine} vs {coarse}"
        );
    }

    #[test]
    fn output_is_never_negative() {
        let img = flat(64, 64, 0.05);
        let out = film_grain(img.view(), &GrainParams::new(5.0, 2.0, 8)).unwrap();
        for &v in out.iter() {
            assert!(v >= 0.0, "got {v}");
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        let img = flat(8, 8, 0.5);
        for bad in [-0.1_f32, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                film_grain(img.view(), &GrainParams::new(bad, 1.0, 0)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        for bad in [0.0_f32, -1.0, f32::NAN] {
            assert!(matches!(
                film_grain(img.view(), &GrainParams::new(0.5, bad, 0)).unwrap_err(),
                PhaiosError::Parameter(_)
            ));
        }
        let rgb = Array3::<f32>::zeros((4, 4, 3));
        assert!(matches!(
            film_grain(rgb.view(), &GrainParams::default()).unwrap_err(),
            PhaiosError::Shape(_)
        ));
    }

    #[test]
    fn accepts_any_layout() {
        let img = Array3::<f32>::from_shape_fn((8, 4, 1), |(y, x, _)| (y * 4 + x) as f32 / 32.0);
        let strided = img.slice(ndarray::s![..;2, .., ..]);
        let params = GrainParams::new(0.3, 2.0, 4);
        let out = film_grain(strided, &params).unwrap();
        assert_eq!(out.dim(), (4, 4, 1));
        assert!(out.is_standard_layout());
        assert_eq!(out, film_grain(strided.to_owned().view(), &params).unwrap());
    }
}
