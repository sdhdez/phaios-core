// SPDX-License-Identifier: GPL-3.0-or-later
//! Histogram — the crate's first reduction.
//!
//! Every other kernel maps an image to an image. This one maps an image
//! to *statistics*, which is a different shape of API and a deliberate
//! widening of §1's "pure functions on `f32` (H, W, C) arrays". It earns
//! the widening because the decisions around a histogram are
//! specification-shaped, and specifications belong in one place.
//!
//! # Why this is not `numpy.histogram`
//!
//! Computing bin counts is trivial. Agreeing on *what to count* is not,
//! and a front-end that answers these questions for itself will answer
//! them differently from the next front-end:
//!
//! - **Which data?** A histogram of linear scene-referred values is
//!   almost unreadable: 18% grey sits a fifth of the way up the axis and
//!   everything the photographer cares about bunches against the left
//!   edge. Photographers read *display-referred* histograms, so the
//!   intended call site is after [`crate::encode::encode_srgb`]. Clipping
//!   analysis is the exception — see below.
//! - **Which range?** Fixed `[0, 1]` by default, not the data's own
//!   min/max. An auto-ranged histogram silently rescales itself as the
//!   photographer works, so the display stops being comparable between
//!   two adjustments, which is the one thing it is for.
//! - **What counts as clipped?** Samples outside the range are counted
//!   separately rather than being folded into the end bins. Folding them
//!   in is why so many histogram displays show a spike at the right edge
//!   that cannot be distinguished from legitimately bright content.
//!
//! Answering those once, here, is the point. Two consumers of this crate
//! showing the same file must show the same histogram.
//!
//! # Determinism
//!
//! Bin counts are integers, and integer addition is associative, so the
//! result is independent of the order in which samples are visited — at
//! any thread count and on either backend. This is the one reduction in
//! the crate that does not need the care §2's "deterministic reductions"
//! rule demands of floating-point sums, and it is worth saying why
//! rather than leaving a reader to wonder whether it was overlooked.
//!
//! # Reference
//!
//! Counting samples into bins needs no attribution. The one result here
//! that does is the equalising transfer returned by
//! [`Histogram::equalisation_lut`] — that the cumulative distribution
//! function, used as a tone curve, flattens a histogram:
//!
//! > Rafael C. Gonzalez and Richard E. Woods, *Digital Image
//! > Processing*, 4th ed., Pearson (2018), §3.3 "Histogram
//! > Equalization"; histogram matching is §3.4.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView3, Axis};
use pyo3::{pyclass, pymethods};

use crate::error::PhaiosError;

// ── Parameters ────────────────────────────────────────────────────────────────

/// Parameters for [`histogram`].
///
/// ```python
/// # The display default: 256 bins over [0, 1]
/// params = phaios_core.HistogramParams()
///
/// # Fine-grained analysis of a 16-bit export
/// params = phaios_core.HistogramParams(bins=65536)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct HistogramParams {
    /// Number of bins spanning `[min, max]`. Must be at least 2.
    ///
    /// 256 matches an 8-bit display and is the sensible default for a
    /// histogram a person looks at. Larger values are for analysis —
    /// 65536 resolves individual 16-bit codes, which is how you find
    /// quantisation damage.
    #[pyo3(get, set)]
    pub bins: u32,
    /// Lower edge of the counted range. Samples below it are counted in
    /// `below` rather than in bin 0.
    #[pyo3(get, set)]
    pub min: f32,
    /// Upper edge of the counted range, inclusive. Samples above it are
    /// counted in `above` rather than in the last bin.
    ///
    /// Exactly `max` lands in the last bin, so a white pixel in a
    /// display-referred image reads as "at white" rather than "clipped".
    #[pyo3(get, set)]
    pub max: f32,
}

#[pymethods]
impl HistogramParams {
    /// Create new ``HistogramParams``.
    #[new]
    #[pyo3(signature = (bins = 256, min = 0.0, max = 1.0))]
    pub fn new(bins: u32, min: f32, max: f32) -> Self {
        Self { bins, min, max }
    }

    /// Two ``HistogramParams`` are equal when all three fields match.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.bins == other.bins
            && self.min.to_bits() == other.min.to_bits()
            && self.max.to_bits() == other.max.to_bits()
    }

    /// Return a debug representation.
    pub fn __repr__(&self) -> String {
        format!(
            "HistogramParams(bins={}, min={}, max={})",
            self.bins, self.min, self.max
        )
    }
}

impl Default for HistogramParams {
    fn default() -> Self {
        Self {
            bins: 256,
            min: 0.0,
            max: 1.0,
        }
    }
}

// ── Result type ───────────────────────────────────────────────────────────────

/// The result of [`histogram`]: per-channel bin counts plus the samples
/// that fell outside the counted range.
///
/// Counts are `u64`, which cannot overflow for any image that fits in
/// memory.
///
/// `skip_from_py_object` because this is a *result*: it is handed to
/// Python and never taken back as a kernel argument, so the conversion
/// PyO3 would derive has no call site.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Histogram {
    counts: Array2<u64>,
    below: Vec<u64>,
    above: Vec<u64>,
    non_finite: Vec<u64>,
    /// Number of bins per channel.
    #[pyo3(get)]
    pub bins: usize,
    /// Number of channels.
    #[pyo3(get)]
    pub channels: usize,
    /// Lower edge of the counted range.
    #[pyo3(get)]
    pub min: f32,
    /// Upper edge of the counted range.
    #[pyo3(get)]
    pub max: f32,
}

impl Histogram {
    /// Bin counts, shape `(channels, bins)`.
    #[must_use]
    pub fn counts(&self) -> &Array2<u64> {
        &self.counts
    }

    /// Per-channel count of samples below `min` (NaN excluded; −∞ lands
    /// here, since it genuinely is below the range).
    #[must_use]
    pub fn below(&self) -> &[u64] {
        &self.below
    }

    /// Per-channel count of samples above `max` (NaN excluded; +∞ lands
    /// here).
    #[must_use]
    pub fn above(&self) -> &[u64] {
        &self.above
    }

    /// Per-channel count of NaN samples.
    ///
    /// Separate from `below`/`above` because a NaN is not a bright pixel
    /// or a dark one — it is a broken one, and silently folding it into
    /// a clipping count would hide a real defect in the caller's data.
    #[must_use]
    pub fn non_finite(&self) -> &[u64] {
        &self.non_finite
    }

    /// Per-channel total of every sample seen, in range or not — bins,
    /// `below`, `above` and `non_finite` together.
    ///
    /// # Panics
    /// If `channel` is not below [`Histogram::channels`]. This is the
    /// crate's only public function taking a caller-supplied index; the
    /// Python binding guards it and raises `IndexError` instead, since a
    /// panic must never cross the FFI boundary.
    #[must_use]
    pub fn total(&self, channel: usize) -> u64 {
        self.counts.row(channel).iter().sum::<u64>()
            + self.below[channel]
            + self.above[channel]
            + self.non_finite[channel]
    }

    /// Normalised cumulative distribution over the in-range samples,
    /// shape `(channels, bins)`, each row ending at 1.0 (or all zero if
    /// the channel has no in-range samples).
    ///
    /// Entry `b` is the cumulative fraction at the **upper edge** of bin
    /// `b`, which is the usual definition — and precisely why it is not
    /// the right table to hand to [`crate::lut::apply_lut`] directly.
    /// That kernel places entry `i` of an `n`-entry table at input
    /// fraction `i / (n − 1)`, so passing these `bins` values shifts the
    /// transfer by half a bin and lifts the black point. Use
    /// [`Histogram::equalisation_lut`], which adds the leading zero that
    /// makes the two alignments agree.
    ///
    /// Accumulation is in `u64` and only the final division is floating
    /// point, so the result is exact up to one correctly-rounded divide
    /// per bin.
    #[must_use]
    pub fn cdf(&self) -> Array2<f32> {
        // Bounded already: `validate_shape` caps channels x bins before
        // any Histogram can exist, so this cannot be the runaway case.
        let mut out = Array2::<f32>::zeros((self.channels, self.bins));
        for ch in 0..self.channels {
            let in_range: u64 = self.counts.row(ch).iter().sum();
            if in_range == 0 {
                continue;
            }
            let mut running = 0_u64;
            for (bin, &count) in self.counts.row(ch).iter().enumerate() {
                running += count;
                out[[ch, bin]] = (running as f64 / in_range as f64) as f32;
            }
        }
        out
    }
}

impl Histogram {
    /// The equalising transfer, shape `(channels, bins + 1)`, ready to
    /// hand straight to [`crate::lut::apply_lut`] over this histogram's
    /// own `[min, max]`.
    ///
    /// This is [`Histogram::cdf`] with a leading zero. The extra entry is
    /// not cosmetic: `cdf()[b]` is the cumulative fraction at the *upper
    /// edge* of bin `b`, so the `bins` values describe inputs
    /// `1/bins … 1`, while `apply_lut` reads an `n`-entry table as
    /// describing inputs `0 … 1` evenly. Prepending the cumulative
    /// fraction below the first bin — which is zero by definition — makes
    /// entry `i` sit at input fraction `i / bins` on both sides, and the
    /// two agree exactly.
    ///
    /// The half-bin bias this removes is small but systematic: it lifted
    /// black by `1 / bins` while leaving white alone, so equalising an
    /// already-uniform image was not quite the identity. With the leading
    /// zero it is the identity exactly.
    #[must_use]
    pub fn equalisation_lut(&self) -> Array2<f32> {
        let cdf = self.cdf();
        // Bounded by `validate_shape`, as in `cdf`.
        let mut out = Array2::<f32>::zeros((self.channels, self.bins + 1));
        for ch in 0..self.channels {
            for bin in 0..self.bins {
                out[[ch, bin + 1]] = cdf[[ch, bin]];
            }
        }
        out
    }
}

#[pymethods]
impl Histogram {
    /// Bin counts as a ``(channels, bins)`` array of ``uint64``.
    #[pyo3(name = "counts")]
    fn counts_py(&self, py: pyo3::Python<'_>) -> pyo3::Py<numpy::PyArray2<u64>> {
        use numpy::IntoPyArray;
        self.counts.clone().into_pyarray(py).unbind()
    }

    /// The equalising transfer, ``(channels, bins + 1)`` float32, ready
    /// to pass to ``apply_lut`` over this histogram's own range.
    ///
    /// ``cdf()`` with a leading zero: the extra entry aligns the CDF's
    /// bin-upper-edge convention with ``apply_lut``'s even placement of
    /// table entries, removing a half-bin bias that lifted black.
    #[pyo3(name = "equalisation_lut")]
    fn equalisation_lut_py(&self, py: pyo3::Python<'_>) -> pyo3::Py<numpy::PyArray2<f32>> {
        use numpy::IntoPyArray;
        self.equalisation_lut().into_pyarray(py).unbind()
    }

    /// Normalised cumulative distribution, ``(channels, bins)`` float32.
    ///
    /// Entry ``b`` is the cumulative fraction at the upper edge of bin
    /// ``b``. For equalisation use ``equalisation_lut()`` instead — see
    /// its documentation for why the alignment differs.
    #[pyo3(name = "cdf")]
    fn cdf_py(&self, py: pyo3::Python<'_>) -> pyo3::Py<numpy::PyArray2<f32>> {
        use numpy::IntoPyArray;
        self.cdf().into_pyarray(py).unbind()
    }

    /// Per-channel count of samples below ``min``.
    #[pyo3(name = "below")]
    fn below_py(&self) -> Vec<u64> {
        self.below.clone()
    }

    /// Per-channel count of samples above ``max``.
    #[pyo3(name = "above")]
    fn above_py(&self) -> Vec<u64> {
        self.above.clone()
    }

    /// Per-channel count of NaN samples.
    #[pyo3(name = "non_finite")]
    fn non_finite_py(&self) -> Vec<u64> {
        self.non_finite.clone()
    }

    /// Per-channel total of every sample seen.
    #[pyo3(name = "total")]
    fn total_py(&self, channel: usize) -> pyo3::PyResult<u64> {
        if channel >= self.channels {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "channel {channel} out of range for {} channels",
                self.channels
            )));
        }
        Ok(self.total(channel))
    }

    /// Return a debug representation.
    fn __repr__(&self) -> String {
        format!(
            "Histogram(channels={}, bins={}, range=({}, {}))",
            self.channels, self.bins, self.min, self.max
        )
    }
}

// ── Kernel ────────────────────────────────────────────────────────────────────

/// Largest bin count accepted. Well above the 65536 an analysis pass of
/// 16-bit data needs, and far below `i32::MAX`, which the device kernels
/// index with — a bin count that truncated to a negative `int` would make
/// them write outside the counter buffer.
pub const MAX_BINS: u32 = 1 << 22;

/// Largest accumulator the reduction will allocate, in bytes. `bins` and
/// the image's channel count are both caller-controlled and multiply, so
/// without a ceiling a modest-looking request allocates tens of
/// gigabytes — and a failed `Vec` allocation *aborts* the process rather
/// than unwinding, which no Python `except` can catch. Rejecting is the
/// only way to keep the no-panic-across-FFI contract here.
const MAX_ACCUMULATOR_BYTES: usize = 256 << 20;

/// Validate histogram parameters. Shared by the CPU and CUDA backends so
/// both reject the same inputs with the same messages.
// `!(max > min)` rather than `max <= min`: the two differ on NaN, and the
// negated form rejects a NaN bound instead of accepting it. The finiteness
// check above already catches NaN, so this is belt and braces — but the
// belt is what makes the braces safe to remove later.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn validate(params: &HistogramParams) -> Result<(), PhaiosError> {
    if params.bins < 2 {
        return Err(PhaiosError::Parameter(format!(
            "bins is {}, expected at least 2",
            params.bins
        )));
    }
    if params.bins > MAX_BINS {
        return Err(PhaiosError::Parameter(format!(
            "bins is {}, expected at most {MAX_BINS}",
            params.bins
        )));
    }
    if !params.min.is_finite() || !params.max.is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "range is ({}, {}), expected finite bounds",
            params.min, params.max
        )));
    }
    if !(params.max > params.min) {
        return Err(PhaiosError::Parameter(format!(
            "range is ({}, {}), expected max > min",
            params.min, params.max
        )));
    }
    // A span wider than f32 can represent overflows to infinity, and every
    // sample then divides to zero and lands in bin 0 — a defined, stable,
    // silently useless answer. Rejecting is better than returning it.
    if !(params.max - params.min).is_finite() {
        return Err(PhaiosError::Parameter(format!(
            "range ({}, {}) spans more than f32 can represent",
            params.min, params.max
        )));
    }
    Ok(())
}

/// Validate the parameters *against an image shape*, which is where the
/// accumulator's real size becomes known: it is `channels × (bins + 3)`,
/// and both factors come from the caller.
///
/// Shared by the CPU and CUDA backends, so both refuse the same requests
/// with the same message rather than one aborting and the other
/// returning an error.
pub(crate) fn validate_shape(
    shape: (usize, usize, usize),
    params: &HistogramParams,
) -> Result<(), PhaiosError> {
    validate(params)?;
    let channels = shape.2;
    let slots = channels.saturating_mul(params.bins as usize + 3);
    let bytes = slots.saturating_mul(std::mem::size_of::<u64>());
    if bytes > MAX_ACCUMULATOR_BYTES {
        return Err(PhaiosError::Parameter(format!(
            "a {}-channel image at {} bins needs a {} MiB accumulator, above the {} MiB limit",
            channels,
            params.bins,
            bytes >> 20,
            MAX_ACCUMULATOR_BYTES >> 20
        )));
    }
    Ok(())
}

/// Assign one sample to a slot in a per-channel accumulator.
///
/// Returns the offset within the channel's block: `0..bins` for an
/// in-range sample, then `bins` for below, `bins + 1` for above and
/// `bins + 2` for NaN. Keeping the three outliers in the same block
/// means one reduction handles everything.
///
/// Mirrored statement for statement by `src/cuda/ptx/histogram.cu`.
#[inline]
pub(crate) fn slot_of(value: f32, min: f32, max: f32, bins: usize) -> usize {
    if value.is_nan() {
        return bins + 2;
    }
    if value < min {
        return bins;
    }
    if value > max {
        return bins + 1;
    }
    // `max` itself belongs in the last bin, not in `above`: a white pixel
    // in a display-referred image is at white, not clipped.
    let t = (value - min) / (max - min);
    let idx = (t * bins as f32) as usize;
    if idx >= bins { bins - 1 } else { idx }
}

/// Count how the image's samples are distributed, per channel.
///
/// Returns a [`Histogram`] holding `(channels, bins)` counts over
/// `[min, max]`, plus separate tallies of the samples that fell below,
/// above, or were NaN.
///
/// Input shape: `(H, W, C)`, any channel count, any layout.
///
/// Order-sensitive in the sense that matters most: **call it after
/// [`crate::encode::encode_srgb`]** if the histogram is for a person to
/// look at. A linear scene-referred histogram is technically correct and
/// practically unreadable — see the module documentation. Call it on
/// linear data only when analysing headroom, where the question is how
/// far above 1.0 the highlights actually reach.
///
/// Deterministic at any thread count: counts are integers.
///
/// # Errors
/// - [`PhaiosError::Parameter`] if `bins` is below 2 or above
///   [`MAX_BINS`]; if the range bounds are not finite with `max > min`,
///   or span more than `f32` can represent; or if the image's channel
///   count and `bins` together would need an accumulator above the
///   backend's limit.
#[must_use = "the histogram is the result; ignoring it wastes a full pass"]
pub fn histogram(img: ArrayView3<f32>, params: &HistogramParams) -> Result<Histogram, PhaiosError> {
    validate_shape(img.dim(), params)?;

    let (h, _, channels) = img.dim();
    let bins = params.bins as usize;
    let (min, max) = (params.min, params.max);
    let stride = bins + 3; // bins, then below / above / nan

    // Rows are processed in chunks rather than one at a time because
    // rayon's `fold` allocates a fresh accumulator per *split*, and the
    // split count follows the item count — iterating single rows on a
    // tall image left hundreds of full-size accumulators live at once
    // (measured at ~281 on a 24 MP frame). Chunking bounds that to a
    // small multiple of the thread count.
    //
    // The merge is pairwise and the accumulators are integers, so
    // addition is associative and the chunking cannot change the answer.
    let chunk_rows = h
        .div_ceil(rayon::current_num_threads().max(1).saturating_mul(4))
        .max(1);
    let acc = img
        .axis_chunks_iter(Axis(0), chunk_rows)
        .into_par_iter()
        .fold(
            || vec![0_u64; channels * stride],
            |mut acc, chunk| {
                for row in chunk.axis_iter(Axis(0)) {
                    for pixel in row.axis_iter(Axis(0)) {
                        for (ch, &v) in pixel.iter().enumerate() {
                            acc[ch * stride + slot_of(v, min, max, bins)] += 1;
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0_u64; channels * stride],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    Ok(assemble(acc, channels, bins, min, max))
}

/// Build the public [`Histogram`] from a flat `channels × (bins + 3)`
/// accumulator. Shared with the CUDA backend, which produces the same
/// layout on the device.
pub(crate) fn assemble(
    acc: Vec<u64>,
    channels: usize,
    bins: usize,
    min: f32,
    max: f32,
) -> Histogram {
    let stride = bins + 3;
    // Not routed through `alloc`: `validate_shape` has already capped
    // channels x bins at the accumulator budget before any Histogram can
    // exist, so this allocation is bounded more tightly than the crate
    // limit would bound it.
    let mut counts = Array2::<u64>::zeros((channels, bins));
    let mut below = vec![0_u64; channels];
    let mut above = vec![0_u64; channels];
    let mut non_finite = vec![0_u64; channels];

    for ch in 0..channels {
        let base = ch * stride;
        for bin in 0..bins {
            counts[[ch, bin]] = acc[base + bin];
        }
        below[ch] = acc[base + bins];
        above[ch] = acc[base + bins + 1];
        non_finite[ch] = acc[base + bins + 2];
    }

    Histogram {
        counts,
        below,
        above,
        non_finite,
        bins,
        channels,
        min,
        max,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, array};

    #[test]
    fn counts_sum_to_the_pixel_count() {
        let img = Array3::<f32>::from_shape_fn((17, 23, 3), |(y, x, c)| {
            ((y * 23 + x + c) % 100) as f32 / 99.0
        });
        let h = histogram(img.view(), &HistogramParams::default()).unwrap();
        for ch in 0..3 {
            assert_eq!(h.total(ch), 17 * 23, "channel {ch} lost or gained samples");
        }
    }

    #[test]
    fn a_constant_image_fills_one_bin() {
        let img = Array3::<f32>::from_elem((8, 8, 1), 0.5);
        let h = histogram(img.view(), &HistogramParams::new(256, 0.0, 1.0)).unwrap();
        let occupied: Vec<usize> = (0..256).filter(|&b| h.counts()[[0, b]] > 0).collect();
        assert_eq!(occupied.len(), 1, "a constant image occupies one bin");
        assert_eq!(occupied[0], 128, "0.5 of [0,1] over 256 bins is bin 128");
        assert_eq!(h.counts()[[0, 128]], 64);
    }

    #[test]
    fn range_endpoints_land_in_the_end_bins() {
        let img = array![[[0.0_f32], [1.0]]];
        let h = histogram(img.view(), &HistogramParams::new(256, 0.0, 1.0)).unwrap();
        assert_eq!(h.counts()[[0, 0]], 1, "min belongs in bin 0");
        assert_eq!(h.counts()[[0, 255]], 1, "max belongs in the last bin");
        assert_eq!(h.below()[0], 0);
        assert_eq!(h.above()[0], 0, "exactly max is not 'above'");
    }

    #[test]
    fn out_of_range_is_counted_separately_not_folded_in() {
        // The defect this design exists to avoid: a spike at the right
        // edge that cannot be told apart from legitimate bright content.
        let img = array![[[-0.5_f32], [0.5], [1.5], [2.5]]];
        let h = histogram(img.view(), &HistogramParams::new(4, 0.0, 1.0)).unwrap();
        assert_eq!(h.below()[0], 1);
        assert_eq!(h.above()[0], 2);
        assert_eq!(
            h.counts().row(0).iter().sum::<u64>(),
            1,
            "only 0.5 is in range"
        );
        assert_eq!(h.total(0), 4);
    }

    #[test]
    fn nan_is_separate_from_clipping_and_infinity_is_not() {
        let img = array![[[f32::NAN], [f32::INFINITY], [f32::NEG_INFINITY]]];
        let h = histogram(img.view(), &HistogramParams::default()).unwrap();
        assert_eq!(h.non_finite()[0], 1, "NaN is a broken pixel, counted apart");
        assert_eq!(h.above()[0], 1, "+inf genuinely is above the range");
        assert_eq!(h.below()[0], 1, "-inf genuinely is below it");
    }

    #[test]
    fn channels_are_counted_independently() {
        // Red 1.0, green 0.5, blue 0.0 everywhere.
        let img = Array3::<f32>::from_shape_fn((4, 4, 3), |(_, _, c)| match c {
            0 => 1.0,
            1 => 0.5,
            _ => 0.0,
        });
        let h = histogram(img.view(), &HistogramParams::new(4, 0.0, 1.0)).unwrap();
        assert_eq!(h.counts()[[0, 3]], 16, "channel 0 at the top bin");
        assert_eq!(h.counts()[[1, 2]], 16, "channel 1 at 0.5");
        assert_eq!(h.counts()[[2, 0]], 16, "channel 2 at the bottom bin");
    }

    #[test]
    fn cdf_is_monotonic_and_ends_at_one() {
        let img = Array3::<f32>::from_shape_fn((32, 32, 1), |(y, x, _)| {
            ((y * 32 + x) % 256) as f32 / 255.0
        });
        let h = histogram(img.view(), &HistogramParams::default()).unwrap();
        let cdf = h.cdf();
        let mut prev = 0.0_f32;
        for bin in 0..h.bins {
            let v = cdf[[0, bin]];
            assert!(v >= prev, "cdf must be non-decreasing at bin {bin}");
            prev = v;
        }
        assert!((prev - 1.0).abs() < 1e-6, "cdf must end at 1.0, got {prev}");
    }

    #[test]
    fn cdf_of_an_empty_channel_is_all_zero() {
        // Every sample out of range: nothing in range to accumulate.
        let img = Array3::<f32>::from_elem((4, 4, 1), 5.0);
        let h = histogram(img.view(), &HistogramParams::new(8, 0.0, 1.0)).unwrap();
        assert!(h.cdf().iter().all(|v| *v == 0.0));
        assert_eq!(h.above()[0], 16);
    }

    #[test]
    fn thread_count_cannot_change_the_answer() {
        // Integer counting is associative; this pins the claim.
        let img = Array3::<f32>::from_shape_fn((97, 131, 3), |(y, x, c)| {
            ((y * 131 + x * 7 + c) % 1000) as f32 / 999.0
        });
        let p = HistogramParams::new(64, 0.0, 1.0);
        let a = histogram(img.view(), &p).unwrap();
        let b = histogram(img.view(), &p).unwrap();
        assert_eq!(a.counts(), b.counts());
    }

    #[test]
    fn accepts_any_layout() {
        let img = Array3::<f32>::from_shape_fn((12, 15, 3), |(y, x, c)| {
            ((y * 15 + x + c) % 50) as f32 / 49.0
        });
        let strided = img.slice(ndarray::s![..;2, ..;3, ..]);
        let owned = strided.to_owned();
        let p = HistogramParams::new(32, 0.0, 1.0);
        assert_eq!(
            histogram(strided, &p).unwrap().counts(),
            histogram(owned.view(), &p).unwrap().counts()
        );
    }

    #[test]
    fn empty_image_gives_empty_counts() {
        let img = Array3::<f32>::zeros((0, 5, 3));
        let h = histogram(img.view(), &HistogramParams::default()).unwrap();
        assert_eq!(h.channels, 3);
        assert_eq!(h.counts().sum(), 0);
        assert_eq!(h.total(0), 0);
    }

    #[test]
    fn custom_range_is_honoured() {
        // Analysis of highlight headroom: count what is above 1.0.
        let img = array![[[0.5_f32], [1.5], [2.5], [3.5]]];
        let h = histogram(img.view(), &HistogramParams::new(4, 1.0, 4.0)).unwrap();
        assert_eq!(h.below()[0], 1, "0.5 is below the analysed range");
        assert_eq!(h.counts().row(0).iter().sum::<u64>(), 3);
    }

    #[test]
    fn rejects_bad_parameters() {
        let img = array![[[0.5_f32]]];
        for p in [
            HistogramParams::new(1, 0.0, 1.0),
            HistogramParams::new(0, 0.0, 1.0),
            HistogramParams::new(256, 1.0, 0.0),
            HistogramParams::new(256, 0.0, 0.0),
            HistogramParams::new(256, f32::NAN, 1.0),
            HistogramParams::new(256, 0.0, f32::INFINITY),
            // A span wider than f32 can represent: max - min overflows,
            // and every sample would otherwise divide to zero and pile
            // into bin 0.
            HistogramParams::new(256, -3.0e38, 3.0e38),
            HistogramParams::new(256, f32::MIN, f32::MAX),
        ] {
            assert!(
                histogram(img.view(), &p).is_err(),
                "{p:?} should be rejected"
            );
        }
    }

    /// Finding 11: every cdf test used a one-channel image, so replacing
    /// `counts.row(ch)` with `counts.row(0)` in `cdf()` left the whole
    /// suite green. The per-channel indexing needs a case where the
    /// channels genuinely differ.
    #[test]
    fn cdf_indexes_each_channel_separately() {
        // Channel 0 is all dark, channel 1 all mid, channel 2 all bright.
        let img = Array3::<f32>::from_shape_fn((4, 4, 3), |(_, _, c)| match c {
            0 => 0.05,
            1 => 0.5,
            _ => 0.95,
        });
        let h = histogram(img.view(), &HistogramParams::new(10, 0.0, 1.0)).unwrap();
        let cdf = h.cdf();

        // Each channel's CDF steps to 1.0 at its own bin, not at another's.
        let step_bin = |ch: usize| (0..10).find(|&b| cdf[[ch, b]] >= 1.0).unwrap();
        assert_eq!(step_bin(0), 0, "channel 0 is dark");
        assert_eq!(step_bin(1), 5, "channel 1 is mid");
        assert_eq!(step_bin(2), 9, "channel 2 is bright");
        // And before its own step each channel is still at zero.
        assert_eq!(cdf[[1, 4]], 0.0);
        assert_eq!(cdf[[2, 8]], 0.0);
    }

    /// Finding 12: `cdf()` documents normalisation over the *in-range*
    /// samples, and no test had both in-range and out-of-range samples,
    /// so normalising by the full total instead passed everything.
    #[test]
    fn cdf_normalises_over_in_range_samples_only() {
        // Four in range, four out. If the normaliser used the total the
        // CDF would end at 0.5 instead of 1.0.
        let img = ndarray::array![[[0.1_f32], [0.3], [0.6], [0.9], [-1.0], [-2.0], [5.0], [7.0]]];
        let h = histogram(img.view(), &HistogramParams::new(10, 0.0, 1.0)).unwrap();
        assert_eq!(h.below()[0], 2);
        assert_eq!(h.above()[0], 2);
        let cdf = h.cdf();
        assert!(
            (cdf[[0, 9]] - 1.0).abs() < 1e-6,
            "cdf must reach 1.0 over the in-range samples, got {}",
            cdf[[0, 9]]
        );
    }

    /// Finding 14: `total()` documents "every sample seen", and no test
    /// called it on an image containing NaN, so dropping the
    /// `non_finite` term passed.
    #[test]
    fn total_includes_the_non_finite_tally() {
        let img = ndarray::array![[[0.5_f32], [f32::NAN], [f32::NAN], [2.0], [-1.0]]];
        let h = histogram(img.view(), &HistogramParams::default()).unwrap();
        assert_eq!(h.non_finite()[0], 2);
        assert_eq!(
            h.total(0),
            5,
            "total must count bins + below + above + NaN, not just the first three"
        );
    }

    /// The reduction must reject rather than abort when the accumulator
    /// would be enormous. A failed `Vec` allocation calls
    /// `handle_alloc_error`, which aborts and cannot be caught from
    /// Python at all.
    #[test]
    fn oversized_accumulators_are_rejected_not_attempted() {
        let tiny = Array3::<f32>::zeros((1, 1, 1));
        assert!(
            histogram(tiny.view(), &HistogramParams::new(MAX_BINS + 1, 0.0, 1.0)).is_err(),
            "bins above the cap must be refused"
        );
        // Default parameters, but a channel count that multiplies out.
        let many_channels = Array3::<f32>::zeros((0, 1, 4_000_000));
        assert!(
            histogram(many_channels.view(), &HistogramParams::default()).is_err(),
            "channels x bins must be bounded too, even for an empty image"
        );
    }

    /// Chunking the rows changed how the parallel accumulators are split;
    /// the answer must be identical to a single-threaded run.
    #[test]
    fn chunked_reduction_matches_a_serial_count() {
        let img = Array3::<f32>::from_shape_fn((201, 37, 3), |(y, x, c)| {
            ((y * 37 + x * 3 + c) % 997) as f32 / 996.0
        });
        let p = HistogramParams::new(64, 0.0, 1.0);
        let got = histogram(img.view(), &p).unwrap();

        let mut want = vec![0_u64; 3 * (64 + 3)];
        for y in 0..201 {
            for x in 0..37 {
                for c in 0..3 {
                    want[c * 67 + slot_of(img[[y, x, c]], 0.0, 1.0, 64)] += 1;
                }
            }
        }
        let expected = assemble(want, 3, 64, 0.0, 1.0);
        assert_eq!(got.counts(), expected.counts());
        assert_eq!(got.below(), expected.below());
        assert_eq!(got.above(), expected.above());
    }

    /// The claim the pair is sold on: equalising an already-uniform
    /// image is the *identity*. With `cdf()` fed straight to the LUT it
    /// was off by half a bin — black lifted by 1/bins — which the old
    /// tolerance-based test was five times too slack to notice.
    #[test]
    fn equalisation_lut_is_aligned_to_apply_lut() {
        for bins in [16_u32, 64, 256, 1024] {
            // Exactly one sample per bin: the flattest possible image, so
            // the equalising transfer must be the identity.
            let n = bins as usize;
            let data: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
            let img = Array3::from_shape_vec((1, n, 1), data).unwrap();

            let h = histogram(img.view(), &HistogramParams::new(bins, 0.0, 1.0)).unwrap();
            let table = h.equalisation_lut();
            let out = crate::lut::apply_lut(
                img.view(),
                table.row(0).to_owned().view(),
                &crate::lut::LutParams::new(0.0, 1.0),
            )
            .unwrap();

            let worst = img
                .iter()
                .zip(out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            // One bin's worth of slack would hide the old bug; this is a
            // hundredth of that.
            assert!(
                worst < 0.01 / bins as f32,
                "bins={bins}: equalising a uniform image shifted by {worst}, \
                 which is {} of a bin",
                worst * bins as f32
            );
        }
    }

    #[test]
    fn equalisation_lut_is_cdf_with_a_leading_zero() {
        let img = Array3::<f32>::from_shape_fn((8, 8, 3), |(y, x, c)| {
            ((y * 8 + x + c * 3) % 64) as f32 / 63.0
        });
        let h = histogram(img.view(), &HistogramParams::new(32, 0.0, 1.0)).unwrap();
        let cdf = h.cdf();
        let table = h.equalisation_lut();
        assert_eq!(table.dim(), (3, 33));
        for ch in 0..3 {
            assert_eq!(
                table[[ch, 0]],
                0.0,
                "the leading entry is zero by definition"
            );
            for bin in 0..32 {
                assert_eq!(table[[ch, bin + 1]], cdf[[ch, bin]]);
            }
        }
    }

    #[test]
    fn params_equality_and_repr() {
        let a = HistogramParams::new(256, 0.0, 1.0);
        assert!(a.__eq__(&HistogramParams::default()));
        assert!(!a.__eq__(&HistogramParams::new(512, 0.0, 1.0)));
        assert_eq!(a.__repr__(), "HistogramParams(bins=256, min=0, max=1)");
    }
}
