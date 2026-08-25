# Changelog

All notable changes to `phaios-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The Rust crate and the Python wheel always carry the same version.

## [Unreleased] — 0.2.0-dev

### Added — histogram and lookup tables

Two primitives that are dull alone and cover a family together.

`histogram(img, HistogramParams)` — the crate's **first reduction**. It
returns a `Histogram` (per-channel `(channels, bins)` counts, plus
separate `below` / `above` / `non_finite` tallies and a `cdf()`), not an
image, which deliberately widens §1's "pure functions on `(H, W, C)`
arrays".

The widening is justified by what the function actually settles.
Counting into bins is trivial; agreeing on *what to count* is not, and
each of these is a decision a front-end would otherwise make alone:
histograms are read display-referred, so the intended call site is after
`encode_srgb`; the range is fixed rather than auto-scaled, so two
adjustments stay comparable; out-of-range samples are tallied *apart
from* the end bins, which is the difference between a display that can
show clipping and one where a bright picture and a clipped one look
alike; and NaN is counted apart from both, because a NaN is a broken
pixel rather than a bright or dark one.

It is also the crate's only reduction needing no determinism care at
all: bin counts are integers and integer addition is associative, so
neither thread count nor the completion order of the device's atomics
can change a total.

`apply_lut(img, lut, LutParams)` — arbitrary tone transfer through a
caller-supplied 1-D table, linearly interpolated and clamped (not
extrapolated) outside its domain. One kernel instead of a kernel per
curve shape: sample a UI spline and it is a curves tool, pass a CDF row
and it is histogram equalisation, tabulate an H&D curve and it is film
emulation. It deliberately does **not** require monotonicity, which
makes it the only tone kernel able to express solarisation — `tone_curve`
and `zone_system` are both monotone by construction.

Equalisation is the proof the factoring is right: `histogram` →
`equalisation_lut()` → `apply_lut` is the whole implementation, with no
third component, and the test suite asserts it end to end.

`equalisation_lut()` is `cdf()` with a leading zero, and the extra entry
is load-bearing. `cdf()[b]` is the cumulative fraction at the *upper
edge* of bin `b`, so the `bins` values describe inputs `1/bins … 1`,
while `apply_lut` reads an `n`-entry table as describing `0 … 1` evenly.
Feeding the CDF straight in biases the transfer by a full bin — black
lifts by `1/bins`, white stays — so equalising an already-uniform image
was not quite the identity. With the leading zero it is the identity
exactly, measured at 16, 64, 256 and 1024 bins.

Both **bit-exact across CPU and CUDA**. The device histogram privatises
per block in shared memory and falls back to global atomics above what
fits (768 KB would be needed for a 65536-bin three-channel pass); both
paths produce identical counts.

24 MP: histogram 3.7 ms (256 bins, 1 channel), 9.4 ms (3 channels),
49.4 ms (65536 bins); `apply_lut` 10.9 ms.

### Added — dithered quantisation, and the export contract

`quantize_u8(img, QuantizeParams)` and `quantize_u16(...)` — the
terminal stage, and the first kernels in the crate that return integers
(`uint8` / `uint16`) rather than `float32`. Two monomorphic functions
rather than one with a depth argument, so the returned dtype is a static
property of the call.

Optional triangular-PDF dither of ±1 LSB, which converts banding into
fine noise: on a 256-pixel ramp spanning two 8-bit codes, undithered
gives one hard transition and dithered gives 105, with the mean shifting
by 0.008 of a code. Triangular specifically, not uniform — a uniform
deviate decorrelates the error's mean but leaves its variance modulated
by the signal (Lipshitz, Wannamaker and Vanderkooy, *JAES* 40(5), 1992).
The unit tests assert the variance is 1/6 rather than 1/3 so the two
cannot be silently interchanged.

Dither defaults to off, and the deviate comes from `splitmix64` keyed on
position, channel and an explicit `seed` — the same hash `film_grain`
uses, and for the same reason: no global RNG, and the same bytes at any
thread count. `Dither.Off` is named `Off` rather than `None` because
`Dither.None` is a syntax error in Python.

**Bit-exact across CPU and CUDA:** the dither is exact 64-bit integer
arithmetic and the rounding is `floor(v + 0.5)` — an exact operation
composed with a correctly-rounded one, rather than a library rounding
routine host and device might implement differently.

~5.9 ms undithered and ~10.2 ms dithered on a 24 MP frame (u8);
~11.3 ms for u16.

New: **`docs/export.md`**, the export contract. It states normatively
what a finished phaios image is, so the layer that writes files has no
image decisions left to make: untoned output is single-channel and MUST
be written with greyscale photometric and a greyscale ICC profile whose
tone curve matches `encode_srgb` (never three duplicated RGB channels);
toned output is RGB/sRGB; 16-bit is the archival default; 8-bit MUST be
dithered unless grain already serves that purpose; float containers
should carry linear scene-referred data and skip both encoding and
quantisation. The document is written to outlast the parked question of
*where* file writing lives, since it constrains the output either way.

### Added — highlight roll-off

`highlight_rolloff(img, RolloffParams)` — the explicit choice between
clipping the highlights and rolling them off, a decision every consumer
was previously making by accident in whichever line called `np.clip`.

Two photographic parameters: `knee`, below which nothing changes, and
`white_point`, the scene value that becomes pure white. Between them the
transfer is a quadratic Bézier whose control point is fixed by the two
continuity requirements rather than tuned — it leaves the identity at
slope 1 and lands on white at slope 0, so neither the knee nor the
clipping point produces a visible edge.

The default `RolloffParams()` is `(1.0, 1.0)`, which collapses to
`min(x, 1.0)` bit-for-bit: adding the stage to an existing pipeline
changes nothing until a shoulder is asked for. On the synthetic checker
pushed +2 EV, eight distinguishable highlights survive as one value
under the default and as eight under `white_point = 8.0`.

**Bit-exact across CPU and CUDA** with no tolerance at all — the curve
uses only add, subtract, multiply, divide and square root, every one of
which IEEE-754-2008 §5.4.1 requires to be correctly rounded, so unlike
the other tone stages there is no libm to disagree with. Asserted with
`assert_eq!` over seven configurations, including the degenerate
`white = 2 − knee` case where the quadratic coefficient is exactly zero
and the solve becomes linear.

Element-wise on any channel count and any layout; ~10.9 ms on a 24 MP
frame, bandwidth-bound like the other element-wise kernels, and the
default clip costs the same as the shoulder.

### Added — resampling geometry (resize, straighten)

`resize(img, ResizeParams)` — separable resampling with three
polynomial filters: `Area` (exact fractional coverage, the correct
downscale filter at any ratio), `Bilinear`, and `CatmullRom` (Keys
1981, the photographic upscale default). Centre-aligned, so a same-size
resize is the exact identity; replicate borders; a constant image
survives to ~1 ULP.

`straighten(img, StraightenParams)` — rotation up to ±45° (positive
clockwise) with 16-tap Catmull-Rom sampling, cropped to the largest
inscribed axis-aligned rectangle (max-area construction). `degrees = 0`
is the exact identity; the outermost ~2-pixel band may include
replicate-clamped frame-edge samples (the cubic's support), documented.

Both are **bit-exact across CPU and CUDA** — the filters are
polynomial and the straighten sin/cos is computed once on the host —
asserted with `assert_eq!` by the conformance suite. Geometry order:
`orient` → `straighten` → `crop` → `resize`.

### Added — exact geometry (crop, orientation)

`crop(img, CropParams)` and `orient(img, Orientation)` — the first
geometry kernels, deliberately in the core because their parameters
live in consumers' sidecar files: two front ends implementing the same
crop differently would make the same sidecar render different images.

Both are pure index permutations — no arithmetic on pixel values — so
they are **bit-exact across every backend unconditionally**, asserted
by the conformance suite for all crop rectangles and all eight
orientations, including a geometry-first resident GPU chain.
`Orientation`'s discriminants are the Exif codes 1..=8 (JEITA CP-3451);
rotations are clockwise; every variant decomposes through one shared
`flags()` definition both backends implement.

**Pipeline position: geometry runs first** — `orient`, then `crop`,
before exposure and the look pipeline. The order is load-bearing, not
stylistic: the vignette centres on the frame it is given (which must be
the cropped frame — asserted by test), and film grain keys its noise to
pixel coordinates (which must be the final grid). The kernel-viewer's
RAW path now applies camera orientation through the core kernel instead
of private code.

### Added — optional CUDA backend (`--features cuda`)

All twelve kernels on NVIDIA GPUs, off by default: published wheels
stay CPU-only and the ordinary dependency graph is unchanged (cudarc
adds three lines to `cargo tree`). Building with the feature needs
`nvcc`; running needs only the NVIDIA driver, dlopened — a CUDA build
imports cleanly on machines with no driver and `gpu.available()` is
simply `False`. One `compute_80` PTX covers sm_80 → sm_120+; older
cards get a clear `RuntimeError`.

Measured on an RTX 5070 Ti, full nine-stage pipeline, one upload and
one download: **24 MP export 468 → 48 ms (9.8×); 2 MP preview
40.5 → 2.9 ms (14×)**. `local_contrast` alone: 182 → 15.5 ms.

- Python: `phaios_core.gpu` — `available()`, `devices()`,
  `GpuContext`, `GpuImage` (upload once, chain kernels resident,
  download once). Kernel signatures mirror the CPU functions
  argument-for-argument after the image; the existing param classes
  are reused verbatim, so sidecars and presets stay backend-neutral.
- Rust: `cuda::Context`, `DeviceImage`, and `<kernel>_device` +
  per-call forms under `cuda::kernels`.
- Determinism is now scoped to a backend (`docs/ffi.md` §6 rewritten):
  bit-identical within a backend unconditionally; across backends,
  transcendental-free kernels are bit-exact (PTX built with
  `-fmad=false`) and the rest hold committed bounds asserted by
  `tests/cuda_conformance.rs` — 33 tests, skipped cleanly without a
  device, `PHAIOS_REQUIRE_GPU=1` to forbid skipping.
  `film_grain`'s splitmix64 hash is asserted bit-identical over 2²⁰
  coordinates. The cross-backend framing corrects a fiction in the old
  contract: the CPU kernels call the platform libm, and the three
  published wheels link three different ones, so "bit-identical on
  another machine" was already untrue across OSes in v0.1.
- New `PhaiosError::Backend` → Python `RuntimeError` (a missing GPU is
  an environment condition, not a bad argument). Source-breaking for
  Rust consumers matching `PhaiosError` exhaustively.
- New dependency: `cudarc` 0.19 (optional, MIT OR Apache-2.0),
  justified in `Cargo.toml`; the driver-API exit strategy (14 entry
  points, replaceable via dlopen + cuGetProcAddress) is documented in
  `src/cuda/context.rs`.

An audit of the v0.1.1 kernels opened this cycle. All six kernels were
verified against independent NumPy reference implementations and agree
to f32 precision; the entries below are what the audit turned up around
them. No v0.1 function signature changed.

### Fixed

- **`zone_system` and `encode_srgb` no longer panic on non-contiguous
  input.** Both called `as_slice().expect(...)`, so any strided view —
  `img[::2, ::2]`, a downsampled preview — raised
  `pyo3_runtime.PanicException`. That class inherits from
  `BaseException`, not `Exception`, so it passes through a consumer's
  `except Exception:` and can kill a GUI worker thread. Every kernel now
  accepts any layout (C, Fortran, strided, negative strides) and returns
  a freshly allocated C-contiguous array.
- **`zone_system` is now bit-reproducible.** Zone offsets were summed in
  `HashMap` iteration order, which varies with the per-instance hash
  seed; since f32 addition is not associative, identical input produced
  different output on every process. Measured: eight processes, eight
  different digests, 12.8% of pixels off by 1 ULP. Offsets are now
  summed in ascending zone order.
- **Examples 01–05 write display-referred output.** They wrote linear
  values straight into 8-bit PPMs, so midtones came out ~2.5× too dark
  (18% grey rendered as code 48 instead of 120). They now apply
  `encode_srgb` before writing. Example 06 still writes both ways, which
  is its purpose.
- Guided-filter variance is clamped at zero. `mean(L²) − mean(L)²` is a
  cancelling subtraction whose error grows with image area and pixel
  magnitude; a negative result made `a = var/(var+ε)` invert or
  over-drive the local linear model.

### Added — kernels

Six new kernels, completing the v0.2 pipeline. Each accepts any array
layout and returns a freshly allocated C-contiguous array.

- **`exposure(img, stops)`** — `out = in · 2^stops`, the first pipeline
  stage. Any channel count. Nothing is clamped, so highlights driven
  above 1.0 remain recoverable; `2^n` being exact for integer `n` makes
  +n EV followed by −n EV bit-exact.
- **`hsl_bw(img, HslWeightedParams)`** — per-hue B&W weighting across
  eight bands (red 0°, orange 30°, yellow 60°, green 120°, aqua 180°,
  blue 240°, purple 270°, magenta 300°), Gaussian-blended in hue space
  with σ = 30° by default. Modulated by the chroma ratio
  `(max − min) / max` rather than HSL saturation — the ratio is
  invariant under exposure changes and stays well-defined above
  L = 1, where HSL saturation is degenerate.
- **`tone_curve(img, ToneCurveParams)`** — parametric slope/offset/power
  curve, the ASC Color Decision List primary correction (ASC Technology
  Committee v1.2, 2009). Monotonic for any positive slope and power.
  Any channel count.
- **`film_grain(img, GrainParams)`** — band-passed procedural grain with
  a `4·L·(1−L)` midtone envelope. The noise is a hash of `(seed, x, y)`
  rather than a sequential generator, so output is bit-identical
  regardless of thread count — verified against rayon pools of 1, 2, 3
  and 8 threads — and no RNG dependency is needed. Grain size comes from
  a difference of box filters, normalised analytically so `intensity`
  means the same thing at every size.
- **`split_toning(img, SplitToningParams)`** — shadow and highlight
  tinting in OKLab (Ottosson 2020), crossfaded by a smoothstep around a
  pivot. **Takes `(H, W, 1)` and returns `(H, W, 3)`** — the only kernel
  that changes the channel count. Lightness is carried through
  untouched, so the tonal rendering survives the tint.
- **`vignette(img, VignetteParams)`** — radial darkening or lightening,
  blending circular and rectangular falloff via `roundness`. Distances
  are normalised to the frame, so a preview and the full-size render
  agree; all channels of a pixel share one factor, so it darkens without
  tinting. Any channel count.

New param classes: `HslWeightedParams`, `ToneCurveParams`,
`GrainParams`, `SplitToningParams`, `VignetteParams` — each with field
getters, `__eq__` and `__repr__`, and each validated by its kernel.

Six new examples (07–12) and a criterion benchmark per kernel.

### Added — other

- `CHANGELOG.md` (this file).
- `ZoneParams.offsets` getter, returning the zone-index → stop-offset
  map as a dict. Needed to serialise settings into sidecar and preset
  files.
- `__eq__` on `ZoneParams` and `GuidedFilterParams`, so consumers can
  compare parameter objects to decide whether a cached render is stale.
- `PhaiosError::Parameter`, mapped to `ValueError`. Raised for a zone
  index outside 0..=10 (previously a silent no-op), a negative or
  non-finite `eps`, and non-finite offsets or `strength`.
- A `verify` gate on `release.yml`: the tag must match the version in
  `Cargo.toml` and `pyproject.toml`, and fmt, clippy and the tests must
  pass on the tagged commit, before anything is built or published.

### Changed

- **`local_contrast` is 2.5× faster and uses 3.5× less memory.** The
  summed-area tables are now built in two parallel passes (rows, then
  512-column blocks) instead of one single-threaded cell-by-cell pass,
  and intermediates are released as soon as they are consumed. At 24 MP,
  r = 8: 452 ms → 178 ms, peak scratch 1332 MB → 381 MB. Output is
  unchanged.
- Benchmarks run on deterministic pseudo-random images rather than
  constant ones, which were the cheapest possible input for three of the
  six kernels. Numbers before and after this change are not comparable;
  see `docs/architecture.md` §7.
- CI lints with `--all-targets` (examples, benches and tests, not just
  the library) and discovers examples from cargo metadata rather than a
  hardcoded list.
- `docs/architecture.md` and `docs/ffi.md` corrected where they
  described behaviour the code did not have: zone offsets are not
  clamped, `encode_srgb` does not clamp to [0, 1], zero-size arrays are
  accepted, `encode_srgb` accepts any channel count, and the sRGB
  transfer is C⁰ but not C¹ at the threshold (the derivatives differ by
  1.68%).

### Removed

- The `log` dependency, which was declared but never used.

### Notes for consumers

- `ZoneParams` and `GuidedFilterParams` are no longer hashable: PyO3
  removes `__hash__` when `__eq__` is defined. This is the correct
  semantic for value-compared objects, but they cannot be dict keys.
- Code that passed an out-of-range zone index, a negative `eps` or a
  non-finite parameter now gets a `ValueError` where it previously got
  silently arbitrary behaviour.

---

## [0.1.1] — 2026-05-16

### Fixed

- `release.yml`: removed a duplicate `with:` block that made the workflow
  invalid YAML; added `skip-existing: true` so re-runs after a partial
  failure are idempotent.

### Changed

- `Cargo.lock` is committed, so `cargo publish` no longer sees a dirty tree.
- CI installs Python via `astral-sh/setup-uv`.
- Documented that `CARGO_REGISTRY_TOKEN` needs both `publish-new` and
  `publish-update` scopes.
- Minimum Python raised to 3.12 (matches the `abi3-py312` wheel).

---

## [0.1.0] — 2026-05-09

First release. Six numerical kernels, all pure functions on `f32`
C-contiguous `(H, W, C)` arrays, exposed to Python as `phaios_core`.

### Added

- `luminance_bw` — standard luminance B&W conversion (BT.601, BT.709,
  BT.2020). ITU-R BT.709-6 (2015) Table 1.
- `channel_mixer_bw` — arbitrary RGB weights, negative values permitted.
- `color_filter_bw` — Wratten-style coloured-filter simulation
  (Yellow #8 K2, Orange #21, Red #25 A, Green #11 X1, Blue #47 C5).
- `zone_system` — Adams/Archer eleven-zone tone curve with Gaussian
  blending in zone-position space (Davis, *Beyond the Zone System*, 1999).
- `local_contrast` — He–Sun–Tang guided filter (ECCV 2010), O(1)
  integral-image formulation, self-guided.
- `encode_srgb` — IEC 61966-2-1 sRGB transfer encoding (terminal stage).
- Six runnable examples writing 8-bit PPMs from a synthetic Macbeth chart.
- Criterion benchmarks on a 24 MP synthetic image.
- Python-side FFI smoke tests (`tests/ffi.py`).

[Unreleased]: https://github.com/sdhdez/phaios-core/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sdhdez/phaios-core/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sdhdez/phaios-core/releases/tag/v0.1.0
