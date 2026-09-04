# Changelog

All notable changes to `phaios-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The Rust crate and the Python wheel always carry the same version.

## [Unreleased] — 0.2.0-dev

### Fixed — two CUDA kernels on high-dynamic-range input

Both were found by widening the test inputs, not by reading the code. Every
conformance test and every sweep had drawn from `[0, 1)`, so no
accumulator's dynamic range was ever stressed. On a dark field with a
specular highlight — ordinary linear scene-referred data, a lamp or a
sun-glint in an otherwise dim frame — two kernels missed the agreement
bound `docs/ffi.md` §6 commits to.

- **The CUDA box blur (σ ≥ 6) lost dark detail beside a highlight.** It
  missed its bound by up to 2.8e7×, and was 164% wrong in relative terms
  next to a 1e4 highlight. A sliding window subtracts as it advances, so
  the small samples are the difference of two large ones; Kahan
  compensation does not help, because its error bound is `2ε·Σ|xᵢ|`, which
  the large values dominate. The device accumulator is now f64, matching
  the host. The kernel is bandwidth-bound, so this costs nothing
  measurable.
- **`local_contrast` read noise as an edge beside a highlight.** It missed
  its bound by 51.8×. The local variance is `mean(L²) − mean(L)²`, a
  cancelling subtraction of two large numbers, and it was computed in f32
  from Kahan-compensated f32 sums — Kahan compensates a sum, never a
  cancelling difference. On a uniformly bright region the result was
  noise, `a = var/(var+ε)` read that noise as an edge, and the filter
  smoothed where it should have sharpened. Only the L and L² path moved to
  f64; the coefficient sums stay f32, which is measured rather than
  assumed. This one is not free — the guided filter is compute-bound, so
  at 24 MP and r = 8 it costs 4.4 → 12.0 ms on the device, still 14× the
  CPU.

Both now have a dynamic-range conformance test sweeping highlights from
1e0 to 1e8, as does `glow`, the third kernel that accumulates over many
samples.

### Fixed — the cross-backend oracle scored NaN as perfect agreement

`worst_violation`, which every bounded conformance assertion goes through,
reduced with `.fold(0.0, f32::max)`. Rust's `f32::max` returns the *other*
operand when one is NaN, so every NaN violation was silently dropped: a
GPU kernel returning nothing but NaN scored 0.000, and so did an empty or
truncated output, because `zip` stops at the shorter side. All tests still
passed once it was fixed, so nothing had been leaning on the leniency —
but a lenient oracle is invisible from reading any test that uses it.

### Added — verifying the CUDA backend without the maintainer

The hosted CI runner has neither `nvcc` nor a device, so `ci.yml` skips
every target gated behind `required-features = ["cuda"]` and never builds
the backend at all. That is not going to change, so the answer is to let
anyone with a device verify it themselves and say so.

- `scripts/gpu-verify.sh` — the mirror image of the CI job: it runs
  exactly the targets CI skips, under `PHAIOS_REQUIRE_GPU=1` so a green
  result cannot mean the suite quietly skipped 54 tests for want of a
  device. It ends by printing a device/driver/version block meant to be
  pasted into an issue. `.github/workflows/gpu.yml` invokes the same
  script on a self-hosted runner, so the hand-run command and the CI job
  cannot drift apart.
- `examples/23_gpu_selftest.rs` — every device entry point against the CPU
  function that is its specification, 112 cases, tolerances taken from
  `docs/ffi.md` §6 rather than invented, exiting non-zero on any failure.
- **A GPU twin for every CPU example**, 19 of them, plus
  `22_gpu_pipeline.rs` for the device-resident chain the backend exists
  for. Each mirrors its counterpart's input and parameters and writes the
  same PPM stems, so the two directories diff file by file.
- `benches/gpu.rs` — the backend shipped with no benchmark at all, so
  `cargo bench` never opened the device. Ids mirror the CPU ones under a
  `gpu/` prefix, on the same image with the same channel count, so the two
  can simply be divided. `docs/architecture.md` §18's GPU table is now
  sourced from it.


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

### Added — light scattering (halation, diffusion, veiling glare)

`glow(img, GlowParams)` — `out = in + amount · blur(max(in − threshold,
0), σ)`. One kernel, because the three named effects are one operation:
what separates them is the parameters and, more to the point, **where in
the pipeline the call sits**.

| Effect | Happens in | threshold | σ | Position |
|---|---|---|---|---|
| Veiling glare | the lens | 0 | frame-spanning | earliest |
| Halation | the emulsion | high | moderate | after `exposure`, before the tone stages |
| Diffusion | the print | mid | large | after the tone stages |

**Veiling glare is the one that needs a kernel rather than a curve
preset**, because it is not a per-pixel transfer. With no threshold and a
frame-spanning σ, the black point lifts by an amount set by the
brightness of the *whole image*: measured in `examples/21_glow`, an
identical black corner pixel ends at 0.019 in a dim frame and 0.380 in a
bright one. No tone curve can do that — a curve sees one pixel at a time
— and it is a large part of why uncoated lenses look the way they do.

Additive rather than conserving, so the result may exceed 1.0. That is
deliberate: headroom is carried to `highlight_rolloff` and spent once,
explicitly. A conserving form would darken the source to pay for the
halo, which is right for a diffuser and wrong for halation, where the
highlight is already saturated and the halo is genuinely extra density.

The weight is `max(in − threshold, 0)` rather than a hard cut, so a
bright region does not acquire an outline where it crosses the threshold.
`amount = 0` is the exact identity; `amount` must be non-negative, so
this cannot be used as an unsharp mask — detail enhancement remains
`local_contrast`, which is edge-aware and does not halo.

Composed rather than fused on both backends: the weight and the add are
element-wise and exact, and the spread between them is the crate's one
Gaussian, so this cannot drift from `blur`. Bounded at (rtol 1e-5, atol
1e-7), inherited entirely from that blur.

24 MP: 106 ms for halation and diffusion, 117 ms for glare on the CPU;
8.3 and 17.9 ms on the GPU.

### Added — Gaussian blur

`blur(img, BlurParams)` — a separable Gaussian, and the primitive the
scattering effects are built on: halation, diffusion and veiling glare
are all *blur, weighted, added back*, so none of them can be written
until this exists.

Two paths, chosen by measurement rather than taste. Below σ = 6 a direct
separable convolution against sampled weights, which is exact. At or
above it, three box passes whose variances sum to σ², which costs the
same at any radius — 90 ms on a 24 MP frame whether σ is 6, 16 or 64,
where a truncated kernel at σ = 32 would need 386 taps per axis.

The crossover sits where the two *cost* the same, because on accuracy the
direct path always wins. That is not where it first went: placed at σ = 4
(where the box path's σ error first drops under 1%), σ ∈ [4, 6) was being
handled by the slower *and* less accurate path. Benchmarking surfaced it,
and moving the crossover made that band numerically exact.

Two traps the width search had to be taught, both found by measurement
and both now documented so they are not retried:

- **More box passes do not help small σ** — they raise the floor, 1.414
  for three passes, 1.633 for four, 1.826 for five.
- **Matching σ alone is the wrong objective.** An unconstrained search
  for σ = 2 returns `[1, 1, 7]`: the right variance from a single box and
  two passes that do nothing. Requiring width ≥ 3 is still not enough —
  σ = 5 then picks `[3, 3, 17]`, best σ match available and still nearly
  a box, shape error 0.0223 against 0.0131 for a balanced triple at the
  same σ error. The search now also caps the width ratio at 3:1.

Borders clamp: a constant image is preserved exactly including its edges,
and an impulse near an edge loses the tail that falls outside — a σ = 5
kernel in a 16×16 frame keeps 0.79 of its energy. Both halves are
asserted, so the second cannot later be mistaken for a defect.

On the GPU the box kernel needed segmenting to be worth having. Written
the obvious way — one thread per row or column — it measured **40 ms**,
four times the direct path and worse than `local_contrast`, because a
24 MP frame has only ~5000 lanes and a thread each leaves the device
about 95% idle. Cutting each lane into segments takes it to **7.5 ms**.
GPU figures: blur 6.0–10.7 ms direct, 7.5–8.8 ms box.

**Bounded, not bit-exact** across backends at (rtol 1e-5, atol 1e-7): the
device accumulates each pass in Kahan-compensated f32 where the host uses
f64, since a consumer card runs f64 at 1/64 rate. Measured worst case is
0.04x of `local_contrast`'s looser bound, which is why a tighter one is
committed — a bound never approached cannot catch a regression. σ = 0 is
the exact identity on both.

### Fixed — an uncatchable abort on zero-stride input

Every kernel sized its output from the *logical* shape of the caller's
array. A numpy `broadcast_to` view has an unbounded logical shape backed
by as little as four bytes, so `ph.exposure(np.broadcast_to(x,
(100000, 100000, 3)), 1.0)` asked for 120 GB — and a failed `Vec`
allocation in Rust calls `handle_alloc_error`, which **aborts**: it
raises nothing at all, not even `PanicException`, and kills the
interpreter. Nothing in Python could catch it, which makes it strictly
worse than the panic CLAUDE.md §2 forbids and for the same reason.

Found by the review of `shadow_rolloff`, but crate-wide and long
pre-existing rather than new to that kernel.

All 28 caller-derived allocations in `src/` now go through
`alloc::zeros3` / `zeros2`, which check a **8 GiB** single-allocation
budget first and return the new `PhaiosError::Allocation` → Python
`MemoryError`. The CUDA `Context::upload` path is bounded too: its
`as_standard_layout` staging copy reached the same abort. Two internal
helpers (`integral::sat`, `local_contrast::guided_filter`) became
fallible so the error propagates rather than being swallowed.

The limit is a constant on purpose. It cannot be delegated to the
allocator — measured on Linux with default heuristic overcommit,
`Vec::try_reserve_exact` *succeeded* for a 111 GiB request on a 60 GiB
machine and the process died later under the OOM killer — and it must
not be derived from free memory, which would make the same call succeed
or fail on ambient machine state, exactly what §2's purity rule forbids.
For scale, 8 GiB is about 28x a 24 MP RGB frame.

`tests/ffi.py`'s layout matrix gained a **zero-stride** variant. Its
absence is why this went unnoticed: every other variant (strided,
Fortran, reversed) has the storage its shape implies.

**Known and deferred to v0.3:** the budget bounds single allocations
only. A kernel holding several full-resolution intermediates, or a
caller looping over many frames, can still exhaust memory with every
individual call inside the limit. Bounding a pipeline's peak needs an
arena or a budget threaded through the API — a design change, not a
constant. Documented in `docs/ffi.md`.

### Added — shadow roll-off, completing the characteristic curve

`shadow_rolloff(img, ShadowRolloffParams)` — the **toe**, and the
counterpart to `highlight_rolloff`'s shoulder. An emulsion does not hold
full contrast down to zero exposure: below some threshold the density
gradient falls away, so the deepest shadows lose separation and run
together into black instead of being cut off at a hard floor.

Two parameters: `knee`, above which nothing changes, and `strength`,
which is one minus the slope at black. On `[0, knee]` the transfer is
the cubic Hermite fixed by passing through the origin at slope
`1 − strength` and meeting the identity at the knee in both value and
slope, so the join shows no crease. Monotone for every strength, and
`strength = 0` is the *exact* identity — short-circuited rather than
trusting `knee · (x / knee)` to round back, which it does not always do.

The measurable effect is compression: two samples 0.02 apart just above
black stay 0.02 apart at strength 0, 0.0119 at 0.5, and 0.0038 at 1.0.
It darkens as it compresses, which is the right direction — lifting them
instead would be a black-lift control, a different thing, and a
separation-only test would not have told the two apart.

**Bit-exact across CPU and CUDA**: a cubic in Horner form, built from
multiply, add, subtract and one divide, with the same grouping on both
sides because a different association would round differently.

The point is the composition it completes. With the straight section —
whose slope is what `tone_curve` sets — the three make the classic
characteristic shape:

    shadow_rolloff  →  tone_curve  →  highlight_rolloff
         toe            straight          shoulder

That composition's local slope, measured by feeding scalar probe pairs
through the three kernels (the table `examples/19_characteristic_curve`
prints), runs 0.45 / 1.09 / 1.20 / 0.03 at inputs 0.01 / 0.05 / 0.40 /
2.00 — low, rising, peak, low. A linear rendering gives
1.00 / 1.00 / 1.00 / 0.00: flat, then clipped abruptly instead of
gradually. The example also renders the checker at +2 EV, but the slope
figures come from the probes, not from the image.

It is not a densitometric H&D model and does not claim to be: no
base-plus-fog, no D-max, no named emulsion, no gamma in the sense a
densitometer measures. For a genuine measured curve, tabulate it and use
`apply_lut`. ~11.0 ms on a 24 MP frame.

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

### Added — type stubs

- `python/phaios_core/__init__.pyi` and `gpu.pyi`: full type stubs for
  every function, class and enum, plus `py.typed` ([PEP
  561](https://peps.python.org/pep-0561/)). mypy and pyright pick them
  up with no configuration. Docstrings are mirrored verbatim from the
  runtime module, not just signatures.
- `tests/stub_contract.py`, a stdlib-only contract test wired into
  `tests/ffi.py` and `tests/ffi_gpu.py`, failing on any drift between a
  stub and the runtime module: names, signatures, defaults,
  docstrings, class shape, and that the installed package actually
  ships the stub files.
- `mypy --strict python/phaios_core` and `python -m mypy.stubtest
  phaios_core --allowlist tests/stubtest-allowlist.txt` in CI and
  `scripts/gpu-verify.sh`, catching what the contract test does not:
  type errors in the stubs themselves, and PyO3-specific signature
  facts like the exact `__eq__`/`__ne__` parameter shape.

### Changed

- The package moved to maturin's mixed layout (`python-source =
  "python"` in `pyproject.toml`) so `phaios_core.gpu` can carry its own
  stub — maturin's pure-Rust layout supports only one stub file, at the
  package root. `python/phaios_core/__init__.py` is now committed to
  the repo, byte-identical to what maturin generated automatically
  before; no behaviour change.
- `mypy>=1.18` added to `requirements-dev.txt` (dev-only; the crate
  itself gains no new dependency).
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
