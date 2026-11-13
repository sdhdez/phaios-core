# Changelog

All notable changes to `phaios-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The Rust crate and the Python wheel always carry the same version.

## [Unreleased] — 0.2.0-dev

Twenty-one new kernels, an optional CUDA backend, type stubs and a
rewritten documentation set. The crate now ships 27 kernels, every one
of which has a GPU twin. No v0.1 function signature changed.

The pipeline order is settled and stated identically everywhere:

```
orient -> hot_pixels -> denoise -> straighten -> crop -> resize
-> exposure -> B&W conversion -> zone_system -> blur -> glow
-> local_contrast -> sharpen -> shadow_rolloff -> tone_curve
-> film_grain -> split_toning -> vignette -> highlight_rolloff
-> encode_srgb -> quantize_u8 | quantize_u16
```

`apply_lut` may run at any point after the B&W stage, and `histogram`
is a reduction rather than a stage.

### Added

**Geometry and restoration**

- `orient(img, Orientation)` — the eight Exif dihedral transforms,
  discriminants 1..=8 (JEITA CP-3451), rotations clockwise. A pure
  index permutation.
- `crop(img, CropParams)` — exact rectangle extraction. Geometry runs
  first, so a sidecar's crop renders the same image in every front end.
- `straighten(img, StraightenParams)` — rotation up to +/-45 degrees
  with 16-tap Catmull-Rom sampling, cropped to the largest inscribed
  axis-aligned rectangle. `degrees = 0` is the exact identity.
- `resize(img, ResizeParams)` — separable resampling with `Area`
  (exact fractional coverage), `Bilinear` and `CatmullRom` (Keys 1981).
  Centre-aligned, so a same-size resize is the exact identity.
- `hot_pixels(img, HotPixelParams)` — an index-clamped 3x3 conditional
  median for sensor defects, replacing a sample only past
  `|p - m| > threshold + relative*|m|`. `relative` defaults to `0.0`,
  which is the absolute-only criterion.
- `denoise(img, DenoiseParams)` — the guided filter's base term,
  self-guided per channel or cross-guided from a shared luminance guide
  at `C = 3`. `amount = 0.0` is the identity. `radius` is capped at
  `MAX_RADIUS = 32`.

**Tone and look**

- `exposure(img, stops)` — `out = in * 2^stops`. Nothing is clamped, so
  highlights above 1.0 stay recoverable, and +n EV then -n EV is
  bit-exact.
- `hsl_bw(img, HslWeightedParams)` — per-hue B&W weighting across eight
  bands (0, 30, 60, 120, 180, 240, 270, 300 degrees), Gaussian-blended
  with `sigma_deg = 30.0`, modulated by the chroma ratio
  `(max - min) / max`.
- `tone_curve(img, ToneCurveParams)` — the ASC CDL slope/offset/power
  primary correction (ASC Technology Committee v1.2, 2009). Monotonic
  for any positive slope and power.
- `shadow_rolloff(img, ShadowRolloffParams)` — the toe: a cubic Hermite
  below `knee`, meeting the identity in value and slope so the join
  shows no crease. `strength = 0.0` is the exact identity.
- `highlight_rolloff(img, RolloffParams)` — the shoulder: a quadratic
  Bezier between `knee` and `white_point`, leaving the identity at
  slope 1 and landing on white at slope 0. The default `(1.0, 1.0)`
  collapses to `min(x, 1.0)` bit for bit, so adding the stage changes
  nothing until a shoulder is asked for.
- `film_grain(img, GrainParams)` — band-passed procedural grain with a
  `4*L*(1-L)` midtone envelope. The noise is a hash of `(seed, x, y)`
  rather than a sequential generator, so the output is bit-identical at
  any thread count. `intensity = 0.0` is a passthrough.
- `split_toning(img, SplitToningParams)` — shadow and highlight tinting
  in OKLab (Ottosson 2020), crossfaded by a smoothstep around a pivot.
  Takes `(H, W, 1)` and returns `(H, W, 3)`: the only kernel that adds
  channels. Lightness is carried through untouched.
- `vignette(img, VignetteParams)` — radial darkening or lightening,
  blending circular and rectangular falloff through `roundness`.
  Distances are normalised to the frame, so a preview and the full-size
  render agree. All channels of a pixel share one factor.

**Spatial**

- `blur(img, BlurParams)` — a separable Gaussian, and the primitive the
  scattering effects are built on. Below sigma 6 a direct convolution;
  at or above it three box passes whose variances sum to sigma squared,
  which costs the same at any radius. Borders clamp. `sigma = 0.0` is
  the exact identity.
- `glow(img, GlowParams)` — `out = in + amount * blur(max(in -
  threshold, 0), sigma)`: halation, diffusion and veiling glare are one
  operation at different parameters and different pipeline positions.
  Additive, so headroom is carried to `highlight_rolloff` and spent
  once. `amount = 0.0` is the exact identity and must be non-negative.
- `sharpen(img, SharpenParams)` — a plain, haloing unsharp mask:
  `out = img + amount * soft_gate(detail, threshold) * detail`. The
  gate is a Hermite smoothstep, so a few ULP of cross-backend
  disagreement cannot flip a pixel between amplified and not. All three
  parameters default to `0.0`, the identity.

**Analysis and export**

- `histogram(img, HistogramParams)` — the crate's first reduction.
  Returns a `Histogram`: per-channel `(channels, bins)` counts, plus
  separate `below`, `above` and `non_finite` tallies and a `cdf()`. The
  range is fixed rather than auto-scaled, and out-of-range samples are
  tallied apart from the end bins, so clipping is visible.
- `apply_lut(img, lut, LutParams)` — arbitrary tone transfer through a
  caller-supplied 1-D table, linearly interpolated and clamped outside
  its domain. Monotonicity is not required, so it is the only tone
  kernel that can express solarisation. `histogram` ->
  `equalisation_lut()` -> `apply_lut` is the whole of histogram
  equalisation.
- `quantize_u8(img, QuantizeParams)` and `quantize_u16(...)` — the
  terminal stage, and the first kernels returning integers. Optional
  triangular-PDF dither of +/-1 LSB, keyed on position, channel and an
  explicit `seed`; dither defaults to off. `Dither.Off` is named `Off`
  because `Dither.None` is a syntax error in Python.

**The CUDA backend (`--features cuda`)**

- Every one of the 27 kernels runs on an NVIDIA GPU through a second,
  device-resident API, off by default: published wheels stay CPU-only.
  Building needs `nvcc`; running needs only the driver, which is
  dlopened. One `compute_80` PTX covers sm_80 and later; an older card
  gets a `RuntimeError`.
- Python: `phaios_core.gpu` with `available()`, `devices()`,
  `GpuContext` and `GpuImage` (upload once, chain resident, download
  once). Kernel signatures mirror the CPU functions argument for
  argument after the image, and the param classes are reused verbatim,
  so sidecars and presets stay backend-neutral.
- Rust: `cuda::Context`, `DeviceImage`, and `<kernel>_device` plus
  per-call forms under `cuda::kernels`.
- Determinism is scoped to a backend: bit-identical within one (and one
  driver), bounded across, with the bound committed per kernel and
  asserted by `tests/cuda_conformance.rs` (79 tests, skipped cleanly
  without a device; `PHAIOS_REQUIRE_GPU=1` forbids skipping).
  `sharpen`'s relative tolerance is taken against
  `max(|out|, amount·|blur(in)|)`: its output cancels where a pixel is
  about `amount/(1 + amount)` of its blur, and there the blur's last
  bit is relative to the term subtracted.
- Each CUDA file is compiled in its own precision mode, chosen from
  measured speed gains (`build.rs`, `PRECISION`; docs/gpu.md, "Precision
  modes"). Fast math for `zone_system`, `hsl_bw`, `encode_srgb` and
  `tone_curve` (13–47 % faster), FMA for `local_contrast` and `denoise`
  (5–12 %), strict IEEE for the other 15 files. `vignette` stays strict:
  fast math saved 23 % (0.1 ms), but its error grew with
  input × amount / feather, so no fixed bound held. `tone_curve` pins
  its affine step to the CPU's two roundings, so its `power == 1` path
  stays bit-exact unless an input or the affine result is subnormal. The
  kernels in docs/ffi.md §6's bit-exact list are bit-exact across
  backends, NaN payload and sign aside; the others stay within one 8-bit
  code of the CPU on every finite test image. `hsl_bw` keeps its hue
  and chroma divisions correctly rounded, and `denoise` pins `var(I)`,
  `cov(I, p_c)` and its intercept in `coeff_ab_cross`: approximated or
  fused, they broke the bound under opposing hue weights, next to
  `FLT_MAX` or −inf, and beside a sample above about 1e12. The build
  removes `NVCC_APPEND_FLAGS`, `NVCC_PREPEND_FLAGS`, `CUDAFE_FLAGS`,
  `INCLUDES`, `SYSTEM_INCLUDES`, `PTXAS_FLAGS` and `LIBRARIES` from
  nvcc's environment, so a developer's shell cannot change the PTX.
- New `PhaiosError::Backend`, mapped to Python `RuntimeError`. A
  missing GPU is an environment condition, not a bad argument.
  Source-breaking for Rust consumers matching `PhaiosError`
  exhaustively.
- New optional dependency `cudarc` 0.19 (MIT OR Apache-2.0), justified
  in `Cargo.toml`. The driver-API exit strategy, 22 entry points
  replaceable through `dlopen` and `dlsym`, is documented in
  `src/cuda/context.rs`.

**Testing, tooling and documentation**

- Type stubs: `python/phaios_core/__init__.pyi` and `gpu.pyi` for every
  function, class and enum, plus `py.typed`
  ([PEP 561](https://peps.python.org/pep-0561/)). Docstrings are
  mirrored verbatim from the runtime module.
  `tests/stub_contract.py` fails on any drift, and CI also runs
  `mypy --strict` and `mypy.stubtest`.
- `tests/properties.rs`: property tests over every kernel's shared
  `validate*` function, generated across the whole input domain rather
  than hand-picked. Valid input is accepted with finite output; invalid
  input is rejected without a panic and names the offending field. New
  dev-dependency `proptest`, fixed seed, no regressions file written.
- `scripts/gpu-verify.sh`: the half of CI that a hosted runner cannot
  do, under `PHAIOS_REQUIRE_GPU=1` so a green result cannot mean the
  CUDA tests were quietly skipped. It keeps the driver's JIT cache in
  `.cache/cuda-jit` unless `CUDA_CACHE_PATH` is set. It prints a PTX
  audit (each file's precision mode and its FMA, approximate,
  flush-to-zero and unpinned float instruction counts), which fails if
  any PTX came from another nvcc than the one found now, and, to paste
  into an issue, a block with the device, driver, backend fingerprint
  and full nvcc release. Its Python step is skipped without
  `.phaios-venv`, without `phaios_core` in it or with a CPU-only build,
  and fails when `phaios_core` does not import or its fingerprint
  differs from the tree's; with `PHAIOS_REQUIRE_GPU` set by the caller,
  to any value, every skip is a failure. `.github/workflows/gpu.yml`
  invokes the same script on a self-hosted runner, when started by
  hand.
- `scripts/determinism-guards.sh`, run by CI and by the release gate:
  fails on `mul_add`, `algebraic_*`, `#[target_feature]`, runtime CPU
  detection (`is_*_feature_detected!`), SIMD intrinsics (`std::arch`,
  `core::arch`) and ndarray matrix products (`.dot(`,
  `general_mat_mul`, whose sgemm picks an FMA kernel at run time) in
  the Rust of `src/`, and on `target-cpu` or `target-feature` in cargo
  config, workflows and `pyproject.toml`. Each wheel therefore computes
  at its target's fixed default CPU (baseline x86-64 on Linux, x86-64
  plus SSE3 and cmpxchg16b on Windows, Apple M1 on macOS), with no
  per-machine dispatch. It skips only lines whose first non-blank
  characters are `//`, as does CI's check for implicit state in
  `src/cuda/`.
- `tests/libm-imports.txt` and `scripts/libm-inventory.sh`, run by CI
  and by the release gate: the ledger of every function the extension
  imports from the platform libm, with the kernels that reach it, per
  pixel or once per call. CI fails on Linux when the set of imports
  changes.
- `scripts/kernel_digests.py` (numpy only): digests every kernel's
  output over generated test images, on either backend, and compares
  two runs case by case, down to ULP distances and changed u8 and u16
  codes. It shows whether a dependency, toolchain or platform change
  moved any output bit, and `compare --expect-bounded` judges a CPU and
  a GPU run against each kernel's class and (rtol, atol) in
  docs/ffi.md §6. CI runs its self-test and checks that a 64×64 run
  does not depend on the thread count.
- `examples/23_gpu_selftest.rs`: every device entry point against the
  CPU function that is its specification, with the committed
  tolerances.
- A GPU twin for every CPU example, 25 of them, plus
  `22_gpu_pipeline.rs` for the device-resident chain. Each mirrors its
  counterpart's input, parameters and output stems, so the two sets
  diff file by file.
- `benches/gpu.rs`: ids mirror the CPU ones under a `gpu/` prefix, on
  the same image with the same channel count.
- New documents: `docs/kernels.md` (what each of the 27 kernels does to
  the image), `docs/gpu.md` (the CUDA backend: build, device-resident
  chain, agreement classes, performance, verification) and
  `docs/export.md` (what a finished phaios image must be in a file,
  normatively: single-channel greyscale untoned, RGB toned, 16-bit
  archival default, 8-bit dithered). `README.md` is rewritten around
  install and use; `docs/architecture.md` keeps the derivations.
- `ZoneParams.offsets` getter, for serialising settings into sidecar
  and preset files.
- `__eq__` on `ZoneParams` and `GuidedFilterParams`, so a consumer can
  decide whether a cached render is stale.
- `PhaiosError::Parameter`, mapped to `ValueError`. Raised for a zone
  index outside 0..=10 (previously a silent no-op), a negative or
  non-finite `eps`, and non-finite offsets or `strength`.
- A `verify` gate on `release.yml`: the tag must match the version in
  `Cargo.toml` and `pyproject.toml`, and fmt, clippy and the tests must
  pass on the tagged commit, before anything is built or published.
- Every artefact is installed and tested before it is published.
  `.github/workflows/wheels.yml` builds the three wheels and the sdist,
  installs each wheel into a new virtual environment on its own OS
  under Python 3.12 and 3.15 (and the Linux wheel under 3.12 with numpy
  1.26), and runs `scripts/wheel-smoke.sh` there: numpy must arrive
  through the wheel's own metadata, the stubs must be installed, a
  kernel must give the exact result, the CPU wheel must have no `gpu`
  submodule, and `tests/ffi.py` must pass against the installed
  package. The sdist must contain exactly the files `cargo package`
  lists and must build from source with `--locked`. Each wheel's SBOM
  is checked against `cargo metadata`. `release.yml` publishes only
  when all of it passed, PyPI first and crates.io after. On a push or
  pull request that changes the packaging, the same workflow runs as a
  rehearsal. The six cells on the newest numpy also upload the kernel
  digests of the published build on their platform.

### Changed

- **Python 3.15 is supported.** The wheel is still the single
  `cp312-abi3` build and installs unchanged on CPython 3.12 through
  3.15; CI's Python job now runs all four, and the 3.15 classifier is
  declared. Free-threaded interpreters (3.14t, 3.15t) are still not
  covered (README "Install"). Building from source on free-threaded
  3.13t, which 0.1.x's PyO3 0.28 allowed, now fails: PyO3 supports
  free-threaded CPython from 3.14 only.
- **The module declares that it needs the GIL**
  (`#[pymodule(gil_used = true)]`). Under PyO3 0.30 a module declares
  the opposite unless told otherwise. On free-threaded 3.14t and 3.15t,
  which build from source, importing `phaios_core` now re-enables the
  GIL, with a `RuntimeWarning`, until the kernels have been reviewed
  for free threading. The `abi3` wheels are unaffected: the 3.12 stable
  ABI they are built against has no such declaration.
- **Wheel metadata follows PEP 639 and carries an SBOM.** The licence is
  the SPDX expression `License-Expression: GPL-3.0-or-later`
  (Metadata-Version 2.4), with `LICENSE` in `.dist-info/licenses/`; the
  licence classifier is gone. Each wheel carries maturin's CycloneDX
  SBOM of the Rust crates it links (PEP 770) in `.dist-info/sboms/`.
  Building from source needs maturin 1.12 or later (was 1.7), the
  first release that writes the SBOM.
- **Built against PyO3 0.30 and rust-numpy 0.30**, up from 0.28 in
  0.1.x. No Python signature or default changed. Two things a caller
  can see:
  - An array of the wrong dtype or number of dimensions is still a
    `TypeError`, but the message now names the mismatch —
    `type mismatch:\n from=float64, to=float32` or
    `dimensionality mismatch:\n from=2, to=3` — where 0.1.x raised a
    generic conversion error that did not say which. Code matching on
    the old message text must be updated.
  - A module function's `__self__` is now the module rather than
    `None`, as for any C extension: PyO3 0.28, which 0.1.x used,
    wrongly marked `#[pyfunction]`s `METH_STATIC`.
- **The backend fingerprint names the CUDA toolkit.**
  `GpuContext.fingerprint` is now
  `cuda/<device>/cc<maj>.<min>/ptx-compute_80/nvcc-<maj>.<min>.<patch>`,
  for example
  `cuda/NVIDIA GeForce RTX 5070 Ti/cc12.0/ptx-compute_80/nvcc-13.4.92`.
  The PTX is not committed and is rebuilt by whatever toolkit is
  present, and the bounded kernels inline toolkit code that changes
  between toolkits (libdevice `cbrtf` in `split_toning`, `logf`/`cosf`
  in `film_grain`, nvcc's fast-math intrinsics), so the toolkit is part of the backend's identity. A
  consumer that records a fingerprint must re-record it. `build.rs`
  re-runs when that nvcc, or its toolkit's libdevice, changes on disk,
  so an in-place toolkit upgrade refreshes the PTX and the fingerprint
  together; what cargo cannot see is listed at `watch_toolkit` in
  `build.rs`, and `touch src/cuda/ptx` forces the rebuild then.
- **`local_contrast` is 2.5x faster and uses 3.5x less memory.** The
  summed-area tables are built in two parallel passes instead of one
  single-threaded pass, and intermediates are released as soon as they
  are consumed. Output is unchanged.
- Both `denoise` paths sum each window directly from its own pixels
  rather than through a global summed-area table, which cancelled at
  extreme highlights. This is what caps `radius` at 32.
- The "no image assets" rule is enforced rather than stated:
  `.gitignore` covers every raster and RAW extension anywhere in the
  tree, CI and the release `verify` job fail on any tracked image file,
  and `Cargo.toml`'s `exclude` carries the same patterns. The one named
  exception is the maintainer's test photographs in
  `tools/kernel-viewer/testdata/` (five RAW files from four cameras, with
  the camera's JPEG where the RAW embeds one): tracked for the viewer and
  for measurements on real images, and kept out of the crate, the sdist
  and source archives.
- The package moved to maturin's mixed layout
  (`python-source = "python"`) so `phaios_core.gpu` can carry its own
  stub; maturin's pure-Rust layout supports only one, at the package
  root. `python/phaios_core/__init__.py` is now committed, byte
  identical to what maturin generated before. No behaviour change.
- Benchmarks run on deterministic pseudo-random images rather than
  constant ones, which were the cheapest possible input for three of
  the six v0.1 kernels. Figures before and after are not comparable.
- CI lints with `--all-targets` and discovers examples from cargo
  metadata rather than a hardcoded list.
- The workflows are hardened. Each action is pinned to a commit SHA,
  each job is granted only the permissions it needs, no checkout keeps
  its token, every cargo command that resolves dependencies passes
  `--locked` (in `scripts/gpu-verify.sh` too), maturin (1.16.0) and
  cargo-audit (0.23.0) are pinned, every job that runs steps has a
  timeout, and a newer CI run cancels an older one on the same branch.
  The release tag filter now matches `vMAJOR.MINOR.PATCH` only, so a
  pre-release tag publishes nothing. `gpu.yml` runs only when started by hand: no
  self-hosted CUDA runner exists, and its push and pull-request
  triggers queued jobs that never ran.
- `mypy>=2.4` added to `requirements-dev.txt`, whose maturin floor is
  raised to 1.12 like `pyproject.toml`'s. Dev-only; the crate gains no
  new dependency.
- Minimum Rust is declared as 1.88.

### Fixed

- **The wheel declared no dependencies.** `pip install phaios-core`
  into a bare environment succeeded and the first kernel call then
  raised `PanicException`. `dependencies = ["numpy>=1.26"]` is now
  declared. `[project]` also declared no `readme`, so the PyPI page
  would have been blank; `readme = "README.md"` fixes it. Added the
  Python 3 / 3.12 / 3.13 / 3.14 / 3.15 classifiers.
- **Without numpy, `import phaios_core` raises `ImportError`.** An
  environment that still lacks numpy (`pip install --no-deps`, a
  vendored copy) imported the module without complaint, and the first
  kernel call raised `PanicException`, which `except Exception:` does
  not catch. The module now imports numpy as it initialises, so the
  import itself fails with `ModuleNotFoundError: No module named
  'numpy'`.
- **A borrowed input array raised `PanicException`.** An input that
  another rust-numpy extension held mutably borrowed through rust-numpy's
  cross-extension borrow tracking panicked on extraction, because
  rust-numpy before 0.30 unwrapped the borrow check. Under 0.30 it is
  a catchable `TypeError` ("The given array is already borrowed").
- **The GPU entry points panicked on a machine with no NVIDIA driver.**
  cudarc's dynamic-loading path panics rather than returning an error
  when no driver library loads. All three entry points now probe with
  `is_culib_present` first, so `gpu.available()` is `False`,
  `gpu.devices()` is `[]` and `gpu.GpuContext(0)` raises a catchable
  `RuntimeError`. No published wheel was affected: they are CPU-only.
- **An uncatchable abort on zero-stride input.** Every kernel sized its
  output from the caller's logical shape, and a numpy `broadcast_to`
  view has an unbounded logical shape backed by as little as four
  bytes. A failed `Vec` allocation aborts the process rather than
  raising, so nothing in Python could catch it. All caller-derived
  allocations now go through `alloc::zeros3` / `zeros2`, which check an
  **8 GiB** single-allocation limit and return the new
  `PhaiosError::Allocation`, mapped to Python `MemoryError`. The CUDA
  `Context::upload` staging copy is bounded too. `tests/ffi.py`'s
  layout matrix gained a zero-stride variant.
- **`zone_system` and `encode_srgb` panicked on non-contiguous input.**
  Both called `as_slice().expect(...)`, so any strided view raised
  `PanicException`. Every kernel now accepts any layout and returns a
  freshly allocated C-contiguous array.
- **`zone_system` is bit-reproducible.** Zone offsets were summed in
  `HashMap` iteration order, so identical input produced different
  output in every process. They are now summed in ascending zone order.
- **The CUDA box blur lost dark detail beside a highlight** at
  sigma >= 6, missing its committed bound on ordinary scene-referred
  data. A sliding window subtracts, and Kahan compensation does not
  help because its error bound is dominated by the large samples. The
  device accumulator is now f64, matching the host, at no measurable
  cost.
- **`local_contrast` read noise as an edge beside a highlight.** The
  local variance `mean(L^2) - mean(L)^2` is a cancelling subtraction
  and was computed from Kahan-compensated f32 sums. The L and L^2 path
  is now f64 on both backends; the coefficient sums stay f32. Both
  kernels, and `glow`, now have a conformance test sweeping highlights
  from 1e0 to 1e8.
- **The cross-backend oracle scored NaN as perfect agreement.**
  `worst_violation` reduced with `f32::max`, which returns the other
  operand when one is NaN, so every NaN violation was dropped and a
  truncated output scored zero. All tests still passed once it was
  fixed.
- Guided-filter variance is clamped at zero: a negative result made
  `a = var/(var+eps)` invert or over-drive the local linear model.
- Examples 01 to 05 write display-referred output. They wrote linear
  values straight into 8-bit PPMs, so midtones came out about 2.5x too
  dark. Example 06 still writes both ways, which is its purpose.
- `docs/architecture.md` and `docs/ffi.md` corrected where they
  described behaviour the code did not have: zone offsets are not
  clamped, `encode_srgb` does not clamp to [0, 1], zero-size arrays are
  accepted, `encode_srgb` accepts any channel count, and the sRGB
  transfer is C0 but not C1 at the threshold.

### Removed

- The `log` dependency, which was declared but never used.

### Notes for consumers

- `ZoneParams` and `GuidedFilterParams` are no longer hashable: PyO3
  removes `__hash__` when `__eq__` is defined. This is correct for
  value-compared objects, but they cannot be dict keys.
- Code that passed an out-of-range zone index, a negative `eps` or a
  non-finite parameter now gets a `ValueError` where it previously got
  silently arbitrary behaviour.
- The determinism contract is scoped to a backend. "Bit-identical on
  another machine" was already untrue across operating systems in v0.1,
  because the CPU kernels call the platform libm and the three
  published wheels link three different ones.

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
