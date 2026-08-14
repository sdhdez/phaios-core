# Changelog

All notable changes to `phaios-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The Rust crate and the Python wheel always carry the same version.

## [Unreleased] — 0.2.0-dev

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

### Added

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
