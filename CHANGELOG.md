# Changelog

All notable changes to `phaios-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The Rust crate and the Python wheel always carry the same version.

## [Unreleased] — 0.2.0-dev

### Added

- `CHANGELOG.md` (this file).

### Fixed

_(nothing yet)_

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
