# phaios-core

Numerical kernels for black-and-white RAW image processing. Rust crate
with PyO3 bindings. The reusable, GUI-free, I/O-free core that any
front-end (desktop, web service, GIMP plugin, CLI script) can build on.

The name is from φαιός (Greek, "dusky grey"), the term Aristotle uses
in *De Sensu* for the intermediate colours between white and black.

Licence: GPLv3. Maintainer: Simon ([github.com/sdhdez](https://github.com/sdhdez)).

## 1. What this crate does — and what it doesn't

**Does:** the four B&W conversion methods, Adams/Archer Zone System
tone curve, He–Sun–Tang guided filter for local contrast, sRGB
transfer encoding, procedural film grain, split-toning, vignette.
All as pure functions on `f32` C-contiguous `(H, W, C)` arrays.

**Doesn't:** open RAW files, write TIFFs, manage settings, draw
pixels on a screen, talk to the network. Those belong to consumers
(`phaios` desktop, `phaios-web`, your scripts).

This separation is load-bearing. If you find yourself reaching for
`std::fs` or pulling in an image-format crate, stop — that work
belongs in a consumer.

## 2. Hard constraints (never violate)

- **Pure functions only.** Public kernels take immutable inputs and
  return new arrays (or write into caller-provided scratch buffers
  explicitly typed as such). No globals, no thread-locals, no hidden
  state.
- **`f32`, linear, scene-referred** for all pipeline math. The sRGB
  transfer is applied only by `encode_srgb`, which is the very last
  stage and is the only kernel that produces display-referred output.
- **Determinism.** Any kernel using randomness takes an explicit
  `seed: u64`. No global RNG.
- **No I/O.** No `std::fs`, no `std::net`, no `println!` outside of
  examples and tests. Logging via the `log` crate facade, never
  direct prints.
- **No panics across the FFI boundary.** Convert errors via
  `thiserror` + `From` impls into `PyErr`. Internal panics on
  invariant violations (e.g. shape mismatch) are acceptable but
  must be documented.
- **Zero-copy at FFI.** Inputs as `PyReadonlyArray3<f32>`. Outputs
  as `Py<PyArray3<f32>>` allocated once and returned. No
  `.to_owned()` on input arrays.
- **GIL release.** Long-running kernels release the GIL via
  `py.detach(...)` (PyO3 ≥ 0.22; `allow_threads` was removed).
  Document any kernel that doesn't and why.
- **Public API stability.** Once a function ships in a tagged
  release, its signature is stable until the next major version.
  Breaking changes go through a deprecation cycle.

## 3. Pipeline math — the rules contributors must know

```
RAW (consumer's problem)
  → linear scene-referred f32 RGB        ← input to kernels
  → exposure                             ← kernel
  → B&W conversion (one of four)         ← kernel
  → tone (zone system, parametric)       ← kernel
  → local contrast (guided filter)       ← kernel
  → finishing (grain, toning, vignette)  ← kernels
  → sRGB encode                          ← kernel (terminal)
  → display-referred f32 RGB             ← output, consumer writes file
```

The pipeline order matters. Document any kernel that has order
sensitivity in its doc comment.

Reference values worth committing to memory:
- BT.709 luminance weights: `(0.2126, 0.7152, 0.0722)`. Default for
  sRGB-primary data.
- Middle grey: 18% reflectance = `0.18` linear.
- sRGB threshold: `0.0031308`.

Full derivations and citations live in `docs/architecture.md`.

## 4. The Python ↔ Rust boundary

See `docs/ffi.md` for the full contract. Summary:

- Inputs: `PyReadonlyArray3<f32>`, C-contiguous, shape `(H, W, C)`
  with C ∈ {1, 3}.
- Outputs: `Py<PyArray3<f32>>`, same layout.
- Param types are `#[pyclass]` Rust structs constructible by name in
  Python. The orchestrator (in `phaios` desktop) builds them once
  per pipeline run.
- Errors: `PyResult<T>`, never panic across FFI.
- The Python module is named `phaios_core`. The crate is `phaios-core`.
  The version of both must match exactly — CI enforces this.

### PyO3 0.28 implementation notes (verified in v0.1)

The following patterns are current as of PyO3 0.28 / numpy 0.28.
Some differ from older tutorials:

- **GIL release**: `py.detach(move || { ... })` — `allow_threads` was
  removed in 0.22. The closure must be `Send`; `ArrayView3<f32>` is
  `Copy + Send` so it can be captured by `move`.
- **Array output**: `use numpy::IntoPyArray;` must be imported explicitly.
  `array.into_pyarray(py)` returns `Bound<'py, PyArray3<f32>>`; call
  `.unbind()` to get the `Py<PyArray3<f32>>` that `#[pyfunction]` returns.
- **`#[pyclass]` with `Clone`**: add `from_py_object` to opt in to the
  `FromPyObject` derive: `#[pyclass(from_py_object)]`. Without it, PyO3
  0.28 emits a deprecation warning and will break in a future release.
- **Enums**: `#[pyclass(eq, eq_int)]` enables Python integer comparison.
  Use `#[derive(Default)]` with `#[default]` on the default variant —
  clippy `-D warnings` rejects a manual `impl Default` when derive works.
- **Module/function name clash**: when a `#[pyfunction]` has the same
  name as its containing Rust module (e.g. `fn local_contrast` inside
  `mod local_contrast`), rename the Rust function (e.g. `local_contrast_fn`)
  and add `#[pyo3(name = "local_contrast")]` to expose it with the right
  Python name.
- **Dtype-mismatch exception type**: when a Python caller passes the
  wrong numpy dtype, PyO3 raises `TypeError` or `ValueError` depending
  on the PyO3/numpy version. In `tests/ffi.py`, use
  `pytest.raises(Exception)` rather than a specific subclass.

## 5. Code conventions

- **Rust 2024 edition**, stable toolchain.
- **`cargo fmt`** + **`cargo clippy -- -D warnings`** are blocking in
  CI.
- **`#![deny(missing_docs)]`** on the public API.
- **Doc comments on every public item.** For algorithms, cite the
  paper, textbook, or spec by full title and year. Examples:

  ```rust
  /// Apply the Adams/Archer Zone System tone curve.
  ///
  /// Eleven zones (0..=10), each one stop apart; Zone V = middle
  /// grey at 18% reflectance. Offsets are blended via a Gaussian
  /// in zone-position space (σ = 0.8 zones).
  ///
  /// Reference: Ansel Adams, *The Negative*, Little, Brown (1948),
  /// chapter 5; modernised in Davis, *Beyond the Zone System*,
  /// Focal Press (1999).
  ```

- **SPDX header on every source file:** `// SPDX-License-Identifier: GPL-3.0-or-later`.
- **`#[must_use]`** on functions returning `Result` or owned data.
- **Tests next to code.** Unit tests in `#[cfg(test)] mod tests`;
  integration tests in `tests/`.

## 6. Dependency policy

- **crates.io only.** Pin via `Cargo.lock` committed to the repo.
- **Adding a dep requires** licence, primary source URL,
  maintainer, and a justification. Record this as a comment in
  `Cargo.toml` next to the dep.
- **Prefer std > established crate > new dep.** "Established" means:
  >1M downloads, active maintenance, used by at least one major
  Rust project.
- **`cargo audit`** runs in CI and is blocking.
- The author is security-conscious — every new dep is a supply-chain
  decision.

Current core deps (do not exceed without justification): `pyo3`,
`numpy`, `ndarray`, `rayon`, `rand`, `rand_distr`, `thiserror`, `log`.

`ndarray` version must match the version pulled in by `numpy` (check
with `cargo tree | grep ndarray` after adding or updating `numpy`).
Enable the `rayon` feature: `ndarray = { version = "...", features = ["rayon"] }`.

`criterion` ≥ 0.5: `criterion::black_box` is deprecated — use
`std::hint::black_box` instead in all benchmark files.

## 7. Build & dev workflow

```
# First time
rustup default stable
uv venv .phaios-venv && source .phaios-venv/bin/activate
uv pip install -r requirements-dev.txt   # installs maturin, pytest

# After Rust changes
maturin develop --release             # rebuilds and installs into venv

# Test everything
cargo test
cargo clippy -- -D warnings
cargo fmt --check
pytest                                # Python-side smoke tests on the bindings

# Benchmarks (criterion)
cargo bench

# Build wheels for distribution
maturin build --release               # local
# CI uses cibuildwheel for manylinux + Windows + macOS
```

## 8. Examples (`examples/`)

The `examples/` directory is the public face of this crate for
non-Python users. Treat it like the OpenGL `examples/` directory:
small, self-contained, one concept per example.

Conventions:
- Each example is a single Rust binary in `examples/<name>.rs` (cargo
  picks them up automatically).
- Each example produces an 8-bit PPM file in `examples/output/`
  (PPM has no dependencies, anyone can view it).
- The synthetic test image is built in code (a Macbeth-style colour
  checker) — no real image inputs in this crate, ever.
- Each example begins with a doc comment explaining what kernel it
  demonstrates and what to look for in the output.

When adding a new kernel, add a new example for it. CI runs every
example as part of the test suite.

## 9. Commit & branch hygiene

- Conventional commits with optional scope: `feat(bw):`,
  `fix(zone):`, `test:`, `docs:`, `bench:`, `chore:`.
- One logical change per commit.
- `main` is always green; feature work in `feat/<slug>`.
- Tag releases as `v0.1.0`, `v0.2.0`. Both crate and Python wheel
  carry the same version.

## 10. Working agreement with Claude Code

- **Plan before code.** Produce a written plan, wait for approval,
  then implement.
- **Cite sources** in doc comments for every algorithm.
- **Ask before adding dependencies.** Justify each.
- **Flag assumptions.** Don't paper over ambiguity by picking a
  default silently.
- **Every question is standalone.** Don't assume context from other
  repos, past sessions, or unrelated files.
- **No telemetry, no network at runtime, ever.**
- If the user requests a feature that violates section 1 ("does
  what / doesn't do what"), surface the conflict before
  implementing. The split with `phaios` desktop is intentional.

## 11. Quick command reference

```
maturin develop --release             # rebuild + install Python bindings
cargo test                            # Rust unit + integration tests
cargo clippy -- -D warnings           # lint
cargo fmt --check                     # format check
cargo audit                           # supply-chain audit
cargo bench                           # criterion benchmarks (bench profile = optimised; no --release flag)
cargo run --example zone_system       # run a single example
pytest                                # Python-side smoke tests
maturin build --release               # local wheel build
```
