# phaios-core

Numerical kernels for black-and-white RAW image processing. Rust crate
with PyO3 bindings. The reusable, GUI-free, I/O-free core that any
front-end (desktop, web service, GIMP plugin, CLI script) can build on.

The name is from φαιός (Greek, "dusky grey"), the term Aristotle uses
in *De Sensu* for the intermediate colours between white and black.

Licence: GPLv3. Maintainer: Simon ([github.com/sdhdez](https://github.com/sdhdez)).

## 1. What this crate does — and what it doesn't

**v0.1 (shipped):** three B&W conversion kernels (standard luminance,
channel mixer, coloured-filter simulation), Adams/Archer Zone System
tone curve, He–Sun–Tang guided filter for local contrast, sRGB transfer
encoding.

**v0.2 (implemented, unreleased):** exposure compensation, HSL-weighted
B&W (8 hue bands), procedural film grain (explicit seed, no RNG
dependency), split-toning in OKLab, radial vignette, parametric tone
curve (ASC CDL); highlight roll-off (the explicit clip-vs-shoulder
decision, defaulting to a hard clip) and its counterpart shadow
roll-off (the toe; together with `tone_curve` they compose the
characteristic curve); dithered quantisation to u8/u16
(the first kernels returning integers); `apply_lut` and `histogram` —
the latter being the crate's first *reduction*, returning statistics
rather than an image; a separable Gaussian `blur` and `glow`, which
covers halation, diffusion and veiling glare in one kernel; the four geometry
kernels — `crop`, `orient` (the eight
Exif transforms), `straighten` (±45° with an inscribed-rectangle crop)
and `resize` (area / bilinear / Catmull-Rom); and an **optional CUDA
backend** behind `--features cuda`, exposing every kernel a second time
through the `phaios_core.gpu` submodule with a device-resident image
type. The GPU backend is strictly additive: the CPU build is unchanged
and needs no CUDA toolkit.

All as pure functions on `f32` `(H, W, C)` arrays: any input layout,
always a C-contiguous result.

**v0.3 (not started, deliberately):** pixel-level local adjustment
(masks, U-Point-style edit propagation) and the Newson et al. (2017)
stochastic grain model. Both are research territory; do not start
either without an explicit decision.

**Never does:** open RAW files, write TIFFs, manage settings, draw
pixels on a screen, talk to the network. Those belong to consumers
(`phaios` desktop, `phaios-web`, your scripts).

This separation is load-bearing. If you find yourself reaching for
`std::fs` or pulling in an image-format crate, stop — that work
belongs in a consumer.

## 2. Hard constraints (never violate)

- **Pure functions only — no *implicit* state.** Public kernels take
  immutable inputs and return new arrays. No globals, no thread-locals,
  no lazily-initialised singletons, no ambient context. State a backend
  genuinely requires — a CUDA device, its stream, its module cache —
  lives in a `Context` the caller constructs, owns and passes
  explicitly (on the GPU path, inside the `DeviceImage`); it is
  reachable only through that argument and cannot affect results. A
  kernel's output depends on its arguments and nothing else. CI-greps
  `src/cuda/` for `static mut|OnceLock|OnceCell|lazy_static!|thread_local!`.
- **`f32`, linear, scene-referred** for all pipeline math. The sRGB
  transfer is applied only by `encode_srgb`, which is the very last
  stage and is the only kernel that produces display-referred output.
- **Determinism is per backend.** Bit-identical output is promised
  within a backend (any thread count, any launch geometry) and bounded
  across backends — see `docs/ffi.md` §6; kernels free of
  transcendentals are bit-exact even across. Any kernel using
  randomness takes an explicit `seed: u64`. No global RNG. A GPU port
  must reproduce the *integer* part of a noise scheme bit-for-bit.
- **The CPU implementation is the specification.** Every GPU kernel is
  validated against it, never the other way round; disagreement beyond
  the committed bound means the GPU kernel is wrong. Validation logic
  is extracted (`pub(crate) validate*`) and shared, so both backends
  reject identical inputs with identical messages.
- **No I/O.** No `std::fs`, no `std::net`, no `println!` outside of
  examples and tests. The crate currently emits no diagnostics at
  all; if a kernel ever needs them, add the `log` facade back with a
  justification — never direct prints.
- **No panics across the FFI boundary.** Convert errors via
  `thiserror` + `From` impls into `PyErr`. Internal panics on
  invariant violations (e.g. shape mismatch) are acceptable but
  must be documented. `PanicException` is *not* a safety net: it
  inherits from `BaseException`, so it slips through a consumer's
  `except Exception:` and kills the calling thread. Never `expect`
  on anything the caller controls — including array layout.
- **Bounded allocation.** No single array may exceed 8 GiB
  (`src/alloc.rs`); beyond that kernels return `PhaiosError::Allocation`
  → Python `MemoryError`. A failed `Vec` allocation *aborts* rather than
  unwinding, which no `except` can catch, and a zero-stride numpy
  broadcast reaches that from four bytes of storage. Never call
  `Array3::zeros` on a caller-derived shape — use `alloc::zeros3`.
  **Deferred to v0.3:** the limit bounds single allocations only, not a
  pipeline's peak footprint or cumulative use across calls. See
  `docs/ffi.md`.
- **Layout-agnostic inputs.** `PyReadonlyArray3` accepts strided,
  Fortran-order and negative-stride arrays, and a consumer passing
  `img[::2, ::2]` is normal. Use `ndarray::Zip`, which walks any
  layout; never `as_slice().expect(...)`.
- **Deterministic reductions.** f32 addition is not associative, so
  a seed alone does not give reproducibility. Sort before reducing
  over a `HashMap`/`HashSet`, and fix the order of any parallel sum
  (or accumulate in f64). `zone_system` shipped in v0.1 summing in
  hash order and produced different bytes on every process.
- **Zero-copy at FFI.** Inputs as `PyReadonlyArray3<f32>`. Outputs
  as `Py<PyArray3<f32>>` allocated once and returned. No
  `.to_owned()` on input arrays.
- **Watch the scratch footprint.** A 24 MP frame is 100 MB as f32
  and 200 MB as f64; a kernel holding a handful of full-resolution
  intermediates reaches gigabytes. Drop each as soon as it is
  consumed, and prefer mapping during accumulation to materialising
  a transformed copy.
- **GIL release.** Long-running kernels release the GIL via
  `py.detach(...)` (PyO3 ≥ 0.22; `allow_threads` was removed).
  Document any kernel that doesn't and why.
- **Public API stability.** Once a function ships in a tagged
  release, its signature is stable until the next major version.
  Breaking changes go through a deprecation cycle.

## 3. Pipeline math — the rules contributors must know

```
RAW (consumer's problem)
  → linear scene-referred f32 RGB     (H, W, 3)   ← input to kernels
  → orient → straighten → crop → resize            ← kernels (geometry,
                                                     in that order)
  → exposure                                       ← kernel
  → B&W conversion (four methods)     → (H, W, 1) ← kernel
  → zone system                                    ← kernel
  → local contrast (guided filter)                 ← kernel
  → film grain                                     ← kernel
  → split-toning                      → (H, W, 3) ← kernel
  → vignette                                       ← kernel
  → shadow roll-off (toe)                          ← kernel
  → parametric tone curve                          ← kernel (straight)
  → highlight roll-off                             ← kernel (last linear)
  → sRGB encode                                    ← kernel
  → quantize (u8 / u16)                            ← kernel (terminal)
  → integer codes                                  ← output, consumer writes
                                                     the file per docs/export.md
```

The channel count collapses at the B&W stage and returns at
split-toning. Kernels after that point (`vignette`, `tone_curve`,
`encode_srgb`) accept any channel count, so a pipeline need not branch
on whether toning is enabled.

The pipeline order matters. Document any kernel that has order
sensitivity in its doc comment.

Reference values worth committing to memory:
- BT.709 luminance weights: `(0.2126, 0.7152, 0.0722)`. Default for
  sRGB-primary data.
- Middle grey: 18% reflectance = `0.18` linear; OKLab lightness ≈ 0.565.
- sRGB threshold: `0.0031308`. The transfer is C⁰ there but **not** C¹ —
  slope 12.920 below, 12.703 above.
- Hue band centres: red 0°, orange 30°, yellow 60°, green 120°,
  aqua 180°, blue 240°, purple 270°, magenta 300°.

Full derivations and citations live in `docs/architecture.md`.

## 4. The Python ↔ Rust boundary

See `docs/ffi.md` for the full contract, and `docs/export.md` for what
a finished image must look like when it reaches a file. Summary:

- Inputs: `PyReadonlyArray3<f32>`, shape `(H, W, C)` with C ∈ {1, 3},
  in any memory layout.
- Outputs: `Py<PyArray3<f32>>`, always freshly allocated and
  C-contiguous.
- Param types are `#[pyclass]` Rust structs constructible by name in
  Python. The orchestrator (in `phaios` desktop) builds them once
  per pipeline run.
- Errors: `PyResult<T>`, never panic across FFI.
- The Python module is named `phaios_core`. The crate is `phaios-core`.
  The version of both must match exactly — CI enforces this.

### PyO3 0.29 implementation notes (verified in v0.1–v0.2)

The following patterns are current as of PyO3 0.29 / numpy 0.29; all
survived the 0.28 → 0.29 bump unchanged.
Some differ from older tutorials:

- **GIL release**: `py.detach(move || { ... })` — `allow_threads` was
  removed in 0.22. The closure must be `Send`; `ArrayView3<f32>` is
  `Copy + Send` so it can be captured by `move`.
- **Array output**: `use numpy::IntoPyArray;` must be imported explicitly.
  `array.into_pyarray(py)` returns `Bound<'py, PyArray3<f32>>`; call
  `.unbind()` to get the `Py<PyArray3<f32>>` that `#[pyfunction]` returns.
- **`#[pyclass]` with `Clone`**: add `from_py_object` to opt in to the
  `FromPyObject` derive: `#[pyclass(from_py_object)]`. Without it, PyO3
  0.28+ emits a deprecation warning and will break in a future release.
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
- **`__eq__` removes `__hash__`**: defining `__eq__` in `#[pymethods]`
  makes the class unhashable, matching Python's own rule for
  value-compared objects. Correct, but it means param objects cannot be
  used as dict keys.
- **Validate in the kernel, not the constructor**: a `#[new]` that
  returns `PyResult` changes the Rust signature, and v0.1 signatures are
  frozen. Kernels already return `Result`, so parameter checks go there.
- **Interpreter-dependent tests don't link**: with the
  `extension-module` feature the test binary has no libpython, so
  `Python::attach` in a `#[cfg(test)]` block fails. Test error *values*
  in Rust and the exception *types* from `tests/ffi.py`.
- **maturin normalises the version**: `0.2.0-dev` in `Cargo.toml` and
  `pyproject.toml` builds a `0.2.0.dev0` wheel (PEP 440). The CI
  consistency check compares the raw TOML strings, so keeping both files
  identical is enough.
- **Fixed-size arrays cross the boundary**: `[f32; 8]` works as a
  `#[pyo3(get, set)]` field and as a `#[new]` argument, converting to
  and from a Python list. Used by `HslWeightedParams.hue_weights` and
  the OKLab triples in `SplitToningParams`.
- **`#[pyo3(signature = ...)]` gives keyword defaults** on `#[new]`, so
  `GrainParams(intensity=0.2)` works without writing a builder. Array
  defaults are written inline: `shadow_oklab = [0.0, 0.0, 0.0]`.

### Kernel-writing checklist (v0.2 practice)

Each kernel shipped as one commit containing all of:

1. `src/<kernel>.rs` with the kernel, its `#[pyclass]` params, and
   `#[cfg(test)] mod tests`.
2. The PyO3 binding in `lib.rs`, plus `m.add_class` / `m.add_function`.
3. An integration test in `tests/kernels.rs` asserting a *property*, not
   just a shape — the thing the kernel exists to do.
4. Python smoke tests in `tests/ffi.py`, including the error paths.
5. `examples/NN_<name>.rs` with an entry in `Cargo.toml`.
6. A criterion benchmark on the 24 MP image.
7. A section in `docs/architecture.md` with the citation.

Prefer properties that would fail loudly if the maths were wrong:
band ordering, exposure invariance, round-trip identity, determinism
across thread counts. A test that only checks the output shape passes
just as happily on a kernel that returns its input.

## 5. Code conventions

- **Rust 2024 edition**, stable toolchain.
- **`cargo fmt`** + **`cargo clippy --all-targets -- -D warnings`**
  are blocking in CI.
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
`numpy`, `ndarray`, `rayon`, `thiserror`, plus `cudarc` — **optional**,
pulled in only by `--features cuda` — for the GPU backend. `log` was
dropped in v0.2-dev: it was declared but never used. `rand` and
`rand_distr` were considered for film grain and **rejected**: the
shipped kernel hashes pixel coordinates with splitmix64, which is
reproducible across thread counts and portable to the GPU bit-for-bit,
neither of which a stateful RNG can promise.

Kernels reach rayon through `ndarray::Zip::par_for_each` and
`ndarray::parallel::prelude`, both enabled by ndarray's `rayon`
feature; the direct dependency is kept for kernels that need rayon's
own iterators.

`ndarray` version must match the version pulled in by `numpy` (check
with `cargo tree | grep ndarray` after adding or updating `numpy`).
Enable the `rayon` feature: `ndarray = { version = "...", features = ["rayon"] }`.

`criterion` ≥ 0.5: `criterion::black_box` is deprecated — use
`std::hint::black_box` instead in all benchmark files.

## 7. Build & dev workflow

```
# First time (requires Python 3.12+; 3.13 recommended for performance)
rustup default stable
uv venv .phaios-venv && source .phaios-venv/bin/activate
uv pip install -r requirements-dev.txt   # installs maturin, pytest

# After Rust changes
maturin develop --release             # rebuilds and installs into venv

# Test everything
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
pytest                                # Python-side smoke tests on the bindings

# Benchmarks (criterion)
cargo bench

# Build wheels for distribution
maturin build --release               # local
# Release workflow uses maturin-action for manylinux_2_17 + Windows + macOS
```

CI (`.github/workflows/ci.yml`) runs the same checks — `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo audit`,
every example (discovered from cargo metadata, not a hardcoded list), and
`pytest`. Green locally ≈ green in CI.

Run clippy with `--all-targets` locally too: the bare form lints only the
library, so warnings in examples, benches and tests go unseen until CI.

`cargo audit` is a separate binary (`cargo install cargo-audit`); CI uses
the `rustsec/audit-check` action instead, so a fresh clone will not have
the local command until it is installed.

### Working files stay inside the repo

Scratch files, throwaway scripts, generated reports, disposable worktrees
and any other residual of a piece of work go in **`.cache/`** at the repo
root — never `/tmp`, never `~/.cache`, never a home directory. `.cache/`
is gitignored (anchored, like `/target/`) and listed in `Cargo.toml`'s
`exclude`, so nothing in it can reach a commit or a published crate. It
holds nothing the crate needs and is safe to delete wholesale.

Layout: `.cache/scratch/` for one-off files, `.cache/worktrees/` for
disposable `git worktree` checkouts — the way to test a deliberate
mutation without touching the working tree.

Two reasons beyond tidiness. A system temp directory here is a small
tmpfs with a quota: a `cargo` build in one dies partway with `Disk quota
exceeded`, and it is cleared without warning mid-session, which has
already lost working data. And a path under `$HOME` is invisible to
`git status` — residuals accumulate there unnoticed, where `.cache/` can
be inspected and removed like any other build artefact.

**Exception: Claude Code's own files.** Its memory, workflow journals and
task outputs live under `~/.claude/` because the harness owns those paths.
That is expected; don't try to relocate them.

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
example as part of the test suite — except those declaring
`required-features`, which discovery skips because the hosted runner has
neither a CUDA toolkit nor a device. `15_gpu_exposure` is the only one
today, and it is therefore the maintainer's job to run it locally
before a release.

The interactive counterpart lives in `tools/kernel-viewer/` — a
standalone GUI (eframe/egui) with live sliders for every kernel, RAW/DNG
loading via rawler, CPU/GPU A/B split and difference view, and a
synthetic photo scene. It has its own `Cargo.toml`, `Cargo.lock` and an
empty `[workspace]` table, and is deliberately **not** a workspace
member: its GUI dependencies must never enter this crate's lockfile or
audit surface. It consumes only the public Rust API. The "no real image
inputs, ever" rule applies to committed assets; the viewer loads images
at runtime, and its `testdata/` directory is gitignored. It is a local
dev tool: no CI.

## 9. Commit & branch hygiene

- Conventional commits with optional scope: `feat(bw):`,
  `fix(zone):`, `test:`, `docs:`, `bench:`, `chore:`.
- One logical change per commit.
- `main` is always green; feature work in `feat/<slug>`.
- Tag releases as `v0.1.0`, `v0.2.0`. Both crate and Python wheel
  carry the same version. CI enforces this with a version-consistency
  check (`Cargo.toml` version == `pyproject.toml` version).

**Cutting a release:**
1. Bump the version in `Cargo.toml`, `pyproject.toml`, **and**
   `Cargo.lock` in a single commit (`chore: bump version to vX.Y.Z`).
   (`Cargo.lock` updates automatically after any `cargo` command;
   stage it explicitly or `cargo publish` will see a dirty tree.)
2. Ensure prerequisites are in place (one-time setup — see README):
   - `CARGO_REGISTRY_TOKEN` GitHub Actions secret (crates.io). The
     token must have both `publish-new` (first upload) **and**
     `publish-update` (subsequent versions) scopes, scoped to the
     crate. Set no expiry or a long one — short-lived tokens cause
     403s on re-runs.
   - PyPI OIDC trusted publisher configured (`release.yml` / env `pypi`)
3. `git tag vX.Y.Z && git push origin vX.Y.Z` — triggers
   `.github/workflows/release.yml`, which builds wheels for
   manylinux_2_17, Windows x86_64, macOS arm64 via
   `PyO3/maturin-action@v1`, then publishes to PyPI (OIDC, no stored
   token) and crates.io (`CARGO_REGISTRY_TOKEN` secret).

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
- If the user requests a feature that violates section 1 ("v0.1
  implemented / v0.2 planned / never does"), surface the conflict
  before implementing. The split with `phaios` desktop is
  intentional.

## 11. Quick command reference

```
maturin develop --release             # rebuild + install Python bindings
cargo test                            # Rust unit + integration tests
cargo clippy --all-targets -- -D warnings   # lint (incl. examples/benches)
cargo fmt --check                     # format check
cargo audit                           # supply-chain audit (needs cargo-audit)
cargo bench                           # criterion benchmarks (bench profile = optimised; no --release flag)
cargo run --example zone_system       # run a single example
pytest                                # Python-side smoke tests
maturin build --release               # local wheel build
```
