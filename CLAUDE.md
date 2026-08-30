# phaios-core

Numerical kernels for black-and-white RAW image processing: a Rust crate
with PyO3 bindings, GUI-free and I/O-free, that any front-end can build
on. What the crate contains is in `README.md` and `CHANGELOG.md`; the
derivations and citations are in `docs/architecture.md`. This file is the
part you need while writing code.

## 1. Scope

**Never does:** open RAW files, write TIFFs, manage settings, draw pixels
on a screen, talk to the network. Those belong to consumers (`phaios`
desktop, your scripts). The separation is load-bearing — if you find
yourself reaching for `std::fs` or an image-format crate, stop: that work
belongs in a consumer.

**Not started, deliberately:** pixel-level local adjustment (masks,
U-Point-style edit propagation) and the Newson et al. (2017) stochastic
grain model. Both are research territory; do not start either without an
explicit decision.

If a request conflicts with any of this, surface the conflict before
implementing.

## 2. Hard constraints (never violate)

- **Pure functions only — no *implicit* state.** Public kernels take
  immutable inputs and return new arrays. No globals, no thread-locals,
  no lazily-initialised singletons, no ambient context. State a backend
  genuinely requires — a CUDA device, its stream, its module cache —
  lives in a `Context` the caller constructs, owns and passes explicitly
  (on the GPU path, inside the `DeviceImage`); it is reachable only
  through that argument and cannot affect results. CI greps `src/cuda/`
  for `static mut|OnceLock|OnceCell|lazy_static!|thread_local!`.
- **`f32`, linear, scene-referred** for all pipeline math. The sRGB
  transfer is applied only by `encode_srgb`, the last stage and the only
  kernel producing display-referred output.
- **Determinism is per backend.** Bit-identical within a backend (any
  thread count, any launch geometry), bounded across — `docs/ffi.md` §6.
  Kernels free of transcendentals are bit-exact even across. Any kernel
  using randomness takes an explicit `seed: u64`. No global RNG. A GPU
  port must reproduce the *integer* part of a noise scheme bit-for-bit.
- **The CPU implementation is the specification.** Every GPU kernel is
  validated against it, never the other way round; disagreement beyond
  the committed bound means the GPU kernel is wrong. Validation is
  extracted (`pub(crate) validate*`) and shared, so both backends reject
  identical inputs with identical messages.
- **No I/O.** No `std::fs`, no `std::net`, no `println!` outside examples
  and tests. The crate emits no diagnostics at all; if a kernel ever
  needs them, add the `log` facade back with a justification — never
  direct prints.
- **No panics across the FFI boundary.** Convert errors via `thiserror`
  + `From` impls into `PyErr`. Internal panics on invariant violations
  are acceptable but must be documented. `PanicException` is *not* a
  safety net: it inherits from `BaseException`, so it slips through a
  consumer's `except Exception:` and kills the calling thread. Never
  `expect` on anything the caller controls — including array layout.
- **Bounded allocation.** No single array may exceed 8 GiB
  (`src/alloc.rs`); beyond that kernels return `PhaiosError::Allocation`
  → Python `MemoryError`. A failed `Vec` allocation *aborts* rather than
  unwinding, which no `except` can catch, and a zero-stride numpy
  broadcast reaches that from four bytes of storage. Never call
  `Array3::zeros` on a caller-derived shape — use `alloc::zeros3`. The
  limit bounds single allocations only, not a pipeline's peak footprint;
  the rest is deferred to v0.3.
- **Layout-agnostic inputs.** `PyReadonlyArray3` accepts strided,
  Fortran-order and negative-stride arrays, and a consumer passing
  `img[::2, ::2]` is normal. Use `ndarray::Zip`, which walks any layout;
  never `as_slice().expect(...)`.
- **Deterministic reductions.** f32 addition is not associative, so a
  seed alone does not give reproducibility. Sort before reducing over a
  `HashMap`/`HashSet`, and fix the order of any parallel sum (or
  accumulate in f64). `zone_system` shipped in v0.1 summing in hash order
  and produced different bytes on every process.
- **Zero-copy at FFI.** Inputs as `PyReadonlyArray3<f32>`. Outputs as
  `Py<PyArray3<f32>>` allocated once and returned. No `.to_owned()` on
  input arrays.
- **Watch the scratch footprint.** A 24 MP frame is 100 MB as f32 and
  200 MB as f64; a kernel holding a handful of full-resolution
  intermediates reaches gigabytes. Drop each as soon as it is consumed,
  and prefer mapping during accumulation to materialising a copy.
- **GIL release.** Long-running kernels release the GIL via
  `py.detach(...)` (PyO3 ≥ 0.22; `allow_threads` was removed). Document
  any kernel that doesn't, and why.
- **Public API stability.** Once a function ships in a tagged release its
  signature is stable until the next major version. Breaking changes go
  through a deprecation cycle.

## 3. Pipeline order

Geometry (`orient` → `straighten` → `crop` → `resize`) → `exposure` →
B&W conversion → `zone_system` → `local_contrast` → `film_grain` →
`split_toning` → `vignette` → `shadow_rolloff` → `tone_curve` →
`highlight_rolloff` → `encode_srgb` → `quantize`.

The channel count collapses to 1 at the B&W stage and returns to 3 at
split-toning. Kernels after that point accept any channel count, so a
pipeline need not branch on whether toning is enabled. Order matters:
document any kernel with order sensitivity in its doc comment.

Values that are easy to get wrong, and that a kernel needs at hand:

- BT.709 luminance weights `(0.2126, 0.7152, 0.0722)` — the default for
  sRGB-primary data.
- Middle grey: 18% reflectance = `0.18` linear.
- sRGB threshold `0.0031308`. The transfer is C⁰ there but **not** C¹ —
  slope 12.920 below, 12.703 above.

Everything else, with derivations and citations, is in
`docs/architecture.md`.

## 4. The Python ↔ Rust boundary

`docs/ffi.md` is the full contract; `docs/export.md` covers what a
finished image must look like in a file. In short: inputs are
`PyReadonlyArray3<f32>`, shape `(H, W, C)` with C ∈ {1, 3}, any layout;
outputs are `Py<PyArray3<f32>>`, freshly allocated and C-contiguous;
errors are `PyResult<T>` and never a panic. Param types are `#[pyclass]`
structs constructible by name in Python. The Python module is
`phaios_core`, the crate is `phaios-core`, and their versions must match
exactly — CI enforces it.

### PyO3 0.29 notes

Current as of PyO3 0.29 / numpy 0.29, and several differ from older
tutorials:

- **GIL release:** `py.detach(move || { ... })`; `allow_threads` was
  removed in 0.22. The closure must be `Send` — `ArrayView3<f32>` is
  `Copy + Send`, so `move` works.
- **Array output:** `use numpy::IntoPyArray;` must be imported
  explicitly. `array.into_pyarray(py)` returns `Bound<'py, PyArray3>`;
  call `.unbind()` for the `Py<PyArray3<f32>>` a `#[pyfunction]` returns.
- **`#[pyclass]` with `Clone`:** add `from_py_object` to opt into the
  `FromPyObject` derive. Without it 0.28+ warns and will break later.
- **Enums:** `#[pyclass(eq, eq_int)]` enables integer comparison. Use
  `#[derive(Default)]` with `#[default]` — clippy `-D warnings` rejects a
  manual `impl Default` where derive would do.
- **Module/function name clash:** when a `#[pyfunction]` shares its
  module's name, rename the Rust fn and add `#[pyo3(name = "...")]`.
- **Validate in the kernel, not the constructor:** a `#[new]` returning
  `PyResult` changes the Rust signature, and shipped signatures are
  frozen. Kernels already return `Result`.
- **`__eq__` removes `__hash__`**, matching Python's own rule for
  value-compared objects — so param objects cannot be dict keys.
- **Interpreter-dependent tests don't link:** with `extension-module` the
  test binary has no libpython, so `Python::attach` in `#[cfg(test)]`
  fails. Test error *values* in Rust, exception *types* from
  `tests/ffi.py` — and there use `pytest.raises(Exception)`, since a
  dtype mismatch raises `TypeError` or `ValueError` depending on version.
- **Fixed-size arrays cross the boundary:** `[f32; 8]` works as a
  `#[pyo3(get, set)]` field and a `#[new]` argument.
- **`#[pyo3(signature = ...)]` gives keyword defaults** on `#[new]`;
  array defaults are written inline.

### Kernel-writing checklist

One commit containing all of:

1. `src/<kernel>.rs` — the kernel, its `#[pyclass]` params, and
   `#[cfg(test)] mod tests`.
2. The PyO3 binding in `lib.rs`, plus `m.add_class` / `m.add_function`.
3. An integration test in `tests/kernels.rs` asserting a *property*.
4. Python smoke tests in `tests/ffi.py`, including the error paths.
5. `examples/NN_<name>.rs`, declared in `Cargo.toml`.
6. A criterion benchmark in `benches/kernels.rs` on the 24 MP image —
   and, if the kernel has a GPU path, in `benches/gpu.rs` too, with the
   same channel count so the two ids can be divided.
7. A section in `docs/architecture.md` with the citation.

Prefer properties that fail loudly if the maths is wrong: band ordering,
exposure invariance, round-trip identity, a signed rather than absolute
comparison. A test that only checks the output shape passes just as
happily on a kernel that returns its input — and a test written on a
constant image passes on one that ignores its parameters entirely.

## 5. Code conventions

- **Rust 2024 edition**, stable toolchain.
- **`cargo fmt`** and **`cargo clippy --all-targets -- -D warnings`** are
  blocking in CI. Run clippy with `--all-targets` locally too: the bare
  form lints only the library, so warnings in examples, benches and tests
  go unseen until CI.
- **`#![deny(missing_docs)]`** on the public API, and a doc comment on
  every public item. Cite algorithms by full title and year.
- **SPDX header on every source file:**
  `// SPDX-License-Identifier: GPL-3.0-or-later`.
- **`#[must_use]`** on functions returning `Result` or owned data.
- Unit tests in `#[cfg(test)] mod tests`; integration tests in `tests/`.

## 6. Dependency policy

Every new dependency is a supply-chain decision.

- **crates.io only**, pinned via the committed `Cargo.lock`.
- **Adding one requires** licence, primary source URL, maintainer and a
  justification, recorded as a comment in `Cargo.toml` beside the dep.
- **Prefer std > established crate > new dep.** "Established" means >1M
  downloads, active maintenance, and use by at least one major project.
- **`cargo audit`** is blocking in CI. It is a separate binary locally
  (`cargo install cargo-audit`); CI uses the `rustsec/audit-check`
  action, so a fresh clone will not have the command.

Core deps, not to be exceeded without justification: `pyo3`, `numpy`,
`ndarray`, `rayon`, `thiserror`, plus `cudarc` behind `--features cuda`.
`rand` and `rand_distr` were considered for film grain and **rejected**:
the shipped kernel hashes pixel coordinates with splitmix64, which is
reproducible across thread counts and portable to the GPU bit-for-bit,
neither of which a stateful RNG can promise.

Two traps: `ndarray`'s version must match the one `numpy` pulls in (check
`cargo tree | grep ndarray` after touching `numpy`), with its `rayon`
feature enabled; and `criterion::black_box` is deprecated — use
`std::hint::black_box`.

## 7. Build & test

```sh
uv venv .phaios-venv && source .phaios-venv/bin/activate
uv pip install -r requirements-dev.txt

maturin develop --release        # after Rust changes; add --features cuda
                                 # or the gpu submodule silently disappears
cargo test                       # add --features cuda for the GPU suite
cargo clippy --all-targets -- -D warnings
cargo fmt --check
pytest
cargo bench                      # bench profile is optimised; no --release
```

CI runs the same checks plus `cargo audit` and every example discovered
from cargo metadata. It never builds `--features cuda` — the hosted
runner has no `nvcc` and no device — so `./scripts/gpu-verify.sh` is the
mirror image: it runs exactly the targets CI skips, under
`PHAIOS_REQUIRE_GPU=1` so a green result cannot mean the suite quietly
skipped for want of a device.

### Working files stay inside the repo

Scratch files, throwaway scripts, generated reports and disposable
worktrees go in **`.cache/`** at the repo root — never `/tmp`, never
`~/.cache`, never a home directory. `.cache/` is gitignored and excluded
from the package, so nothing in it can reach a commit or a published
crate; `.cache/scratch/` for one-off files, `.cache/worktrees/` for
throwaway checkouts.

Two reasons beyond tidiness. A system temp directory here is a small
tmpfs with a quota: a `cargo` build in one dies partway with `Disk quota
exceeded`, and it is cleared without warning mid-session, which has
already lost working data. And a path under `$HOME` is invisible to
`git status`, so residuals accumulate there unnoticed.

Claude Code's own memory, journals and task outputs live under
`~/.claude/` because the harness owns those paths. Don't relocate them.

## 8. Examples

`examples/` is the public face of the crate for non-Python users:
small, self-contained, one concept each.

- One Rust binary per example, writing an 8-bit PPM to `examples/output/`
  so it needs no dependencies to view.
- The test image is a Macbeth-style chart built in code. **No real image
  inputs in this crate, ever** — that rule is about committed assets;
  `tools/kernel-viewer/` loads images at runtime and its `testdata/` is
  gitignored.
- A doc comment saying what it demonstrates and what to look for.
- A new kernel gets a new example. A GPU kernel gets a twin under the
  same number series, gated on `required-features = ["cuda"]`, printing
  its agreement with the CPU kernel and naming the file to diff against.

CI runs every example without `required-features`; the GPU ones run only
via `scripts/gpu-verify.sh`.

`tools/kernel-viewer/` is a standalone GUI (eframe/egui) with live
sliders, RAW loading and a CPU/GPU split view. It has its own
`Cargo.toml`, `Cargo.lock` and an empty `[workspace]` table and is
deliberately **not** a workspace member: its GUI dependencies must never
enter this crate's lockfile or audit surface. It consumes only the public
API, and has no CI.

## 9. Commits & releases

- Conventional commits with optional scope: `feat(bw):`, `fix(zone):`,
  `test:`, `docs:`, `bench:`, `chore:`.
- One logical change per commit.
- `main` is always green; feature work on `feat/<slug>`.
- Crate and wheel carry the same version; CI enforces the match.

Releasing is documented in `README.md` — including the pre-tag
`scripts/gpu-verify.sh` run, since nothing in CI would notice a CUDA
backend that fails to compile.

## 10. Working agreement

- **Plan before code.** Produce a written plan, wait for approval, then
  implement.
- **Ask before adding dependencies**, and justify each.
- **Flag assumptions.** Don't paper over ambiguity by picking a default
  silently.
- **Every question is standalone.** Don't assume context from other
  repos, past sessions, or unrelated files.
- **No telemetry, no network at runtime, ever.**
