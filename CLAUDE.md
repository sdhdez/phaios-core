# phaios-core — working notes for Claude Code

Numerical kernels for black-and-white RAW image processing: a Rust crate
with PyO3 bindings, GUI-free and I/O-free.

This file is loaded into context on every task, so it holds only what
changes what you write. Everything else has a home:

| | |
|---|---|
| What the crate contains, dev setup, releasing | `README.md` |
| Conventions, the kernel checklist, dependency rules | `CONTRIBUTING.md` |
| Why a kernel works as it does, with citations | `docs/architecture.md` |
| The FFI contract and CPU/GPU agreement classes | `docs/ffi.md` |
| What a finished image must look like in a file | `docs/export.md` |
| Version history | `CHANGELOG.md` |

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

If a request conflicts with either, surface the conflict before
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
  the rest is deferred.
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
B&W conversion → `zone_system` → `local_contrast` → `shadow_rolloff` →
`tone_curve` → `film_grain` → `split_toning` → `vignette` →
`highlight_rolloff` → `encode_srgb` → `quantize`.

Grain and vignette come after the tone stages: a curve applied
afterwards would reshape the grain off the midtones and act on already
darkened corners. The optional stages (`hot_pixels`, `denoise`, `blur`,
`glow`, `sharpen`) sit where their doc comments say. The channel count
collapses to 1 at the B&W stage and returns to 3 at split-toning;
kernels after that accept any channel count, so a pipeline need not
branch on whether toning is enabled. Order matters — document any
kernel with order sensitivity in its doc comment.

## 4. PyO3 0.29 traps

Current as of PyO3 0.29 / numpy 0.29, and several contradict older
tutorials you may have seen:

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
  fails. Test error *values* in Rust and exception *types* from
  `tests/ffi.py` — and there use `pytest.raises(Exception)`, since a
  dtype mismatch raises `TypeError` or `ValueError` depending on version.
- **Fixed-size arrays cross the boundary:** `[f32; 8]` works as a
  `#[pyo3(get, set)]` field and a `#[new]` argument.
- **`#[pyo3(signature = ...)]` gives keyword defaults** on `#[new]`;
  array defaults are written inline.

## 5. Toolchain traps

- `maturin develop --release` **silently drops the GPU** unless you add
  `--features cuda`; `from phaios_core import gpu` then fails.
- `cargo test` and `cargo clippy` skip the CUDA backend entirely without
  `--features cuda` — it is behind `#[cfg]`, so even a type error there
  passes. `./scripts/gpu-verify.sh` runs what CI cannot.
- Run clippy with `--all-targets`: the bare form lints only the library,
  so examples, benches and tests go unchecked until CI.
- `cargo bench` needs no `--release`; the bench profile is already
  optimised.

## 6. Working practice

- **Plan before code.** Produce a written plan, wait for approval, then
  implement.
- **Adding or changing a kernel:** follow the eight-item checklist in
  `CONTRIBUTING.md`. The work is not complete until all eight exist —
  the benchmark, the `docs/architecture.md` section and the
  `docs/kernels.md` section are the ones most often forgotten.
- **Python-visible changes need the stub.** A new or changed
  `#[pyfunction]`, `#[pyclass]` field, default or docstring is not
  complete until `python/phaios_core/__init__.pyi` (or `gpu.pyi`)
  mirrors it verbatim; `tests/ffi.py` fails on any drift.
- **Ask before adding a dependency**, and justify it. Rules in
  `CONTRIBUTING.md`.
- **Scratch files go in `.cache/`** at the repo root — never `/tmp`,
  never `~/.cache`, never a home directory. `.cache/scratch/` for one-off
  files, `.cache/worktrees/` for throwaway checkouts. It is gitignored
  and excluded from the package. Two reasons beyond tidiness: a system
  temp directory here is a quota-limited tmpfs that kills `cargo` builds
  partway and is cleared without warning mid-session, and a path under
  `$HOME` is invisible to `git status`, so residuals accumulate unseen.
  Claude Code's own memory and journals live under `~/.claude/` because
  the harness owns those paths — don't relocate them.
- **Flag assumptions.** Don't paper over ambiguity by picking a default
  silently.
- **Every question is standalone.** Don't assume context from other
  repos, past sessions, or unrelated files.
- **No telemetry, no network at runtime, ever.**
