# Contributing to phaios-core

Setting up a development environment is covered by **Quick start** in
`README.md`. This file is the process around a change: conventions, what
a complete kernel looks like, and the rules for dependencies.

The design rationale — why a kernel works the way it does, with citations
— lives in `docs/architecture.md`. The FFI contract, including the
per-kernel agreement classes between the CPU and CUDA backends, is
`docs/ffi.md`. What a finished image must look like in a file is
`docs/export.md`.

## Code conventions

- **Rust 2024 edition**, stable toolchain.
- **SPDX header on every source file:**
  `// SPDX-License-Identifier: GPL-3.0-or-later`
- **`#![deny(missing_docs)]`** on the public API: every public item needs
  a doc comment, and every algorithm cites its paper, textbook or spec by
  full title and year.
- **`#[must_use]`** on functions returning `Result` or owned data.
- Unit tests in `#[cfg(test)] mod tests` beside the code; integration
  tests in `tests/`.
- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` are
  blocking in CI. Use `--all-targets` locally too — the bare form lints
  only the library, so warnings in examples, benches and tests go unseen
  until CI.

## What a complete kernel looks like (the seven-item checklist)

One commit containing all of:

1. `src/<kernel>.rs` — the kernel, its `#[pyclass]` params, and
   `#[cfg(test)] mod tests`.
2. The PyO3 binding in `lib.rs`, plus `m.add_class` / `m.add_function`,
   **and its entry in `python/phaios_core/__init__.pyi`** (and `gpu.pyi`
   for a GPU twin).
3. An integration test in `tests/kernels.rs` asserting a *property*.
4. Python smoke tests in `tests/ffi.py`, including the error paths.
5. `examples/NN_<name>.rs`, declared in `Cargo.toml`.
6. A criterion benchmark in `benches/kernels.rs` on the 24 MP image —
   and, if the kernel has a GPU path, one in `benches/gpu.rs` too, using
   the same channel count so the two ids can be divided.
7. A section in `docs/architecture.md` with the citation.

### Writing the property test

Prefer properties that fail loudly when the maths is wrong: band
ordering, exposure invariance, round-trip identity, a signed rather than
an absolute comparison.

Two failure modes worth naming, because both have shipped here:

- A test that only checks the output shape passes just as happily on a
  kernel that returns its input.
- A test written on a *constant* image passes on a kernel that ignores
  its parameters entirely — on a flat field the guided filter's detail
  term is identically zero, so the whole edge-preserving model could be
  arbitrarily wrong and six tests still passed.

When strengthening a weak test, break the code deliberately and confirm
the new test fails. Check the mutation is *live* first: one that leaves
output unchanged proves nothing, and has fooled us more than once.

## Examples

`examples/` is the crate's public face for non-Python users: small,
self-contained, one concept each.

- One Rust binary per example, writing an 8-bit PPM to `examples/output/`
  so it needs no dependencies to view.
- The test image is a Macbeth-style chart built in code. **No real image
  inputs in this crate, ever.** That rule is about committed assets;
  `tools/kernel-viewer/` loads images at runtime and its `testdata/` is
  gitignored. Three layers enforce it: `.gitignore` ignores every raster
  and RAW extension, CI fails on any tracked image file (so a
  `git add -f` is caught), and `Cargo.toml`'s `exclude` keeps one out of
  the crate even then.
- A doc comment saying what the example demonstrates and what to look for
  in the output.
- A new kernel gets a new example. A kernel with a GPU path gets a twin
  in the same number series, gated on `required-features = ["cuda"]`,
  printing its agreement with the CPU kernel and naming the file to diff
  against.

CI runs every example without `required-features`. The GPU ones run only
through `scripts/gpu-verify.sh`, since the hosted runner has neither
`nvcc` nor a device.

## Type stubs

`python/phaios_core/__init__.pyi` (top level) and `python/phaios_core/
gpu.pyi` (the optional CUDA submodule) ship the type information the
compiled extension itself carries none of. They live in `python/`
because the mixed layout (`python-source = "python"` in
`pyproject.toml`) is the only way maturin supports a submodule stub —
its pure-Rust layout supports exactly one stub file, at the package
root (maturin issue #2507).

**The mirror rule:** every signature, default and docstring in the stub
is copied verbatim from the runtime module. This is checked, not just
asked for — `tests/stub_contract.py` is a stdlib-only (`ast` +
`inspect`) checker run through `tests/ffi.py` and `tests/ffi_gpu.py`.
It names every drift it finds and quotes the exact runtime text to
paste back into the stub, so fixing a failure is copy-paste, not
guesswork.

Two more checks run in CI and `scripts/gpu-verify.sh`, on top of the
contract test:

```sh
mypy --strict python/phaios_core
python -m mypy.stubtest phaios_core --allowlist tests/stubtest-allowlist.txt
```

`mypy --strict` type-checks the stubs on their own terms (an unimported
name, a bad `NDArray` parameter, a property/setter type clash).
`stubtest` cross-checks every stub signature against the built module
directly, catching things the contract test does not look at, such as
PyO3's exact `__eq__`/`__ne__` parameter shape.

**Allowlist policy** (`tests/stubtest-allowlist.txt`): entries are for
PyO3 artefacts only — a fact about how PyO3 compiles this extension
that cannot be expressed in a `.pyi` file, never a shortcut around real
drift. Every entry carries a one-line comment saying why it's there.
If `stubtest` reports something and you're not sure whether it's a
PyO3 artefact or a genuine stub error, fix the stub first; allowlist
only what turns out to be unfixable.

## Dependencies

Every new dependency is a supply-chain decision. Ask before adding one.

- **crates.io only**, pinned via the committed `Cargo.lock`.
- **Adding one requires** licence, primary source URL, maintainer and a
  justification, recorded as a comment in `Cargo.toml` beside the dep.
- **Prefer std > established crate > new dep.** "Established" means >1M
  downloads, active maintenance, and use by at least one major project.
- **`cargo audit`** is blocking in CI. Locally it is a separate binary
  (`cargo install cargo-audit`); CI uses the `rustsec/audit-check`
  action, so a fresh clone will not have the command.

Core dependencies, not to be exceeded without justification: `pyo3`,
`numpy`, `ndarray`, `rayon`, `thiserror`, plus `cudarc` behind
`--features cuda`.

`rand` and `rand_distr` were considered for film grain and **rejected**:
the shipped kernel hashes pixel coordinates with splitmix64, which is
reproducible across thread counts and portable to the GPU bit-for-bit,
neither of which a stateful RNG can promise.

Two version traps:

- `ndarray`'s version must match the one `numpy` pulls in — check
  `cargo tree | grep ndarray` after touching `numpy` — with its `rayon`
  feature enabled.
- `criterion::black_box` is deprecated; use `std::hint::black_box`.

## Commits and branches

- Conventional commits with optional scope: `feat(bw):`, `fix(zone):`,
  `test:`, `docs:`, `bench:`, `chore:`.
- One logical change per commit.
- `main` is always green; feature work on `feat/<slug>`.
- The crate and the Python wheel always carry the same version, and CI
  enforces the match.

Releasing is documented under **Releasing** in `README.md`, including the
pre-tag `scripts/gpu-verify.sh` run — nothing in CI builds
`--features cuda`, so nothing there would notice a CUDA backend that
fails to compile.

## Testing against a GPU

The CUDA backend needs the toolkit at build time and a device at run
time. Neither is present on the hosted CI runner, so verification is
something a contributor with a device does locally:

```sh
./scripts/gpu-verify.sh
```

It runs exactly the targets CI skips, under `PHAIOS_REQUIRE_GPU=1` so a
green result cannot mean the suite quietly skipped for want of a device,
and prints a device/driver/version block at the end. Reporting that block
on an issue is genuinely useful — the project has one card to test on,
and a second is evidence it cannot generate itself.

`examples/23_gpu_selftest.rs` is the same idea at kernel granularity:
every device entry point against the CPU function that is its
specification, with the per-kernel tolerances from `docs/ffi.md` §6.
