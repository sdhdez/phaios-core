# Contributing to phaios-core

Setting up a development environment is covered by **Build from
source** in `README.md`. This file is the process around a change:
conventions, what a complete kernel looks like, the rules for
dependencies, and how a release is cut.

The other documents: [docs/kernels.md](docs/kernels.md) says what each
kernel does to the image, [docs/architecture.md](docs/architecture.md)
why it works that way, with citations,
[docs/ffi.md](docs/ffi.md) the Python/Rust boundary contract,
[docs/gpu.md](docs/gpu.md) the CUDA backend, and
[docs/export.md](docs/export.md) what a finished image must be in a
file.

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

## What a complete kernel looks like (the eight-item checklist)

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
8. A section in `docs/kernels.md`: shape contract, visible effect, every
   parameter with its identity value, order constraints, backend class.

### Writing the property test

Prefer properties that fail loudly when the maths is wrong: band
ordering, exposure invariance, round-trip identity, a signed rather than
an absolute comparison.

Two failure modes to avoid:

- A test that only checks the output shape passes just as happily on a
  kernel that returns its input.
- A test written on a *constant* image passes on a kernel that ignores
  its parameters entirely — on a flat field the guided filter's detail
  term is identically zero, so the whole edge-preserving model could be
  arbitrarily wrong and six tests still passed.

When strengthening a weak test, break the code deliberately and confirm
the new test fails. Check the mutation is *live* first: one that leaves
the output unchanged proves nothing.

`tests/properties.rs` is the other half of this: property tests over a
kernel's *validation* contract — the shared `validate*` function every
kernel and its CUDA twin both call — generated across the whole input
domain rather than picked by hand. Add a property there when you touch a
`validate*` function or change its documented domain; add an
example-based test here, or beside the kernel, when the point is a
specific numerical claim (a coefficient, a round-trip, a signed
comparison) that a random search would find by accident or not at all.

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

Every dependency that links into the library is permissively licensed
and GPLv3-compatible. Of the 36 such crates (build and runtime,
excluding dev-only tooling), 32 are `MIT OR Apache-2.0` in some
spelling and four are not:

| Crate | Licence | Reached via |
|---|---|---|
| `numpy` | BSD-2-Clause | direct dependency |
| `libloading` | ISC | `cudarc`, only with `--features cuda` |
| `target-lexicon` | Apache-2.0 WITH LLVM-exception | build graph |
| `unicode-ident` | (MIT OR Apache-2.0) AND Unicode-3.0 | build graph |

BSD-2-Clause, ISC and the Unicode licence are permissive and impose no
condition GPLv3 cannot satisfy; the LLVM exception only widens
Apache-2.0. Regenerate the table with `cargo metadata` after any
dependency change.

**Dev-only:** `criterion` (statistics-driven micro-benchmarking,
`benches/`) and `proptest` (strategies and shrinking over the
`validate*` functions' whole input domain, `tests/properties.rs`) — both
dev-dependencies only, never shipped in the published crate or wheel.

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

## Releasing

The crate version in `Cargo.toml` and the wheel version in
`pyproject.toml` are always identical, and CI enforces the match. Bump
both, and `Cargo.lock`, in one commit.

Before tagging, verify the CUDA backend on a machine that has a device:

```sh
./scripts/gpu-verify.sh
```

Neither `ci.yml` nor `release.yml` builds `--features cuda`, because the
hosted runner has no `nvcc` and no GPU. Nothing else in the pipeline
would notice a backend that fails to compile, while `cargo publish`
ships the source to crates.io regardless. The wheels are built without
the feature and are unaffected either way.

Then tag:

```sh
git tag v0.2.0
git push origin v0.2.0   # triggers release.yml
```

`.github/workflows/release.yml` fires on any `v*` tag and starts with a
`verify` job: the tag must match both versions in the tree, and
`cargo fmt`, `cargo clippy --all-targets`, `cargo test` and the
no-image-assets guard must pass on the tagged commit. Only then does it
build wheels for manylinux_2_17, Windows x86_64 and macOS arm64, build
the sdist, and publish to PyPI and crates.io. Nothing is built until
`verify` is green, because neither registry lets a version be
re-uploaded.

One-time setup, already in place for this repository:

- **crates.io**: `CARGO_REGISTRY_TOKEN` in *Settings, Secrets,
  Actions*. The token needs both `publish-new` and `publish-update`
  scopes; one missing `publish-update` returns 403 on every version
  after the first.
- **PyPI**: OIDC trusted publishing, no stored token. In the PyPI
  project, *Manage, Publishing, Add a new publisher*, with owner
  `sdhdez`, repository `phaios-core`, workflow `release.yml`,
  environment `pypi`.

## Testing against a GPU

The CUDA backend needs the toolkit at build time and a device at run
time. Neither is present on the hosted CI runner, so verification is
something a contributor with a device does locally:

```sh
./scripts/gpu-verify.sh
```

It runs the targets CI skips, under `PHAIOS_REQUIRE_GPU=1` so a green
result cannot mean the suite quietly skipped for want of a device, and
prints a device/driver/version block at the end. What it runs, step by
step, is in [docs/gpu.md](docs/gpu.md#verifying-on-your-gpu). Reporting
that block on an issue adds a row to the confirmed-device table in
[docs/gpu.md](docs/gpu.md#confirmed-on-other-gpus).

`examples/23_gpu_selftest.rs` is the same idea at kernel granularity:
every device entry point against the CPU function that is its
specification, with the per-kernel tolerances from `docs/ffi.md` §6.
