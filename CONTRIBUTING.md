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
- **No fused or machine-specific float code.** The Rust in `src/` uses
  no `mul_add`, `algebraic_*`, `#[target_feature]`, runtime CPU
  detection (`is_*_feature_detected!`), SIMD intrinsics (`std::arch`,
  `core::arch`) or ndarray matrix products (`.dot(`,
  `general_mat_mul`), and no cargo config, workflow or `pyproject.toml`
  passes `target-cpu` or `target-feature`: the CPU is the
  specification, and every wheel computes at its target's fixed
  default CPU (baseline x86-64 on Linux, x86-64 plus SSE3 and
  cmpxchg16b on Windows, Apple M1 on macOS), with no per-machine
  dispatch. `scripts/determinism-guards.sh` checks this in CI and in
  the release gate; run it with `bash`. In `src/` it skips only lines
  whose first non-blank characters are `//` (`src/` has no block
  comments), so a comment after code is still checked. It exits 1 on a
  hit, and 2 when the root it is given is missing or has no `src/`.

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
   the same channel count so the two ids can be divided. A new CUDA file
   also needs an entry in `build.rs`'s `PRECISION` table (the build
   fails without one): strict unless a whole-build measurement in FMA
   or fast math shows a gain worth more than about 0.04 ms at 24 MP for
   at most one 8-bit code against the CPU, with an error that stays
   within a fixed (rtol, atol) bound at every input and parameter
   (`vignette`'s grew with input × amount / feather) and is at most
   about 10× the strict build's difference from the CPU (`film_grain`'s
   was about 45×).
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
  inputs in this crate, ever.** That rule is about committed assets, and
  it has one exception: the maintainer's test photographs in
  `tools/kernel-viewer/testdata/`, with the camera's own JPEG where the
  RAW embeds one, tracked for the viewer and for measurements on real
  images and never packaged (their README gives the terms). Three layers
  enforce the rule: `.gitignore` ignores every raster and RAW extension
  and re-includes only the normalised names there
  (`photoNN.<dng|arw|nef>`, `photoNN.camera.jpg`), CI fails on any
  tracked image file not named, with a matching checksum, in that
  README's list (so a `git add -f` is caught), and `Cargo.toml`'s
  `exclude` keeps every image, the photographs included, out of the
  crate and the sdist; `.gitattributes` keeps the photographs out of
  source archives as well.
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

Two more checks run in CI, and the second also in
`scripts/gpu-verify.sh`, on top of the contract test:

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
- **`cargo audit`** is blocking in CI. It is a separate binary, so a
  fresh clone will not have the command: install it with
  `cargo install cargo-audit --version 0.23.0 --locked`, which is
  exactly what CI runs.

Core dependencies, not to be exceeded without justification: `pyo3`,
`numpy`, `ndarray`, `rayon`, `thiserror`, plus `cudarc` behind
`--features cuda`.

The platform math library is a dependency too, and the one that can
make the same wheel give different bits on different machines. Every
function the extension imports from it is listed, with the kernels
that reach it, in `tests/libm-imports.txt`, and CI and the release
gate fail when that set changes on Linux, the one platform the check
covers (`cargo build --release --lib && bash scripts/libm-inventory.sh
target/release/libphaios_core.so`). A new transcendental in a kernel
(`exp`, `powf`, `ln`, `sin`, …) is a deliberate decision: add its
symbol and call sites to the ledger, per pixel or once per call, and to
the list in `docs/ffi.md` §6, which CI does not compare and is kept in
step by hand.

`ndarray` carries a hazard of the same kind: its 2-D `.dot()` and
`general_mat_mul` run `matrixmultiply`'s sgemm, which picks an FMA
kernel at run time on a CPU that has one, so
`scripts/determinism-guards.sh` forbids both in `src/`.

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

## CI workflows

`.github/workflows/` holds four: `ci.yml` (every push to `main` and
every pull request), `wheels.yml` (builds and smoke-tests the
artefacts a release publishes; on a push or pull request that touches
the packaging, and from `release.yml`), `release.yml` (version tags)
and `gpu.yml` (by hand only, on a self-hosted CUDA runner that does not
exist yet). All four follow the same rules:

- `permissions: {}` at the top, and each job grants itself only what
  it needs: `contents: read`, and `id-token: write` for the PyPI upload
  alone.
- Every `uses:` names a full commit SHA, with the release tag in a
  comment (`actions/checkout@3d3c42e… # v7.0.1`). To move one, find the
  new tag's commit with `git ls-remote --tags
  https://github.com/<owner>/<repo>` (for an annotated tag, the line
  ending in `^{}`), read the release notes, and change the SHA and the
  comment together.
- Every `actions/checkout` sets `persist-credentials: false`, so the
  job's token is not left in the checkout's git config, where a later
  step or an uploaded artefact could pick it up.
- Every cargo command that resolves dependencies passes `--locked`, in
  the workflows and in `scripts/gpu-verify.sh`: a stale `Cargo.lock`
  fails rather than being rewritten on the runner.
- Tools are pinned too: maturin 1.16.0 in `wheels.yml`, cargo-audit
  0.23.0 in `ci.yml`. The Rust toolchain is `stable` of the day,
  installed with `rustup`.

`scripts/wheel-smoke.sh` is what `wheels.yml` runs on each wheel and on
the sdist; it runs locally too, on a wheel from `maturin build`, with
the virtual environment and uv's cache inside `.cache/`:

```sh
maturin build --release --locked -o .cache/scratch/dist
UV_CACHE_DIR=.cache/uv bash scripts/wheel-smoke.sh \
    .cache/scratch/dist/phaios_core-*.whl 3.12 .cache/venvs/smoke
```

## Releasing

The crate version in `Cargo.toml` and the wheel version in
`pyproject.toml` are always identical, and CI enforces the match. Bump
both, and `Cargo.lock`, in one commit.

Before tagging, verify the CUDA backend on a machine that has a device,
with a current `maturin develop --release --features cuda` build in
`.phaios-venv` so the Python step cannot be skipped:

```sh
PHAIOS_REQUIRE_GPU=1 ./scripts/gpu-verify.sh
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

`.github/workflows/release.yml` fires only on a tag of the form
`vMAJOR.MINOR.PATCH`. A pre-release tag such as `v0.2.0-rc1` does not
match and publishes nothing, so there is no release-candidate step: do
not push one. The workflow starts with a `verify` job: the tag must
match both versions in the tree, and `cargo fmt`, `cargo clippy
--all-targets`, `cargo test`, the libm ledger, the determinism guards
and the no-image-assets guard must pass on the tagged commit. Only then
does it call `wheels.yml`, which builds wheels for manylinux_2_17
x86_64, Windows x86_64 and macOS arm64 and the sdist; installs each
wheel into a new virtual environment on its own OS, under Python 3.12
and 3.15 (and on Linux under 3.12 with numpy 1.26, the declared floor),
and runs `scripts/wheel-smoke.sh` there; builds the sdist from source
the same way; and checks each wheel's SBOM against `cargo metadata`.
If every one of those jobs passes, it publishes to PyPI, and only once
PyPI has succeeded to crates.io. Nothing is built until `verify` is
green, and nothing is published until every artefact has been
installed and tested, because neither registry lets a version be
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

It runs the targets CI skips, with the conformance suite under
`PHAIOS_REQUIRE_GPU=1` so a green result cannot mean the suite quietly
skipped for want of a device, and prints a block at the end with the
device, driver, backend fingerprint and full nvcc release. It keeps the
driver's JIT cache in `.cache/cuda-jit` unless `CUDA_CACHE_PATH` is
already set. Its PTX audit prints each file's precision counts and
fails if any PTX in the build the tests used came from another nvcc
than the one it finds now: after a toolkit change cargo could not see,
run `touch src/cuda/ptx`, which makes `build.rs` re-run in every
profile and under maturin. Its Python step runs against `.phaios-venv`
and needs a CUDA build installed there (`maturin develop --release
--features cuda`). Without `.phaios-venv`, without `phaios_core` in it,
or with a CPU-only build that has no `gpu` submodule, the step is
skipped and the block says so; a `phaios_core` that fails to import (a
broken build) or a CUDA build whose fingerprint differs from the
tree's fails the run. For a report or a release check, run it as
`PHAIOS_REQUIRE_GPU=1 ./scripts/gpu-verify.sh`: with the variable set
to any value, every skip is a failure. What it runs, step by step, is
in [docs/gpu.md](docs/gpu.md#verifying-on-your-gpu). Reporting that
block on an issue adds a row to the confirmed-device table in
[docs/gpu.md](docs/gpu.md#confirmed-on-other-gpus).

`examples/23_gpu_selftest.rs` is the same idea at kernel granularity:
every device entry point against the CPU function that is its
specification, with the per-kernel tolerances from `docs/ffi.md` §6.

## Checking that a change moved no output bits

`scripts/kernel_digests.py` runs every kernel, plus the README
pipeline, over fixed test images it generates itself, and writes one
JSON line per case with the SHA-256 of the output. It then compares two
such runs case by case. It needs only numpy. Use it whenever a change
should leave outputs alone (a dependency or toolchain upgrade, a
refactor, a build flag), and to measure a change that is expected to
move them: another libm, OS or machine, or the CUDA backend against the
CPU. The script's docstring is the reference for every option and exit
code.

```sh
python scripts/kernel_digests.py selftest
python scripts/kernel_digests.py run --out .cache/scratch/before.jsonl   # old build
python scripts/kernel_digests.py run --out .cache/scratch/after.jsonl    # new build
python scripts/kernel_digests.py compare .cache/scratch/before.jsonl .cache/scratch/after.jsonl --expect-identical
```

Within one backend, case lines must be byte-identical at any
`RAYON_NUM_THREADS`; CI checks that at `--size 64`. `compare` exits 1
when `--expect-identical` finds a difference, and 2 when nothing can be
compared (for example a run whose setup failed, an empty selection, or
a case whose input differs between the runs). `--write-ids FILE` saves
the ids that are not identical and `run --ids FILE` re-runs only those;
`--label` records free text, such as the commit built, and
`--extra-channels` adds C = 2 and C = 4 cases.

`--gpu` runs the same cases through `phaios_core.gpu`. With both
backends run under `--dump DIR`, `compare --dumps DIR_A DIR_B` gives
element-level detail (differing elements, ULP distances, changed u8 and
u16 codes). `--expect-bounded` (one CPU and one GPU run) also judges
each differing case against its kernel's class and (rtol, atol) in
`docs/ffi.md` §6, and exits 1 on a failure whose id `--allow REGEX`
does not match:

```sh
python scripts/kernel_digests.py run --size 256 --dump .cache/scratch/cpu --out .cache/scratch/cpu.jsonl
python scripts/kernel_digests.py run --size 256 --gpu --cuda-cache "$PWD/.cache/cuda-jit" \
    --dump .cache/scratch/gpu --out .cache/scratch/gpu.jsonl
python scripts/kernel_digests.py compare .cache/scratch/cpu.jsonl .cache/scratch/gpu.jsonl \
    --dumps .cache/scratch/cpu .cache/scratch/gpu --expect-bounded
```

`sharpen`'s tolerance is scaled by its blur (`docs/ffi.md` §6), which
`compare` computes from the case's input and the CPU blur of the
`phaios_core` it imports.

Across backends only the bit-exact kernels should match, and the six
of them that do arithmetic on a NaN (`exposure`, `luminance_bw`,
`channel_mixer_bw`, `color_filter_bw`, `resize`, `straighten`) differ
in NaN payload and sign, because GPU arithmetic yields `0x7fffffff`;
kernels that pass a NaN through unchanged match. A new kernel adds its
cases to `catalog()` in the script.
