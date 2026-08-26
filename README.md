# phaios-core

Numerical kernels for black-and-white RAW image processing.

The name is from φαιός (Greek, "dusky grey"), the term Aristotle uses
in *De Sensu* for the intermediate colours between white and black.

[![CI](https://github.com/sdhdez/phaios-core/actions/workflows/ci.yml/badge.svg)](https://github.com/sdhdez/phaios-core/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## What it does

A Rust crate with PyO3 bindings (`phaios_core` Python module). Pure
functions on `f32` linear scene-referred `(H, W, C)` arrays — no I/O,
no GUI, no hidden state. Any front-end can build on it.

An **optional CUDA backend** (`--features cuda`) runs every kernel on an
NVIDIA GPU through a second, device-resident API. It is strictly
additive: without the feature nothing changes, and no CUDA toolkit is
needed to build or use the crate.

### Kernels

In pipeline order. Every kernel takes any array layout and returns a
freshly allocated C-contiguous array.

| Kernel | Description | Since |
|--------|-------------|-------|
| `orient` | The eight Exif dihedral transforms (rotations, flips) | v0.2 |
| `straighten` | Small-angle rotation (±45°) with inscribed-rectangle crop | v0.2 |
| `crop` | Exact rectangle extraction | v0.2 |
| `resize` | Separable resampling: area / bilinear / Catmull-Rom | v0.2 |
| `exposure` | Exposure compensation in EV stops | v0.2 |
| `luminance_bw` | Standard B&W conversion: BT.601, BT.709 (default), BT.2020 | v0.1 |
| `channel_mixer_bw` | Arbitrary RGB weights in −2..+2 (infrared-like effects) | v0.1 |
| `color_filter_bw` | Wratten-style filter simulation (Yellow, Orange, Red, Green, Blue) | v0.1 |
| `hsl_bw` | Per-hue weighting across 8 bands, Gaussian-blended | v0.2 |
| `zone_system` | Adams/Archer Zone System tone curve, 11 zones, Gaussian-blended | v0.1 |
| `blur` | Separable Gaussian; direct below σ 6, three box passes above | v0.2 |
| `local_contrast` | He–Sun–Tang guided filter for local contrast enhancement | v0.1 |
| `film_grain` | Band-passed procedural grain, deterministic from an explicit seed | v0.2 |
| `split_toning` | Shadow/highlight tinting in OKLab — returns `(H, W, 3)` | v0.2 |
| `vignette` | Radial darkening or lightening, resolution-independent | v0.2 |
| `tone_curve` | Parametric slope/offset/power curve (ASC CDL) | v0.2 |
| `shadow_rolloff` | Cubic shadow toe; with the shoulder, the characteristic curve | v0.2 |
| `highlight_rolloff` | Bézier highlight shoulder; defaults to a hard clip | v0.2 |
| `encode_srgb` | IEC 61966-2-1 sRGB transfer encoding | v0.1 |
| `quantize_u8` / `quantize_u16` | Dithered integer conversion (terminal) | v0.2 |
| `apply_lut` | Arbitrary tone transfer through a 1-D table | v0.2 |
| `histogram` | Per-channel counts + clipping tallies (a *reduction*) | v0.2 |

### GPU backend (optional)

Build with `--features cuda` (needs the CUDA toolkit at build time and a
driver at run time) and the module gains a `phaios_core.gpu` submodule:

```python
import numpy as np, phaios_core as ph
from phaios_core import gpu

ctx = gpu.GpuContext(0)              # explicit — no global device state
img = gpu.GpuImage(ctx, array)       # upload once
img = gpu.exposure(img, 0.5)         # chain on-device, no round trips
img = gpu.luminance_bw(img)
out = img.to_numpy()                 # download once
```

Determinism is promised **per backend**: bit-identical output within a
backend at any thread count or launch geometry, and bounded across them.
Kernels free of transcendentals — including all four geometry kernels —
are bit-exact across CPU and GPU as well; `tests/cuda_conformance.rs`
asserts this with `assert_eq!` and skips cleanly when no device is
present. See [docs/ffi.md](docs/ffi.md) §6 for the exact contract.

### Not in scope

Pixel-level local adjustment (masks, U-Point-style edit propagation) and
the Newson et al. (2017) stochastic grain model are research territory,
deferred past v0.2. RAW decoding, file I/O, settings management and
anything with a user interface belong to consumers, permanently.

---

## Build prerequisites

- **Rust** stable toolchain (`rustup default stable`)
- **Python** 3.12 or later (3.13 recommended for performance)
- **maturin** (installed via requirements-dev.txt)
- *Optional, for `--features cuda`:* the CUDA toolkit (`nvcc`, which
  compiles the kernels to PTX at build time) and an NVIDIA driver. The
  driver library is loaded dynamically, so a CUDA-enabled build still
  runs on a machine without a GPU — it reports no devices rather than
  failing to load.

## Quick start

```sh
# Clone and set up
git clone https://github.com/sdhdez/phaios-core
cd phaios-core

# Python environment
uv venv .phaios-venv
source .phaios-venv/bin/activate   # Windows: .phaios-venv\Scripts\activate
uv pip install -r requirements-dev.txt

# Build and install the Python extension
maturin develop --release

# Run all tests
cargo test
pytest

# Run an example (writes examples/output/01_luminance.ppm)
cargo run --example 01_luminance
```

---

## Usage (Python)

```python
import numpy as np
import phaios_core as ph

# Linear scene-referred f32 RGB, shape (H, W, 3)
img = np.random.rand(1080, 1920, 3).astype(np.float32)

# 1. Exposure, in EV stops
x = ph.exposure(img, 0.5)

# 2. B&W conversion. Four methods; this one weights by hue —
#    bands are red, orange, yellow, green, aqua, blue, purple, magenta.
x = ph.hsl_bw(x, ph.HslWeightedParams([0, 0, 0.4, 0.2, 0, -0.5, 0, 0]))

# 3. Zone System tone curve
x = ph.zone_system(x, ph.ZoneParams({3: -0.3, 7: 0.4}))

# 4. Local contrast
x = ph.local_contrast(x, ph.GuidedFilterParams(radius=8, eps=0.01), strength=0.4)

# 5. Finishing. Grain is deterministic: same seed, same bytes.
x = ph.film_grain(x, ph.GrainParams(intensity=0.12, size_pixels=1.5, seed=20260815))
x = ph.split_toning(x, ph.SplitToningParams([0.0, -0.02, -0.04],   # cool shadows
                                            [0.0,  0.03,  0.03]))  # warm highlights
x = ph.vignette(x, ph.VignetteParams(amount=0.35, feather=0.8))
x = ph.tone_curve(x, ph.ToneCurveParams(slope=1.1, power=0.9))

# 6. Finish. The roll-off replaces the clip a caller used to write by
#    hand: its default is a hard clip, so this is explicit, not new.
x = ph.highlight_rolloff(x, ph.RolloffParams(knee=0.75, white_point=4.0))
x = ph.encode_srgb(x)

# 7. Terminal: continuous -> integer codes. 16-bit is the archival
#    default; 8-bit should be dithered. See docs/export.md.
output = ph.quantize_u16(x)
```

Each stage is optional and each is a pure function; skip any of them and
the rest still compose. `split_toning` is the one that changes shape,
taking `(H, W, 1)` to `(H, W, 3)` — the kernels after it accept either.

---

## Licence

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

All dependencies are permissively licensed and GPLv3-compatible, but
they are not all `MIT OR Apache-2.0` as an earlier version of this
sentence claimed. Of the 36 crates that actually link into the library
(build and runtime, excluding dev-only tooling such as criterion), 32 are
`MIT OR Apache-2.0` in some spelling and four are not:

| Crate | Licence | Reached via |
|---|---|---|
| `numpy` | BSD-2-Clause | direct dependency |
| `libloading` | ISC | `cudarc`, only with `--features cuda` |
| `target-lexicon` | Apache-2.0 WITH LLVM-exception | build graph |
| `unicode-ident` | (MIT OR Apache-2.0) AND Unicode-3.0 | build graph |

BSD-2-Clause, ISC and the Unicode licence are permissive and impose no
condition GPLv3 cannot satisfy; the LLVM exception only widens
Apache-2.0. Regenerate this table with `cargo metadata` after any
dependency change — the claim is checkable, so it should be checked
rather than assumed.

### Algorithm provenance

Every kernel cites the paper, textbook or standard it implements; the
citations were checked against primary sources rather than from memory.
Two kernels carry a fuller **provenance note** in their module
documentation, because the licence of a *reference implementation* is a
separate question from the licence of a *paper*:

- **`local_contrast`** (`src/local_contrast.rs`) — the guided filter.
  The authors' own MATLAB is restricted to non-commercial academic use
  and **must not be ported**; this crate is an independent
  reimplementation from the published equations. The note also records
  which patents were searched and read against the kernel.
- **`split_toning`** (`src/split_toning.rs`) — OKLab. The coefficients
  come verbatim from Ottosson's reference implementation, offered as MIT
  or public domain; this crate elects the public-domain branch, and says
  so rather than relying on a citation to discharge a notice obligation.

Nothing found in that review constrains use or distribution of this
crate. It is a record of what was checked, not legal advice.
Corresponding source is available at the repository URL above
(satisfies GPLv3 §6(d)).

---

## Architecture & FFI contract

- [docs/architecture.md](docs/architecture.md) — pipeline diagram,
  full mathematical derivations, algorithm citations.
- [docs/ffi.md](docs/ffi.md) — Python↔Rust boundary contract,
  array layout, zero-copy rules, error handling.
- [docs/export.md](docs/export.md) — the export contract: what a
  finished phaios image is, normatively. Grey means grey (single-channel
  photometric, greyscale ICC), 16-bit archival default, dither rules,
  and the terminal stage sequence.

---

## Releasing

Version history is in [CHANGELOG.md](CHANGELOG.md).

The release workflow (`.github/workflows/release.yml`) fires on any
`v*` tag. It first runs a `verify` job — the tag must match the version
in both `Cargo.toml` and `pyproject.toml`, and `fmt`, `clippy` and the
test suite must pass on the tagged commit — and only then builds wheels
for manylinux_2_17, Windows x86_64 and macOS arm64 and publishes them.
Nothing is built until `verify` is green, because neither PyPI nor
crates.io allows a version to be re-uploaded.

```sh
# Bump Cargo.toml, pyproject.toml and Cargo.lock in one commit, then:
git tag v0.2.0
git push origin v0.2.0   # triggers release.yml
```

One-time setup, already in place for this repository:

**crates.io** — `CARGO_REGISTRY_TOKEN` in *Settings → Secrets →
Actions*. The token needs both `publish-new` and `publish-update`
scopes; one missing `publish-update` returns 403 on every version after
the first.

**PyPI** — OIDC trusted publishing, no stored token. In the PyPI
project: *Manage → Publishing → Add a new publisher*, with owner
`sdhdez`, repository `phaios-core`, workflow `release.yml`, environment
`pypi`.
