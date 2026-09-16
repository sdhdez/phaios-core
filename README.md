# phaios-core

Numerical kernels for black-and-white RAW image processing: 27 pure
functions on linear, scene-referred `f32` images, as a Rust crate with
Python bindings. No I/O, no GUI, no hidden state, so any front end can
build on it. An optional CUDA backend runs every kernel on an NVIDIA
GPU through a second, device-resident API.

The name is from φαιός (Greek, "dusky grey"), the term Aristotle uses
in *De Sensu* for the intermediate colours between white and black.

[![CI](https://github.com/sdhdez/phaios-core/actions/workflows/ci.yml/badge.svg)](https://github.com/sdhdez/phaios-core/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## Install

```sh
pip install phaios-core     # Python, CPU backend
```

```sh
cargo add phaios-core       # Rust, CPU backend
```

The published wheel is `cp312-abi3`: one build serves every CPython
3.12 or later that has the GIL. Free-threaded interpreters (3.13t,
3.14t) are not covered, because PyO3 ignores `abi3` when building
against them; `pip install` there compiles the sdist and needs a Rust
toolchain.

The wheel is CPU-only. The CUDA backend is a source build with
`--features cuda`, needs `nvcc` at build time and an NVIDIA driver at
run time, and is described in
[docs/gpu.md](https://github.com/sdhdez/phaios-core/blob/main/docs/gpu.md).

## Kernels

All 27, in pipeline order. Every kernel takes any array layout and
returns a freshly allocated C-contiguous array. What each one does to
the image, with its parameters and identity values, is in
[docs/kernels.md](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md).

| Kernel | Description | Since |
|---|---|---|
| [`orient`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#orient) | The eight Exif dihedral transforms (rotations, flips) | v0.2 |
| [`hot_pixels`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#hot_pixels) | Conditional 3x3 median for stuck or hot sensor pixels | v0.2 |
| [`denoise`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#denoise) | Guided-filter noise reduction, self- and cross-guided | v0.2 |
| [`straighten`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#straighten) | Small-angle rotation with inscribed-rectangle crop | v0.2 |
| [`crop`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#crop) | Exact rectangle extraction | v0.2 |
| [`resize`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#resize) | Separable resampling: area, bilinear, Catmull-Rom | v0.2 |
| [`exposure`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#exposure) | Exposure compensation in EV stops | v0.2 |
| [`luminance_bw`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#luminance_bw) | Standard B&W conversion: BT.601, BT.709 (default), BT.2020 | v0.1 |
| [`channel_mixer_bw`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#channel_mixer_bw) | Arbitrary RGB weights, negatives allowed (infrared-like) | v0.1 |
| [`color_filter_bw`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#color_filter_bw) | Six presets: the unfiltered reference and five Wratten filters | v0.1 |
| [`hsl_bw`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#hsl_bw) | Per-hue weighting across 8 bands, Gaussian-blended | v0.2 |
| [`zone_system`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#zone_system) | Adams/Archer Zone System tone curve, 11 zones | v0.1 |
| [`blur`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#blur) | Separable Gaussian: direct below sigma 6, three box passes above | v0.2 |
| [`glow`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#glow) | Light scattering: halation, diffusion, veiling glare | v0.2 |
| [`local_contrast`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#local_contrast) | He-Sun-Tang guided filter for local contrast | v0.1 |
| [`sharpen`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#sharpen) | Threshold-gated Gaussian unsharp mask | v0.2 |
| [`shadow_rolloff`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#shadow_rolloff) | Cubic shadow toe; with the shoulder, the characteristic curve | v0.2 |
| [`tone_curve`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#tone_curve) | Parametric slope/offset/power curve (ASC CDL) | v0.2 |
| [`film_grain`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#film_grain) | Band-passed procedural grain, deterministic from an explicit seed | v0.2 |
| [`split_toning`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#split_toning) | Shadow and highlight tinting in OKLab; returns `(H, W, 3)` | v0.2 |
| [`vignette`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#vignette) | Radial darkening or lightening, resolution-independent | v0.2 |
| [`highlight_rolloff`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#highlight_rolloff) | Bezier highlight shoulder; defaults to a hard clip | v0.2 |
| [`encode_srgb`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#encode_srgb) | IEC 61966-2-1 sRGB transfer encoding | v0.1 |
| [`apply_lut`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#apply_lut) | Arbitrary tone transfer through a 1-D table | v0.2 |
| [`histogram`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#histogram) | Per-channel counts and clipping tallies (a reduction) | v0.2 |
| [`quantize_u8`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#quantize_u8) | Dithered conversion to `uint8` (terminal) | v0.2 |
| [`quantize_u16`](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md#quantize_u16) | Dithered conversion to `uint16` (terminal) | v0.2 |

### The pipeline

```
 C  stage
 3  orient -> [hot_pixels] -> [denoise] -> straighten -> crop -> resize
 3  exposure
 3  luminance_bw | channel_mixer_bw | color_filter_bw | hsl_bw
 |  -------------- 3 -> 1 --------------
 1  zone_system -> [blur] -> [glow] -> local_contrast -> [sharpen]
 1  shadow_rolloff -> tone_curve -> film_grain
 1  split_toning
 |  -------------- 1 -> 3 --------------
 3  vignette -> highlight_rolloff
 3  encode_srgb                  (the only display-referred stage)
 3  quantize_u8 | quantize_u16   -> uint8 / uint16, terminal

    [brackets] = optional. Everything from vignette on accepts any C,
    so skipping split_toning leaves a 1-channel image and still runs.
    apply_lut: any point after the B&W stage.
    histogram: a reduction, callable anywhere.
```

### Not in scope

Pixel-level local adjustment (masks, U-Point-style edit propagation)
and the Newson et al. (2017) stochastic grain model are research
territory, deferred past v0.2. RAW decoding, file I/O, settings
management and anything with a user interface belong to consumers,
permanently.

---

## Usage

```python
import numpy as np
import phaios_core as ph

# Linear scene-referred f32 RGB, shape (H, W, 3)
img = np.random.rand(1080, 1920, 3).astype(np.float32)

# 1. Exposure, in EV stops
x = ph.exposure(img, 0.5)

# 2. B&W conversion. Four methods; this one weights by hue. The bands
#    are red, orange, yellow, green, aqua, blue, purple, magenta.
x = ph.hsl_bw(x, ph.HslWeightedParams([0, 0, 0.4, 0.2, 0, -0.5, 0, 0]))

# 3. Zone System tone curve
x = ph.zone_system(x, ph.ZoneParams({3: -0.3, 7: 0.4}))

# 4. Local contrast
x = ph.local_contrast(x, ph.GuidedFilterParams(radius=8, eps=0.01), strength=0.4)

# 5. Tone stages, then grain and toning on the tones the viewer sees
x = ph.shadow_rolloff(x, ph.ShadowRolloffParams(knee=0.2, strength=0.5))
x = ph.tone_curve(x, ph.ToneCurveParams(slope=1.1, power=0.9))
x = ph.film_grain(x, ph.GrainParams(intensity=0.12, size_pixels=1.5, seed=20260815))
x = ph.split_toning(x, ph.SplitToningParams([0.0, -0.02, -0.04],   # cool shadows
                                            [0.0,  0.03,  0.03]))  # warm highlights

# 6. Finishing. highlight_rolloff decides what happens above 1.0; its
#    default is a hard clip.
x = ph.vignette(x, ph.VignetteParams(amount=0.35, feather=0.8))
x = ph.highlight_rolloff(x, ph.RolloffParams(knee=0.75, white_point=4.0))
x = ph.encode_srgb(x)

# 7. Terminal: continuous -> integer codes. 16-bit is the archival
#    default; 8-bit should be dithered. See docs/export.md.
output = ph.quantize_u16(x)
```

Each stage is optional and each is a pure function; skip any of them
and the rest still compose. `split_toning` is the one that changes
shape, taking `(H, W, 1)` to `(H, W, 3)`. The kernels after it accept
either.

### GPU

Build with `--features cuda` and the module gains a `phaios_core.gpu`
submodule. Upload once, chain on the device, download once:

```python
import numpy as np
from phaios_core import gpu

frame = np.random.rand(1080, 1920, 3).astype(np.float32)

ctx = gpu.GpuContext(0)        # explicit: no global device state
img = ctx.upload(frame)        # one PCIe transfer
img = gpu.exposure(img, 0.5)   # chained on-device, no round trips
img = gpu.luminance_bw(img)
out = img.download()           # one PCIe transfer back
```

`quantize_u8`, `quantize_u16` and `histogram` return host objects and
end the chain. Offload pays when the frame stays resident; a single
kernel per upload does not.

**No hosted CI runs the GPU suite.** GitHub's runners have neither
`nvcc` nor a device. The backend is verified by hand with
`./scripts/gpu-verify.sh`, most recently on an NVIDIA GeForce RTX 5070
Ti (cc 12.0, driver 615.71.09, nvcc 13.4, rustc 1.98.1, Linux). Build
instructions, the agreement classes, measured performance and the table
of confirmed devices are in
[docs/gpu.md](https://github.com/sdhdez/phaios-core/blob/main/docs/gpu.md).

### Type hints

The wheel is [PEP 561](https://peps.python.org/pep-0561/) typed: it
ships `py.typed` and full `.pyi` stubs for every function, class and
enum. mypy and pyright pick them up with no configuration.
`phaios_core.gpu` is typed too, but exists only in a `--features cuda`
build, so guard the import:

```python
try:
    from phaios_core import gpu
except ImportError:
    gpu = None
```

### Determinism

Same backend, same input, same parameters, same seed: the same bytes,
at any thread count and any launch geometry. Across backends the
difference is bounded, and the bound is committed per kernel. A backend
has a name: the CPU target triple, or `GpuContext.fingerprint`. A
consumer that promises exact reproduction should record it beside the
parameters. Kernels free of transcendentals are bit-exact across CPU
and GPU as well. The full contract is
[docs/ffi.md](https://github.com/sdhdez/phaios-core/blob/main/docs/ffi.md)
section 6.

### Errors

| Condition | Python exception |
|---|---|
| A parameter outside its domain | `ValueError` |
| A wrong shape or channel count | `ValueError` |
| An output above the 8 GiB single-allocation limit | `MemoryError` |
| No CUDA device, no driver, or a device below compute capability 8.0 | `RuntimeError` |

No kernel panics on anything a caller can pass, and none validates
pixel values: they must be finite. With a NaN or an infinity present
the result is unspecified.

---

## Build from source

- **Rust** 1.88 or later (`rustup default stable`), the crate's
  declared `rust-version`.
- **Python** 3.12 or later; 3.14 is what the crate is developed
  against.
- **maturin**, installed by `requirements-dev.txt`.
- Optional, for `--features cuda`: the CUDA toolkit (`nvcc` compiles
  the kernels to PTX at build time) and an NVIDIA driver. The driver
  library is loaded dynamically, so a CUDA-enabled build still runs on
  a machine without a GPU: `gpu.available()` is `False`,
  `gpu.devices()` is empty, and `gpu.GpuContext(...)` raises
  `RuntimeError`.

```sh
git clone https://github.com/sdhdez/phaios-core
cd phaios-core

uv venv .phaios-venv
source .phaios-venv/bin/activate   # Windows: .phaios-venv\Scripts\activate
uv pip install -r requirements-dev.txt

maturin develop --release          # add --features cuda for the GPU backend

cargo test
pytest

cargo run --example 01_luminance   # writes examples/output/01_luminance.ppm
```

`maturin develop` is an editable install: the compiled extension is
written to `python/phaios_core/` inside the checkout (gitignored) and
the virtual environment gets a `.pth` pointing there, so a rebuild is
picked up without reinstalling. Two virtual environments developing
from the same checkout overwrite each other's build; use one per
checkout.

---

## Documentation

| Document | What it covers |
|---|---|
| [docs/kernels.md](https://github.com/sdhdez/phaios-core/blob/main/docs/kernels.md) | What each of the 27 kernels does to the image: shapes, parameters, identity values, order |
| [docs/gpu.md](https://github.com/sdhdez/phaios-core/blob/main/docs/gpu.md) | The CUDA backend: build, device-resident chain, agreement classes, performance, verification |
| [docs/architecture.md](https://github.com/sdhdez/phaios-core/blob/main/docs/architecture.md) | Why each kernel works as it does: derivations and citations |
| [docs/ffi.md](https://github.com/sdhdez/phaios-core/blob/main/docs/ffi.md) | The Python/Rust boundary: layout, zero-copy, errors, the GIL, determinism |
| [docs/export.md](https://github.com/sdhdez/phaios-core/blob/main/docs/export.md) | What a finished phaios image must be in a file, normatively |
| [examples/README.md](https://github.com/sdhdez/phaios-core/blob/main/examples/README.md) | The 48 runnable examples, CPU and GPU |
| [CONTRIBUTING.md](https://github.com/sdhdez/phaios-core/blob/main/CONTRIBUTING.md) | Conventions, the kernel checklist, dependency rules, releasing |
| [CHANGELOG.md](https://github.com/sdhdez/phaios-core/blob/main/CHANGELOG.md) | Version history |

---

## Licence

GNU General Public License v3.0 or later. See
[LICENSE](https://github.com/sdhdez/phaios-core/blob/main/LICENSE).
Corresponding source is available at this repository (GPLv3 section
6(d)).

Every dependency is permissively licensed and GPLv3-compatible; the
list and the check are in
[CONTRIBUTING.md](https://github.com/sdhdez/phaios-core/blob/main/CONTRIBUTING.md#dependencies).

Two kernels carry a provenance note in their module documentation,
because the licence of a reference implementation is a separate
question from the licence of a paper. `local_contrast` is an
independent reimplementation from the guided filter's published
equations: the authors' own MATLAB is restricted to non-commercial
academic use and must not be ported. `split_toning` takes the OKLab
coefficients verbatim from Ottosson's reference implementation, offered
as MIT or public domain, and elects the public-domain branch.
