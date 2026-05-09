# phaios-core

Numerical kernels for black-and-white RAW image processing.

The name is from φαιός (Greek, "dusky grey"), the term Aristotle uses
in *De Sensu* for the intermediate colours between white and black.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

## What it does

A Rust crate with PyO3 bindings (`phaios_core` Python module). Pure
functions on `f32` linear scene-referred `(H, W, C)` arrays — no I/O,
no GUI, no hidden state. Any front-end can build on it.

### v0.1 feature set

| Kernel | Description |
|--------|-------------|
| `luminance_bw` | Standard B&W conversion: BT.601, BT.709 (default), BT.2020 |
| `channel_mixer_bw` | Arbitrary RGB weights in −2..+2 (infrared-like effects) |
| `color_filter_bw` | Wratten-style filter simulation (Yellow, Orange, Red, Green, Blue) |
| `zone_system` | Adams/Archer Zone System tone curve, 11 zones, Gaussian-blended |
| `local_contrast` | He–Sun–Tang guided filter for local contrast enhancement |
| `encode_srgb` | IEC 61966-2-1 sRGB transfer encoding (terminal stage) |

---

## Build prerequisites

- **Rust** stable toolchain (`rustup default stable`)
- **Python** 3.9 or later
- **maturin** (installed via requirements-dev.txt)

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
import phaios_core

# Linear f32 RGB image, shape (H, W, 3)
img = np.random.rand(1080, 1920, 3).astype(np.float32)

# B&W conversion — BT.709 luminance (default)
bw = phaios_core.luminance_bw(img)          # shape (H, W, 1)

# Zone System tone curve
params = phaios_core.ZoneParams({5: 0.5})   # lift Zone V by half a stop
toned = phaios_core.zone_system(bw, params)

# Local contrast enhancement
lc_params = phaios_core.GuidedFilterParams(radius=8, eps=0.01)
enhanced = phaios_core.local_contrast(toned, lc_params, strength=0.5)

# sRGB encode (always last)
output = phaios_core.encode_srgb(enhanced)
```

---

## Licence

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).

All dependencies are MIT OR Apache-2.0, both compatible with GPLv3.
Corresponding source is available at the repository URL above
(satisfies GPLv3 §6(d)).

---

## Architecture & FFI contract

- [docs/architecture.md](docs/architecture.md) — pipeline diagram,
  full mathematical derivations, algorithm citations.
- [docs/ffi.md](docs/ffi.md) — Python↔Rust boundary contract,
  array layout, zero-copy rules, error handling.
