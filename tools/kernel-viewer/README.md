# kernel-viewer

Interactive parameter explorer for the phaios-core kernels — the
interactive counterpart of the crate's `examples/` directory, in the
spirit of the classic OpenGL sample programs: open a kernel on an image,
drag the sliders, watch it act.

```sh
cd tools/kernel-viewer

cargo run --release -- vignette                 # CPU, preloaded photo
cargo run --release --features cuda -- grain    # + GPU toggle and A/B view
cargo run --release -- exposure ~/pics/x.jpg    # your image
cargo run --release -- --help
```

## Usage

```
kernel-viewer [KERNEL] [IMAGE] [--scene photo|macbeth] [--gpu] [--full]
```

- **KERNEL** — any of the sixteen kernel names (`exposure`, `hsl_bw`,
  `zone_system`, …); the in-window dropdown or `[` `]` switch at runtime.
- **IMAGE** — a JPEG, PNG or RAW/DNG. Defaults to
  `testdata/R0000096.DNG` when present, else the synthetic scene.
  Drag-and-drop a file onto the window to load it at any time.
- `--gpu` — start on the CUDA backend (`--features cuda` + a device).
- `--full` — disable the 1600-px working-copy cap. Interaction defaults
  to a linear-space downscale so slider drags stay fluid; use this when
  judging pixel-scale effects (grain size!) at export resolution.

Keys: `[` `]` cycle kernel · `G` toggle backend · `D` difference ×64 ·
`R` reroll the grain seed · drag the white line in A/B split view.

## Colour pipeline (the part worth knowing)

Kernels contract for **linear scene-referred f32**; the screen wants
display-referred. The viewer keeps one canonical path:

- **DNG/RAW** (via `rawler`, the tool's own decoder — phaios-core never
  opens RAW files, by design): rawler's develop pipeline *minus its
  final sRGB gamma step*, so the result is linear — a RAW is exactly the
  data the kernels want, no transfer guessing involved. Viewer-grade
  development only; real RAW development is the desktop app's job.
- **JPEG/PNG**: decoded u8 → inverse sRGB transfer → linear.
- **Display**: kernel output → clamp [0, 1] → `encode_srgb` → texture.
  The `encode_srgb` *kernel view* skips that second encode (its output is
  already display-referred).
- The luminance-input kernels (`zone_system`, `local_contrast`,
  `film_grain`, `split_toning`) are fed through a fixed BT.709
  `luminance_bw` prep stage; the prep is not a parameter.
- In A/B mode the display encode always runs on the CPU for both
  backends, so any visible difference originates in the kernel under
  test. `D` shows `|cpu − gpu| × 64`: black is conformance, anything
  visible is divergence ≥ ~0.016 linear.

## Why this crate is standalone

It has its own `Cargo.toml` **and `Cargo.lock`** and an empty
`[workspace]` table: the GUI/RAW dependency stack never enters
phaios-core's lockfile, `cargo audit` surface, CI, or published crate.
It consumes only the crate's public Rust API. `testdata/` is gitignored —
the repo's "no real image inputs, ever" rule governs committed assets;
this tool loads images at runtime only.
