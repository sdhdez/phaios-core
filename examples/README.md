# phaios-core Examples

Each example demonstrates one kernel on a synthetic Macbeth-style colour
checker. No real image inputs are ever used. Output is written as an
8-bit binary PPM file to `examples/output/` — PPM requires no external
viewer on Linux/macOS and is supported by most image tools on Windows.

---

## Running an example

```sh
# Build and run a single example
cargo run --example 01_luminance

# Run all examples
for n in 01 02 03 04 05 06; do
    cargo run --example ${n}_*
done
```

Output files appear in `examples/output/`. PPM files are listed in
`.gitignore` and are not committed to the repo.

---

## Viewing the output

**Linux:** `feh examples/output/01_luminance.ppm` or
`display examples/output/01_luminance.ppm` (ImageMagick).

**macOS:** `open examples/output/01_luminance.ppm` (Preview understands PPM).

**Windows:** Rename to `.pnm` and open with IrfanView, GIMP, or
`magick convert 01_luminance.ppm 01_luminance.png`.

**Any platform:** `cargo run --example 01_luminance && python -c
"from PIL import Image; Image.open('examples/output/01_luminance.ppm').show()"`

---

## The synthetic test image

`examples/shared/mod.rs` generates a 24-patch Macbeth ColorChecker
in scene-linear sRGB. The patches are arranged in a 6×4 grid, each
patch 64×64 pixels, giving a 384×256 image. Patch values come from the
BabelColor average measurements (D65) with the sRGB transfer removed,
giving linear values in [0, 1].

This is intentionally not a real photograph: the crate has no I/O
and should never depend on external image files.

## Output is display-referred

Kernels operate on linear scene-referred data, but an 8-bit PPM is
display-referred. Examples 01–05 therefore apply `encode_srgb` — the
terminal pipeline stage — before writing, via
`shared::write_ppm_grey_display`. Without it the midtones come out far
too dark: 18% grey would land on code 48 instead of 120.

Example 06 writes one file each way, on purpose, to show the
difference.

---

## Examples

| File | Kernel | What to look for |
|------|--------|-----------------|
| `01_luminance.rs` | `luminance_bw` | Grey patches with BT.709 weighting — greens render brighter than reds |
| `02_channel_mixer.rs` | `channel_mixer_bw` | Compare standard vs. boosted-red weights |
| `03_color_filter.rs` | `color_filter_bw` | All six Wratten presets side by side |
| `04_zone_system.rs` | `zone_system` | Tone-mapped result vs. unprocessed |
| `05_local_contrast.rs` | `local_contrast` | Detail enhanced vs. flat original |
| `06_srgb_encode.rs` | `encode_srgb` | Linear vs. encoded (the "gamma lift" on shadows) |
| `07_exposure.rs` | `exposure` | ±2 EV, and why the visible step is smaller than the arithmetic one |
| `08_hsl_weighted.rs` | `hsl_bw` | A blue weight darkens sky while foliage stays put; neutrals never move |
| `09_tone_curve.rs` | `tone_curve` | Slope, offset and power isolated — offset is the one that lifts black |
| `10_vignette.rs` | `vignette` | Circular vs. rectangular falloff; identical on a half-size preview |
| `11_split_toning.rs` | `split_toning` | Sepia, selenium and cross-process; the untinted round trip stays neutral |
| `12_film_grain.rs` | `film_grain` | The `4·L·(1−L)` envelope printed as a histogram; same seed, same bytes |
| `13_geometry.rs` | `crop`, `orient` | All eight Exif orientations; crop-then-vignette vs vignette-then-crop |
| `14_resample.rs` | `resize`, `straighten` | Three filters at two ratios; inscribed-crop dimensions printed |
| `15_gpu_exposure.rs` | CUDA backend | Needs `--features cuda`; bit-exactness and the resident-image pattern |

Several examples print measurements as well as writing files — the
round-trip error in 07 and 11, the preview/full-frame agreement in 10,
the grain envelope in 12. Those numbers are the point of the example as
much as the image is.
