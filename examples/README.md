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
patch 64×64 pixels, giving a 384×256 image. Patch values are the
standard CIE colorimetric values (D50 illuminant) converted to
linear sRGB and clamped to [0, 1].

This is intentionally not a real photograph: the crate has no I/O
and should never depend on external image files.

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
