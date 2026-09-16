# phaios-core examples

48 runnable examples, one concept each, on a synthetic Macbeth-style
colour checker. No real image inputs are ever used. Each writes an
8-bit binary PPM to `examples/output/`, which needs no external viewer
on Linux or macOS and is read by most image tools on Windows.

23 examples run on the CPU. The other 25 need `--features cuda`: 22 of
them are twins of a CPU example, and 15, 22 and 23 stand on their own.

---

## Running an example

```sh
cargo run --example 01_luminance

# every CPU example
for f in examples/[0-9]*.rs; do
    name="$(basename "${f%.rs}")"
    case "$name" in *_gpu_*) continue;; esac
    cargo run --example "$name"
done

# a GPU example: needs the CUDA toolkit at build time and a device at run time
cargo run --release --features cuda --example 23_gpu_selftest
```

Output files appear in `examples/output/`. PPM files are gitignored and
never committed.

`scripts/gpu-verify.sh` runs every GPU example in one pass, along with
the CUDA test suite. See [../docs/gpu.md](../docs/gpu.md).

---

## Viewing the output

**Linux:** `feh examples/output/01_luminance.ppm` or
`display examples/output/01_luminance.ppm` (ImageMagick).

**macOS:** `open examples/output/01_luminance.ppm` (Preview reads PPM).

**Windows:** rename to `.pnm` and open with IrfanView or GIMP, or
`magick convert 01_luminance.ppm 01_luminance.png`.

**Any platform:** `python -c "from PIL import Image;
Image.open('examples/output/01_luminance.ppm').show()"`

---

## The synthetic test image

`examples/shared/mod.rs` generates a 24-patch Macbeth ColorChecker in
scene-linear sRGB: a 6x4 grid of 64x64 patches, so a 384x256 image.
Patch values come from the BabelColor average measurements (D65) with
the sRGB transfer removed, giving linear values in [0, 1].

The crate has no I/O and must never depend on an external image file,
so the chart is built in code.

## Output is display-referred

Kernels operate on linear scene-referred data; an 8-bit PPM is
display-referred. Nearly every example therefore applies `encode_srgb`
before writing, through `shared::write_ppm_grey_display` or
`write_ppm_display`. Without it the midtones come out far too dark:
18% grey would land on code 48 instead of 120. There are three
deliberate exceptions: example 06 writes one file each way to show the
difference, and examples 17 and 38 write quantiser output, where the
codes are already the file.

---

## CPU examples

| File | Kernel | What to look for |
|---|---|---|
| `01_luminance.rs` | `luminance_bw` | Grey patches with BT.709 weighting; greens render brighter than reds |
| `02_channel_mixer.rs` | `channel_mixer_bw` | A BT.709 reference against red-boosted and green+blue mixes |
| `03_color_filter.rs` | `color_filter_bw` | Six presets: the unfiltered reference and five Wratten filters |
| `04_zone_system.rs` | `zone_system` | A no-op pass, a pulled Zone V, and a pushed Zone VII |
| `05_local_contrast.rs` | `local_contrast` | Detail enhanced at two radii against the unfiltered original |
| `06_srgb_encode.rs` | `encode_srgb` | Linear against encoded: the gamma lift on shadows |
| `07_exposure.rs` | `exposure` | +/-2 EV, and why the visible step is smaller than the arithmetic one |
| `08_hsl_weighted.rs` | `hsl_bw` | A blue weight darkens sky while foliage stays put; neutrals never move |
| `09_tone_curve.rs` | `tone_curve` | Slope, offset and power isolated; offset is the one that lifts black |
| `10_vignette.rs` | `vignette` | Circular against rectangular falloff; identical on a half-size preview |
| `11_split_toning.rs` | `split_toning` | Sepia, selenium and cross-process; the untinted round trip stays neutral |
| `12_film_grain.rs` | `film_grain` | The `4L(1-L)` envelope printed as a bar chart of grain sigma; same seed, same bytes |
| `13_geometry.rs` | `orient`, `crop` | All eight Exif orientations; crop-then-vignette against vignette-then-crop |
| `14_resample.rs` | `resize`, `straighten` | Three filters at two ratios; the inscribed-crop dimensions printed |
| `16_highlight_rolloff.rs` | `highlight_rolloff` | Clip against shoulder on a +2 EV push; how many of eight highlights survive each |
| `17_quantize.rs` | `quantize_u8`, `quantize_u16` | Where banding comes from: a ramp spanning two 8-bit codes, plain and dithered |
| `18_histogram_lut.rs` | `histogram`, `apply_lut` | Equalisation from two calls; solarisation; a tabulated film curve |
| `19_characteristic_curve.rs` | `shadow_rolloff` | Toe, straight section and shoulder composed; the slope table that shows the shape |
| `20_blur.rs` | `blur` | Impulse response against a true Gaussian on both paths; the border-clamp case |
| `21_glow.rs` | `glow` | Halation against diffusion against glare; why glare cannot be a tone curve |
| `43_sharpen.rs` | `sharpen` | The halo at a step edge; the same threshold gate turning off inside a flat patch |
| `45_hot_pixels.rs` | `hot_pixels` | Planted defects removed; a genuine fine line and a highlight texture bump survive |
| `47_denoise.rs` | `denoise` | Cross-guided on RGB: flat-patch noise quieted, patch edges kept; a channel smoothed by the shared guide, not its own signal |

## GPU examples

Every example below needs `--features cuda` at build time and an NVIDIA
device at run time, and prints its agreement with the CPU kernel that is
its specification. Each twin also names the CPU example's output file to
diff against. Examples 15, 22 and 23 are not twins of one CPU example:
15 measures transfer economics, 22 runs the whole chain three ways, and
23 checks all 27 device entry points at once.

| File | Kernel | What to look for |
|---|---|---|
| `15_gpu_exposure.rs` | `exposure` | Device enumeration, a context, one kernel; bit-exactness and why per-call offload does not pay |
| `22_gpu_pipeline.rs` | the whole chain | Upload once, download once, in the canonical order, against a per-call offload chain: 7x the bytes moved |
| `23_gpu_selftest.rs` | all 27 | Every device entry point against its CPU oracle, with the committed per-kernel bounds. The one to run on a new card |
| `24_gpu_luminance.rs` | `luminance_bw` | The twin of 01: same pixels, different processor |
| `25_gpu_channel_mixer.rs` | `channel_mixer_bw` | The twin of 02 |
| `26_gpu_color_filter.rs` | `color_filter_bw` | The twin of 03, all six presets |
| `27_gpu_zone_system.rs` | `zone_system` | The twin of 04 |
| `28_gpu_local_contrast.rs` | `local_contrast` | The twin of 05; the guided filter's 1e-4 / 1e-6 class in practice |
| `29_gpu_srgb_encode.rs` | `encode_srgb` | The twin of 06 |
| `30_gpu_hsl_weighted.rs` | `hsl_bw` | The twin of 08; eight `expf` per pixel, and among the largest CPU/GPU ratios in the crate |
| `31_gpu_tone_curve.rs` | `tone_curve` | The twin of 09; bit-exact at `power == 1`, bounded otherwise |
| `32_gpu_vignette.rs` | `vignette` | The twin of 10, computed from a device-resident monochrome intermediate |
| `33_gpu_split_toning.rs` | `split_toning` | The twin of 11; the 1 to 3 channel step on the device |
| `34_gpu_film_grain.rs` | `film_grain` | The twin of 12; the integer hash is bit-exact, the Box-Muller half bounded |
| `35_gpu_geometry.rs` | `orient`, `crop` | The twin of 13; one upload feeds all eight transforms |
| `36_gpu_resample.rs` | `resize`, `straighten` | The twin of 14; polynomial filters, bit-exact |
| `37_gpu_highlight_rolloff.rs` | `highlight_rolloff` | The twin of 16; ten files to compare five pairs at a time |
| `38_gpu_quantize.rs` | `quantize_u8`, `quantize_u16` | The twin of 17; device kernels whose output is codes, not an image |
| `39_gpu_histogram_lut.rs` | `histogram`, `apply_lut` | The twin of 18; atomics that reduce across threads and still commute |
| `40_gpu_characteristic_curve.rs` | `shadow_rolloff` | The twin of 19; read 19 first for why the curve has that shape |
| `41_gpu_blur.rs` | `blur` | The twin of 20; the direct path and the three-box path |
| `42_gpu_glow.rs` | `glow` | The twin of 21 |
| `44_gpu_sharpen.rs` | `sharpen` | The twin of 43; the gate reads the derived detail signal, not the raw pixel |
| `46_gpu_hot_pixels.rs` | `hot_pixels` | The twin of 45; a comparator network, so every sample must match bit for bit |
| `48_gpu_denoise.rs` | `denoise` | The twin of 47; cross-guided at `C == 3`, 14 launches |

Several examples print measurements as well as writing files: the
round-trip error in 07 and 11, the preview and full-frame agreement in
10, the grain envelope in 12, the bound multiples in every GPU twin.
Those numbers are the point of the example as much as the image is.
