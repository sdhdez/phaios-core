# The export contract

What a finished phaios image *is*, stated normatively, so that the layer
which writes files has no image decisions left to make.

This document exists because "save a black-and-white photograph" is not
one decision but six, and five of them are numerical — which means they
belong here, in the core, where they are tested and reproducible. Only
the sixth, the file container, belongs to a consumer. A front-end that
follows this document cannot accidentally produce an RGB grey JPEG with
a mismatched profile and call it a black-and-white photograph.

The key words MUST, MUST NOT, SHOULD and MAY are used in the sense of
RFC 2119.

---

## 1. The terminal sequence

A pipeline that intends to write a file MUST end with these stages, in
this order:

```
… look stages …
  → highlight_rolloff      last stage on linear scene-referred data
  → encode_srgb            the only stage that produces display-referred data
  → quantize_u8 | quantize_u16    continuous → integer codes
  → [container writing]    outside the core
```

Each of the three is skippable only with its consequence understood:

- Skipping `highlight_rolloff` means the caller has chosen a hard clip.
  That is legitimate — it is the kernel's own default — but it should be
  a choice rather than an omission. See §3.
- Skipping `encode_srgb` writes linear data into a file that will be
  interpreted as encoded, which renders the midtones far too dark: 18%
  grey lands on code 48 instead of 120.
- Skipping `quantize_*` is correct only when the container itself is
  floating-point (see §5).

Nothing may run *after* `encode_srgb` except quantisation. The output of
`encode_srgb` is display-referred, and every other kernel in the crate
is specified on linear scene-referred data.

---

## 2. Grey means grey

This is the contract that makes phaios a black-and-white tool rather
than a colour tool used monochromatically.

**Untoned output** — a pipeline in which `split_toning` did not run —
is `(H, W, 1)`. When such an image is written to a file:

- The file MUST use a single-channel greyscale photometric
  interpretation. In TIFF that is `PhotometricInterpretation = 1`
  (BlackIsZero) with `SamplesPerPixel = 1`; in PNG it is colour type 0.
- The file MUST NOT store three duplicated channels. An RGB file whose
  channels happen to be equal is not a greyscale image: it is three
  times the size, it invites a colour-managed application to apply a
  matrix transform that has no business being applied, and any
  subsequent edit can pull the channels apart silently.
- The embedded ICC profile MUST be a **greyscale** profile whose tone
  response curve matches `encode_srgb` — that is, the IEC 61966-2-1
  transfer, not a pure 2.2 power law. The two differ most in the
  shadows, exactly where black-and-white work lives.

**Toned output** — a pipeline in which `split_toning` did run — is
`(H, W, 3)` and MUST be written as RGB with a standard sRGB profile.
Toning is a colour operation; the file has to say so.

A consumer MUST determine which case applies from the channel count of
the array it received, never from a setting, so the file cannot
disagree with the pixels.

---

## 3. Highlights are a decision, not a default

Every stage before `highlight_rolloff` preserves values above 1.0.
Something must map them into the displayable range, and the crate now
makes that explicit rather than leaving it to whichever line calls
`clip`.

- A consumer MUST NOT clamp the highlight end itself. It MUST route the
  decision through `highlight_rolloff`, whose default parameters
  (`knee = 1.0`, `white_point = 1.0`) reproduce a hard clip bit for bit.
  The point is not to forbid clipping; it is to make clipping visible in
  the settings, and therefore recorded in the sidecar.
- The black end is different: values below zero can arise from negative
  channel-mixer weights, and `quantize_*` clamps them to code 0. No
  separate stage is needed.

---

## 4. Depth and dither

- 16-bit MUST be the default for archival output. It leaves enough
  precision that a consumer can grade the file further without tearing
  it.
- 8-bit output MUST be dithered (`Dither::Tpdf`) unless `film_grain` ran
  at visible intensity, in which case the grain already serves as
  dither and `Dither::Off` is correct. Undithered 8-bit banding is the
  most common visible defect in black-and-white export.
- The dither `seed` MUST be recorded alongside the other settings. It is
  part of the reproducibility key: same seed, same file.

---

## 5. Floating-point containers

A consumer writing a floating-point container (OpenEXR, 32-bit TIFF)
SHOULD stop after the last look stage and write **linear scene-referred**
data, skipping both `encode_srgb` and `quantize_*`. Encoding and
quantisation exist to fit an image into integer codes for a display; a
float container has no such constraint, and baking a display transfer
into one throws away the headroom that made the format worth choosing.

Such a file is not a finished photograph but an intermediate, and should
be labelled as one.

---

## 6. Reproducibility

A file written by this pipeline is reproducible from its settings only
if the settings record everything that varied. That means:

- every kernel's parameters, including the geometry stages, whose
  coordinates are meaningless without the crop and orientation that
  produced them;
- every `seed` (`film_grain`, `quantize_*`);
- the **backend fingerprint**, per `ffi.md` §6. Determinism is promised
  per backend; a file rendered on CUDA and one rendered on the CPU agree
  within the committed bound but are not guaranteed byte-identical for
  kernels containing transcendentals.

A consumer that promises exact reproduction MUST either record the
fingerprint or designate `cpu` as its archival backend.

---

## 7. What the core does not do

It does not open or write files, embed profiles, or know what a TIFF is.
Everything above is expressed as constraints on the array a consumer
receives and on the metadata it must carry, precisely so that the file
layer is a mechanical translation with no image decisions in it.

Where that file layer should live, in each consumer, in a sibling
`phaios-io` crate, or behind a feature flag here, is an open question.
This document is written to outlast that decision: it constrains the
output regardless of who writes the bytes.
