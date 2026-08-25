# phaios-core Architecture

Mathematical derivations, algorithm citations, and design rationale for
every kernel in the phaios-core pipeline. This document is the reference
a new contributor reads to understand the math without reading the code.

---

## 1. Pipeline overview

```
Consumer delivers:
  linear scene-referred f32 RGB  (H, W, 3)
        │
        ▼
  ┌─────────────┐
  │  geometry   │  orient → straighten → crop → resize   src/geometry.rs
  └─────────────┘  (all bit-exact across backends)
        │
        ▼
  ┌─────────────┐
  │  exposure   │  × 2^stops                             src/exposure.rs
  └─────────────┘
        │  (H, W, 3)
        ▼
  ┌─────────────┐
  │  B&W conv.  │  luminance / channel mixer /           src/bw.rs
  └─────────────┘  colour filter / HSL-weighted
        │  (H, W, 1)   ← the image becomes monochrome here
        ▼
  ┌─────────────┐
  │  zone tone  │  Adams/Archer + Davis Gaussian         src/tone.rs
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  local      │  He–Sun–Tang guided filter             src/local_contrast.rs
  │  contrast   │
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  grain      │  band-passed hashed noise              src/film_grain.rs
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ split-tone  │  OKLab chroma blend                    src/split_toning.rs
  └─────────────┘
        │  (H, W, 3)   ← and colour again here
        ▼
  ┌─────────────┐
  │  vignette   │  radial falloff                        src/vignette.rs
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ tone curve  │  ASC CDL slope/offset/power            src/tone.rs
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │ sRGB encode │  IEC 61966-2-1 terminal stage          src/encode.rs
  └─────────────┘
        │
        ▼
Consumer receives:
  display-referred f32 RGB  (H, W, 3)  → write to file
```

**Channel count along the pipeline.** The B&W stage collapses three
channels to one; split-toning takes it back to three. Everything between
those two points is single-channel, and everything after split-toning
(`vignette`, `tone_curve`, `encode_srgb`) accepts any channel count so
that the same pipeline runs whether or not toning is enabled.

**Order sensitivity.** Some of the ordering is forced, and some of it is
a judgement that the kernels document but do not enforce:

| Constraint | Why |
|---|---|
| **geometry first** (`orient`, then `crop`) | `vignette` centres on the frame it is given, which must be the *cropped* frame; `film_grain` keys noise to pixel coordinates, which must be the final grid; crop rectangles are expressed in the upright (oriented) frame |
| `encode_srgb` last | every other kernel assumes linear input |
| `exposure` opens the look pipeline (right after geometry) | a stop is a factor of two only in linear light |
| B&W before zone/local-contrast/grain | those kernels take `(H, W, 1)` |
| `split_toning` after B&W | it takes `(H, W, 1)` and returns `(H, W, 3)` |
| grain after the tone stages | a tone curve applied afterwards reshapes the grain, and the `4·L·(1−L)` envelope would no longer sit on the midtones the viewer sees |
| vignette after the tone stages | otherwise the tone curve acts on already-darkened corners |

Within those constraints the remaining orderings are equivalent.

**Layout.** Every kernel accepts any array layout — C-contiguous,
Fortran-order, strided views, negative strides — and always returns a
freshly allocated C-contiguous array. See `docs/ffi.md` §1.

---

## 2. Exposure

```
out = in · 2^stops
```

A stop is a doubling of exposure, so in linear scene-referred data the
operation is a single multiply. That is the whole reason the pipeline
keeps its data linear until the final encode: applied after a transfer
function, the same adjustment would be a curve whose shape depended on
the transfer, and "+1 EV" would no longer mean one stop.

Nothing is clamped. Highlights driven above 1.0 stay above 1.0, where
the tone stages can still pull them back; clipping here would throw away
information irrecoverably. Since `2^n` is exact for integer `n`, +n EV
followed by −n EV is bit-exact, not merely close.

Reference: the stop as a doubling of luminous exposure is standard
photographic practice; Ansel Adams, *The Negative*, Little, Brown
(1948), chapter 4.

---

## 3. Luminance weights (B&W method 1)

### Derivation

The CIE 1931 XYZ tristimulus values define luminance Y as a linear
combination of the primaries. For each RGB colour space, the Y row of
the colour-space-to-XYZ matrix gives the luminance weights.

**BT.709 (sRGB primaries)**

The ITU-R BT.709 standard (Rec. 709) defines primaries at:

| Primary | CIE xy chromaticity |
|---------|---------------------|
| Red | (0.640, 0.330) |
| Green | (0.300, 0.600) |
| Blue | (0.150, 0.060) |
| White (D65) | (0.3127, 0.3290) |

Solving the 3×3 system `RGB_to_XYZ · [1,1,1]ᵀ = D65_XYZ` gives the
normalisation factors; the Y row of the resulting matrix is:

```
Y_709 = 0.2126·R + 0.7152·G + 0.0722·B
```

These values are specified verbatim in ITU-R BT.709-6 (2015), Table 1.
They are the default in phaios-core because sRGB-primary RAW data (the
vast majority of consumer cameras) is defined on the Rec. 709 primaries.

**BT.601 (SDTV primaries)**

Older standard, still relevant for digitised film and SDTV content.
Primaries:

| Primary | CIE xy chromaticity |
|---------|---------------------|
| Red | (0.630, 0.340) |
| Green | (0.310, 0.595) |
| Blue | (0.155, 0.070) |

Solving as above:

```
Y_601 = 0.2990·R + 0.5870·G + 0.1140·B
```

Specified in ITU-R BT.601-7 (2011), §2.5.1.

**BT.2020 (UHDTV primaries)**

Wide-colour-gamut standard for 4K/8K production:

| Primary | CIE xy chromaticity |
|---------|---------------------|
| Red | (0.708, 0.292) |
| Green | (0.170, 0.797) |
| Blue | (0.131, 0.046) |

```
Y_2020 = 0.2627·R + 0.6780·G + 0.0593·B
```

Specified in ITU-R BT.2020-2 (2015), Table 4.

### Why the green weight is always the largest

Humans are most sensitive to light at ~555 nm, which corresponds to
the CIE 1931 V(λ) luminosity function peak. Green primaries are
positioned near this peak for all three standards, so green always
dominates the luminance mix.

### Implementation note

The weights are applied as a dot product across the channel axis of
the `(H, W, 3)` array, producing `(H, W, 1)`:

```
Y[h, w, 0] = w[0]·R[h,w] + w[1]·G[h,w] + w[2]·B[h,w]
```

---

## 4. Coloured-filter simulation (B&W method 3)

### Spectral basis

A coloured filter in front of the lens attenuates light reaching the
film according to its spectral transmission curve T(λ). For three-
channel linear RGB, we approximate this with a per-channel transmission
vector (t_R, t_G, t_B):

```
R' = t_R · R
G' = t_G · G
B' = t_B · B
```

Then Y is computed on the filtered RGB using the chosen luminance
standard. The net effect is a dot product with weights (t_R·w_R, t_G·w_G,
t_B·w_B), but keeping the two steps separate is cleaner and lets the
user mix any luminance standard with any filter.

### Wratten presets

Kodak Wratten 2 filters (named after Frederick Wratten, 1906) are the
industry-standard reference for B&W contrast control. The channel
transmission values below are derived from the Kodak Wratten Gelatin
Filters datasheet (Publication No. B3-203, 5th ed.), sampling at the
centroid wavelengths of a nominal sRGB camera's R/G/B channels (~620,
~540, ~450 nm):

| Name | Designation | (t_R, t_G, t_B) | Effect |
|------|-------------|-----------------|--------|
| None | — | (1.00, 1.00, 1.00) | No filtration |
| Yellow | #8 K2 | (1.00, 0.90, 0.30) | Moderate sky darkening |
| Orange | #21 | (1.00, 0.55, 0.10) | Strong sky, foliage contrast |
| Red | #25 A | (1.00, 0.10, 0.02) | Very dark sky, snow bright |
| Green | #11 X1 | (0.20, 1.00, 0.30) | Natural foliage, skin darker |
| Blue | #47 C5 | (0.10, 0.30, 1.00) | Haze enhancement |

The values are **approximate**. Real Wratten transmission is a continuous
spectral function; this three-point sampling is a useful approximation
that matches the perceptual character of each filter for typical outdoor
scenes. For spectrally accurate rendering, a multispectral pipeline is
required — outside this crate's scope.

### Exposure factor

Real filters absorb light and require exposure compensation (the "filter
factor" printed on Wratten packaging, e.g. "2×" for Yellow #8 K2).
phaios-core does **not** apply this correction automatically: the
exposure kernel is the right place to compensate, and the correction
depends on the scene's spectral distribution. The filter simulation
kernel is purely multiplicative.

---

## 5. HSL-weighted conversion (B&W method 4)

The three methods above assign one weight per *channel*. This one
assigns a weight per *hue*, which is what a photographer means by
"darken the sky without darkening the foliage".

### Bands

Eight bands, spaced more finely at the warm end where skin, foliage and
sky separations matter most:

| Index | Band | Centre |
|-------|------|--------|
| 0 | red | 0° |
| 1 | orange | 30° |
| 2 | yellow | 60° |
| 3 | green | 120° |
| 4 | aqua | 180° |
| 5 | blue | 240° |
| 6 | purple | 270° |
| 7 | magenta | 300° |

### Formula

```
Y_out = max(Y_base · (1 + chroma · Σ_i  w_i · exp(-d_i² / (2σ²))), 0)
```

where `Y_base` is the luminance from the chosen ITU-R standard, `d_i` is
the **circular** distance from the pixel's hue to band `i` (350° is 10°
from red, not 350°), and σ is 30° by default, at which adjacent bands
overlap at roughly half weight.

The eight terms are summed in fixed array order, so the result is
bit-reproducible — see `docs/ffi.md` §6.

### Saturation measure

`chroma` is `(max − min) / max`, the HSV-style chroma ratio, **not** HSL
saturation `(max − min) / (1 − |2L − 1|)`. This is a deliberate
departure from the usual formulation, for two reasons:

1. **Scale invariance.** The chroma ratio does not change when the
   image is brightened. Exposure is a kernel two stages upstream, and a
   saturation measure that moved with it would silently change the B&W
   conversion whenever exposure was adjusted.
2. **Scene-referred data.** HSL saturation's denominator passes through
   zero at L = 1 and goes negative above it. Highlights above 1.0 are
   normal here, so the formula is not merely imprecise in that range —
   it is degenerate.

The visible consequence: a bright saturated yellow responds strongly to
the yellow band, where HSL saturation would have judged it barely
saturated on account of its lightness.

Negative components are clamped to zero before the hue geometry, since
white balance can push a channel slightly negative and a negative `min`
would inflate the chroma ratio past 1.

Reference: the hue/chroma geometry is the standard hexagonal projection
of Joblove & Greenberg, "Color spaces for computer graphics",
*SIGGRAPH '78*, pp. 20–25.

---

## 6. Zone System tone curve

### Historical context

Ansel Adams and Fred Archer developed the Zone System circa 1939–1941
as a practical framework for pre-visualising and controlling tonal
relationships in B&W photography. Adams codified the mathematics in
*The Negative* (Little, Brown, 1948), chapter 5. Phil Davis extended
it with a continuous mathematical model in *Beyond the Zone System*,
4th ed. (Focal Press, 1999), introducing the Gaussian-blending approach
implemented here.

### Zone definitions

Eleven zones (Roman numerals 0..=10):

| Zone | Description | Linear value |
|------|-------------|-------------|
| 0 | Maximum black | ≈ 0.0056 |
| I | Near-black detail threshold | ≈ 0.0113 |
| II | First textured shadow | ≈ 0.0225 |
| III | Dark shadow, full texture | ≈ 0.045 |
| IV | Dark skin, foliage, shadow detail | ≈ 0.09 |
| V | Middle grey, clear sky | **0.18** |
| VI | Light skin, light concrete | ≈ 0.36 |
| VII | Light grey, highlights with texture | ≈ 0.72 |
| VIII | Textured white | ≈ 1.44 |
| IX | Glaring white | ≈ 2.88 |
| X | Paper white | ≈ 5.76 |

Each zone is one stop (factor of 2) apart. Zone V = 18% reflectance
is the photographic middle grey, matching the calibration of incident
light meters.

### Zone position function

For a pixel with linear luminance L, its zone position is:

```
zone_pos(L) = 5 + log₂(L / 0.18)
```

This maps 0.18 → 5 (Zone V), 0.36 → 6 (Zone VI), 0.09 → 4 (Zone IV),
etc. Values outside 0..10 are valid (they describe shadows darker than
Zone 0 or highlights brighter than Zone X) and are handled gracefully
by the Gaussian tails.

### Gaussian blending (Davis 1999)

The user supplies a map `{zone_index: stops_offset}` assigning a tonal
correction (in stops) to each zone. The total offset at a pixel is the
Gaussian-weighted sum of all zone offsets:

```
total_offset(p) = Σ_z  offset[z] · exp(-(zone_pos(p) - z)² / (2·σ²))
```

where σ = 0.8 zones. The Gaussian ensures smooth transitions: a zone
offset primarily affects pixels at that zone position and tails off
over roughly ±1.5 zones.

The output luminance is:

```
L_out = L · 2^total_offset(p)
```

### Parameter type

```python
params = phaios_core.ZoneParams({5: +1.0, 7: -0.5})
# Zone V brightened by 1 stop; Zone VII darkened by 0.5 stops.
```

The zone index is an integer 0..=10; anything outside that range is
rejected with a `ValueError` (`PhaiosError::Parameter`). An index of 99
would otherwise be a silent no-op — its Gaussian is zero everywhere in
range — which hides what is almost always a caller bug.

The offset is a number of stops. [−3, +3] is the useful range, but the
kernel does **not** clamp: an offset of +10 really does multiply that
zone by 2¹⁰. Only non-finite offsets are rejected. Omitted zones default
to 0.

### Numerical notes

- The zone position is computed as `(L / 0.18).log2()` — one division
  and one `log2`, rather than two logarithms.
- The Gaussian is evaluated for each zone index in the map; zones not
  in the map contribute 0.
- Input values ≤ 0 are clamped to a small positive epsilon before
  the log to avoid −∞. A negative input keeps its sign in the output:
  the multiplier is applied to the original value, not the clamped one.
- **The zone offsets are summed in ascending zone order.** f32 addition
  is not associative, so summing them in `HashMap` order would make the
  output depend on the per-process hash seed — the same image and the
  same parameters would produce different bytes on every run. Any future
  kernel that reduces over an unordered collection must impose an order
  the same way.

---

## 7. Parametric tone curve (ASC CDL)

```
out = max(in · slope + offset, 0)^power
```

Three controls, under the names photographers already use: `slope` is
gain, `offset` is lift, `power` is gamma. The identity is
`(1.0, 0.0, 1.0)`.

Chosen over a four-point spline — the alternative the design brief
offered — for three reasons:

- It is a published interchange standard, so a grade means the same
  thing in other tools.
- Three numbers cover the adjustment; a spline needs four control points
  and a UI to place them.
- It is monotonic for any positive `slope` and `power`, so it cannot
  invert tonal order. A spline has to be constrained to guarantee that.

The clamp before the exponent is required rather than defensive: a
negative base raised to a fractional power has no real value, so without
it a negative `offset` would produce NaN across the shadows. With it,
those values crush to black — and *irreversibly*, which is the one thing
to know when reaching for a negative offset.

Applies to any channel count, since it sits after split-toning where the
data may have become three-channel again.

Reference: American Society of Cinematographers Technology Committee,
"ASC Color Decision List (ASC CDL) Transfer Functions and Interchange
Syntax", version 1.2 (2009), §2.1.

---

## 8. Guided filter for local contrast

### Background

The guided filter (He, Sun, Tang, "Guided Image Filtering," *ECCV 2010*,
LNCS 6311, pp. 1–14) is a linear-time edge-preserving smoothing filter.
It is patent-free. The key property for local contrast use: it
preserves edges while smoothing flat regions, making it superior to a
Gaussian blur for the unsharp-masking operation.

### Local linear model

Given guide image I and input image p, the filter models the output q
as a locally linear function of I in each square window Ω_k of radius r:

```
q_i = a_k · I_i + b_k    ∀i ∈ Ω_k
```

The coefficients a_k, b_k are found by minimising:

```
E(a_k, b_k) = Σ_{i∈Ω_k} [(a_k·I_i + b_k - p_i)² + ε·a_k²]
```

ε is a regularisation term that controls the degree of smoothing.
Large ε → more smoothing, smaller a_k (less edge-preservation).
The closed-form solution is:

```
a_k = (cov_k(I, p)) / (var_k(I) + ε)
b_k = mean_k(p) - a_k · mean_k(I)
```

where cov_k and var_k are the covariance and variance in Ω_k.

### Self-guided formulation

For local contrast, the guide I = p = L (luminance). This simplifies:

```
a_k = var_k(L) / (var_k(L) + ε)
b_k = (1 - a_k) · mean_k(L)
```

### Integral-image O(1) formulation

Naively computing mean and variance over every window is O(r²·HW).
The integral image (summed-area table) reduces each window statistic to
four additions, making the total complexity O(HW) regardless of r.

For an image f of size H×W, define the integral image:

```
S[y, x] = Σ_{j≤y, i≤x} f[j, i]
```

Then the sum over any rectangle (y1,x1)–(y2,x2) is:

```
Σ = S[y2,x2] - S[y1-1,x2] - S[y2,x1-1] + S[y1-1,x1-1]
```

We compute integral images for L and L² simultaneously, then:

```
mean_k(L)   = Σ(L)   / |Ω_k|
mean_k(L²)  = Σ(L²)  / |Ω_k|
var_k(L)    = mean_k(L²) - mean_k(L)²
```

Window counts |Ω_k| must be computed per-pixel at boundaries (partial
windows), then the coefficient images a and b are box-filtered to
average overlapping windows. All box filters use the same integral-
image trick.

`var_k(L)` is clamped at zero. The subtraction above cancels two nearly
equal quantities, and the rounding error of the tables grows with both
image area and pixel magnitude; where it exceeds the true variance the
result goes slightly negative, which would make `a` negative (the local
linear model inverted) or greater than one (over-driven). Zero variance
means "no measurable structure here", whose correct limit is `a = 0`,
i.e. pure smoothing.

### Precision and memory

The tables are accumulated in f64. A window statistic is the difference
of two large partial sums — at 24 MP the bottom-right entry of the L²
table is of order 10⁷ — and an f32 table would lose exactly the low bits
the variance is computed from.

That costs 8 bytes per pixel per table, and the algorithm needs four
tables (L, L², a, b), so the order of operations matters as much as the
formula. The tables of L and L² are dropped as soon as a and b exist,
and each coefficient array is dropped as soon as its own table is built;
the L² table is accumulated through a mapping closure rather than from a
materialised squared copy of the image. Peak scratch is about 24 bytes
per pixel — roughly 600 MB for a 24 MP frame, against 1.3 GB when all
the intermediates were left live.

### Parallelism

Each table is built in two passes, both parallel:

1. **Horizontal prefix sums.** Rows are independent, so they are summed
   in parallel.
2. **Vertical prefix sums.** Columns are independent. Rather than one
   task per column — which would stride across the full row pitch on
   every step — the table is split into blocks of 512 columns and each
   block is accumulated row by row, keeping the traversal row-major and
   cache-friendly.

The per-pixel coefficient and output passes are parallel over pixels.
A single-threaded cell-by-cell table build was the dominant cost of the
kernel before this split: 452 ms of the original 24 MP measurement.

### Local contrast output

```
output = L + strength · (L - guided_filter(L, r, ε))
```

`guided_filter(L, r, ε)` is the smooth (low-frequency) version of L.
Subtracting it from L gives the high-frequency (detail) component.
`strength` controls how much detail is added back: 0 = no change,
1 = standard unsharp mask, >1 = over-sharpening.

### Parameters

| Parameter | Type | Range | Meaning |
|-----------|------|-------|---------|
| `radius` | u32 | any; 1..512 useful | Window half-size in pixels. 0 makes the filter the identity; a radius larger than the image is legal, since windows clamp to the image extent, and makes every window the whole image |
| `eps` | f32 | ≥ 0, finite | Regularisation; try 0.01. Negative values are rejected: they make `a = var/(var+ε)` singular wherever the local variance approaches −ε |
| `strength` | f32 | finite; 0..2 useful | Detail amplification. Negative values smooth instead of sharpening |

---

## 9. Film grain

```
out = max(L + intensity · 4·t·(1−t) · bandpass(noise, size), 0),  t = clamp(L, 0, 1)
```

### Deterministic noise without a generator

Each pixel's noise is a hash of `(seed, x, y)` — splitmix64, then
Box–Muller — rather than a draw from a sequential RNG. No pixel depends
on another's state, which buys three properties:

- identical output on one thread or thirty-two;
- identical output across platforms, since integer hashing has no
  floating-point tolerance;
- no hidden contract. A per-tile `SeedableRng` is reproducible too, but
  only while the tiling holds still: the tile size would quietly become
  part of the output, and changing it later would alter every rendered
  image.

This is also why the kernel needs no RNG dependency.

The coordinates are mixed once before meeting the seed. Without that
step `(x, y)` and `(y, x)` collide and the grain shows a diagonal
mirror line.

### Band-pass, and why grain has a size

White noise added per pixel is sensor noise, not grain: it has no
characteristic scale, so it vanishes when the image is downsampled and
turns to mush when it is enlarged. Real grain clumps.

The band-pass is a difference of two box filters over the same noise
field, with radii `floor(size/2)` and `round(max(size, 1))`, the inner
capped one below the outer. The `max(size, 1)` matters: `size_pixels`
only has to be positive, and a sub-half-pixel size would otherwise round
to an outer radius of 0. Grain finer than a pixel cannot be resolved and
behaves as one pixel. The energy that survives is concentrated around
`size_pixels`.

Both filters read one summed-area table (§8), so the cost is independent
of the grain size.

**Normalisation.** For nested windows of `n₁` and `n₂` pixels over
unit-variance input, the difference of box means has variance
`1/n₁ − 1/n₂`. Dividing by its square root makes `intensity` mean the
same thing at every grain size — analytically, with no second pass over
the image and no order-dependent reduction. `intensity` is then the
standard deviation of the grain, in linear units, where the envelope
peaks.

### The envelope

`4·t·(1−t)` peaks at mid-grey and vanishes at both ends: film has no
grain where no silver was developed, and none where the emulsion
saturated. `t` is clamped to `[0, 1]` first — above 1.0 the parabola
goes negative, which would invert the grain rather than fade it out.

Reference: Sebastiano Vigna, "Further scramblings of Marsaglia's
xorshift generators", *J. Comput. Appl. Math.* 315 (2017), pp. 175–181;
G. E. P. Box and Mervin E. Muller, "A Note on the Generation of Random
Normal Deviates", *Ann. Math. Statist.* 29(2) (1958), pp. 610–611.

The Newson et al. (2017) stochastic grain model — which simulates
individual silver grains rather than filtering noise — is deliberately
out of scope; it is v0.3 material at the earliest.

---

## 10. Split-toning

Takes `(H, W, 1)` monochrome and returns `(H, W, 3)` linear sRGB. The
only kernel that adds channels.

### Why OKLab

Tinting means adding chroma without disturbing the lightness the tone
stages just established. That needs a space whose lightness axis is
genuinely independent of its colour axes. In linear sRGB, adding to the
red channel makes a pixel both redder and brighter. CIELAB separates
them in principle, but its blue-hue non-uniformity bends a
constant-chroma sweep visibly towards purple.

OKLab was fitted to fix exactly that, and costs two 3×3 matrices with a
cube root between them.

### Algorithm

1. Take the OKLab lightness `L` of the luminance sample. For a neutral
   the three cone responses are equal, so the forward transform
   collapses to one cube root instead of three.
2. Crossfade the two tints with a smoothstep centred on the pivot:
   `mix = smoothstep(pivot − 0.25, pivot + 0.25, L)`.
3. Convert `(L, a, b)` back to linear sRGB.

Only the `a` and `b` components of each tint are used. The parameters
are full OKLab triples so that a colour picked in an OKLab picker can be
passed through unchanged, but honouring the `L` component would move the
tonal rendering — the very thing the space was chosen to protect.

`balance` shifts the crossover: positive moves it down, so more of the
image reads as highlight and takes the highlight tint.

The crossover half-width is fixed at 0.25 rather than exposed. A fourth
control would let the user reproduce the pivot's job by another route.

### Behaviour at black

OKLab lightness 0 with non-zero chroma is not a colour — nothing is both
black and tinted. Asked for one, the inverse transform returns the
nearest thing it can, slightly outside the sRGB cube, mostly as a small
*negative* blue. The kernel does not clamp; the sign means a clamping
consumer sees black rather than a lifted shadow.

| tint (applied to both a and b) | worst channel at L = 0 |
|------|---------------|
| 0.02 | 3.6e−5 |
| 0.05 | 5.6e−4 |
| 0.10 | 4.5e−3 |
| 0.20 | 3.6e−2 |

The error grows as the cube of the tint, so it is negligible across the
range anyone tones in.

Reference: Björn Ottosson, "A perceptual color space for image
processing" (2020), <https://bottosson.github.io/posts/oklab/>.

---

## 11. Vignette

```
out = max(in · (1 − amount · falloff(distance)), 0)
```

Positive `amount` darkens the corners, negative lightens them.

### Distance

Normalised frame coordinates put the centre at the origin and the
corners at `(±1, ±1)`. Two measures are blended by `roundness`:

- **Euclidean**, `√(nx² + ny²) / √2` — a circle, reaching 1 only at the
  corners, so the middle of each edge darkens less than the corners do;
- **Chebyshev**, `max(|nx|, |ny|)` — a rectangle following the frame,
  reaching 1 along the whole border.

`roundness = 0` is the circle, `1` the rectangle.

Because the coordinates are normalised, the kernel is
**resolution-independent**: the same parameters give the same picture on
a preview and on the full-size frame. A consumer can render a small
preview and trust it. Pixel *centres* are used, so a single-row image
sits on the centre line rather than at an edge.

### Falloff

A Hermite smoothstep from `1 − feather` to `1`. The cubic `3t² − 2t³`
has zero derivative at both ends, so the vignette meets the untouched
centre and the fully-applied corner without a seam — even a narrow
feather has no banding edge.

Every channel of a pixel gets the same factor, so the vignette darkens
without tinting.

### What this is not

Real lenses vignette because off-axis illumination falls as cos⁴ of the
field angle. Correcting *that* needs the lens, the aperture and the
focal length, and belongs in the RAW decoder. This kernel is the
darkroom gesture — burning the edges to hold the eye in the frame —
with a shape the photographer picks directly.

Reference for the falloff: Ebert et al., *Texturing & Modeling: A
Procedural Approach*, 3rd ed., Morgan Kaufmann (2003), §2.3. For the
cos⁴ law: Sidney F. Ray, *Applied Photographic Optics*, 3rd ed., Focal
Press (2002), §14.

---

## 12. sRGB transfer encoding

### Specification

IEC 61966-2-1:1999 defines the sRGB colour space. The transfer function
(also called the "gamma" in informal usage) maps linear scene-referred
values to display-referred values:

```
f(x) = 12.92 · x                      if x ≤ 0.0031308
f(x) = 1.055 · x^(1/2.4) − 0.055     if x > 0.0031308
```

This kernel does **not** clamp. Values outside [0, 1] pass through the
transfer as they are: a negative input takes the linear branch and stays
negative (`encode_srgb(−0.05)` = −0.646), and a value above 1 encodes
above 1. Clamping is the caller's decision, because whether out-of-range
data is an error or headroom depends on what the consumer is doing with
it — the kernel cannot know.

Negative inputs are why the branch is written `x ≤ threshold` rather
than as a `powf` with a sign fix-up: `(−0.05)^(1/2.4)` is NaN.

### Continuity at the threshold

At x = 0.0031308:

| | Value | Derivative |
|---|---|---|
| **Linear branch** `12.92·x` | 0.040449936 | 12.920 |
| **Power branch** `1.055·x^(1/2.4) − 0.055` | 0.040449908 | 12.703 |

The values agree to 2.9 × 10⁻⁸, so the function is C⁰ — visually and
numerically seamless.

It is **not** C¹. The derivatives differ by 1.68%. Making the junction
C¹ as well requires the threshold to satisfy `φ·x₀ = a/1.4`, i.e.
x₀ = 0.055/(1.4 · 12.92) = 0.0030407, and IEC 61966-2-1 specifies
0.0031308 instead. The kink is far below the visible-difference
threshold at that luminance, which is why the standard tolerates it, but
anything differentiating the transfer (a gradient-domain operator, an
analytic inverse used in optimisation) should know the corner is there.

`src/encode.rs` asserts the C⁰ property by evaluating the kernel on both
sides of the threshold, rather than by re-deriving the formula in the
test.

### Why not a simple power law (γ = 2.2)?

The linear segment near black exists to avoid infinite slope at x = 0:
d/dx [x^(1/2.4)] → ∞ as x → 0. A pure power law would amplify noise
in very dark values and cause quantisation banding when encoding to
8-bit. The linear piece passes through the origin with finite slope
12.92, avoiding both problems.

### Implementation note

The kernel is applied element-wise, one branch at the threshold:

```rust
let encoded = if x <= 0.0031308_f32 {
    12.92 * x
} else {
    1.055 * x.powf(1.0 / 2.4) - 0.055
};
```

The branch is worth keeping rather than flattening into a branchless
blend: it is perfectly predicted over the long runs of same-side pixels
a photograph produces, and `f32::powf` dominates the cost either way.
The pixel loop is parallelised with `ndarray::Zip::par_for_each`, which
walks any input layout — see §2's layout rule.

---

## 13. Performance (v0.2-dev)

Measured with `cargo bench` (criterion, bench profile) on a synthetic
4323 × 5765 (≈ 24 MP) `f32` image filled with deterministic
pseudo-random values. Not CI gates — informational only.

| Kernel | Measured mean | Benchmark id | Notes |
|--------|--------------|--------------|-------|
| `crop` | **7.7 ms** | `crop/24MP/centre-half` | pure copy of the half-area rectangle |
| `orient` | **112.7 ms** | `orient/24MP/rotate90` | known-slow: the transposing copy is cache-hostile and currently single-threaded; quarter-turn-heavy pipelines should batch it with straighten |
| `resize` | **32.6 ms** | `resize/24MP/to-2048-area` | separable area filter to a 2048-wide export |
| `straighten` | **47.1 ms** | `straighten/24MP/2deg` | 16-tap Catmull-Rom per pixel |
| `exposure` | **29.3 ms** | `exposure/24MP/+1EV` | 3 channels in *and* out — 300 MB of traffic, twice the B&W kernels' |
| `luminance_bw` | **14.8 ms** | `luminance_bw/24MP/BT709` | Memory-bandwidth bound |
| `channel_mixer_bw` | **15.0 ms** | `channel_mixer_bw/24MP` | Same bandwidth pattern |
| `color_filter_bw` | **14.9 ms** | `color_filter_bw/24MP/Red25A` | Combined dot product |
| `hsl_bw` | **52.0 ms** | `hsl_bw/24MP/8-bands` | Hue geometry plus 8 `exp` per pixel |
| `zone_system` | **13.0 ms** | `zone_system/24MP/1-zone-offset` | `exp` + `log2` per pixel |
| `tone_curve` | **10.4 ms** | `tone_curve/24MP/slope-offset-power` | One `powf` per pixel |
| `local_contrast` | **177.5 ms** | `local_contrast/24MP/r=8` | 4 parallel SAT builds |
| `film_grain` | **58.1 ms** | `film_grain/24MP/size=2` | Hash + Box–Muller per pixel, then one SAT |
| `split_toning` | **23.4 ms** | `split_toning/24MP` | 1 cube root in, 3 cubes out, per pixel |
| `vignette` | **10.6 ms** | `vignette/24MP` | `sqrt` per pixel |
| `encode_srgb` | **10.7 ms** | `encode_srgb/24MP` | `powf` per pixel |

A full v0.2 pipeline — exposure, HSL conversion, zone system, local
contrast, grain, toning, vignette, curve, encode — is therefore around
400 ms for a 24 MP frame on this machine, dominated by `local_contrast`.

Machine: AMD Ryzen 9 9950X 16-Core (32 threads), Linux, `cargo bench`
(optimised profile, rayon parallelism enabled).

`local_contrast` remains SAT-dominated: four prefix-sum passes each
touch every pixel once, setting a memory-bandwidth floor of roughly
200 MB per f64 table. Making those passes parallel (§8) took the kernel
from 452 ms to 178 ms, and peak scratch from 1.3 GB to ~600 MB. Runtime
is independent of radius, as the O(1) formulation requires: r = 32
measures the same 178 ms as r = 8.

Comparisons with numbers published before v0.2 need care: benchmarks up
to v0.1.1 ran on a *constant* image, which is the cheapest possible
input for `encode_srgb` (one branch), `zone_system` (one zone position)
and `local_contrast` (zero variance everywhere). Against the same flat
input, the three B&W kernels measured 10.1 ms rather than 15 ms.

---

## 14. Geometry (crop, orientation)

Two exact operations, deliberately in the core rather than in front
ends: their parameters live in consumers' sidecar files, and if two
front ends implemented the same crop differently, the same sidecar
would render different images.

Both are **pure index permutations** — no arithmetic on pixel values —
so they are bit-identical across every backend unconditionally, unlike
the transcendental-bearing kernels of §13.

**`crop(img, CropParams{x, y, width, height})`** extracts the
rectangle; it must lie entirely within the frame (validated with u64
arithmetic so near-`u32::MAX` coordinates cannot wrap). Zero-size
rectangles are legal, per the crate's zero-size policy.

**`orient(img, Orientation)`** applies one of the eight dihedral
transforms. The enum discriminants are the Exif orientation codes 1..=8
(JEITA CP-3451, tag 0x0112); rotations are clockwise. Internally every
variant decomposes to `(transpose, flip_y, flip_x)` with the flips
acting on the *output* axes — one shared definition
(`Orientation::flags`) that both backends implement, so they cannot
drift. The composition properties (each orientation undone by its
`inverse()`, four quarter-turns closing to the identity) are asserted
in the test suite.

**Pipeline position: first**, `orient` before `crop`, both before
exposure and the look pipeline. See the order table in §1 — the
vignette centre and the grain coordinate grid make this load-bearing,
not stylistic.

### Resampling: `resize` and `straighten`

Both resample with **polynomial filters only**, and the one
transcendental — sin/cos of the straighten angle — is evaluated once on
the host and passed to both backends as identical f32 scalars. Every
per-pixel operation is then a correctly rounded mul/add/div/floor in a
fixed accumulation order, so both kernels are **bit-exact across
backends**, verified by `assert_eq!` in the conformance suite over
twelve resize configurations and five angles.

`resize(img, ResizeParams{width, height, filter})` is separable
(horizontal pass, then vertical over the transposed view — identical
arithmetic per axis). Centre alignment `c = (i + 0.5)·scale − 0.5`
makes a same-size resize the *exact* identity. Filters:

| Filter | Use | Definition |
|---|---|---|
| `Area` | downscale | exact fractional pixel coverage — true area averaging at any ratio (degenerates to nearest when upscaling) |
| `Bilinear` | cheap | triangle, radius 1 (scaled under minification) |
| `CatmullRom` | upscale | Keys 1981 cubic, a = −0.5, radius 2 — third-order accurate, so it reproduces polynomials up to *quadratic* exactly (a linear ramp upscales exactly; a cubic does not — f(x) = x³ interpolates to t − 3t² + 3t³ between samples) |

A constant image survives to ~1 ULP (weighted sum and weight sum round
separately before the normalising division); tap coordinates clamp to
the frame (replicate borders).

`straighten(img, StraightenParams{degrees})` rotates by up to ±45°
(positive clockwise, matching `Rotate90`; compose with `orient` beyond
that) and crops to the **largest axis-aligned rectangle inscribed** in
the rotated frame (the standard max-area two-case construction,
computed in f64 on the host). Sampling is 16-tap Catmull-Rom in a fixed
4×4 order; `degrees = 0` is the exact identity. The cubic's ±2-pixel
support can reach frame-edge pixels in the outermost ~2-pixel band of
the result, where taps clamp (replicate) — standard practice,
documented rather than hidden. Geometry order within the group:
`orient` → `straighten` → `crop` → `resize`.
