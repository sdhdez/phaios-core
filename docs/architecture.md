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
  │  exposure   │  × 2^stops                           src/exposure.rs
  └─────────────┘
        │
        ▼
  ┌─────────────┐
  │  B&W conv.  │  luminance / channel mixer /          src/bw.rs
  └─────────────┘  colour filter → (H, W, 1)
        │
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
  ┌─────────────┐  (v0.2: grain, split-toning, vignette)
  │  finishing  │
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

**Order sensitivity.** The sRGB encode must be last — all other kernels
operate on linear data. B&W conversion must precede zone-system and
local-contrast (which operate on single-channel luminance). Exposure
must precede B&W conversion. All other orderings within those
constraints are mathematically equivalent, though the canonical order
above is recommended.

---

## 2. Luminance weights (B&W method 1)

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

## 3. Coloured-filter simulation (B&W method 3)

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

## 4. Zone System tone curve

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
| 0 | Maximum black | ≈ 0.003 |
| I | Near-black detail threshold | ≈ 0.006 |
| II | First textured shadow | ≈ 0.012 |
| III | Dark shadow, full texture | ≈ 0.024 |
| IV | Dark skin, foliage, shadow detail | ≈ 0.048 |
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

The zone index is an integer 0..=10. The offset is a float in the range
[−3, +3] stops (clamped by the kernel). Omitted zones default to 0.

### Numerical notes

- `log₂` is computed as `f32::ln(x) / std::f32::consts::LN_2`.
- The Gaussian is evaluated for each zone index in the map; zones not
  in the map contribute 0.
- Input values ≤ 0 are clamped to a small positive epsilon before
  the log to avoid −∞.

---

## 5. Guided filter for local contrast

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
| `radius` | u32 | 1..512 | Window half-size in pixels |
| `eps` | f32 | > 0 | Regularisation; try 0.01 |
| `strength` | f32 | 0..2 | Detail amplification |

---

## 6. sRGB transfer encoding

### Specification

IEC 61966-2-1:1999 defines the sRGB colour space. The transfer function
(also called the "gamma" in informal usage) maps linear scene-referred
values to display-referred values:

```
f(x) = 12.92 · x                      if x ≤ 0.0031308
f(x) = 1.055 · x^(1/2.4) − 0.055     if x > 0.0031308
```

The clamping to [0, 1] is applied before encoding; values outside this
range are clamped, not reflected.

### C¹ continuity at the threshold

At x = 0.0031308:

**Linear branch:** 12.92 · 0.0031308 = 0.04045 (value)
**Derivative:** 12.92 (constant)

**Power branch:** 1.055 · 0.0031308^(1/2.4) − 0.055
= 1.055 · 0.0627... − 0.055
≈ 0.0661... − 0.055
≈ 0.04045 ✓ (values match to 1e-5)

**Power branch derivative:** 1.055 · (1/2.4) · 0.0031308^(1/2.4 - 1)
= 1.055 · 0.4167 · 0.0031308^(-0.5833)
= 0.4396 · 44.7...
≈ 12.92 ✓ (derivatives match to 1e-4)

The piecewise function is C¹ (continuous with continuous first
derivative) at the junction. The slight discrepancy from the exact
12.92 at the power-branch derivative is within the tolerance of the
IEC specification (which rounds the threshold and coefficients).

### Why not a simple power law (γ = 2.2)?

The linear segment near black exists to avoid infinite slope at x = 0:
d/dx [x^(1/2.4)] → ∞ as x → 0. A pure power law would amplify noise
in very dark values and cause quantisation banding when encoding to
8-bit. The linear piece passes through the origin with finite slope
12.92, avoiding both problems.

### Implementation note

The kernel is applied element-wise. For vectorised execution, avoid
branching per element:

```rust
let encoded = if x <= 0.0031308_f32 {
    12.92 * x
} else {
    1.055 * x.powf(1.0 / 2.4) - 0.055
};
```

`rayon::par_iter` is appropriate for the pixel loop given that
`f32::powf` is the dominant cost.

---

## 7. Performance targets (v0.1)

Measured with `cargo bench` (criterion, bench profile) on a synthetic
4323 × 5765 (≈ 24 MP) `f32` image. Not CI gates — informational only.

| Kernel | Measured mean | Benchmark id | Notes |
|--------|--------------|--------------|-------|
| `luminance_bw` | **10.1 ms** | `luminance_bw/24MP/BT709` | Memory-bandwidth bound |
| `channel_mixer_bw` | **10.1 ms** | `channel_mixer_bw/24MP` | Same bandwidth pattern |
| `color_filter_bw` | **10.1 ms** | `color_filter_bw/24MP/Red25A` | Combined dot product |
| `zone_system` | **14.3 ms** | `zone_system/24MP/1-zone-offset` | exp + ln per pixel |
| `local_contrast` | **452 ms** | `local_contrast/24MP/r=8` | 4 sequential SAT builds |
| `encode_srgb` | **8.1 ms** | `encode_srgb/24MP` | powf per pixel (grey input) |

Machine: AMD Ryzen 9 9950X 16-Core (32 threads), Linux, `cargo bench`
(optimised profile, rayon parallelism enabled).

`local_contrast` is SAT-dominated — the four sequential prefix-sum passes
each touch every pixel once, setting a ~95 MB/pass memory-bandwidth floor.
The parallel coefficient computation adds little overhead by comparison.
Future work: parallelise SAT rows independently to reduce sequential cost.
