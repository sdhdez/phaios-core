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
  │  exposure   │  × 2^stops    (planned, v0.2)        src/exposure.rs
  └─────────────┘  not implemented — consumers apply
        │          their own exposure for now
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

**Layout.** Every kernel accepts any array layout — C-contiguous,
Fortran-order, strided views, negative strides — and always returns a
freshly allocated C-contiguous array. See `docs/ffi.md` §1.

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

## 6. sRGB transfer encoding

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

## 7. Performance (v0.2-dev)

Measured with `cargo bench` (criterion, bench profile) on a synthetic
4323 × 5765 (≈ 24 MP) `f32` image filled with deterministic
pseudo-random values. Not CI gates — informational only.

| Kernel | Measured mean | Benchmark id | Notes |
|--------|--------------|--------------|-------|
| `luminance_bw` | **15.0 ms** | `luminance_bw/24MP/BT709` | Memory-bandwidth bound |
| `channel_mixer_bw` | **15.0 ms** | `channel_mixer_bw/24MP` | Same bandwidth pattern |
| `color_filter_bw` | **14.9 ms** | `color_filter_bw/24MP/Red25A` | Combined dot product |
| `zone_system` | **13.0 ms** | `zone_system/24MP/1-zone-offset` | `exp` + `log2` per pixel |
| `local_contrast` | **177.8 ms** | `local_contrast/24MP/r=8` | 4 parallel SAT builds |
| `encode_srgb` | **10.5 ms** | `encode_srgb/24MP` | `powf` per pixel |

Machine: AMD Ryzen 9 9950X 16-Core (32 threads), Linux, `cargo bench`
(optimised profile, rayon parallelism enabled).

`local_contrast` remains SAT-dominated: four prefix-sum passes each
touch every pixel once, setting a memory-bandwidth floor of roughly
200 MB per f64 table. Making those passes parallel (§5) took the kernel
from 452 ms to 178 ms, and peak scratch from 1.3 GB to ~600 MB. Runtime
is independent of radius, as the O(1) formulation requires: r = 32
measures the same 178 ms as r = 8.

Comparisons with numbers published before v0.2 need care: benchmarks up
to v0.1.1 ran on a *constant* image, which is the cheapest possible
input for `encode_srgb` (one branch), `zone_system` (one zone position)
and `local_contrast` (zero variance everywhere). Against the same flat
input, the three B&W kernels measured 10.1 ms rather than 15 ms.
