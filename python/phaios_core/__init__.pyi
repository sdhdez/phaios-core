"""The `phaios_core` Python extension module.

Exposes the numerical kernels as Python-callable functions. All
arrays are `numpy.float32`, C-contiguous, shape `(H, W, C)`."""

import numpy as np
from numpy.typing import NDArray
from typing import ClassVar
from collections.abc import Sequence

__version__: str

class LuminanceStandard:
    """ITU-R luminance standard for B&W conversion.

Selects which RGB primaries' Y-row coefficients to use when computing
the perceptual luminance of a scene-referred linear f32 image.
`Bt709` is the default and the correct choice for sRGB-primary data
(the vast majority of consumer RAW pipelines)."""

    Bt601: ClassVar[LuminanceStandard]  # 0
    Bt709: ClassVar[LuminanceStandard]  # 1
    Bt2020: ClassVar[LuminanceStandard]  # 2

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ColorFilter:
    """Wratten-style coloured-filter preset for B&W contrast control.

Each variant represents a gel filter placed in front of the lens.
The filter attenuates light by channel, shifting the relative tonal
values of differently coloured subjects.

Channel transmission values are sampled at the centroid wavelengths
of a nominal sRGB camera's R/G/B channels (~620, ~540, ~450 nm) from
the Kodak Wratten Gelatin Filters datasheet, B3-203 (5th ed.).
These are approximations — real Wratten curves are continuous spectra.

**Note:** real filters reduce total exposure. This kernel does not
apply exposure compensation — do that in the exposure kernel."""

    NoFilter: ClassVar[ColorFilter]  # 0
    Yellow8K2: ClassVar[ColorFilter]  # 1
    Orange21: ClassVar[ColorFilter]  # 2
    Red25A: ClassVar[ColorFilter]  # 3
    Green11X1: ClassVar[ColorFilter]  # 4
    Blue47C5: ClassVar[ColorFilter]  # 5

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class CropParams:
    """A crop rectangle, in pixels of the frame it is applied to — after
`orient` and `straighten` in the standard geometry order, i.e. the
upright, levelled frame.

`x`, `y` locate the top-left corner; the rectangle must lie entirely
within the image.

```python
params = phaios_core.CropParams(x=100, y=50, width=3000, height=2000)
```"""

    def __new__(cls, x: int, y: int, width: int, height: int) -> CropParams: ...

    x: int
    """Left edge, pixels from the left of the frame."""

    y: int
    """Top edge, pixels from the top of the frame."""

    width: int
    """Width of the result in pixels."""

    height: int
    """Height of the result in pixels."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class Orientation:
    """One of the eight dihedral transforms of the frame, encoded as its
Exif orientation value (JEITA CP-3451, tag 0x0112).

Rotations are **clockwise**, matching the Exif reading ("Rotate 90
CW" is what a viewer must apply to display the frame upright)."""

    Normal: ClassVar[Orientation]  # 1
    FlipHorizontal: ClassVar[Orientation]  # 2
    Rotate180: ClassVar[Orientation]  # 3
    FlipVertical: ClassVar[Orientation]  # 4
    Transpose: ClassVar[Orientation]  # 5
    Rotate90: ClassVar[Orientation]  # 6
    Transverse: ClassVar[Orientation]  # 7
    Rotate270: ClassVar[Orientation]  # 8

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ResizeFilter:
    """Resampling filter for [`resize`]."""

    Area: ClassVar[ResizeFilter]  # 0
    Bilinear: ClassVar[ResizeFilter]  # 1
    CatmullRom: ClassVar[ResizeFilter]  # 2

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ResizeParams:
    """Parameters for [`resize`].

```python
params = phaios_core.ResizeParams(2048, 1365, phaios_core.ResizeFilter.Area)
```"""

    def __new__(cls, width: int, height: int, filter: ResizeFilter = ResizeFilter.Area) -> ResizeParams: ...

    width: int
    """Target width in pixels."""

    height: int
    """Target height in pixels."""

    filter: ResizeFilter
    """The resampling filter."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class StraightenParams:
    """Parameters for [`straighten`].

```python
params = phaios_core.StraightenParams(degrees=-1.8)  # level a tilted horizon
```"""

    def __new__(cls, degrees: float) -> StraightenParams: ...

    degrees: float
    """Rotation in degrees, **positive clockwise**, limited to ±45°
(compose with [`orient`] for quarter turns)."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class HslWeightedParams:
    """Parameters for [`hsl_bw`].

```python
# Darken blues (sky), lift yellows (foliage in autumn light)
params = phaios_core.HslWeightedParams([0, 0, 0.6, 0, 0, -0.7, 0, 0])
```"""

    def __new__(cls, hue_weights: Sequence[float], standard: LuminanceStandard = LuminanceStandard.Bt709, sigma_deg: float = 30.0) -> HslWeightedParams: ...

    @property
    def hue_weights(self) -> list[float]:
        """Per-band luminance multiplier, in the order of
[`HUE_BAND_CENTRES_DEG`]: red, orange, yellow, green, aqua, blue,
purple, magenta. −1..+1 is the useful range; not clamped."""
        ...

    @hue_weights.setter
    def hue_weights(self, value: Sequence[float]) -> None: ...

    standard: LuminanceStandard
    """Luminance standard used for the base greyscale value."""

    sigma_deg: float
    """Gaussian width in hue space, in degrees. Larger values blend
neighbouring bands together; 30° makes adjacent bands overlap at
roughly half weight."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ZoneParams:
    """Zone System tone offsets.

Maps zone index (0..=10) to a stop offset in −3..+3. Zones not
present in the map are treated as having a 0-stop offset.

```python
# Lift Zone V by half a stop, deepen Zone III by 0.3 stops
params = phaios_core.ZoneParams({5: 0.5, 3: -0.3})
```"""

    def __new__(cls, offsets: dict[int, float]) -> ZoneParams: ...

    @property
    def offsets(self) -> dict[int, float]:
        """The zone-index → stop-offset map, as a dict.

Returns a copy: mutating it does not change the ``ZoneParams``.
Consumers need this to serialise settings — the desktop app's
sidecar and preset files round-trip through it."""
        ...

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ToneCurveParams:
    """Parameters for [`tone_curve`], in ASC CDL terms.

The three knobs a colourist expects, under their photographic names:

| Field | Also called | Effect |
|-------|-------------|--------|
| `slope` | gain | scales the whole range about black |
| `offset` | lift | shifts the whole range, black included |
| `power` | gamma | bends the midtones, leaving 0 and 1 fixed |

The identity is `(1.0, 0.0, 1.0)`.

```python
# Lift the blacks slightly and open up the midtones
params = phaios_core.ToneCurveParams(slope=1.0, offset=0.02, power=0.85)
```"""

    def __new__(cls, slope: float = 1.0, offset: float = 0.0, power: float = 1.0) -> ToneCurveParams: ...

    slope: float
    """Multiplier applied before the offset. 1.0 is neutral."""

    offset: float
    """Added after the slope. 0.0 is neutral. Positive values lift black."""

    power: float
    """Exponent applied last. 1.0 is neutral; below 1 brightens the
midtones, above 1 darkens them. Must be positive."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class GuidedFilterParams:
    """Parameters for the guided filter.

```python
params = phaios_core.GuidedFilterParams(radius=8, eps=0.01)
```"""

    def __new__(cls, radius: int, eps: float) -> GuidedFilterParams: ...

    radius: int
    """Filter radius in pixels. The window is `(2r+1) × (2r+1)`."""

    eps: float
    """Regularisation term ε. Controls the degree of smoothing.
Larger values → more smoothing, less edge preservation."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class GrainParams:
    """Parameters for [`film_grain`].

```python
params = phaios_core.GrainParams(intensity=0.25, size_pixels=1.5, seed=20260815)
```"""

    def __new__(cls, intensity: float = 0.0, size_pixels: float = 1.0, seed: int = 0) -> GrainParams: ...

    intensity: float
    """Grain amplitude at the midtones, 0..=1 for the useful range.

The band-passed noise is normalised to unit variance, so this is
the **standard deviation** of the grain in linear units where the
`4·L·(1−L)` envelope peaks, at L = 0.5. Individual pixels reach
further, as Gaussian tails do. Not clamped, but values above 1
swamp the image.

The measured spread falls slightly below `intensity` in dark
tones, where the output clamp at zero truncates the lower tail."""

    size_pixels: float
    """Characteristic grain size in pixels; 0.5..=4.0 is typical.

This is a *scale*, not a radius: the noise is band-passed around
it, so raising it makes the clumps coarser rather than merely
blurrier. Sizes below 1 pixel cannot be resolved and behave as 1."""

    seed: int
    """Explicit RNG seed. Same seed, same parameters, same input →
bit-identical output, on any thread count and any platform."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class SplitToningParams:
    """Parameters for [`split_toning`].

```python
# Cool shadows, warm highlights — the classic cross-process look.
params = phaios_core.SplitToningParams(
    shadow_oklab=[0.0, -0.02, -0.06],
    highlight_oklab=[0.0, 0.03, 0.05],
    pivot=0.5,
    balance=0.0,
)
```"""

    def __new__(cls, shadow_oklab: Sequence[float] = [0.0, 0.0, 0.0], highlight_oklab: Sequence[float] = [0.0, 0.0, 0.0], pivot: float = 0.5, balance: float = 0.0) -> SplitToningParams: ...

    @property
    def shadow_oklab(self) -> list[float]:
        """Shadow tint as an OKLab triple `[L, a, b]`.

**Only `a` and `b` are used.** The lightness component is ignored
deliberately: toning must not move the tonal rendering that the
zone system and the tone curve just established. The field takes
a full triple so a colour picked in an OKLab picker can be passed
through unmodified.

Typical chroma magnitudes are small — 0.02 is a clear tint, 0.1 is
heavy-handed."""
        ...

    @shadow_oklab.setter
    def shadow_oklab(self, value: Sequence[float]) -> None: ...

    @property
    def highlight_oklab(self) -> list[float]:
        """Highlight tint as an OKLab triple `[L, a, b]`. As above, only `a`
and `b` are used."""
        ...

    @highlight_oklab.setter
    def highlight_oklab(self, value: Sequence[float]) -> None: ...

    pivot: float
    """Lightness at which the two tints mix equally, 0..=1, in OKLab
lightness (not linear luminance — OKLab L of middle grey is about
0.57, not 0.18)."""

    balance: float
    """Shifts the crossover, −1..=1. Positive favours the highlight
tint by moving the crossover down; negative favours the shadow
tint. `0.0` leaves the pivot where it is."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class VignetteParams:
    """Parameters for [`vignette`].

```python
# Classic subtle corner burn
params = phaios_core.VignetteParams(amount=0.35, feather=0.6, roundness=0.0)
```"""

    def __new__(cls, amount: float = 0.0, feather: float = 0.5, roundness: float = 0.0) -> VignetteParams: ...

    amount: float
    """Strength at the corners. **Positive darkens, negative lightens.**

−1..+1 is the useful range: at `+1.0` the corners reach black, at
`−1.0` they are doubled. Values beyond that are allowed; the
result is clamped at zero so the image never goes negative."""

    feather: float
    """Width of the transition, 0..=1.

`1.0` spreads the falloff from the centre all the way to the
corners — the gentlest, most natural-looking option. Smaller
values push the transition outwards, concentrating it near the
corners; `0.0` is a hard edge with no gradient at all."""

    roundness: float
    """Corner shape, 0..=1. `0.0` is a circle, `1.0` follows the frame.

A circular vignette darkens the middle of each edge as well as
the corners; at `1.0` the iso-lines are rectangles parallel to
the frame, so only the border darkens, evenly."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class BlurShape:
    """Blur kernel shape."""

    Gaussian: ClassVar[BlurShape]  # 0

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class BlurParams:
    """Parameters for [`blur`].

```python
params = phaios_core.BlurParams(sigma=8.0)
```"""

    def __new__(cls, sigma: float = 0.0, shape: BlurShape = BlurShape.Gaussian) -> BlurParams: ...

    sigma: float
    """Standard deviation in pixels. `0.0` is the identity.

Measured in pixels of the image as given, so a blur applied to a
half-size preview is *not* the same picture as the same σ on the
full frame — unlike [`crate::vignette`], which is
resolution-independent by construction. Scale σ with the image if
you are previewing.

Bounded above by [`MAX_SIGMA`]."""

    shape: BlurShape
    """Kernel shape. Default [`BlurShape::Gaussian`]."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class GlowParams:
    """Parameters for [`glow`].

```python
# Halation: bright areas only, moderate spread, applied early
params = phaios_core.GlowParams(threshold=0.8, sigma=8.0, amount=0.35)

# Veiling glare: everything scatters, frame-wide, applied earlier still
params = phaios_core.GlowParams(threshold=0.0, sigma=400.0, amount=0.06)
```"""

    def __new__(cls, threshold: float = 0.0, sigma: float = 8.0, amount: float = 0.0) -> GlowParams: ...

    threshold: float
    """Level above which light scatters. `0.0` means all of it does.

Subtracted rather than used as a hard cut — the weight is
`max(in − threshold, 0)` — so the contribution fades in smoothly
and a bright region does not acquire an outline at the threshold."""

    sigma: float
    """Standard deviation of the scatter, in pixels.

Small for a tight halo, frame-spanning for glare. Being in pixels,
it is resolution-dependent: scale it with the image when working
on a preview."""

    amount: float
    """How much scattered light is added back. `0.0` is the identity and
the default.

Non-negative: this kernel adds light, it does not subtract it."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class RolloffParams:
    """Parameters for [`highlight_rolloff`].

```python
# Hold two stops of highlight detail above the knee
params = phaios_core.RolloffParams(knee=0.75, white_point=4.0)

# The default is a hard clip at 1.0 — identical to np.clip(x, 0, 1)
params = phaios_core.RolloffParams()
```"""

    def __new__(cls, knee: float = 1.0, white_point: float = 1.0) -> RolloffParams: ...

    knee: float
    """Where compression begins, 0..=1.

Values at or below the knee pass through untouched, so this is
the promise that midtones are not disturbed. Lowering it recruits
more of the tonal range into the shoulder, which buys smoother
highlights at the cost of some contrast just below white."""

    white_point: float
    """The scene value that becomes pure white, ≥ 1.0.

`1.0` means "clip at 1.0" and gives back the hard clip. `4.0`
means a value two stops above nominal white is what finally
reaches 1.0, so those two stops of highlight survive as detail
instead of collapsing to a flat patch. Anything above the white
point is white."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class ShadowRolloffParams:
    """Parameters for [`shadow_rolloff`].

```python
# A moderate toe over the bottom fifth of the scale
params = phaios_core.ShadowRolloffParams(knee=0.2, strength=0.6)

# The default leaves the image alone
params = phaios_core.ShadowRolloffParams()
```"""

    def __new__(cls, knee: float = 0.2, strength: float = 0.0) -> ShadowRolloffParams: ...

    knee: float
    """Input value above which nothing changes, 0..=1.

The toe occupies `[0, knee]`, and everything above it is returned
bit-identical — the knee is a hard boundary on *where* the curve
acts, not a fade. Larger values recruit more of the tonal range
into the compression, which softens a wider band of shadows; the
cost is paid inside that band, in the tones just below the knee
that were previously untouched, not above it."""

    strength: float
    """How hard the shadows are compressed, 0..=1.

This is one minus the slope at black. `0.0` is the identity and
the default. `1.0` takes the slope at black to zero, so tones near
zero run together completely — the deepest shadows become a single
black rather than a graded near-black."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class Dither:
    """Dither strategy for [`quantize_u8`] / [`quantize_u16`]."""

    Off: ClassVar[Dither]  # 0
    Tpdf: ClassVar[Dither]  # 1

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class QuantizeParams:
    """Parameters for [`quantize_u8`] and [`quantize_u16`].

```python
# 16-bit archival export, no dither needed
params = phaios_core.QuantizeParams()

# 8-bit web export, dithered
params = phaios_core.QuantizeParams(phaios_core.Dither.Tpdf, seed=20260825)
```"""

    def __new__(cls, dither: Dither = Dither.Off, seed: int = 0) -> QuantizeParams: ...

    dither: Dither
    """Dither strategy. Default [`Dither::Off`]."""

    seed: int
    """Seed for the dither pattern. Ignored when `dither` is
[`Dither::Off`].

Explicit, like every other source of randomness in the crate: the
same seed reproduces the same file, and the seed belongs in the
settings that travel with the image."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class HistogramParams:
    """Parameters for [`histogram`].

```python
# The display default: 256 bins over [0, 1]
params = phaios_core.HistogramParams()

# Fine-grained analysis of a 16-bit export
params = phaios_core.HistogramParams(bins=65536)
```"""

    def __new__(cls, bins: int = 256, min: float = 0.0, max: float = 1.0) -> HistogramParams: ...

    bins: int
    """Number of bins spanning `[min, max]`. Must be at least 2.

256 matches an 8-bit display and is the sensible default for a
histogram a person looks at. Larger values are for analysis —
65536 resolves individual 16-bit codes, which is how you find
quantisation damage."""

    min: float
    """Lower edge of the counted range. Samples below it are counted in
`below` rather than in bin 0."""

    max: float
    """Upper edge of the counted range, inclusive. Samples above it are
counted in `above` rather than in the last bin.

Exactly `max` lands in the last bin, so a white pixel in a
display-referred image reads as "at white" rather than "clipped"."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

class Histogram:
    """The result of [`histogram`]: per-channel bin counts plus the samples
that fell outside the counted range.

Counts are `u64`, which cannot overflow for any image that fits in
memory.

`skip_from_py_object` because this is a *result*: it is handed to
Python and never taken back as a kernel argument, so the conversion
PyO3 would derive has no call site."""

    @property
    def bins(self) -> int:
        """Number of bins per channel."""
        ...

    @property
    def channels(self) -> int:
        """Number of channels."""
        ...

    @property
    def min(self) -> float:
        """Lower edge of the counted range."""
        ...

    @property
    def max(self) -> float:
        """Upper edge of the counted range."""
        ...

    def counts(self) -> NDArray[np.uint64]:
        """Bin counts as a ``(channels, bins)`` array of ``uint64``."""
        ...

    def cdf(self) -> NDArray[np.float32]:
        """Normalised cumulative distribution, ``(channels, bins)`` float32.

Entry ``b`` is the cumulative fraction at the upper edge of bin
``b``. For equalisation use ``equalisation_lut()`` instead — see
its documentation for why the alignment differs."""
        ...

    def equalisation_lut(self) -> NDArray[np.float32]:
        """The equalising transfer, ``(channels, bins + 1)`` float32, ready
to pass to ``apply_lut`` over this histogram's own range.

``cdf()`` with a leading zero: the extra entry aligns the CDF's
bin-upper-edge convention with ``apply_lut``'s even placement of
table entries, removing a half-bin bias that lifted black."""
        ...

    def below(self) -> list[int]:
        """Per-channel count of samples below ``min``."""
        ...

    def above(self) -> list[int]:
        """Per-channel count of samples above ``max``."""
        ...

    def non_finite(self) -> list[int]:
        """Per-channel count of NaN samples."""
        ...

    def total(self, channel: int) -> int:
        """Per-channel total of every sample seen."""
        ...

    def __repr__(self) -> str: ...

class LutParams:
    """Domain of the table passed to [`apply_lut`].

```python
# A table covering the display range
params = phaios_core.LutParams()

# A table covering three stops of highlight headroom
params = phaios_core.LutParams(min=0.0, max=8.0)
```"""

    def __new__(cls, min: float = 0.0, max: float = 1.0) -> LutParams: ...

    min: float
    """Input value mapped to the table's first entry.

Anything at or below this takes the first entry — the table is
clamped, not extrapolated. Extrapolating a curve the photographer
drew past the range they drew it over invents data."""

    max: float
    """Input value mapped to the table's last entry."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    __hash__: ClassVar[None]

def crop(img: NDArray[np.float32], params: CropParams) -> NDArray[np.float32]:
    """Crop to a rectangle.

A pure index copy — no pixel value is touched, so the result is
bit-identical on every backend. Geometry runs first in the pipeline:
the vignette centres on the frame it is given (which must be the
cropped frame) and film grain keys its noise to pixel coordinates
(which must be the final grid).

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : CropParams
    The rectangle: ``x``, ``y`` top-left corner, ``width``,
    ``height``. Must lie entirely within the frame.

Returns
-------
numpy.ndarray
    Shape ``(height, width, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If the rectangle exceeds the frame."""
    ...

def orient(img: NDArray[np.float32], orientation: Orientation) -> NDArray[np.float32]:
    """Apply one of the eight dihedral orientations (Exif tag 0x0112).

A pure index permutation — bit-identical on every backend. Rotations
are clockwise; the transposing variants swap width and height.
Orientation precedes crop in the pipeline, so crop rectangles are
expressed in the upright frame.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
orientation : Orientation
    ``Orientation.Normal`` … ``Orientation.Rotate270``; the enum
    values are the Exif orientation codes 1..=8.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)`` or ``(W, H, C)``, dtype ``float32``,
    C-contiguous."""
    ...

def resize(img: NDArray[np.float32], params: ResizeParams) -> NDArray[np.float32]:
    """Resample to a new size with a separable polynomial filter.

``ResizeFilter.Area`` computes exact fractional pixel coverage — the
correct choice for downscaling; ``ResizeFilter.CatmullRom`` (Keys
1981) is the photographic default for upscaling. Bit-exact across
backends. A same-size resize is the exact identity.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)``, dtype ``float32``, any layout.
params : ResizeParams
    Target width, height and the filter.

Returns
-------
numpy.ndarray
    Shape ``(height, width, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If a target dimension is zero or the input is empty."""
    ...

def straighten(img: NDArray[np.float32], params: StraightenParams) -> NDArray[np.float32]:
    """Rotate by a small angle (±45°, positive clockwise) and crop to the
largest inscribed rectangle.

16-tap Catmull-Rom resampling; the only transcendentals (sin/cos of
the one angle) are computed on the host, so the kernel is bit-exact
across backends. ``degrees=0`` is the exact identity. Compose with
``orient`` for quarter turns.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)``, dtype ``float32``, any layout.
params : StraightenParams
    The angle in degrees.

Returns
-------
numpy.ndarray
    The inscribed rectangle, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If the angle is not finite, exceeds ±45°, or leaves no whole
    pixel inscribed."""
    ...

def exposure(img: NDArray[np.float32], stops: float) -> NDArray[np.float32]:
    """Apply exposure compensation in EV stops.

Computes ``out = img * 2**stops``. This is the first pipeline stage:
it operates on linear scene-referred data, where a stop is by
definition a factor of two.

Values are not clamped — highlights pushed above 1.0 stay there so
later tone stages can recover them.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
stops : float
    Exposure adjustment in EV. Positive brightens, negative darkens,
    ``0.0`` is the identity.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``stops`` is not finite."""
    ...

def luminance_bw(img: NDArray[np.float32], standard: LuminanceStandard = LuminanceStandard.Bt709) -> NDArray[np.float32]:
    """Convert a linear RGB image to greyscale using standard luminance weights.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory
    layout, scene-referred linear sRGB values.
standard : LuminanceStandard, optional
    Which ITU-R standard to use. Default: ``LuminanceStandard.Bt709``.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``, linear luminance.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 3)``."""
    ...

def channel_mixer_bw(img: NDArray[np.float32], wr: float, wg: float, wb: float) -> NDArray[np.float32]:
    """Convert a linear RGB image to greyscale using arbitrary channel weights.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory layout.
wr, wg, wb : float
    Per-channel weights. Range −2..+2 is conventional; negative weights
    produce infrared-like inversions. Weights need not sum to one.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 3)``."""
    ...

def color_filter_bw(img: NDArray[np.float32], filter: ColorFilter = ColorFilter.NoFilter, standard: LuminanceStandard = LuminanceStandard.Bt709) -> NDArray[np.float32]:
    """Convert a linear RGB image to greyscale using a Wratten-style filter.

Applies the filter's per-channel transmission vector then collapses
to luminance using ``standard``.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 3)``, dtype ``float32``, any memory layout.
filter : ColorFilter, optional
    Wratten-style preset. Default: ``ColorFilter.NoFilter``.
standard : LuminanceStandard, optional
    Which ITU-R standard to use. Default: ``LuminanceStandard.Bt709``.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 3)``."""
    ...

def hsl_bw(img: NDArray[np.float32], params: HslWeightedParams) -> NDArray[np.float32]:
    """Convert a linear RGB image to greyscale with per-hue-band weighting.

Computes a base luminance, then scales it by
``1 + Σ w_i · gaussian_i(hue) · chroma`` over eight hue bands
(red 0°, orange 30°, yellow 60°, green 120°, aqua 180°, blue 240°,
purple 270°, magenta 300°). The result is clamped at zero.

The saturation measure is the chroma ratio ``(max − min) / max``,
which is invariant under exposure changes — see the Rust docs for
why HSL saturation is not used.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 3)``, dtype ``float32``, any layout.
params : HslWeightedParams
    Eight band weights, luminance standard, and Gaussian width.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 3)``, if ``sigma_deg`` is not
    finite and positive, or if any weight is not finite."""
    ...

def zone_system(img: NDArray[np.float32], params: ZoneParams) -> NDArray[np.float32]:
    """Apply the Adams/Archer Zone System tone curve.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 1)``, dtype ``float32``, linear
    luminance (output of a B&W conversion kernel). Any memory layout
    is accepted; the returned array is always C-contiguous.
params : ZoneParams
    Zone offsets mapping zone index 0..10 → stop offset.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``, tone-adjusted luminance.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 1)``."""
    ...

def tone_curve(img: NDArray[np.float32], params: ToneCurveParams) -> NDArray[np.float32]:
    """Apply a parametric slope/offset/power tone curve.

Computes ``out = max(img * slope + offset, 0) ** power``,
element-wise. This is the ASC Color Decision List primary
correction — "gain, lift, gamma" in photographic terms.

Monotonic for any positive ``slope`` and ``power``, so it cannot
invert tonal order. The clamp before the exponent means a negative
``offset`` crushes to black rather than producing NaN.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : ToneCurveParams
    Slope, offset and power. The identity is ``(1.0, 0.0, 1.0)``.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If any parameter is not finite, or ``power`` is not positive."""
    ...

def blur(img: NDArray[np.float32], params: BlurParams | None = None) -> NDArray[np.float32]:
    """Blur an image with an isotropic Gaussian of standard deviation
sigma, in pixels.

sigma = 0.0 is the exact identity. Borders clamp, so a constant
image is preserved everywhere including its edges — and an impulse
near an edge loses the tail that falls outside.

Below sigma 4 this is a direct separable convolution; at or above it,
three box passes whose variances sum to sigma-squared, which costs
the same at any radius.

The result is in **pixels of the image as given**, so a blur on a
half-size preview is not the same picture as the same sigma on the
full frame. Scale sigma with the image when previewing.

Order-sensitive, and which way depends on what the blur is for: as a
capture-side effect (halation) it belongs early on linear data; as a
print-side one (diffusion) after the tone stages. Light adds
linearly, and only the linear frame gets that right.

Parameters
----------
img : numpy.ndarray
    Input array, shape (H, W, C) for any channel count, dtype
    float32, any memory layout.
params : BlurParams
    Sigma in pixels and kernel shape. Default: sigma 0, the identity.

Returns
-------
numpy.ndarray
    Shape (H, W, C), dtype float32, C-contiguous.

Raises
------
ValueError
    If sigma is negative, not finite, or above 4096 — a blur wider
    than any frame this crate is built for, and the point past which
    choosing the box widths stops being cheap.
MemoryError
    If the output exceeds the single-allocation limit."""
    ...

def glow(img: NDArray[np.float32], params: GlowParams | None = None) -> NDArray[np.float32]:
    """Spread light above ``threshold`` and add it back — halation,
diffusion and veiling glare, which are one operation at three sets of
parameters.

Computes ``out = in + amount * blur(max(in - threshold, 0), sigma)``.

What separates the three effects is the parameters and **where in the
pipeline the call sits**:

- **Veiling glare** — the lens. ``threshold=0`` so all light
  scatters, a frame-spanning ``sigma``, applied earliest. Blacks lift
  by an amount that depends on how bright the *whole frame* is, which
  is precisely what a per-pixel tone curve cannot do.
- **Halation** — the emulsion. A high ``threshold``, moderate
  ``sigma``, applied after ``exposure`` and *before* the tone stages,
  because it happens at capture.
- **Diffusion** — the print. Mid ``threshold``, large ``sigma``,
  applied after the tone stages.

The result may exceed 1.0, deliberately: headroom is carried to
``highlight_rolloff`` rather than clamped here. ``amount = 0.0`` is
the exact identity.

Not an unsharp mask — ``amount`` must be non-negative. For detail
enhancement use ``local_contrast``, which is edge-aware and does not
halo.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : GlowParams
    Threshold, sigma in pixels, and amount. Default: amount 0.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``threshold`` or ``amount`` is negative or not finite, or
    ``sigma`` is outside the blur's domain.
MemoryError
    If the intermediates exceed the single-allocation limit."""
    ...

def highlight_rolloff(img: NDArray[np.float32], params: RolloffParams) -> NDArray[np.float32]:
    """Roll highlights off into a shoulder instead of clipping them.

Below ``knee`` nothing changes. Between ``knee`` and ``white_point``
the transfer follows a quadratic Bézier that leaves the identity at
slope 1 and reaches exactly 1.0 at ``white_point`` with slope 0, so
neither end produces a visible edge. At and above ``white_point`` the
result is 1.0.

This is the stage that decides what becomes of the highlight headroom
every earlier stage preserved. The default ``RolloffParams()`` is a
hard clip at 1.0, reproducing ``numpy.clip(img, None, 1.0)`` exactly,
so adding the stage changes nothing until it is asked to.

Values below zero are left alone: clamping black is a separate
decision.

Order-sensitive: the last stage on linear scene-referred data,
immediately before ``encode_srgb``.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : RolloffParams
    Knee and white point. The default ``(1.0, 1.0)`` is a hard clip.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``knee`` is outside 0..=1, or ``white_point`` is not finite or
    is below 1.0."""
    ...

def shadow_rolloff(img: NDArray[np.float32], params: ShadowRolloffParams | None = None) -> NDArray[np.float32]:
    """Roll the deepest shadows off into black instead of holding full
contrast down to zero.

Above ``knee`` nothing changes. Below it the curve bends away,
reaching the origin with slope ``1 - strength``, so shadow separation
is compressed and tones run together as they approach black. The join
at the knee is C-1 in value and slope, so it does not show as a crease
in a gradient.

This is the **toe** of the characteristic curve.
``highlight_rolloff`` is the shoulder, and ``tone_curve`` sets the
slope of the straight section between them; applied in that order
they compose the classic three-part film tone scale. For a *measured*
emulsion curve rather than a parametric one, tabulate the data and
use ``apply_lut``.

The default ``ShadowRolloffParams()`` has ``strength=0`` and is the
exact identity.

Order-sensitive: apply at the start of the tone stages, on linear
scene-referred data, before the contrast is set.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : ShadowRolloffParams
    Knee and strength, both in 0..=1. Default: no compression.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``knee`` or ``strength`` is outside 0..=1, or is not finite."""
    ...

def quantize_u8(img: NDArray[np.float32], params: QuantizeParams | None = None) -> NDArray[np.uint8]:
    """Quantise display-referred float data to 8-bit integer codes.

Each sample is scaled to ``0..=255``, optionally perturbed by ±1 LSB
of triangular dither, and rounded. Values outside ``[0, 1]`` clamp to
the end codes; NaN maps to 0.

Eight bits is where dither earns its keep: without it a smooth sky
bands visibly, because the quantisation error of a smooth gradient is
itself smooth and collects into contour lines. Pass
``Dither.Tpdf`` with a seed unless the image already carries grain,
which dithers it as a side effect.

Order-sensitive: terminal, and the input must already be
display-referred — ``encode_srgb`` has to have run. Quantising linear
data throws away most of the shadow range.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout. Expected in ``[0, 1]``.
params : QuantizeParams
    Dither strategy and seed. Default: no dither.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``uint8``, C-contiguous."""
    ...

def quantize_u16(img: NDArray[np.float32], params: QuantizeParams | None = None) -> NDArray[np.uint16]:
    """Quantise display-referred float data to 16-bit integer codes.

As ``quantize_u8`` but to ``0..=65535``. This is the archival
default: 16 bits leaves enough headroom that dither is a refinement
rather than a necessity, and enough precision that a consumer can
grade the file further without tearing it.

Order-sensitive: terminal, after ``encode_srgb``.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout. Expected in ``[0, 1]``.
params : QuantizeParams
    Dither strategy and seed. Default: no dither.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``uint16``, C-contiguous."""
    ...

def histogram(img: NDArray[np.float32], params: HistogramParams | None = None) -> Histogram:
    """Count how the image's samples are distributed, per channel.

Returns a ``Histogram`` with ``(channels, bins)`` counts over
``[min, max]``, plus separate tallies of the samples that fell below
the range, above it, or were NaN. Those three are kept apart from the
bins on purpose: folding out-of-range samples into the end bins is
why so many histogram displays show a spike at the right edge that
cannot be told apart from legitimately bright content.

**Call this after ``encode_srgb``** if the histogram is for a person
to look at. A linear scene-referred histogram is correct and
unreadable — 18% grey sits a fifth of the way up the axis. Call it on
linear data only when analysing headroom.

Deterministic at any thread count and on either backend: bin counts
are integers, and integer addition is associative.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
params : HistogramParams
    Bin count and counted range. Default: 256 bins over ``[0, 1]``.

Returns
-------
Histogram
    With ``counts()``, ``cdf()``, ``below()``, ``above()``,
    ``non_finite()`` and ``total(channel)``.

Raises
------
ValueError
    If ``bins`` is below 2 or above 4194304; if the range is not
    finite with ``max > min`` or spans more than float32 can
    represent; or if the channel count and ``bins`` together would
    need an accumulator above the backend's limit."""
    ...

def apply_lut(img: NDArray[np.float32], lut: NDArray[np.float32], params: LutParams | None = None) -> NDArray[np.float32]:
    """Apply a 1-D lookup table as a tone transfer.

The table's entries are spread evenly across ``[params.min,
params.max]``; a sample between two entries is linearly interpolated,
and one outside the domain takes the nearest end entry — clamped, not
extrapolated. The same table is applied to every channel.

This one kernel covers the whole curve family. Sample a UI spline
into a table and it is a curves tool; pass a ``Histogram.cdf()`` row
and it is histogram equalisation; tabulate an H&D curve and it is
film emulation.

The table is **not** required to be monotone — non-monotone tables
are how solarisation is expressed, and it is the one thing
``tone_curve`` and ``zone_system`` structurally cannot do.

Note that a table is *data*, not a parameter: a consumer promising
exact reproduction must store the whole table in its sidecar. See
``docs/export.md``.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout.
lut : numpy.ndarray
    1-D ``float32`` table, at least 2 entries, all finite.
params : LutParams
    The input range the table spans. Default: ``[0, 1]``.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If the table has fewer than 2 entries or a non-finite value, or
    the domain is not finite with ``max > min`` or spans more than
    float32 can represent."""
    ...

def local_contrast(img: NDArray[np.float32], params: GuidedFilterParams, strength: float) -> NDArray[np.float32]:
    """Enhance local contrast using the He–Sun–Tang guided filter.

Computes ``output = L + strength · (L - guided_filter(L))``.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 1)``, dtype ``float32``, any memory layout.
params : GuidedFilterParams
    Filter radius and epsilon regularisation term.
strength : float
    Detail amplification factor (0 = no change, 1 = standard unsharp
    mask, > 1 = over-sharpening).

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 1)``."""
    ...

def film_grain(img: NDArray[np.float32], params: GrainParams) -> NDArray[np.float32]:
    """Add procedural film grain.

Computes ``out = max(L + intensity * 4*t*(1-t) * bandpass(noise), 0)``
with ``t = clip(L, 0, 1)``, so the grain peaks in the midtones and
vanishes at both ends of the range.

Deterministic by construction: each pixel's noise is a hash of
``(seed, x, y)``, not a draw from a sequential generator, so the
output is bit-identical on any thread count and any platform.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 1)``, dtype ``float32``, any layout.
params : GrainParams
    Intensity, grain size in pixels, and the explicit seed.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 1)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 1)``, if ``intensity`` is
    negative or non-finite, or if ``size_pixels`` is not positive."""
    ...

def split_toning(img: NDArray[np.float32], params: SplitToningParams) -> NDArray[np.float32]:
    """Tint shadows and highlights separately, returning linear sRGB.

**Changes the shape of the data**: takes ``(H, W, 1)`` monochrome
luminance and returns ``(H, W, 3)`` linear sRGB. It is the only
kernel that adds channels, which is why ``vignette``, ``tone_curve``
and ``encode_srgb`` all accept any channel count.

Works in OKLab (Ottosson 2020), so the tint adds chroma without
moving the lightness that the tone stages established. Only the
``a`` and ``b`` components of each tint are used; the ``L``
component is ignored.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, 1)``, dtype ``float32``, any layout.
params : SplitToningParams
    Shadow and highlight tints as OKLab triples, plus pivot and
    balance.

Returns
-------
numpy.ndarray
    Shape ``(H, W, 3)``, dtype ``float32``, C-contiguous, linear sRGB.

Raises
------
ValueError
    If ``img`` is not shape ``(H, W, 1)``, if ``pivot`` is outside
    0..=1, if ``balance`` is outside -1..=1, or if a tint component
    is not finite."""
    ...

def vignette(img: NDArray[np.float32], params: VignetteParams) -> NDArray[np.float32]:
    """Apply a radial vignette.

Scales each pixel by ``1 - amount * falloff(distance)``, where the
distance is measured in normalised frame coordinates: the centre is
0 and the corners are 1. Positive ``amount`` darkens the corners,
negative lightens them; the result is clamped at zero.

Because the coordinates are normalised, the result is
resolution-independent — a preview and the full-size frame get the
same picture from the same parameters.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)`` for any channel count, dtype
    ``float32``, any memory layout. Every channel of a pixel gets the
    same factor, so the vignette darkens without tinting.
params : VignetteParams
    Amount, feather and roundness.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, C-contiguous.

Raises
------
ValueError
    If any parameter is not finite, or if ``feather`` or
    ``roundness`` is outside 0..=1."""
    ...

def encode_srgb(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the IEC 61966-2-1 sRGB transfer encoding.

This is always the last kernel in the pipeline. Converts scene-referred
linear f32 values to display-referred sRGB. Values are not clamped —
caller should clamp to [0, 1] beforehand if required.

Parameters
----------
img : numpy.ndarray
    Input array, shape ``(H, W, C)``, dtype ``float32``. Any channel
    count and any memory layout are accepted; the returned array is
    always C-contiguous.

Returns
-------
numpy.ndarray
    Shape ``(H, W, C)``, dtype ``float32``, display-referred sRGB."""
    ...

# `phaios_core.gpu` exists only in builds compiled with `--features cuda`.
# Consumers that support CPU-only installs must guard the import:
#
#     try:
#         from phaios_core import gpu
#     except ImportError:
#         gpu = None
from . import gpu as gpu
