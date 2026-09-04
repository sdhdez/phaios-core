# SPDX-License-Identifier: GPL-3.0-or-later
"""CUDA backend for phaios-core (built with `--features cuda`).

Every kernel in the parent module appears here a second time, operating
on a device-resident ``GpuImage`` instead of a numpy array, so a
pipeline uploads once and downloads once rather than round-tripping
per stage::

    ctx = gpu.GpuContext(0)
    img = ctx.upload(array)
    img = gpu.exposure(img, 0.5)
    img = gpu.luminance_bw(img)
    out = img.download()

The context is explicit: there is no global device state, and a context
the caller drops releases its device memory. Determinism is promised
per backend — bit-identical within a backend, bounded across — and
kernels free of transcendentals are bit-exact against the CPU as well.
See docs/ffi.md section 6.

Call ``available()`` before constructing a context; ``devices()`` lists
what is present."""

import numpy as np
from numpy.typing import NDArray
from typing import final

from phaios_core import (
    BlurParams,
    BlurShape,
    ColorFilter,
    CropParams,
    Dither,
    GlowParams,
    GrainParams,
    GuidedFilterParams,
    Histogram,
    HistogramParams,
    HslWeightedParams,
    LuminanceStandard,
    LutParams,
    Orientation,
    QuantizeParams,
    ResizeFilter,
    ResizeParams,
    RolloffParams,
    ShadowRolloffParams,
    SplitToningParams,
    StraightenParams,
    ToneCurveParams,
    VignetteParams,
    ZoneParams,
)

@final
class GpuInfo:
    """Information about one CUDA device.

Plain data: enumeration is free of side effects, and no method on
this class touches the device."""

    @property
    def ordinal(self) -> int:
        """Device index, accepted by ``GpuContext(index=...)``."""
        ...

    @property
    def name(self) -> str:
        """Marketing name, e.g. ``"NVIDIA GeForce RTX 5070 Ti"``."""
        ...

    @property
    def compute_capability(self) -> tuple[int, int]:
        """Compute capability ``(major, minor)``."""
        ...

    @property
    def supported(self) -> bool:
        """Whether this crate's kernels can run on it (requires ≥ 8.0)."""
        ...

    def __repr__(self) -> str: ...

@final
class GpuContext:
    """An owned handle to one CUDA device.

Construct once, pass to every GPU kernel call. Dropping it releases
nothing shared: contexts are independent, and images produced by one
context are not valid with another."""

    def __new__(cls, index: int = 0) -> GpuContext: ...

    @property
    def info(self) -> GpuInfo:
        """Information about the device this context owns."""
        ...

    @property
    def fingerprint(self) -> str:
        """The backend fingerprint — the reproducibility key of
``docs/ffi.md`` §6. Record it wherever exact reproduction is
promised."""
        ...

    def upload(self, img: NDArray[np.float32]) -> GpuImage:
        """Upload an image to this device, returning a ``GpuImage`` that
stays resident until downloaded. Accepts any layout."""
        ...

    def __repr__(self) -> str: ...

@final
class GpuImage:
    """An image resident in GPU memory.

Produced by ``GpuContext.upload`` or returned by a GPU kernel; comes
back to numpy through ``download()``. Chaining kernels over
``GpuImage`` values pays PCIe once at each end of the pipeline
instead of per stage. The image keeps its context alive, so dropping
the ``GpuContext`` first is safe."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as ``(H, W, C)``."""
        ...

    @property
    def fingerprint(self) -> str:
        """The fingerprint of the context this image lives on."""
        ...

    def download(self) -> NDArray[np.float32]:
        """Download to a freshly allocated, C-contiguous float32 array."""
        ...

    def __repr__(self) -> str: ...

def available() -> bool:
    """True if at least one supported CUDA device is present. Never raises."""
    ...

def devices() -> list[GpuInfo]:
    """Enumerate CUDA devices. Never raises; an empty list means none."""
    ...

def exposure(img: GpuImage, stops: float) -> GpuImage:
    """Apply exposure compensation in EV stops on the GPU.

Bit-identical to ``phaios_core.exposure`` — the conformance suite
asserts equality with no tolerance. Signature-compatible with the
CPU function: same arguments after the image, so a pipeline can
switch backends by switching what it passes.

Parameters
----------
img : GpuImage
    Device-resident input, from ``GpuContext.upload`` or a previous
    kernel.
stops : float
    Exposure adjustment in EV.

Returns
-------
GpuImage
    Device-resident result; call ``download()`` to retrieve it.

Raises
------
ValueError
    If ``stops`` is not finite (same message as the CPU kernel).
RuntimeError
    If a device operation fails."""
    ...

def local_contrast(img: GpuImage, params: GuidedFilterParams, strength: float) -> GpuImage:
    """Enhance local contrast using the guided filter, on the GPU.

Same algorithm as ``phaios_core.local_contrast`` with one documented
reformulation: the CPU's global f64 summed-area tables become
separable f32 box filters with compensated summation. Agreement with
the CPU is bounded at 1e-4 relative (asserted by the conformance
suite); output is bit-reproducible within this backend.

Parameters
----------
img : GpuImage
    Device-resident ``(H, W, 1)`` input.
params : GuidedFilterParams
    The same parameter object the CPU kernel takes.
strength : float
    Detail amplification factor.

Returns
-------
GpuImage
    Device-resident ``(H, W, 1)`` result.

Raises
------
ValueError
    Same conditions and messages as the CPU kernel.
RuntimeError
    If a device operation fails."""
    ...

def encode_srgb(img: GpuImage) -> GpuImage:
    """IEC 61966-2-1 sRGB transfer on the GPU. Mirrors
``phaios_core.encode_srgb``; agreement bounded by one ``powf``."""
    ...

def tone_curve(img: GpuImage, params: ToneCurveParams) -> GpuImage:
    """ASC CDL tone curve on the GPU. Mirrors ``phaios_core.tone_curve``;
the ``power == 1`` and identity paths are bit-exact."""
    ...

def vignette(img: GpuImage, params: VignetteParams) -> GpuImage:
    """Radial vignette on the GPU. Mirrors ``phaios_core.vignette``;
bit-exact against the CPU."""
    ...

def highlight_rolloff(img: GpuImage, params: RolloffParams) -> GpuImage:
    """Highlight roll-off on the GPU. Mirrors
``phaios_core.highlight_rolloff``; bit-exact.

The default ``RolloffParams()`` is a hard clip at 1.0."""
    ...

def shadow_rolloff(img: GpuImage, params: ShadowRolloffParams | None = None) -> GpuImage:
    """Shadow toe on the GPU. Mirrors ``phaios_core.shadow_rolloff``;
bit-exact.

The default ``ShadowRolloffParams()`` is the identity."""
    ...

def blur(img: GpuImage, params: BlurParams | None = None) -> GpuImage:
    """Gaussian blur on the GPU. Mirrors ``phaios_core.blur``; agreement is
bounded rather than bit-exact, because the device accumulates each
separable pass in Kahan-compensated float32 where the host uses
float64 — a consumer card runs float64 at 1/64 rate.

``sigma = 0.0`` is the exact identity on both backends."""
    ...

def glow(img: GpuImage, params: GlowParams | None = None) -> GpuImage:
    """Light scattering on the GPU — halation, diffusion and veiling glare.
Mirrors ``phaios_core.glow``; agreement is bounded, inherited from
the blur between the two element-wise halves."""
    ...

def quantize_u8(img: GpuImage, params: QuantizeParams | None = None) -> NDArray[np.uint8]:
    """Quantise a device image to 8-bit codes, returning a numpy array.

Terminal: quantisation is where the device-resident chain ends, so
this returns host data rather than another ``GpuImage``. Mirrors
``phaios_core.quantize_u8``; bit-exact."""
    ...

def quantize_u16(img: GpuImage, params: QuantizeParams | None = None) -> NDArray[np.uint16]:
    """Quantise a device image to 16-bit codes, returning a numpy array.

Terminal, like ``quantize_u8``. Mirrors ``phaios_core.quantize_u16``;
bit-exact."""
    ...

def histogram(img: GpuImage, params: HistogramParams | None = None) -> Histogram:
    """Per-channel histogram of a device image, returned on the host.

A reduction, so it returns a ``Histogram`` rather than a
``GpuImage``: the counts are small and their destination is a display
or an auto-correction, both of which live on the host. Mirrors
``phaios_core.histogram``; bit-identical, because bin counts are
integers and integer addition commutes."""
    ...

def apply_lut(img: GpuImage, lut: NDArray[np.float32], params: LutParams | None = None) -> GpuImage:
    """Apply a 1-D lookup table on the GPU. Mirrors
``phaios_core.apply_lut``; bit-exact."""
    ...

def luminance_bw(img: GpuImage, standard: LuminanceStandard = LuminanceStandard.Bt709) -> GpuImage:
    """Standard-luminance B&W conversion on the GPU: ``(H, W, 3)`` in,
``(H, W, 1)`` out. Mirrors ``phaios_core.luminance_bw``; bit-exact."""
    ...

def channel_mixer_bw(img: GpuImage, wr: float, wg: float, wb: float) -> GpuImage:
    """Arbitrary-weight channel mixer on the GPU. Mirrors
``phaios_core.channel_mixer_bw``; bit-exact.

Collapses ``(H, W, 3)`` to ``(H, W, 1)``."""
    ...

def color_filter_bw(img: GpuImage, filter: ColorFilter = ColorFilter.NoFilter, standard: LuminanceStandard = LuminanceStandard.Bt709) -> GpuImage:
    """Wratten-style colour-filter conversion on the GPU. Mirrors
``phaios_core.color_filter_bw``; bit-exact.

Collapses ``(H, W, 3)`` to ``(H, W, 1)``."""
    ...

def hsl_bw(img: GpuImage, params: HslWeightedParams) -> GpuImage:
    """HSL-weighted B&W conversion on the GPU. Mirrors
``phaios_core.hsl_bw``; agreement bounded by ``expf``.

Collapses ``(H, W, 3)`` to ``(H, W, 1)``."""
    ...

def zone_system(img: GpuImage, params: ZoneParams) -> GpuImage:
    """Zone System tone curve on the GPU. Mirrors
``phaios_core.zone_system``; the dense in-order offset sum inherits
the CPU's ordered-reduction guarantee mechanically."""
    ...

def split_toning(img: GpuImage, params: SplitToningParams) -> GpuImage:
    """Split-toning on the GPU: ``(H, W, 1)`` in, ``(H, W, 3)`` out.
Mirrors ``phaios_core.split_toning``; agreement bounded by ``cbrtf``."""
    ...

def film_grain(img: GpuImage, params: GrainParams) -> GpuImage:
    """Procedural film grain on the GPU. Mirrors
``phaios_core.film_grain``: the splitmix64 hash is bit-exact against
the CPU (asserted over 2**20 coordinates); Box-Muller and the box
filters are bounded."""
    ...

def crop(img: GpuImage, params: CropParams) -> GpuImage:
    """Crop on the GPU. Mirrors ``phaios_core.crop``; bit-exact (a pure
index copy)."""
    ...

def orient(img: GpuImage, orientation: Orientation) -> GpuImage:
    """Dihedral orientation on the GPU. Mirrors ``phaios_core.orient``;
bit-exact (a pure index permutation)."""
    ...

def resize(img: GpuImage, params: ResizeParams) -> GpuImage:
    """Resize on the GPU. Mirrors ``phaios_core.resize``; bit-exact."""
    ...

def straighten(img: GpuImage, params: StraightenParams) -> GpuImage:
    """Straighten on the GPU. Mirrors ``phaios_core.straighten``; bit-exact."""
    ...
