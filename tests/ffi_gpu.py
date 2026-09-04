# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-side tests for the optional phaios_core.gpu submodule.

Two regimes, both covered:

- No GPU (or a CPU-only build): the import must succeed, the probe
  functions must answer without raising, and constructing a context must
  raise RuntimeError — never PanicException. This file's most important
  job is asserting that the absence of hardware is boring.
- GPU present: the CUDA kernels must agree with the CPU kernels
  (bit-exactly, for exposure) and obey the same layout and error
  contracts.

Run with: pytest tests/ffi_gpu.py
"""

from pathlib import Path

import numpy as np
import pytest

import phaios_core as ph
import stub_contract

# The gpu submodule only exists when the wheel was built with the cuda
# feature. A CPU-only build skips this whole file — that build's contract
# is covered by ffi.py.
gpu = pytest.importorskip("phaios_core.gpu")

HAS_DEVICE = gpu.available()

needs_device = pytest.mark.skipif(not HAS_DEVICE, reason="no supported CUDA device")


def test_documented_example_names_apis_that_exist():
    """The gpu module docstring and the README ship the same five-line
    example. Two of its lines named APIs that never existed: `GpuImage`
    is frozen with no constructor, so `gpu.GpuImage(ctx, array)` raises
    TypeError, and the download method is `download`, not `to_numpy`.

    Nothing in the suite referenced either name, so the shipped example
    — the first thing a reader would type — was also the first thing to
    fail. This test needs no hardware: it checks names, not kernels.
    """
    assert hasattr(gpu, "GpuContext")
    assert hasattr(gpu.GpuContext, "upload"), "the example uploads via the context"
    assert hasattr(gpu.GpuImage, "download"), "the example downloads via the image"
    assert not hasattr(gpu.GpuImage, "to_numpy"), "stale name back in the API"
    for name in ("exposure", "luminance_bw"):
        assert hasattr(gpu, name), f"the example calls gpu.{name}"

    # A GpuImage comes from ctx.upload(); it is deliberately not
    # constructible, which is exactly what the old example got wrong.
    with pytest.raises(TypeError):
        gpu.GpuImage(None, None)


def test_shipped_docstring_example_matches_the_real_api():
    """Pin the docstring itself, not just the API it describes.

    The test above would notice the API drifting away from the docs; this
    one notices the docs drifting away from the API, which is the
    direction that actually happened — the example named `GpuImage(ctx,
    array)` and `to_numpy()`, neither of which ever existed.
    """
    doc = gpu.__doc__ or ""
    assert doc, "the gpu submodule ships a module docstring"

    for stale in ("to_numpy", "gpu.GpuImage("):
        assert stale not in doc, f"the docstring example names {stale!r}, which does not exist"
    for real in ("ctx.upload(", "img.download()"):
        assert real in doc, f"the docstring example should show {real!r}"


_GPU_PYI = Path(__file__).resolve().parent.parent / "python" / "phaios_core" / "gpu.pyi"


def test_gpu_stub_matches_the_runtime_submodule():
    """Every public name, signature, docstring and class shape in
    `python/phaios_core/gpu.pyi` must match the built `phaios_core.gpu`
    submodule exactly. Needs no device: it only inspects the module
    object, which exists whenever this file was not already skipped by
    the file-level `importorskip` above."""
    errors = stub_contract.check_stub(_GPU_PYI, gpu, top_level=False)
    assert not errors, "\n".join(errors)


@needs_device
def test_documented_example_runs_end_to_end():
    """Run the documented example verbatim, so it cannot rot silently."""
    array = np.zeros((8, 8, 3), dtype=np.float32)
    array[2:6, 2:6, :] = 0.4

    ctx = gpu.GpuContext(0)
    img = ctx.upload(array)
    img = gpu.exposure(img, 0.5)
    img = gpu.luminance_bw(img)
    out = img.download()

    assert out.shape == (8, 8, 1)
    assert out.dtype == np.float32
    want = ph.luminance_bw(ph.exposure(array, 0.5))
    np.testing.assert_allclose(out, want, rtol=1e-6, atol=1e-7)


# ── The no-hardware contract (runs everywhere) ────────────────────────────────


def test_available_never_raises():
    assert isinstance(gpu.available(), bool)


def test_devices_never_raises():
    devs = gpu.devices()
    assert isinstance(devs, list)
    for d in devs:
        assert isinstance(d.name, str)
        assert isinstance(d.supported, bool)
        assert len(d.compute_capability) == 2


def test_bad_index_raises_runtime_error_not_panic():
    """A missing device is an environment condition: RuntimeError.

    Explicitly not PanicException, which inherits from BaseException and
    slips through `except Exception:` — the failure mode docs/ffi.md §4
    exists to prevent.
    """
    with pytest.raises(RuntimeError):
        gpu.GpuContext(index=9999)


# ── With hardware ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ctx():
    return gpu.GpuContext()


@needs_device
def test_context_reports_identity(ctx):
    assert ctx.info.supported
    assert ctx.info.compute_capability >= (8, 0)
    assert ctx.fingerprint.startswith("cuda/")


@needs_device
def test_upload_download_round_trip(ctx):
    rng = np.random.default_rng(42)
    img = rng.random((64, 48, 3)).astype(np.float32)
    for view in (img, img[::2], img[:, ::3], np.asfortranarray(img), img[::-1]):
        out = ctx.upload(view).download()
        np.testing.assert_array_equal(out, view)
        assert out.flags["C_CONTIGUOUS"]


@needs_device
def test_exposure_bit_exact_vs_cpu(ctx):
    rng = np.random.default_rng(42)
    img = rng.random((256, 384, 3)).astype(np.float32)
    cpu = ph.exposure(img, 0.5)
    out = gpu.exposure(ctx.upload(img), 0.5).download()
    np.testing.assert_array_equal(out, cpu)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]


@needs_device
def test_exposure_same_validation_as_cpu(ctx):
    img = np.ones((4, 4, 1), dtype=np.float32)
    resident = ctx.upload(img)
    with pytest.raises(ValueError) as gpu_err:
        gpu.exposure(resident, float("nan"))
    with pytest.raises(ValueError) as cpu_err:
        ph.exposure(img, float("nan"))
    assert str(gpu_err.value) == str(cpu_err.value)


@needs_device
def test_local_contrast_agrees_with_cpu(ctx):
    rng = np.random.default_rng(3)
    img = rng.random((256, 384, 1)).astype(np.float32)
    params = ph.GuidedFilterParams(8, 0.01)
    cpu = ph.local_contrast(img, params, 0.5)
    out = gpu.local_contrast(ctx.upload(img), params, 0.5).download()
    np.testing.assert_allclose(out, cpu, rtol=1e-4, atol=1e-6)
    assert out.flags["C_CONTIGUOUS"]


@needs_device
def test_kernels_reuse_cpu_param_classes(ctx):
    """The same param objects drive both backends — the property that
    keeps sidecars and presets backend-neutral."""
    img = np.full((32, 32, 1), 0.3, dtype=np.float32)
    gf = ph.GuidedFilterParams(4, 0.01)
    tc = ph.ToneCurveParams(1.1, 0.0, 0.9)
    vg = ph.VignetteParams(0.35, 0.8, 0.1)
    a = ph.vignette(ph.tone_curve(ph.local_contrast(img, gf, 0.5), tc), vg)
    x = ctx.upload(img)
    b = gpu.vignette(gpu.tone_curve(gpu.local_contrast(x, gf, 0.5), tc), vg).download()
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-6)


@needs_device
def test_vignette_and_luminance_bit_exact(ctx):
    rng = np.random.default_rng(9)
    img = rng.random((128, 96, 3)).astype(np.float32)
    vg = ph.VignetteParams(0.5, 0.8, 0.2)
    np.testing.assert_array_equal(
        gpu.vignette(ctx.upload(img), vg).download(), ph.vignette(img, vg)
    )
    np.testing.assert_array_equal(
        gpu.luminance_bw(ctx.upload(img)).download(), ph.luminance_bw(img)
    )


@needs_device
def test_resident_chain_one_upload_one_download(ctx):
    """The Stage C property: five stages, PCIe paid once at each end."""
    rng = np.random.default_rng(5)
    img = rng.random((256, 384, 3)).astype(np.float32)
    gf = ph.GuidedFilterParams(8, 0.01)
    tc = ph.ToneCurveParams(1.15, 0.01, 0.85)
    vg = ph.VignetteParams(0.35, 0.8, 0.1)

    c = ph.luminance_bw(img)
    c = ph.local_contrast(c, gf, 0.4)
    c = ph.tone_curve(c, tc)
    c = ph.vignette(c, vg)
    cpu = ph.encode_srgb(c)

    x = ctx.upload(img)
    x = gpu.luminance_bw(x)
    x = gpu.local_contrast(x, gf, 0.4)
    x = gpu.tone_curve(x, tc)
    x = gpu.vignette(x, vg)
    out = gpu.encode_srgb(x).download()

    assert x.shape == (256, 384, 1)
    np.testing.assert_allclose(out, cpu, rtol=2e-4, atol=1e-6)


@needs_device
def test_full_pipeline_resident(ctx):
    """All nine v0.2 stages on the GPU, one upload, one download."""
    rng = np.random.default_rng(0)
    img = rng.random((256, 384, 3)).astype(np.float32)
    hsl = ph.HslWeightedParams([0, 0, 0.4, 0.2, 0, -0.5, 0, 0])
    zones = ph.ZoneParams({3: -0.3, 7: 0.4})
    gf = ph.GuidedFilterParams(8, 0.01)
    grain = ph.GrainParams(0.12, 1.5, 20260815)
    toning = ph.SplitToningParams([0.0, -0.02, -0.04], [0.0, 0.03, 0.03])
    vg = ph.VignetteParams(0.35, 0.8, 0.1)
    tc = ph.ToneCurveParams(1.1, 0.0, 0.9)

    c = ph.exposure(img, 0.5)
    c = ph.hsl_bw(c, hsl)
    c = ph.zone_system(c, zones)
    c = ph.local_contrast(c, gf, 0.4)
    c = ph.film_grain(c, grain)
    c = ph.split_toning(c, toning)
    c = ph.vignette(c, vg)
    c = ph.tone_curve(c, tc)
    cpu = ph.encode_srgb(c)

    x = ctx.upload(img)
    x = gpu.exposure(x, 0.5)
    x = gpu.hsl_bw(x, hsl)
    x = gpu.zone_system(x, zones)
    x = gpu.local_contrast(x, gf, 0.4)
    x = gpu.film_grain(x, grain)
    x = gpu.split_toning(x, toning)
    x = gpu.vignette(x, vg)
    x = gpu.tone_curve(x, tc)
    out = gpu.encode_srgb(x).download()

    np.testing.assert_allclose(out, cpu, rtol=1e-3, atol=1e-5)


@needs_device
def test_image_survives_context_drop():
    """A GpuImage keeps its context alive; dropping the GpuContext first
    must not invalidate the image."""
    c = gpu.GpuContext()
    img = c.upload(np.ones((16, 16, 1), dtype=np.float32))
    del c
    out = img.download()
    np.testing.assert_array_equal(out, np.ones((16, 16, 1), dtype=np.float32))


@needs_device
def test_local_contrast_same_validation_as_cpu(ctx):
    rgb = np.ones((4, 4, 3), dtype=np.float32)
    params = ph.GuidedFilterParams(4, 0.01)
    resident = ctx.upload(rgb)
    with pytest.raises(ValueError) as gpu_err:
        gpu.local_contrast(resident, params, 0.5)
    with pytest.raises(ValueError) as cpu_err:
        ph.local_contrast(rgb, params, 0.5)
    assert str(gpu_err.value) == str(cpu_err.value)


@needs_device
def test_fingerprint_is_stable_across_contexts():
    a = gpu.GpuContext()
    b = gpu.GpuContext()
    assert a.fingerprint == b.fingerprint


# ── Bindings with no Python smoke coverage before the v0.2 audit ──────────────
#
# Six of the twenty functions registered on phaios_core.gpu were never
# called from Python: the four geometry kernels and two of the B&W
# conversions. Their binding code — signature, param conversion, and in
# orient's case the H/W swap on a resident image — was reachable only
# through Rust. These tests exercise each once and check it against the
# CPU kernel, which is the specification.


@needs_device
def test_gpu_crop_matches_cpu(ctx):
    rng = np.random.default_rng(11)
    img = rng.random((23, 31, 3)).astype(np.float32)
    params = ph.CropParams(4, 6, 12, 9)
    got = gpu.crop(ctx.upload(img), params).download()
    np.testing.assert_array_equal(got, ph.crop(img, params))
    assert got.shape == (9, 12, 3)


@needs_device
@pytest.mark.parametrize(
    "name",
    [
        "Normal",
        "FlipHorizontal",
        "Rotate180",
        "FlipVertical",
        "Transpose",
        "Rotate90",
        "Transverse",
        "Rotate270",
    ],
)
def test_gpu_orient_matches_cpu(ctx, name):
    """All eight transforms, including the four that swap H and W on the
    resident image — the case the binding has to get right."""
    rng = np.random.default_rng(12)
    img = rng.random((7, 11, 3)).astype(np.float32)
    orientation = getattr(ph.Orientation, name)
    got = gpu.orient(ctx.upload(img), orientation).download()
    want = ph.orient(img, orientation)
    np.testing.assert_array_equal(got, want)
    assert got.shape == want.shape


@needs_device
@pytest.mark.parametrize("filter_name", ["Area", "Bilinear", "CatmullRom"])
def test_gpu_resize_matches_cpu(ctx, filter_name):
    rng = np.random.default_rng(13)
    img = rng.random((19, 27, 3)).astype(np.float32)
    params = ph.ResizeParams(13, 9, getattr(ph.ResizeFilter, filter_name))
    got = gpu.resize(ctx.upload(img), params).download()
    np.testing.assert_array_equal(got, ph.resize(img, params))
    assert got.shape == (9, 13, 3)


@needs_device
def test_gpu_straighten_matches_cpu(ctx):
    rng = np.random.default_rng(14)
    img = rng.random((21, 29, 3)).astype(np.float32)
    params = ph.StraightenParams(7.5)
    got = gpu.straighten(ctx.upload(img), params).download()
    np.testing.assert_array_equal(got, ph.straighten(img, params))


@needs_device
def test_gpu_channel_mixer_matches_cpu(ctx):
    rng = np.random.default_rng(15)
    img = rng.random((12, 16, 3)).astype(np.float32)
    got = gpu.channel_mixer_bw(ctx.upload(img), 0.4, -0.3, 1.1).download()
    np.testing.assert_array_equal(got, ph.channel_mixer_bw(img, 0.4, -0.3, 1.1))
    assert got.shape == (12, 16, 1)


@needs_device
@pytest.mark.parametrize(
    "preset",
    ["NoFilter", "Yellow8K2", "Orange21", "Red25A", "Green11X1", "Blue47C5"],
)
def test_gpu_color_filter_matches_cpu(ctx, preset):
    rng = np.random.default_rng(16)
    img = rng.random((12, 16, 3)).astype(np.float32)
    flt = getattr(ph.ColorFilter, preset)
    got = gpu.color_filter_bw(ctx.upload(img), flt).download()
    np.testing.assert_array_equal(got, ph.color_filter_bw(img, flt))
    assert got.shape == (12, 16, 1)


@needs_device
@pytest.mark.parametrize(
    "knee,white",
    [(1.0, 1.0), (0.8, 2.0), (0.5, 4.0), (0.6, 1.4), (0.0, 64.0)],
)
def test_gpu_highlight_rolloff_matches_cpu(ctx, knee, white):
    """Bit-exact, including the a == 0 degenerate solve at white = 2 - knee."""
    rng = np.random.default_rng(21)
    img = (rng.random((17, 23, 3)).astype(np.float32) * 5.0) - 0.5
    params = ph.RolloffParams(knee, white)
    got = gpu.highlight_rolloff(ctx.upload(img), params).download()
    np.testing.assert_array_equal(got, ph.highlight_rolloff(img, params))


@needs_device
@pytest.mark.parametrize("seed", [0, 20260825, 2**64 - 1])
def test_gpu_quantize_matches_cpu(ctx, seed):
    """Terminal kernels: they return numpy arrays, not GpuImages."""
    rng = np.random.default_rng(31)
    img = (rng.random((23, 31, 3)).astype(np.float32) * 1.4) - 0.2
    for dither in (ph.Dither.Tpdf,):
        params = ph.QuantizeParams(dither, seed)
        resident = ctx.upload(img)
        got8 = gpu.quantize_u8(resident, params)
        got16 = gpu.quantize_u16(resident, params)
        assert got8.dtype == np.uint8 and got16.dtype == np.uint16
        np.testing.assert_array_equal(got8, ph.quantize_u8(img, params))
        np.testing.assert_array_equal(got16, ph.quantize_u16(img, params))


@needs_device
def test_gpu_quantize_default_params(ctx):
    rng = np.random.default_rng(32)
    img = rng.random((9, 11, 1)).astype(np.float32)
    resident = ctx.upload(img)
    np.testing.assert_array_equal(gpu.quantize_u8(resident), ph.quantize_u8(img))
    np.testing.assert_array_equal(gpu.quantize_u16(resident), ph.quantize_u16(img))


@needs_device
@pytest.mark.parametrize("bins", [2, 256, 1024, 65536])
def test_gpu_histogram_matches_cpu(ctx, bins):
    """Both device paths: shared-memory privatised and the global-atomic
    fallback for tables too large to privatise."""
    rng = np.random.default_rng(41)
    img = (rng.random((37, 53, 3)).astype(np.float32) * 1.4) - 0.2
    params = ph.HistogramParams(bins, 0.0, 1.0)
    c = ph.histogram(img, params)
    g = gpu.histogram(ctx.upload(img), params)
    np.testing.assert_array_equal(c.counts(), g.counts())
    assert c.below() == g.below()
    assert c.above() == g.above()
    assert c.non_finite() == g.non_finite()


@needs_device
def test_gpu_apply_lut_matches_cpu(ctx):
    rng = np.random.default_rng(42)
    img = (rng.random((23, 31, 3)).astype(np.float32) * 1.4) - 0.2
    lut = (np.linspace(0, 1, 256) ** 0.7).astype(np.float32)
    params = ph.LutParams(0.0, 1.0)
    got = gpu.apply_lut(ctx.upload(img), lut, params).download()
    np.testing.assert_array_equal(got, ph.apply_lut(img, lut, params))


@needs_device
def test_gpu_equalisation_round_trip(ctx):
    """The pair composing on-device: one upload, histogram, LUT, download."""
    rng = np.random.default_rng(43)
    img = (rng.random((64, 64, 1)).astype(np.float32) * 0.2 + 0.4)
    resident = ctx.upload(img)
    h = gpu.histogram(resident)
    out = gpu.apply_lut(resident, h.equalisation_lut()[0], ph.LutParams()).download()
    np.testing.assert_array_equal(
        out, ph.apply_lut(img, ph.histogram(img).equalisation_lut()[0])
    )


@needs_device
@pytest.mark.parametrize("knee,strength", [(0.2, 0.0), (0.2, 0.5), (0.2, 1.0), (1.0, 0.7), (0.0, 1.0)])
def test_gpu_shadow_rolloff_matches_cpu(ctx, knee, strength):
    rng = np.random.default_rng(51)
    img = (rng.random((19, 27, 3)).astype(np.float32) * 1.3) - 0.15
    params = ph.ShadowRolloffParams(knee, strength)
    got = gpu.shadow_rolloff(ctx.upload(img), params).download()
    np.testing.assert_array_equal(got, ph.shadow_rolloff(img, params))


@needs_device
def test_gpu_upload_refuses_oversized_instead_of_aborting(ctx):
    """`Context::upload` materialises a host copy of the *logical* shape
    via as_standard_layout, so it reached the same abort the CPU kernels
    did. Both surfaces must refuse identically."""
    huge = np.broadcast_to(np.float32(0.05), (100_000, 100_000, 3))
    with pytest.raises(MemoryError):
        ctx.upload(huge)


@needs_device
@pytest.mark.parametrize("sigma", [0.0, 1.5, 3.9, 4.0, 8.0])
def test_gpu_blur_matches_cpu_within_bound(ctx, sigma):
    rng = np.random.default_rng(61)
    img = rng.random((37, 53, 3)).astype(np.float32)
    params = ph.BlurParams(sigma)
    got = gpu.blur(ctx.upload(img), params).download()
    want = ph.blur(img, params)
    if sigma == 0.0:
        np.testing.assert_array_equal(got, want)   # identity is a copy
    else:
        assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max() + 1e-7

# ── the six kernels covered only by the composed pipeline ────────────────────
#
# `test_full_pipeline_resident` runs all nine stages CPU-against-GPU at
# rtol 1e-3, which exercises every binding but only in composition: a
# transposed parameter in one stage can be masked by the loose end-to-end
# tolerance or by a compensating error downstream. The Rust conformance
# suite pins the maths per kernel; what these add is the binding, which is
# the layer that has actually rotted here before — `gpu.GpuImage(ctx, arr)`
# and `img.to_numpy()` shipped in the module docstring for a whole release
# naming APIs that never existed, referenced by no test.
#
# Tolerances follow docs/ffi.md §6 rather than being invented here.


@needs_device
def test_gpu_zone_system_matches_cpu(ctx):
    rng = np.random.default_rng(71)
    img = rng.random((23, 31, 1)).astype(np.float32)
    zones = ph.ZoneParams({3: -0.3, 5: 0.4, 8: 0.2})
    got = gpu.zone_system(ctx.upload(img), zones).download()
    want = ph.zone_system(img, zones)
    # log2/exp: bounded, not bit-exact.
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


@needs_device
def test_gpu_hsl_bw_matches_cpu(ctx):
    rng = np.random.default_rng(72)
    img = rng.random((23, 31, 3)).astype(np.float32)
    params = ph.HslWeightedParams([0.3, -0.2, 0.5, 0.0, 0.1, -0.4, 0.2, 0.0])
    got = gpu.hsl_bw(ctx.upload(img), params).download()
    np.testing.assert_allclose(got, ph.hsl_bw(img, params), rtol=1e-5, atol=1e-7)


@needs_device
def test_gpu_film_grain_matches_cpu(ctx):
    rng = np.random.default_rng(73)
    img = rng.random((23, 31, 1)).astype(np.float32)
    params = ph.GrainParams(0.25, 1.8, 20260828)
    got = gpu.film_grain(ctx.upload(img), params).download()
    # The integer hash must reproduce bit-for-bit across backends; the
    # Box-Muller transform on top of it is what widens this to a bound.
    np.testing.assert_allclose(got, ph.film_grain(img, params), rtol=1e-3, atol=1e-5)


@needs_device
def test_gpu_film_grain_uses_the_seed_it_is_given(ctx):
    """A seed that reached the kernel as a constant would still agree with
    the CPU if the CPU binding dropped it the same way. Two seeds must
    differ, and the same seed must repeat."""
    img = np.full((16, 16, 1), 0.5, dtype=np.float32)
    a = gpu.film_grain(ctx.upload(img), ph.GrainParams(0.3, 2.0, 1)).download()
    b = gpu.film_grain(ctx.upload(img), ph.GrainParams(0.3, 2.0, 2)).download()
    again = gpu.film_grain(ctx.upload(img), ph.GrainParams(0.3, 2.0, 1)).download()
    assert not np.array_equal(a, b), "different seeds produced identical grain"
    np.testing.assert_array_equal(a, again)


@needs_device
def test_gpu_split_toning_matches_cpu(ctx):
    rng = np.random.default_rng(74)
    img = rng.random((23, 31, 1)).astype(np.float32)
    params = ph.SplitToningParams([0.01, -0.03, -0.04], [0.0, 0.03, 0.05], 0.45, 0.2)
    got = gpu.split_toning(ctx.upload(img), params).download()
    assert got.shape == (23, 31, 3), "split_toning restores three channels"
    np.testing.assert_allclose(got, ph.split_toning(img, params), rtol=1e-5, atol=1e-7)


@needs_device
def test_gpu_tone_curve_matches_cpu(ctx):
    rng = np.random.default_rng(75)
    img = rng.random((23, 31, 3)).astype(np.float32)
    params = ph.ToneCurveParams(1.15, -0.02, 0.85)
    got = gpu.tone_curve(ctx.upload(img), params).download()
    np.testing.assert_allclose(got, ph.tone_curve(img, params), rtol=1e-5, atol=1e-7)


@needs_device
def test_gpu_encode_srgb_matches_cpu(ctx):
    # Straddle the 0.0031308 breakpoint deliberately: the transfer is C0
    # there but not C1, so a binding that took the wrong branch would show
    # up only near it.
    img = np.array(
        [[[0.0, 0.001, 0.0031308], [0.0031309, 0.5, 1.0]]], dtype=np.float32
    )
    got = gpu.encode_srgb(ctx.upload(img)).download()
    np.testing.assert_allclose(got, ph.encode_srgb(img), rtol=1e-5, atol=1e-7)



@needs_device
@pytest.mark.parametrize(
    "threshold,sigma,amount",
    [(0.0, 8.0, 0.0), (0.8, 8.0, 0.35), (0.5, 20.0, 0.4), (0.0, 400.0, 0.06)],
)
def test_gpu_glow_matches_cpu(ctx, threshold, sigma, amount):
    rng = np.random.default_rng(71)
    img = (rng.random((31, 43, 3)).astype(np.float32) * 2.0)
    params = ph.GlowParams(threshold, sigma, amount)
    got = gpu.glow(ctx.upload(img), params).download()
    want = ph.glow(img, params)
    if amount == 0.0:
        np.testing.assert_array_equal(got, want)
    else:
        assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max() + 1e-7


@needs_device
@pytest.mark.parametrize(
    "amount,sigma,threshold",
    [(0.0, 3.0, 0.1), (0.35, 1.5, 0.0), (0.5, 8.0, 0.0), (0.4, 1.2, 0.05)],
)
def test_gpu_sharpen_matches_cpu(ctx, amount, sigma, threshold):
    rng = np.random.default_rng(76)
    img = rng.random((31, 43, 3)).astype(np.float32) * 2.0
    params = ph.SharpenParams(amount, sigma, threshold)
    got = gpu.sharpen(ctx.upload(img), params).download()
    want = ph.sharpen(img, params)
    if amount == 0.0 or sigma == 0.0:
        np.testing.assert_array_equal(got, want)   # identity fast path, a copy
    else:
        assert np.abs(got - want).max() <= 1e-5 * np.abs(want).max() + 1e-7


@needs_device
def test_gpu_sharpen_default_is_the_identity(ctx):
    rng = np.random.default_rng(77)
    img = rng.random((17, 23, 3)).astype(np.float32)
    got = gpu.sharpen(ctx.upload(img)).download()
    np.testing.assert_array_equal(got, img)


@needs_device
def test_gpu_sharpen_same_validation_as_cpu(ctx):
    img = np.ones((4, 4, 1), dtype=np.float32)
    resident = ctx.upload(img)
    with pytest.raises(ValueError) as gpu_err:
        gpu.sharpen(resident, ph.SharpenParams(-1.0, 1.0, 0.0))
    with pytest.raises(ValueError) as cpu_err:
        ph.sharpen(img, ph.SharpenParams(-1.0, 1.0, 0.0))
    assert str(gpu_err.value) == str(cpu_err.value)
