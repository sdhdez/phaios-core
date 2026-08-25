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

import numpy as np
import pytest

import phaios_core as ph

# The gpu submodule only exists when the wheel was built with the cuda
# feature. A CPU-only build skips this whole file — that build's contract
# is covered by ffi.py.
gpu = pytest.importorskip("phaios_core.gpu")

HAS_DEVICE = gpu.available()

needs_device = pytest.mark.skipif(not HAS_DEVICE, reason="no supported CUDA device")


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
