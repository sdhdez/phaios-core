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
