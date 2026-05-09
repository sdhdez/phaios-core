# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-side FFI smoke tests for phaios_core.

Verifies that each PyO3 binding:
- Is callable from Python with a numpy.float32 C-contiguous input.
- Returns numpy.float32, C-contiguous output.
- Raises a clean Python exception on wrong dtype — no Rust panic.
- Raises ValueError on wrong shape (shape validation in the kernel).

Run with: pytest tests/ffi.py
"""

import numpy as np
import pytest

import phaios_core as ph

# ── Constants ─────────────────────────────────────────────────────────────────

H, W = 64, 64


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rgb_f32():
    """C-contiguous float32 (H, W, 3) — input for B&W kernels."""
    rng = np.random.default_rng(42)
    arr = rng.random((H, W, 3)).astype(np.float32)
    assert arr.flags["C_CONTIGUOUS"]
    return arr


@pytest.fixture
def rgb_f64():
    """float64 (H, W, 3) — used to test dtype rejection."""
    rng = np.random.default_rng(42)
    return rng.random((H, W, 3)).astype(np.float64)


@pytest.fixture
def grey_f32(rgb_f32):
    """C-contiguous float32 (H, W, 1) — input for tone/contrast/encode."""
    out = ph.luminance_bw(rgb_f32)
    assert out.flags["C_CONTIGUOUS"]
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────


def assert_valid_output(arr, expected_shape):
    """Assert output is float32, C-contiguous, and has the expected shape."""
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"
    assert arr.shape == expected_shape, f"expected {expected_shape}, got {arr.shape}"
    assert arr.flags["C_CONTIGUOUS"], "output is not C-contiguous"


# ── Basic import ──────────────────────────────────────────────────────────────


def test_import():
    """The module must be importable after `maturin develop`."""
    assert ph is not None


# ── B&W kernels ───────────────────────────────────────────────────────────────


def test_luminance_bw_shape_dtype(rgb_f32):
    out = ph.luminance_bw(rgb_f32)
    assert_valid_output(out, (H, W, 1))


def test_luminance_bw_default_standard(rgb_f32):
    """Calling without explicit standard should not raise."""
    out = ph.luminance_bw(rgb_f32)
    assert out.shape == (H, W, 1)


def test_luminance_bw_explicit_standards(rgb_f32):
    for std in (ph.LuminanceStandard.Bt601, ph.LuminanceStandard.Bt709, ph.LuminanceStandard.Bt2020):
        out = ph.luminance_bw(rgb_f32, standard=std)
        assert_valid_output(out, (H, W, 1))


def test_channel_mixer_bw_shape_dtype(rgb_f32):
    out = ph.channel_mixer_bw(rgb_f32, 0.3, 0.59, 0.11)
    assert_valid_output(out, (H, W, 1))


def test_color_filter_bw_shape_dtype(rgb_f32):
    out = ph.color_filter_bw(rgb_f32, ph.ColorFilter.Red25A)
    assert_valid_output(out, (H, W, 1))


def test_color_filter_bw_all_presets(rgb_f32):
    for flt in (
        ph.ColorFilter.NoFilter,
        ph.ColorFilter.Yellow8K2,
        ph.ColorFilter.Orange21,
        ph.ColorFilter.Red25A,
        ph.ColorFilter.Green11X1,
        ph.ColorFilter.Blue47C5,
    ):
        out = ph.color_filter_bw(rgb_f32, flt)
        assert_valid_output(out, (H, W, 1))


# ── Type-error rejection (no Rust panic) ──────────────────────────────────────


def test_luminance_bw_wrong_dtype(rgb_f64):
    with pytest.raises(Exception):
        ph.luminance_bw(rgb_f64)


def test_channel_mixer_bw_wrong_dtype(rgb_f64):
    with pytest.raises(Exception):
        ph.channel_mixer_bw(rgb_f64, 1.0, 0.0, 0.0)


def test_color_filter_bw_wrong_dtype(rgb_f64):
    with pytest.raises(Exception):
        ph.color_filter_bw(rgb_f64, ph.ColorFilter.NoFilter)


# ── Shape-error rejection ─────────────────────────────────────────────────────


def test_luminance_bw_wrong_channels():
    bad = np.ones((H, W, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        ph.luminance_bw(bad)


def test_zone_system_wrong_channels(rgb_f32):
    with pytest.raises(ValueError):
        ph.zone_system(rgb_f32, ph.ZoneParams({}))


# ── Zone System ───────────────────────────────────────────────────────────────


def test_zone_params_constructible():
    params = ph.ZoneParams({3: -0.3, 5: 0.0, 7: 1.0})
    assert params is not None


def test_zone_system_shape_dtype(grey_f32):
    params = ph.ZoneParams({5: 0.5})
    out = ph.zone_system(grey_f32, params)
    assert_valid_output(out, (H, W, 1))


def test_zone_system_empty_params_is_identity(grey_f32):
    """Empty ZoneParams must leave the image unchanged."""
    out = ph.zone_system(grey_f32, ph.ZoneParams({}))
    np.testing.assert_allclose(out, grey_f32, atol=1e-5)


# ── Local contrast ────────────────────────────────────────────────────────────


def test_guided_filter_params_constructible():
    params = ph.GuidedFilterParams(radius=8, eps=0.01)
    assert params.radius == 8
    assert abs(params.eps - 0.01) < 1e-6


def test_local_contrast_shape_dtype(grey_f32):
    params = ph.GuidedFilterParams(4, 0.01)
    out = ph.local_contrast(grey_f32, params, 0.5)
    assert_valid_output(out, (H, W, 1))


def test_local_contrast_wrong_channels(rgb_f32):
    params = ph.GuidedFilterParams(4, 0.01)
    with pytest.raises(ValueError):
        ph.local_contrast(rgb_f32, params, 0.5)


# ── sRGB encode ───────────────────────────────────────────────────────────────


def test_encode_srgb_shape_dtype(grey_f32):
    out = ph.encode_srgb(grey_f32)
    assert_valid_output(out, (H, W, 1))


def test_encode_srgb_range(grey_f32):
    """For input in [0, 1], output must also be in [0, 1]."""
    input_clamped = np.clip(grey_f32, 0.0, 1.0)
    out = ph.encode_srgb(input_clamped)
    assert float(out.min()) >= 0.0, f"output min {out.min()} < 0"
    assert float(out.max()) <= 1.0, f"output max {out.max()} > 1"


def test_encode_srgb_wrong_dtype():
    bad = np.ones((H, W, 1), dtype=np.float64)
    with pytest.raises(Exception):
        ph.encode_srgb(bad)
