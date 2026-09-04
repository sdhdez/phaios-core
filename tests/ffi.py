# SPDX-License-Identifier: GPL-3.0-or-later
"""Python-side FFI smoke tests for phaios_core.

Verifies that each PyO3 binding:
- Is callable from Python with a numpy.float32 C-contiguous input.
- Returns numpy.float32, C-contiguous output.
- Raises a clean Python exception on wrong dtype — no Rust panic.
- Raises ValueError on wrong shape (shape validation in the kernel).

Run with: pytest tests/ffi.py
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

import phaios_core as ph
import stub_contract

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


# ── Geometry ──────────────────────────────────────────────────────────────────


def test_crop_shape_and_values(rgb_f32):
    out = ph.crop(rgb_f32, ph.CropParams(4, 2, 32, 16))
    assert_valid_output(out, (16, 32, 3))
    np.testing.assert_array_equal(out, rgb_f32[2:18, 4:36])


def test_crop_rejects_out_of_frame(rgb_f32):
    with pytest.raises(ValueError):
        ph.crop(rgb_f32, ph.CropParams(0, 0, W + 1, H))


def test_crop_params_round_trip():
    p = ph.CropParams(1, 2, 3, 4)
    assert (p.x, p.y, p.width, p.height) == (1, 2, 3, 4)
    assert p == ph.CropParams(1, 2, 3, 4)
    assert p != ph.CropParams(0, 2, 3, 4)


def test_orient_all_eight(rgb_f32):
    """Exif enum values, dimension swaps, and inverse round trips."""
    variants = [
        ph.Orientation.Normal,
        ph.Orientation.FlipHorizontal,
        ph.Orientation.Rotate180,
        ph.Orientation.FlipVertical,
        ph.Orientation.Transpose,
        ph.Orientation.Rotate90,
        ph.Orientation.Transverse,
        ph.Orientation.Rotate270,
    ]
    for exif_value, o in enumerate(variants, start=1):
        assert o == exif_value  # eq_int: the discriminants ARE the Exif codes
        out = ph.orient(rgb_f32, o)
        transposed = exif_value >= 5
        assert out.shape == ((W, H, 3) if transposed else (H, W, 3))
        assert out.flags["C_CONTIGUOUS"]
    # Quarter turns invert each other.
    back = ph.orient(ph.orient(rgb_f32, ph.Orientation.Rotate90), ph.Orientation.Rotate270)
    np.testing.assert_array_equal(back, rgb_f32)


def test_orient_matches_numpy_rot90(rgb_f32):
    """Pin the direction against numpy: Rotate90 CW == np.rot90(k=-1)."""
    out = ph.orient(rgb_f32, ph.Orientation.Rotate90)
    np.testing.assert_array_equal(out, np.rot90(rgb_f32, k=-1, axes=(0, 1)))


def test_resize_shape_and_identity(rgb_f32):
    out = ph.resize(rgb_f32, ph.ResizeParams(32, 16))
    assert_valid_output(out, (16, 32, 3))
    same = ph.resize(rgb_f32, ph.ResizeParams(W, H, ph.ResizeFilter.CatmullRom))
    np.testing.assert_array_equal(same, rgb_f32)  # exact identity


def test_resize_area_matches_numpy_block_mean(rgb_f32):
    out = ph.resize(rgb_f32, ph.ResizeParams(W // 2, H // 2, ph.ResizeFilter.Area))
    ref = rgb_f32.reshape(H // 2, 2, W // 2, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_resize_rejects_zero(rgb_f32):
    with pytest.raises(ValueError):
        ph.resize(rgb_f32, ph.ResizeParams(0, 16))


def test_straighten_zero_is_identity(rgb_f32):
    np.testing.assert_array_equal(ph.straighten(rgb_f32, ph.StraightenParams(0.0)), rgb_f32)


def test_straighten_crops_and_validates(rgb_f32):
    out = ph.straighten(rgb_f32, ph.StraightenParams(5.0))
    assert out.shape[0] < H and out.shape[1] < W
    assert out.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        ph.straighten(rgb_f32, ph.StraightenParams(46.0))


def test_resize_straighten_params_round_trip():
    r = ph.ResizeParams(100, 50, ph.ResizeFilter.Bilinear)
    assert (r.width, r.height, r.filter) == (100, 50, ph.ResizeFilter.Bilinear)
    assert r == ph.ResizeParams(100, 50, ph.ResizeFilter.Bilinear)
    st = ph.StraightenParams(-2.5)
    assert st.degrees == pytest.approx(-2.5)
    assert st == ph.StraightenParams(-2.5)


# ── Exposure ──────────────────────────────────────────────────────────────────


def test_exposure_shape_dtype(rgb_f32):
    out = ph.exposure(rgb_f32, 1.0)
    assert_valid_output(out, (H, W, 3))


def test_exposure_one_stop_doubles(rgb_f32):
    np.testing.assert_allclose(ph.exposure(rgb_f32, 1.0), rgb_f32 * 2.0, rtol=1e-6)


def test_exposure_zero_is_identity(rgb_f32):
    np.testing.assert_array_equal(ph.exposure(rgb_f32, 0.0), rgb_f32)


def test_exposure_round_trips(rgb_f32):
    """2**n is exact for integer n, so this is bit-exact."""
    np.testing.assert_array_equal(ph.exposure(ph.exposure(rgb_f32, 2.0), -2.0), rgb_f32)


def test_exposure_does_not_clip(rgb_f32):
    """Highlights above 1.0 must survive for later stages to recover."""
    assert float(ph.exposure(rgb_f32, 4.0).max()) > 1.0


def test_exposure_accepts_luminance(grey_f32):
    assert_valid_output(ph.exposure(grey_f32, -1.0), (H, W, 1))


def test_exposure_rejects_non_finite(rgb_f32):
    with pytest.raises(ValueError):
        ph.exposure(rgb_f32, float("nan"))


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


# Retyped from the ITU-R recommendations, deliberately not imported from
# the extension: an oracle that reads the same table it is checking
# asserts nothing. BT.601-7 (2011) 2.5.1; BT.709-6 (2015) 3;
# BT.2020-2 (2015) Table 4.
LUMINANCE_WEIGHTS = {
    "Bt601": (0.2990, 0.5870, 0.1140),
    "Bt709": (0.2126, 0.7152, 0.0722),
    "Bt2020": (0.2627, 0.6780, 0.0593),
}


@pytest.mark.parametrize("name", sorted(LUMINANCE_WEIGHTS))
def test_luminance_bw_weight_values(name):
    """Each standard must apply its own coefficients, not merely return
    the right shape.

    `test_luminance_bw_explicit_standards` above sweeps all three and
    checks only dtype and shape, so before this every coefficient
    outside BT.709 could be transposed with the whole suite green — on
    *both* backends, because `LuminanceStandard::weights()` is host-side
    Rust that the CUDA path also calls. The two backends move together,
    so cross-backend conformance cannot see it either.
    """
    weights = LUMINANCE_WEIGHTS[name]
    std = getattr(ph.LuminanceStandard, name)

    for channel in range(3):
        pure = np.zeros((1, 1, 3), np.float32)
        pure[0, 0, channel] = 1.0
        got = float(ph.luminance_bw(pure, standard=std)[0, 0, 0])
        assert abs(got - weights[channel]) < 1e-6, (
            f"{name} on pure channel {channel}: got {got}, want {weights[channel]}"
        )


@pytest.mark.parametrize("name", sorted(LUMINANCE_WEIGHTS))
def test_luminance_bw_preserves_neutrals(name):
    """Every row sums to 1, so a neutral grey survives conversion. This
    catches a transposed digit even where the coefficient itself still
    looks plausible."""
    std = getattr(ph.LuminanceStandard, name)
    assert abs(sum(LUMINANCE_WEIGHTS[name]) - 1.0) < 1e-6
    grey = np.full((1, 1, 3), 0.18, np.float32)
    got = float(ph.luminance_bw(grey, standard=std)[0, 0, 0])
    assert abs(got - 0.18) < 1e-6, f"{name} shifted 18% grey to {got}"


def test_channel_mixer_bw_shape_dtype(rgb_f32):
    out = ph.channel_mixer_bw(rgb_f32, 0.3, 0.59, 0.11)
    assert_valid_output(out, (H, W, 1))


def test_color_filter_bw_shape_dtype(rgb_f32):
    out = ph.color_filter_bw(rgb_f32, ph.ColorFilter.Red25A)
    assert_valid_output(out, (H, W, 1))


# The documented Wratten transmissions (src/bw.rs). Kept here as an
# independent copy on purpose: a test that imported the table from the
# code under test would pass however the table changed.
FILTER_TRANSMISSIONS = {
    "NoFilter": (1.00, 1.00, 1.00),
    "Yellow8K2": (1.00, 0.90, 0.30),
    "Orange21": (1.00, 0.55, 0.10),
    "Red25A": (1.00, 0.10, 0.02),
    "Green11X1": (0.20, 1.00, 0.30),
    "Blue47C5": (0.10, 0.30, 1.00),
}
BT709 = (0.2126, 0.7152, 0.0722)


@pytest.mark.parametrize("name", sorted(FILTER_TRANSMISSIONS))
def test_color_filter_bw_transmission_values(name):
    """Each preset must apply its own transmission table, not merely
    return the right shape. Five of the six had no value-level coverage
    anywhere before the v0.2 audit."""
    transmission = FILTER_TRANSMISSIONS[name]
    flt = getattr(ph.ColorFilter, name)

    for channel in range(3):
        pure = np.zeros((1, 1, 3), np.float32)
        pure[0, 0, channel] = 1.0
        got = float(ph.color_filter_bw(pure, flt)[0, 0, 0])
        want = transmission[channel] * BT709[channel]
        assert abs(got - want) < 1e-6, (
            f"{name} on pure channel {channel}: got {got}, want {want}"
        )


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


# ── HSL-weighted B&W ──────────────────────────────────────────────────────────


def test_hsl_bw_shape_dtype(rgb_f32):
    params = ph.HslWeightedParams([0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
    assert_valid_output(ph.hsl_bw(rgb_f32, params), (H, W, 1))


def test_hsl_bw_zero_weights_match_luminance(rgb_f32):
    """All-zero weights must reduce exactly to the plain luminance kernel."""
    out = ph.hsl_bw(rgb_f32, ph.HslWeightedParams([0.0] * 8))
    np.testing.assert_allclose(out, ph.luminance_bw(rgb_f32), atol=1e-6)


def test_hsl_bw_leaves_neutrals_alone():
    """Zero chroma means zero modulation, whatever the weights are."""
    grey = np.full((8, 8, 3), 0.5, dtype=np.float32)
    out = ph.hsl_bw(grey, ph.HslWeightedParams([1.0] * 8))
    np.testing.assert_allclose(out, 0.5, atol=1e-6)


def test_hsl_bw_is_clamped_at_zero():
    red = np.zeros((4, 4, 3), dtype=np.float32)
    red[..., 0] = 1.0
    out = ph.hsl_bw(red, ph.HslWeightedParams([-3.0, 0, 0, 0, 0, 0, 0, 0]))
    assert float(out.min()) >= 0.0


def test_hsl_params_round_trip():
    weights = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
    params = ph.HslWeightedParams(weights, ph.LuminanceStandard.Bt601, 25.0)
    assert list(params.hue_weights) == pytest.approx(weights)
    assert params.standard == ph.LuminanceStandard.Bt601
    assert params.sigma_deg == pytest.approx(25.0)
    assert params == ph.HslWeightedParams(weights, ph.LuminanceStandard.Bt601, 25.0)
    assert params != ph.HslWeightedParams([0.0] * 8)


def test_hsl_bw_rejects_invalid_sigma(rgb_f32):
    for bad in (0.0, -5.0, float("inf")):
        with pytest.raises(ValueError):
            ph.hsl_bw(rgb_f32, ph.HslWeightedParams([0.0] * 8, ph.LuminanceStandard.Bt709, bad))


def test_hsl_bw_rejects_wrong_channels(grey_f32):
    with pytest.raises(ValueError):
        ph.hsl_bw(grey_f32, ph.HslWeightedParams([0.0] * 8))


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


def test_zone_params_round_trip():
    """Settings serialisation needs to read the offsets back out."""
    offsets = {3: -0.5, 5: 1.0, 8: 0.25}
    params = ph.ZoneParams(offsets)
    assert params.offsets == offsets
    assert params == ph.ZoneParams(offsets)
    assert params != ph.ZoneParams({})
    # The getter hands back a copy, not a live view.
    params.offsets[5] = 99.0
    assert params.offsets[5] == 1.0


def test_guided_filter_params_equality():
    assert ph.GuidedFilterParams(8, 0.01) == ph.GuidedFilterParams(8, 0.01)
    assert ph.GuidedFilterParams(8, 0.01) != ph.GuidedFilterParams(4, 0.01)


@pytest.mark.parametrize("bad_zone", [-1, 11, 99])
def test_zone_system_rejects_out_of_range_zone(grey_f32, bad_zone):
    """An out-of-range zone index used to be a silent no-op."""
    with pytest.raises(ValueError):
        ph.zone_system(grey_f32, ph.ZoneParams({bad_zone: 1.0}))


def test_zone_system_rejects_non_finite_offset(grey_f32):
    with pytest.raises(ValueError):
        ph.zone_system(grey_f32, ph.ZoneParams({5: float("nan")}))


def test_local_contrast_rejects_negative_eps(grey_f32):
    with pytest.raises(ValueError):
        ph.local_contrast(grey_f32, ph.GuidedFilterParams(4, -0.01), 0.5)


def test_zone_system_is_deterministic(grey_f32):
    """Same input, same params, same bytes — every time, every process."""
    params = ph.ZoneParams({z: (z - 5) * 0.1 for z in range(11)})
    first = ph.zone_system(grey_f32, params)
    for _ in range(4):
        np.testing.assert_array_equal(ph.zone_system(grey_f32, params), first)


def test_zone_system_shape_dtype(grey_f32):
    params = ph.ZoneParams({5: 0.5})
    out = ph.zone_system(grey_f32, params)
    assert_valid_output(out, (H, W, 1))


def test_zone_system_empty_params_is_identity(grey_f32):
    """Empty ZoneParams must leave the image unchanged."""
    out = ph.zone_system(grey_f32, ph.ZoneParams({}))
    np.testing.assert_allclose(out, grey_f32, atol=1e-5)


# ── Parametric tone curve ─────────────────────────────────────────────────────


def test_tone_curve_shape_dtype(grey_f32):
    assert_valid_output(ph.tone_curve(grey_f32, ph.ToneCurveParams(1.1, 0.02, 0.9)), (H, W, 1))


def test_tone_curve_default_is_identity(grey_f32):
    np.testing.assert_array_equal(ph.tone_curve(grey_f32, ph.ToneCurveParams()), grey_f32)


def test_tone_curve_matches_the_formula(grey_f32):
    params = ph.ToneCurveParams(1.3, -0.05, 1.7)
    expected = np.maximum(grey_f32 * 1.3 - 0.05, 0.0) ** 1.7
    np.testing.assert_allclose(ph.tone_curve(grey_f32, params), expected, atol=1e-6)


def test_tone_curve_accepts_rgb(rgb_f32):
    """It runs after split-toning, where the data is three-channel again."""
    assert_valid_output(ph.tone_curve(rgb_f32, ph.ToneCurveParams(1.0, 0.0, 0.8)), (H, W, 3))


def test_tone_curve_no_nan_from_negative_offset(grey_f32):
    out = ph.tone_curve(grey_f32, ph.ToneCurveParams(1.0, -0.5, 0.5))
    assert not np.isnan(out).any()
    assert float(out.min()) >= 0.0


def test_tone_curve_rejects_non_positive_power(grey_f32):
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            ph.tone_curve(grey_f32, ph.ToneCurveParams(1.0, 0.0, bad))


def test_tone_curve_params_round_trip():
    p = ph.ToneCurveParams(1.2, 0.03, 0.75)
    assert p.slope == pytest.approx(1.2)
    assert p.offset == pytest.approx(0.03)
    assert p.power == pytest.approx(0.75)
    assert p == ph.ToneCurveParams(1.2, 0.03, 0.75)
    assert p != ph.ToneCurveParams()


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


# ── Film grain ────────────────────────────────────────────────────────────────


def test_film_grain_shape_dtype(grey_f32):
    assert_valid_output(ph.film_grain(grey_f32, ph.GrainParams(0.2, 2.0, 42)), (H, W, 1))


def test_film_grain_zero_intensity_is_identity(grey_f32):
    np.testing.assert_array_equal(ph.film_grain(grey_f32, ph.GrainParams(0.0, 2.0, 42)), grey_f32)


def test_film_grain_is_deterministic(grey_f32):
    """The guarantee the kernel is built around: same seed, same bytes."""
    params = ph.GrainParams(0.3, 2.0, 20260815)
    first = ph.film_grain(grey_f32, params)
    for _ in range(4):
        np.testing.assert_array_equal(ph.film_grain(grey_f32, params), first)


def test_film_grain_different_seeds_differ(grey_f32):
    a = ph.film_grain(grey_f32, ph.GrainParams(0.3, 2.0, 1))
    b = ph.film_grain(grey_f32, ph.GrainParams(0.3, 2.0, 2))
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_film_grain_envelope_silences_the_ends(level):
    flat = np.full((32, 32, 1), level, dtype=np.float32)
    out = ph.film_grain(flat, ph.GrainParams(1.0, 2.0, 5))
    np.testing.assert_allclose(out, level, atol=1e-6)


def test_film_grain_peaks_in_midtones():
    def spread(level):
        flat = np.full((64, 64, 1), level, dtype=np.float32)
        return float(ph.film_grain(flat, ph.GrainParams(0.3, 2.0, 7)).std())

    assert spread(0.5) > spread(0.1) * 2
    assert spread(0.5) > spread(0.9) * 2


@pytest.mark.parametrize("size", [0.5, 1.0, 1.5, 2.0, 4.0, 8.0])
def test_film_grain_every_size_produces_grain(size):
    """Regression: equal band-pass radii silently produced no grain."""
    flat = np.full((64, 64, 1), 0.5, dtype=np.float32)
    assert float(ph.film_grain(flat, ph.GrainParams(0.3, size, 17)).std()) > 0.01


def test_film_grain_rejects_invalid_parameters(grey_f32):
    with pytest.raises(ValueError):
        ph.film_grain(grey_f32, ph.GrainParams(-0.1, 2.0, 0))
    with pytest.raises(ValueError):
        ph.film_grain(grey_f32, ph.GrainParams(0.2, 0.0, 0))


def test_film_grain_rejects_rgb(rgb_f32):
    with pytest.raises(ValueError):
        ph.film_grain(rgb_f32, ph.GrainParams(0.2, 2.0, 0))


def test_grain_params_round_trip():
    p = ph.GrainParams(0.25, 1.5, 20260815)
    assert p.intensity == pytest.approx(0.25)
    assert p.size_pixels == pytest.approx(1.5)
    assert p.seed == 20260815
    assert p == ph.GrainParams(0.25, 1.5, 20260815)
    assert p != ph.GrainParams(0.25, 1.5, 1)


# ── Split-toning ──────────────────────────────────────────────────────────────


def test_split_toning_returns_three_channels(grey_f32):
    """The one kernel that adds channels: (H, W, 1) in, (H, W, 3) out."""
    out = ph.split_toning(grey_f32, ph.SplitToningParams())
    assert_valid_output(out, (H, W, 3))


def test_split_toning_untinted_is_a_faithful_round_trip(grey_f32):
    """With no chroma the result must be neutral, not merely close."""
    out = ph.split_toning(grey_f32, ph.SplitToningParams())
    for c in range(3):
        np.testing.assert_allclose(out[..., c], grey_f32[..., 0], atol=1e-5)


def test_split_toning_tints_shadows_and_highlights_differently():
    img = np.array([[[0.005], [0.9]]], dtype=np.float32)
    params = ph.SplitToningParams([0.0, 0.05, 0.05], [0.0, -0.05, -0.05], 0.5, 0.0)
    out = ph.split_toning(img, params)
    assert out[0, 0, 0] > out[0, 0, 2], "shadow should be warm"
    assert out[0, 1, 2] > out[0, 1, 0], "highlight should be cool"


def test_split_toning_rejects_rgb_input(rgb_f32):
    with pytest.raises(ValueError):
        ph.split_toning(rgb_f32, ph.SplitToningParams())


@pytest.mark.parametrize("pivot", [-0.1, 1.1, float("nan")])
def test_split_toning_rejects_bad_pivot(grey_f32, pivot):
    with pytest.raises(ValueError):
        ph.split_toning(grey_f32, ph.SplitToningParams([0.0] * 3, [0.0] * 3, pivot, 0.0))


@pytest.mark.parametrize("balance", [-1.5, 1.5, float("inf")])
def test_split_toning_rejects_bad_balance(grey_f32, balance):
    with pytest.raises(ValueError):
        ph.split_toning(grey_f32, ph.SplitToningParams([0.0] * 3, [0.0] * 3, 0.5, balance))


def test_split_toning_params_round_trip():
    p = ph.SplitToningParams([0.0, 0.02, -0.03], [0.0, -0.01, 0.04], 0.45, 0.2)
    assert list(p.shadow_oklab) == pytest.approx([0.0, 0.02, -0.03])
    assert list(p.highlight_oklab) == pytest.approx([0.0, -0.01, 0.04])
    assert p.pivot == pytest.approx(0.45)
    assert p.balance == pytest.approx(0.2)
    assert p == ph.SplitToningParams([0.0, 0.02, -0.03], [0.0, -0.01, 0.04], 0.45, 0.2)
    assert p != ph.SplitToningParams()


def test_split_toning_feeds_the_rest_of_the_pipeline(grey_f32):
    """Downstream kernels must accept the three-channel result."""
    toned = ph.split_toning(grey_f32, ph.SplitToningParams([0.0, 0.02, 0.03], [0.0, -0.02, 0.01]))
    out = ph.encode_srgb(ph.vignette(ph.tone_curve(toned, ph.ToneCurveParams(1.1, 0.0, 0.9)),
                                     ph.VignetteParams(0.3, 0.8, 0.0)))
    assert_valid_output(out, (H, W, 3))


# ── Vignette ──────────────────────────────────────────────────────────────────


def test_vignette_shape_dtype(grey_f32):
    assert_valid_output(ph.vignette(grey_f32, ph.VignetteParams(0.5, 0.7, 0.0)), (H, W, 1))


def test_vignette_zero_amount_is_identity(grey_f32):
    np.testing.assert_array_equal(ph.vignette(grey_f32, ph.VignetteParams()), grey_f32)


def test_vignette_darkens_corners_not_centre():
    flat = np.ones((64, 64, 1), dtype=np.float32)
    out = ph.vignette(flat, ph.VignetteParams(0.5, 1.0, 0.0))
    assert out[32, 32, 0] == pytest.approx(1.0, abs=1e-3)
    assert out[0, 0, 0] < 0.6


def test_vignette_negative_amount_lightens():
    flat = np.ones((64, 64, 1), dtype=np.float32)
    out = ph.vignette(flat, ph.VignetteParams(-0.5, 1.0, 0.0))
    assert out[0, 0, 0] > 1.4


def test_vignette_does_not_tint_rgb():
    """Every channel of a pixel must get the same factor."""
    img = np.empty((32, 32, 3), dtype=np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.2, 0.5, 0.8
    out = ph.vignette(img, ph.VignetteParams(0.6, 1.0, 0.0))
    ratios = out / img
    np.testing.assert_allclose(ratios[..., 0], ratios[..., 1], atol=1e-5)
    np.testing.assert_allclose(ratios[..., 1], ratios[..., 2], atol=1e-5)


def test_vignette_is_resolution_independent():
    """A preview and the full frame must agree, so previews are faithful."""
    params = ph.VignetteParams(0.6, 0.8, 0.2)
    small = ph.vignette(np.ones((64, 64, 1), np.float32), params)
    large = ph.vignette(np.ones((512, 512, 1), np.float32), params)
    for sy, sx in [(0, 0), (16, 16), (32, 32), (63, 63)]:
        assert large[sy * 8 + 4, sx * 8 + 4, 0] == pytest.approx(small[sy, sx, 0], abs=0.02)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_vignette_rejects_out_of_range_shape_params(grey_f32, bad):
    with pytest.raises(ValueError):
        ph.vignette(grey_f32, ph.VignetteParams(0.5, bad, 0.0))
    with pytest.raises(ValueError):
        ph.vignette(grey_f32, ph.VignetteParams(0.5, 0.5, bad))


def test_vignette_params_round_trip():
    p = ph.VignetteParams(0.35, 0.6, 0.25)
    assert p.amount == pytest.approx(0.35)
    assert p.feather == pytest.approx(0.6)
    assert p.roundness == pytest.approx(0.25)
    assert p == ph.VignetteParams(0.35, 0.6, 0.25)
    assert p != ph.VignetteParams()


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


# ── Memory layout (regression: PanicException on strided input) ───────────────
#
# `zone_system` and `encode_srgb` used to call `as_slice().expect(...)`, which
# panicked on any non-C-contiguous input. PyO3 turns a Rust panic into
# `pyo3_runtime.PanicException`, which inherits from BaseException — NOT from
# Exception — so it slips straight through a caller's `except Exception:` and
# can take down a GUI worker thread. Downsampled previews (`img[::2, ::2]`) hit
# this on the most ordinary of call sites.


def _layout_variants(arr):
    """Yield (label, array) pairs covering the layouts a caller may pass."""
    yield "c_contiguous", arr
    yield "strided_rows", arr[::2]
    yield "strided_cols", arr[:, ::2]
    yield "fortran", np.asfortranarray(arr)
    yield "reversed", arr[::-1]
    # Zero-stride. Absent from this matrix until the v0.2 audit, which is
    # exactly why an oversized broadcast could abort the interpreter
    # unnoticed: every other variant has the storage its shape implies.
    yield "broadcast", np.broadcast_to(arr[:1], arr.shape)
    # A second zero-stride case, because the first is degenerate for a
    # value check: broadcasting row 0 makes every row identical, so a
    # kernel that reordered rows would still return the right numbers.
    # Broadcasting column 0 keeps the rows distinct and the stride zero.
    yield "broadcast_cols", np.broadcast_to(arr[:, :1], arr.shape)


_LAYOUT_LABELS = [v[0] for v in _layout_variants(np.zeros((4, 4, 1), np.float32))]

# A monotonic 17-entry table, steep enough that a misread sample lands on
# a visibly different output value.
_LAYOUT_LUT = np.linspace(0.0, 1.0, 17, dtype=np.float32) ** 2

# The kernels the layout matrix covers, grouped by what each does to the
# shape. Both the shape test and the value test below consume these
# tables, so the two lists cannot drift apart.

_BW_KERNELS = {  # (H, W, 3) in, (H, W, 1) out
    "luminance_bw": lambda x: ph.luminance_bw(x),
    "channel_mixer_bw": lambda x: ph.channel_mixer_bw(x, 0.3, 0.59, 0.11),
    "color_filter_bw": lambda x: ph.color_filter_bw(x, ph.ColorFilter.Yellow8K2),
    "hsl_bw": lambda x: ph.hsl_bw(x, ph.HslWeightedParams([0.2] * 8)),
}

_SAME_SHAPE_KERNELS = {  # shape-preserving, RGB and luminance alike
    "zone_system": lambda x: ph.zone_system(x, ph.ZoneParams({5: 0.5})),
    "local_contrast": lambda x: ph.local_contrast(x, ph.GuidedFilterParams(4, 0.01), 0.5),
    "encode_srgb": lambda x: ph.encode_srgb(x),
    "exposure": lambda x: ph.exposure(x, 0.5),
    "tone_curve": lambda x: ph.tone_curve(x, ph.ToneCurveParams(1.2, 0.0, 1.0)),
    "vignette": lambda x: ph.vignette(x, ph.VignetteParams(0.4, 0.7)),
    "film_grain": lambda x: ph.film_grain(x, ph.GrainParams(0.2, 2.0, 99)),
    "orient": lambda x: ph.orient(x, ph.Orientation.Rotate180),
    "highlight_rolloff": lambda x: ph.highlight_rolloff(x, ph.RolloffParams(0.7, 3.0)),
    "shadow_rolloff": lambda x: ph.shadow_rolloff(x, ph.ShadowRolloffParams(0.2, 0.5)),
    "blur": lambda x: ph.blur(x, ph.BlurParams(2.0)),
    "glow": lambda x: ph.glow(x, ph.GlowParams(0.3, 3.0, 0.4)),
    # The three kernels whose own strided tests use a C-order view only,
    # which takes the same code path as its contiguous copy.
    "apply_lut": lambda x: ph.apply_lut(x, _LAYOUT_LUT, ph.LutParams()),
    "quantize_u8": lambda x: ph.quantize_u8(x, ph.QuantizeParams(ph.Dither.Tpdf, 11)),
    "quantize_u16": lambda x: ph.quantize_u16(x),
}

_SHAPE_CHANGING_KERNELS = {  # the output shape is not the input shape
    "split_toning": lambda x: ph.split_toning(
        x, ph.SplitToningParams([0.0, -0.02, -0.04], [0.0, 0.03, 0.03])
    ),
    "crop": lambda x: ph.crop(
        x, ph.CropParams(0, 0, max(x.shape[1] // 2, 1), max(x.shape[0] // 2, 1))
    ),
    "resize": lambda x: ph.resize(x, ph.ResizeParams(7, 5, ph.ResizeFilter.Area)),
    "straighten": lambda x: ph.straighten(x, ph.StraightenParams(3.0)),
}


@pytest.mark.parametrize("label", _LAYOUT_LABELS)
def test_all_kernels_accept_any_layout(label, rgb_f32, grey_f32):
    """No kernel may panic on a non-C-contiguous input, whatever its layout."""
    rgb = dict(_layout_variants(rgb_f32))[label]
    grey = dict(_layout_variants(grey_f32))[label]

    for name, kernel in _BW_KERNELS.items():
        out = kernel(rgb)
        assert out.shape == rgb.shape[:2] + (1,), name
        assert out.flags["C_CONTIGUOUS"], f"{name}: output must be C-contiguous"

    for name, kernel in _SAME_SHAPE_KERNELS.items():
        out = kernel(grey)
        assert out.shape == grey.shape, name
        assert out.flags["C_CONTIGUOUS"], f"{name}: output must be C-contiguous"

    # Geometry kernels change the shape by construction, so they assert
    # the contract rather than shape preservation: no panic, C-contiguous
    # out, and the dimensions the parameters ask for.
    gh, gw = grey.shape[:2]
    out = {name: kernel(grey) for name, kernel in _SHAPE_CHANGING_KERNELS.items()}
    for name, arr in out.items():
        assert arr.flags["C_CONTIGUOUS"], f"{name}: output must be C-contiguous"

    # split_toning is the one kernel that grows a channel.
    assert out["split_toning"].shape == grey.shape[:2] + (3,)
    assert out["crop"].shape == (max(gh // 2, 1), max(gw // 2, 1), 1)
    assert out["resize"].shape == (5, 7, 1)
    assert out["straighten"].ndim == 3 and out["straighten"].shape[2] == 1


@pytest.mark.parametrize("label", _LAYOUT_LABELS)
def test_all_kernels_are_layout_agnostic_by_value(label, rgb_f32, grey_f32):
    """Every kernel must return the same *numbers* for a given layout as it
    does for the contiguous copy of the same logical input.

    v0.2 audit finding F10: the matrix above asserts only shape and
    C-contiguity, so a kernel could scramble every pixel of a Fortran-order
    or reversed input and the suite would stay green. Measured with a flat
    `as_slice_memory_order` fast path spliced into `exposure` — a plausible
    optimisation — on a (3, 2, 1) array holding 0..5 at gain 2.0:

        C order:  [0, 2, 4, 6, 8, 10]
        F order:  [0, 4, 8, 2, 6, 10]   <- wrong, and nothing failed

    `test_strided_matches_contiguous_copy` cannot cover this. It passes one
    C-order strided view, whose logical order matches its memory order, so
    view and copy take the same path through such a fast path and agree
    however wrong both are. Here the reference is the kernel's output on
    `np.ascontiguousarray` of the variant, and the Fortran, reversed and
    zero-stride variants reach it by a different route.

    CLAUDE.md section 2 makes layout-agnostic input a hard constraint:
    `PyReadonlyArray3` accepts strided, Fortran-order and negative-stride
    arrays, and a consumer passing `img[::2, ::2]` is normal.

    Equality is exact, not approximate: every kernel here was measured
    bit-identical between each variant and its contiguous copy.
    """
    rgb = dict(_layout_variants(rgb_f32))[label]
    grey = dict(_layout_variants(grey_f32))[label]

    for src, table in ((rgb, _BW_KERNELS), (grey, _SAME_SHAPE_KERNELS),
                       (grey, _SHAPE_CHANGING_KERNELS)):
        reference_input = np.ascontiguousarray(src)
        for name, kernel in table.items():
            np.testing.assert_array_equal(
                kernel(src), kernel(reference_input),
                err_msg=f"{name} on the {label} layout disagrees with its contiguous copy",
            )

    # `histogram` is the crate's one reduction: it returns tallies rather
    # than an image, so the array comparison above cannot reach it.
    for src in (rgb, grey):
        got, want = ph.histogram(src), ph.histogram(np.ascontiguousarray(src))
        np.testing.assert_array_equal(got.counts(), want.counts())
        np.testing.assert_array_equal(got.cdf(), want.cdf())
        assert (got.below(), got.above(), got.non_finite()) == (
            want.below(),
            want.above(),
            want.non_finite(),
        ), f"histogram tallies differ on the {label} layout"

    # One oracle that does not go through the kernel twice, so the matrix
    # is not purely kernel-against-kernel: 2^1 is exact in binary floating
    # point, so one stop of exposure is a bit-exact doubling of every
    # sample, in whatever layout the caller hands it over.
    for src in (rgb, grey):
        np.testing.assert_array_equal(
            ph.exposure(src, 1.0), np.asarray(src) * np.float32(2.0),
            err_msg=f"exposure is not an exact doubling on the {label} layout",
        )


def test_resampling_is_layout_agnostic(rgb_f32):
    """resize and straighten join the layout matrix (review finding)."""
    view = rgb_f32[::2, ::3]
    copy = np.ascontiguousarray(view)
    rp = ph.ResizeParams(17, 11, ph.ResizeFilter.CatmullRom)
    np.testing.assert_array_equal(ph.resize(view, rp), ph.resize(copy, rp))
    sp = ph.StraightenParams(4.0)
    np.testing.assert_array_equal(ph.straighten(view, sp), ph.straighten(copy, sp))


def test_strided_matches_contiguous_copy(grey_f32):
    """A strided view and its contiguous copy must give identical results."""
    view = grey_f32[::2, ::3]
    copy = np.ascontiguousarray(view)
    assert not view.flags["C_CONTIGUOUS"]

    params = ph.ZoneParams({3: -0.4, 6: 0.8})
    np.testing.assert_array_equal(ph.zone_system(view, params), ph.zone_system(copy, params))
    np.testing.assert_array_equal(ph.encode_srgb(view), ph.encode_srgb(copy))

    gp = ph.GuidedFilterParams(3, 0.01)
    np.testing.assert_array_equal(
        ph.local_contrast(view, gp, 0.7), ph.local_contrast(copy, gp, 0.7)
    )


# ── highlight_rolloff ────────────────────────────────────────────────────────


def test_rolloff_default_is_exactly_a_hard_clip(rgb_f32):
    """The default must be byte-identical to np.clip's highlight end, so
    inserting the stage into an existing pipeline changes nothing."""
    img = rgb_f32 * 3.0
    got = ph.highlight_rolloff(img, ph.RolloffParams())
    want = np.minimum(img, 1.0)
    np.testing.assert_array_equal(got, want)


def test_rolloff_keeps_highlights_separable():
    vals = np.array([[[1.0], [1.5], [2.0], [3.0]]], dtype=np.float32)
    clipped = ph.highlight_rolloff(vals, ph.RolloffParams())
    rolled = ph.highlight_rolloff(vals, ph.RolloffParams(0.7, 4.0))
    assert len(np.unique(clipped)) == 1, "default clips them together"
    assert len(np.unique(rolled)) == 4, "the shoulder keeps them apart"
    assert (rolled <= 1.0).all()


def test_rolloff_endpoints_are_exact():
    knee, white = 0.75, 4.0
    probe = np.array([[[knee, white]]], dtype=np.float32).reshape(1, 2, 1)
    out = ph.highlight_rolloff(probe, ph.RolloffParams(knee, white))
    assert out[0, 0, 0] == np.float32(knee), "f(knee) must be knee"
    assert out[0, 1, 0] == np.float32(1.0), "f(white) must be exactly 1.0"


def test_rolloff_below_knee_is_untouched(grey_f32):
    small = grey_f32 * 0.5  # everything well below the knee
    out = ph.highlight_rolloff(small, ph.RolloffParams(0.9, 4.0))
    np.testing.assert_array_equal(out, small)


@pytest.mark.parametrize(
    "knee,white",
    [(1.5, 2.0), (-0.1, 2.0), (0.5, 0.5), (0.5, 0.0), (float("nan"), 2.0), (0.5, float("inf"))],
)
def test_rolloff_rejects_out_of_domain(rgb_f32, knee, white):
    with pytest.raises(ValueError):
        ph.highlight_rolloff(rgb_f32, ph.RolloffParams(knee, white))


def test_rolloff_params_repr_and_eq():
    a, b = ph.RolloffParams(0.8, 2.0), ph.RolloffParams(0.8, 2.0)
    assert a == b
    assert a != ph.RolloffParams(0.8, 2.5)
    assert "knee=0.8" in repr(a)
    assert ph.RolloffParams().knee == 1.0


# ── quantize ─────────────────────────────────────────────────────────────────


def test_quantize_dtypes_and_ranges(grey_f32):
    u8 = ph.quantize_u8(grey_f32)
    u16 = ph.quantize_u16(grey_f32)
    assert u8.dtype == np.uint8 and u16.dtype == np.uint16
    assert u8.shape == grey_f32.shape and u16.shape == grey_f32.shape
    assert u8.flags["C_CONTIGUOUS"] and u16.flags["C_CONTIGUOUS"]
    assert 0 <= u8.min() and u8.max() <= 255
    assert 0 <= u16.min() and u16.max() <= 65535


def test_quantize_params_default_to_no_dither(grey_f32):
    """Calling without params must equal explicit Dither.None."""
    np.testing.assert_array_equal(
        ph.quantize_u8(grey_f32),
        ph.quantize_u8(grey_f32, ph.QuantizeParams(ph.Dither.Off, 0)),
    )


def test_dither_enum_is_spellable_from_python():
    """`Dither.None` would be a syntax error — `None` is a keyword. The
    variant is named `Off` so the Python surface can actually name it."""
    assert ph.Dither.Off != ph.Dither.Tpdf
    assert ph.QuantizeParams().dither == ph.Dither.Off


def test_quantize_endpoints_and_clamping():
    probe = np.array([[[-1.0, 0.0, 0.5, 1.0, 2.0, np.nan]]], dtype=np.float32).reshape(1, 6, 1)
    out = ph.quantize_u8(probe)
    assert list(out[0, :, 0]) == [0, 0, 128, 255, 255, 0], f"got {list(out[0, :, 0])}"


def test_quantize_roundtrips_every_8bit_code():
    codes = np.arange(256, dtype=np.float32) / 255.0
    out = ph.quantize_u8(codes.reshape(1, 256, 1))
    np.testing.assert_array_equal(out[0, :, 0], np.arange(256, dtype=np.uint8))


def test_quantize_dither_is_seeded_and_bounded(grey_f32):
    a = ph.quantize_u8(grey_f32, ph.QuantizeParams(ph.Dither.Tpdf, 5))
    b = ph.quantize_u8(grey_f32, ph.QuantizeParams(ph.Dither.Tpdf, 5))
    c = ph.quantize_u8(grey_f32, ph.QuantizeParams(ph.Dither.Tpdf, 6))
    plain = ph.quantize_u8(grey_f32)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.abs(a.astype(int) - plain.astype(int)).max() <= 1


def test_quantize_dither_breaks_banding():
    ramp = ((60.0 + np.arange(512) / 511.0) / 255.0).astype(np.float32).reshape(1, 512, 1)
    plain = ph.quantize_u8(ramp)
    dithered = ph.quantize_u8(ramp, ph.QuantizeParams(ph.Dither.Tpdf, 3))
    assert len(np.unique(plain)) <= 2
    assert (np.diff(dithered[0, :, 0].astype(int)) != 0).sum() > 50


def test_quantize_accepts_strided_input(rgb_f32):
    view = rgb_f32[::2, ::3]
    params = ph.QuantizeParams(ph.Dither.Tpdf, 11)
    np.testing.assert_array_equal(
        ph.quantize_u8(view, params), ph.quantize_u8(np.ascontiguousarray(view), params)
    )


def test_quantize_params_repr_and_eq():
    a = ph.QuantizeParams(ph.Dither.Tpdf, 7)
    assert a == ph.QuantizeParams(ph.Dither.Tpdf, 7)
    assert a != ph.QuantizeParams(ph.Dither.Tpdf, 8)
    assert "seed=7" in repr(a)
    assert ph.QuantizeParams().seed == 0


# ── histogram ────────────────────────────────────────────────────────────────


def test_histogram_counts_every_sample(rgb_f32):
    h = ph.histogram(rgb_f32)
    assert h.channels == 3 and h.bins == 256
    assert h.counts().shape == (3, 256)
    assert h.counts().dtype == np.uint64
    for ch in range(3):
        assert h.total(ch) == H * W


def test_histogram_separates_clipping_from_content():
    """Out-of-range samples must not be folded into the end bins — that
    is the defect this design exists to avoid."""
    probe = np.array([[[-0.5], [0.5], [1.5], [2.5]]], dtype=np.float32)
    h = ph.histogram(probe, ph.HistogramParams(4, 0.0, 1.0))
    assert h.below() == [1]
    assert h.above() == [2]
    assert h.counts().sum() == 1, "only the in-range sample is binned"
    assert h.total(0) == 4


def test_histogram_nan_is_not_clipping():
    probe = np.array([[[np.nan], [np.inf], [-np.inf]]], dtype=np.float32)
    h = ph.histogram(probe)
    assert h.non_finite() == [1], "NaN is broken, not bright or dark"
    assert h.above() == [1] and h.below() == [1], "infinities are genuinely out of range"


def test_histogram_endpoints():
    probe = np.array([[[0.0], [1.0]]], dtype=np.float32)
    h = ph.histogram(probe, ph.HistogramParams(256, 0.0, 1.0))
    assert h.counts()[0, 0] == 1 and h.counts()[0, 255] == 1
    assert h.above() == [0], "exactly max is at white, not clipped"


def test_histogram_cdf_is_monotonic_and_normalised(grey_f32):
    cdf = ph.histogram(grey_f32).cdf()
    assert cdf.shape == (1, 256) and cdf.dtype == np.float32
    assert (np.diff(cdf[0]) >= -1e-7).all(), "cdf must be non-decreasing"
    assert abs(cdf[0, -1] - 1.0) < 1e-6


def test_histogram_accepts_strided_input(rgb_f32):
    view = rgb_f32[::2, ::3]
    np.testing.assert_array_equal(
        ph.histogram(view).counts(), ph.histogram(np.ascontiguousarray(view)).counts()
    )


def test_histogram_rejects_bad_params(rgb_f32):
    for bins, lo, hi in [(1, 0.0, 1.0), (0, 0.0, 1.0), (256, 1.0, 0.0), (256, 0.0, 0.0)]:
        with pytest.raises(ValueError):
            ph.histogram(rgb_f32, ph.HistogramParams(bins, lo, hi))


def test_histogram_total_rejects_bad_channel(grey_f32):
    with pytest.raises(IndexError):
        ph.histogram(grey_f32).total(5)


# ── apply_lut ────────────────────────────────────────────────────────────────


def test_apply_lut_identity_table_is_identity(grey_f32):
    lut = np.linspace(0, 1, 256).astype(np.float32)
    out = ph.apply_lut(grey_f32, lut)
    assert np.abs(out - grey_f32).max() < 1e-5


def test_apply_lut_clamps_outside_the_domain():
    probe = np.array([[[-5.0], [0.5], [5.0]]], dtype=np.float32)
    lut = np.array([0.2, 0.8], dtype=np.float32)
    out = ph.apply_lut(probe, lut)
    assert out[0, 0, 0] == np.float32(0.2)
    assert out[0, 2, 0] == np.float32(0.8)


def test_apply_lut_nan_propagates():
    probe = np.array([[[np.nan]]], dtype=np.float32)
    assert np.isnan(ph.apply_lut(probe, np.array([0.0, 1.0], np.float32))[0, 0, 0])


def test_apply_lut_accepts_a_non_contiguous_table(grey_f32):
    """A cdf() row, or any slice, must work without the caller copying."""
    wide = np.linspace(0, 1, 512).astype(np.float32)
    strided = wide[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        ph.apply_lut(grey_f32, strided), ph.apply_lut(grey_f32, np.ascontiguousarray(strided))
    )


def test_apply_lut_custom_domain():
    probe = np.array([[[2.0]]], dtype=np.float32)
    lut = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(ph.apply_lut(probe, lut, ph.LutParams(0.0, 4.0))[0, 0, 0] - 0.5) < 1e-6


def test_apply_lut_rejects_bad_tables(grey_f32):
    for bad in [
        np.array([0.5], np.float32),
        np.array([], np.float32),
        np.array([0.0, np.nan], np.float32),
        np.array([0.0, np.inf], np.float32),
    ]:
        with pytest.raises(ValueError):
            ph.apply_lut(grey_f32, bad)
    good = np.linspace(0, 1, 8).astype(np.float32)
    for lo, hi in [(1.0, 0.0), (0.0, 0.0)]:
        with pytest.raises(ValueError):
            ph.apply_lut(grey_f32, good, ph.LutParams(lo, hi))


def test_histogram_and_lut_compose_into_equalisation():
    """The composition that justifies shipping these two rather than an
    `equalise` kernel: cdf -> apply_lut IS equalisation."""
    low = (np.random.default_rng(3).random((128, 128, 1)).astype(np.float32) * 0.2 + 0.4)
    h = ph.histogram(low)
    equalised = ph.apply_lut(low, h.equalisation_lut()[0])
    assert equalised.max() - equalised.min() > 3 * (low.max() - low.min())
    after = ph.histogram(equalised)
    assert after.total(0) == 128 * 128, "no samples invented or lost"


def test_equalisation_lut_is_aligned_and_cdf_is_not():
    """`cdf()` is cumulative at bin upper edges; `apply_lut` spreads its
    entries evenly. The leading zero reconciles the two, and without it
    equalising a uniform image lifts black by a full bin."""
    bins = 256
    img = ((np.arange(bins) + 0.5) / bins).astype(np.float32).reshape(1, bins, 1)
    h = ph.histogram(img, ph.HistogramParams(bins, 0.0, 1.0))

    aligned = ph.apply_lut(img, h.equalisation_lut()[0])
    assert np.abs(aligned - img).max() < 1e-6, "aligned table must be the identity here"

    raw = ph.apply_lut(img, h.cdf()[0])
    assert np.abs(raw - img).max() > 0.5 / bins, "the raw cdf is misaligned by ~a bin"
    assert h.equalisation_lut().shape == (1, bins + 1)
    assert h.equalisation_lut()[0, 0] == 0.0


def test_apply_lut_accepts_a_reversed_table(grey_f32):
    """`np.flip(cdf)` is the documented way to build a histogram-matching
    transfer, and a reversed array is negative-stride — the case that
    used to raise an uncatchable PanicException."""
    cdf = ph.histogram(grey_f32).equalisation_lut()[0]
    reversed_table = np.flip(cdf)
    assert not reversed_table.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        ph.apply_lut(grey_f32, reversed_table),
        ph.apply_lut(grey_f32, np.ascontiguousarray(reversed_table)),
    )


def test_histogram_rejects_oversized_requests_instead_of_aborting():
    """An allocation failure in Rust aborts the process rather than
    unwinding, so nothing in Python could catch it. These must be
    rejected up front."""
    tiny = np.zeros((1, 1, 1), np.float32)
    with pytest.raises(ValueError):
        ph.histogram(tiny, ph.HistogramParams(500_000_000, 0.0, 1.0))
    # Default parameters, but a channel count that multiplies out.
    with pytest.raises(ValueError):
        ph.histogram(np.zeros((0, 1, 20_000_000), np.float32))


def test_apply_lut_nan_keeps_its_bit_pattern():
    negative_nan = np.array([[[np.float32(np.frombuffer(b"\x00\x00\xc0\xff", dtype=np.float32)[0])]]], dtype=np.float32)
    out = ph.apply_lut(negative_nan, np.linspace(0, 1, 16).astype(np.float32))
    assert out.view(np.uint32)[0, 0, 0] == negative_nan.view(np.uint32)[0, 0, 0]


# ── shadow_rolloff (the toe) ─────────────────────────────────────────────────


def test_shadow_rolloff_default_is_the_identity(grey_f32):
    np.testing.assert_array_equal(ph.shadow_rolloff(grey_f32), grey_f32)
    np.testing.assert_array_equal(
        ph.shadow_rolloff(grey_f32, ph.ShadowRolloffParams(0.2, 0.0)), grey_f32
    )


def test_shadow_rolloff_params_defaults_and_repr():
    """The documented `ShadowRolloffParams()` default is the identity, and
    the pyo3 signature defaults are only reachable from Python — nothing
    on the Rust side exercises them."""
    p = ph.ShadowRolloffParams()
    assert p.knee == pytest.approx(0.2)
    assert p.strength == 0.0
    assert p == ph.ShadowRolloffParams(0.2, 0.0)
    assert p != ph.ShadowRolloffParams(0.2, 0.5)
    assert "strength=0" in repr(p)
    # Keyword-only construction, as the docstring advertises.
    assert ph.ShadowRolloffParams(strength=0.5).strength == 0.5
    assert ph.ShadowRolloffParams(knee=0.4).knee == pytest.approx(0.4)


def test_shadow_rolloff_non_finite_at_every_strength():
    """-inf must survive at strength 1.0, where the continuation slope is
    zero and `0.0 * -inf` would make it NaN."""
    probe = np.array([[[np.nan], [np.inf], [-np.inf]]], dtype=np.float32)
    for st in (0.0, 0.5, 1.0):
        out = ph.shadow_rolloff(probe, ph.ShadowRolloffParams(0.2, st))
        assert np.isnan(out[0, 0, 0])
        assert out[0, 1, 0] == np.inf
        assert out[0, 2, 0] == -np.inf, f"-inf became {out[0, 2, 0]} at strength {st}"


def test_shadow_rolloff_compresses_shadow_separation():
    probe = np.array([[[0.0], [0.02]]], dtype=np.float32)
    sep = lambda st: float(
        np.diff(ph.shadow_rolloff(probe, ph.ShadowRolloffParams(0.2, st))[0, :, 0])[0]
    )
    assert abs(sep(0.0) - 0.02) < 1e-6
    assert sep(0.5) < 0.02 * 0.65
    assert sep(1.0) < 0.02 * 0.25
    assert sep(1.0) < sep(0.5)


def test_shadow_rolloff_pins_its_endpoints_and_leaves_the_rest():
    knee = 0.2
    probe = np.array([[[0.0], [knee], [0.5], [1.0], [4.0]]], dtype=np.float32)
    out = ph.shadow_rolloff(probe, ph.ShadowRolloffParams(knee, 0.9))
    assert out[0, 0, 0] == np.float32(0.0)
    assert out[0, 1, 0] == np.float32(knee)
    np.testing.assert_array_equal(out[0, 2:, 0], probe[0, 2:, 0])


def test_shadow_rolloff_darkens_rather_than_lifting():
    probe = (np.arange(64) / 320.0).astype(np.float32).reshape(1, 64, 1)
    out = ph.shadow_rolloff(probe, ph.ShadowRolloffParams(0.2, 0.7))
    assert (out <= probe + 1e-7).all()


@pytest.mark.parametrize(
    "knee,strength",
    [(1.5, 0.5), (-0.1, 0.5), (0.2, 1.5), (0.2, -0.1), (float("nan"), 0.5), (0.2, float("inf"))],
)
def test_shadow_rolloff_rejects_out_of_domain(grey_f32, knee, strength):
    with pytest.raises(ValueError):
        ph.shadow_rolloff(grey_f32, ph.ShadowRolloffParams(knee, strength))


def test_characteristic_curve_composes_from_three_kernels():
    """Toe, straight section, shoulder: low slope at both ends, contrast
    held in the middle. That shape is the whole point of the toe."""
    top = 4.0
    x = np.linspace(0, top, 4001).astype(np.float32).reshape(1, -1, 1)
    y = ph.highlight_rolloff(
        ph.tone_curve(
            ph.shadow_rolloff(x, ph.ShadowRolloffParams(0.18, 0.8)),
            ph.ToneCurveParams(1.2, 0.0, 1.0),
        ),
        ph.RolloffParams(0.7, 3.0),
    )
    assert (np.diff(y[0, :, 0]) >= -1e-7).all(), "must stay monotone"
    assert y.max() <= 1.0 + 1e-6

    slope = np.gradient(y[0, :, 0].astype(np.float64), x[0, :, 0].astype(np.float64))
    at = lambda v: slope[np.argmin(np.abs(x[0, :, 0] - v))]
    assert at(0.01) < at(0.4) * 0.6, "the toe holds less contrast than the midtones"
    assert at(2.0) < at(0.4) * 0.6, "and so does the shoulder"
    assert at(0.4) > 1.0, "the straight section carries the contrast it was given"


# ── Oversized allocations must raise, never abort ────────────────────────────


def _oversized_shape():
    """A logical shape that exceeds both the crate's allocation cap and
    this machine's memory.

    Two constraints, and only the first is obvious:

    1. It must exceed `phaios_core`'s 8 GiB single-allocation cap, or the
       guard these tests exist to check never fires.
    2. If that guard ever regresses, the allocation that follows must be
       *refused* outright rather than granted. Linux's heuristic overcommit
       refuses a single request larger than RAM + swap; one that merely fits
       is granted, and the process then faults pages in until the OOM killer
       picks a victim — which on a desktop is whatever the user was doing,
       not this test process.

    A fixed constant cannot satisfy the second everywhere, and this is not
    hypothetical: the previous 120 GB constant is refused on a 64 GB laptop
    but *granted* on a machine with 65 GB of RAM and 137 GB of swap, where
    it takes the desktop down instead of failing. The floor covers hosts
    where the query is unavailable (Windows has no `sysconf`), and is set
    high enough to be refused by any machine that currently exists.
    """
    floor = 16 << 40  # 16 TiB
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
        total = 0
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("SwapTotal:"):
                    total += int(line.split()[1]) * 1024
                    break
    except OSError:  # pragma: no cover - not Linux
        pass

    want = max(floor, 2 * total)
    side = math.isqrt(want // (4 * 3))
    return (side, side, 3)


# 4 bytes of real storage behind a shape too large for any allocator to
# satisfy. Printed on failure so a bug report from another machine is
# interpretable.
OVERSIZED = _oversized_shape()


@pytest.mark.parametrize(
    "name,call",
    [
        ("exposure", lambda a: ph.exposure(a, 1.0)),
        ("encode_srgb", ph.encode_srgb),
        ("tone_curve", lambda a: ph.tone_curve(a, ph.ToneCurveParams(1.1, 0.0, 1.0))),
        ("vignette", lambda a: ph.vignette(a, ph.VignetteParams(0.4, 0.7))),
        ("highlight_rolloff", lambda a: ph.highlight_rolloff(a, ph.RolloffParams(0.8, 2.0))),
        ("shadow_rolloff", lambda a: ph.shadow_rolloff(a, ph.ShadowRolloffParams(0.2, 0.5))),
        ("luminance_bw", ph.luminance_bw),
        ("channel_mixer_bw", lambda a: ph.channel_mixer_bw(a, 0.3, 0.6, 0.1)),
        ("hsl_bw", lambda a: ph.hsl_bw(a, ph.HslWeightedParams([0.1] * 8))),
        ("apply_lut", lambda a: ph.apply_lut(a, np.linspace(0, 1, 16).astype(np.float32))),
        ("quantize_u8", ph.quantize_u8),
        ("film_grain", lambda a: ph.film_grain(a[:, :, :1], ph.GrainParams(0.2, 2.0, 1))),
        ("local_contrast", lambda a: ph.local_contrast(a[:, :, :1], ph.GuidedFilterParams(2, 0.01), 0.5)),
        ("split_toning", lambda a: ph.split_toning(a[:, :, :1], ph.SplitToningParams([0]*3, [0]*3))),
    ],
)
def test_oversized_output_raises_instead_of_aborting(name, call):
    """A zero-stride numpy view has an unbounded logical shape backed by
    almost no memory. Allocating it directly calls Rust's
    handle_alloc_error, which *aborts* — raising nothing at all, not even
    PanicException, and killing the interpreter. Nothing in Python could
    catch that, so the kernels must refuse up front.

    If this regresses, the test process dies rather than failing, which is
    itself the signal.
    """
    huge = np.broadcast_to(np.float32(0.05), OVERSIZED)
    with pytest.raises(MemoryError):
        call(huge)


def test_oversized_error_is_memoryerror_not_valueerror():
    """MemoryError, as numpy raises for the same request — the arguments
    are well-formed, there is simply too much of them."""
    huge = np.broadcast_to(np.float32(0.05), OVERSIZED)
    with pytest.raises(MemoryError) as e:
        ph.exposure(huge, 1.0)
    assert "above the" in str(e.value)
    # MemoryError is not a ValueError, so a consumer catching bad
    # arguments does not accidentally swallow this.
    assert not isinstance(e.value, ValueError)


def test_a_real_frame_is_never_refused():
    """The limit must be nowhere near a photograph. 24 MP RGB is the
    crate's benchmark size."""
    frame = np.zeros((4323, 5764, 3), np.float32)
    assert ph.exposure(frame, 0.5).shape == frame.shape


def test_resize_refuses_an_oversized_target_instead_of_aborting():
    """`resize` is the only kernel the sweep above cannot cover.

    Every other kernel's oversize arrives in the *input* view, which the
    sweep supplies as a zero-stride broadcast. `resize` takes its output
    size from its *parameters*, so a 2x2 image can ask for 200000x200000
    — 480 GB — with a perfectly ordinary input. Nothing in the parameter
    validation bounds the target; only the allocation guard does.

    If this regresses the interpreter is killed by SIGABRT rather than
    raising, so the test process dies instead of failing. That is itself
    the signal.
    """
    small = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(MemoryError) as e:
        ph.resize(small, ph.ResizeParams(200_000, 200_000))
    assert "above the" in str(e.value)


def test_resize_scratch_buffer_is_bounded_too():
    """Both passes allocate. An extreme width alone overruns the
    intermediate (in_h, out_w, c) buffer while the final output would
    still fit, so the guard has to sit on both."""
    small = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(MemoryError):
        ph.resize(small, ph.ResizeParams(400_000_000, 1))


def test_resize_within_the_limit_is_unaffected():
    """The guard must be nowhere near an ordinary enlargement."""
    small = np.zeros((2, 2, 3), dtype=np.float32)
    assert ph.resize(small, ph.ResizeParams(64, 48)).shape == (48, 64, 3)


# ── blur ─────────────────────────────────────────────────────────────────────


def test_blur_sigma_zero_is_the_exact_identity(rgb_f32):
    np.testing.assert_array_equal(ph.blur(rgb_f32), rgb_f32)
    np.testing.assert_array_equal(ph.blur(rgb_f32, ph.BlurParams(0.0)), rgb_f32)


@pytest.mark.parametrize("sigma", [0.5, 1.5, 3.9, 4.0, 8.0])
def test_blur_preserves_a_constant_image(sigma):
    """Borders clamp, so this must hold at the edges too — the usual place
    a blur leaks darkness in."""
    img = np.full((17, 23, 1), 0.375, np.float32)
    out = ph.blur(img, ph.BlurParams(sigma))
    assert np.abs(out - 0.375).max() < 2e-6


@pytest.mark.parametrize("sigma", [1.0, 2.0, 5.0])
def test_blur_matches_an_independent_gaussian(sigma):
    """Compared against a direct numpy convolution, in the interior where
    border conventions cannot differ."""
    rng = np.random.default_rng(9)
    img = rng.random((96, 96, 1)).astype(np.float32)
    r = int(np.ceil(4 * sigma))
    k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(img[:, :, 0].astype(np.float64), r, mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, pad)
    want = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, tmp)
    got = ph.blur(img, ph.BlurParams(sigma))[:, :, 0]
    c = slice(24, 72)
    # Below the crossover the direct path is exact; above it the box
    # approximation is good to about 1e-2 of peak, which is invisible once
    # a glow scales it by `amount`.
    tol = 1e-6 if sigma < 4.0 else 2e-2
    assert np.abs(got[c, c] - want[c, c]).max() < tol


def test_blur_energy_is_preserved_while_the_kernel_fits():
    sigma = 4.0
    n = int(np.ceil(4 * sigma)) * 2 + 9
    img = np.zeros((n, n, 1), np.float32)
    img[n // 2, n // 2, 0] = 1.0
    assert abs(float(ph.blur(img, ph.BlurParams(sigma)).sum()) - 1.0) < 2e-3


def test_blur_larger_sigma_spreads_further():
    n = 41
    img = np.zeros((n, n, 1), np.float32)
    img[n // 2, n // 2, 0] = 1.0
    peaks = [float(ph.blur(img, ph.BlurParams(s))[n // 2, n // 2, 0]) for s in (0.5, 1.0, 2.0, 4.0, 6.0)]
    assert all(b < a for a, b in zip(peaks, peaks[1:])), peaks


@pytest.mark.parametrize("sigma", [-1.0, float("nan"), float("inf")])
def test_blur_rejects_bad_sigma(grey_f32, sigma):
    with pytest.raises(ValueError):
        ph.blur(grey_f32, ph.BlurParams(sigma))


@pytest.mark.parametrize("sigma", [4096.5, 1e4, 1e5, 3.4e38])
def test_blur_rejects_sigma_above_the_maximum(grey_f32, sigma):
    # These used to run an unbounded search inside py.detach: no result and
    # no way to interrupt it, since the GIL was released. sigma = 1e4 never
    # returned at all. A caller-controlled parameter must not hang the
    # calling thread.
    with pytest.raises(ValueError):
        ph.blur(grey_f32, ph.BlurParams(sigma))


def test_blur_accepts_the_largest_permitted_sigma(grey_f32):
    out = ph.blur(grey_f32, ph.BlurParams(4096.0))
    assert out.shape == grey_f32.shape
    assert np.isfinite(out).all()


def test_glow_inherits_the_blur_sigma_bound(grey_f32):
    # glow delegates its sigma to blur's validator, so the bound has to
    # reach it without glow restating anything.
    with pytest.raises(ValueError):
        ph.glow(grey_f32, ph.GlowParams(0.5, 1e5, 0.5))


def test_blur_params_repr_and_defaults():
    p = ph.BlurParams()
    assert p.sigma == 0.0
    assert p.shape == ph.BlurShape.Gaussian
    assert p == ph.BlurParams(0.0, ph.BlurShape.Gaussian)
    assert "sigma=2" in repr(ph.BlurParams(2.0))


# ── glow: halation, diffusion, veiling glare ─────────────────────────────────


def test_glow_amount_zero_is_the_exact_identity(rgb_f32):
    np.testing.assert_array_equal(ph.glow(rgb_f32), rgb_f32)
    np.testing.assert_array_equal(
        ph.glow(rgb_f32, ph.GlowParams(0.5, 8.0, 0.0)), rgb_f32
    )


def test_glow_only_adds_light(rgb_f32):
    out = ph.glow(rgb_f32, ph.GlowParams(0.3, 4.0, 0.5))
    assert (out >= rgb_f32 - 1e-6).all()


def test_glow_puts_light_beside_a_bright_region():
    """The halo is the point: light where there was none."""
    n = 31
    img = np.full((n, n, 1), 0.05, np.float32)
    img[n // 2 - 1 : n // 2 + 2, n // 2 - 1 : n // 2 + 2] = 1.0
    out = ph.glow(img, ph.GlowParams(0.5, 4.0, 0.6))
    near = out[n // 2, n // 2 + 4, 0] - img[n // 2, n // 2 + 4, 0]
    far = out[n // 2, n // 2 + 10, 0] - img[n // 2, n // 2 + 10, 0]
    assert near > 0.005, f"no halo four pixels out: {near}"
    assert near > far, f"halo should fall off: {near} vs {far}"


def test_veiling_glare_depends_on_the_whole_frame():
    """The property a per-pixel tone curve structurally cannot have: the
    black point lifts by an amount set by the rest of the image."""
    n = 33
    params = ph.GlowParams(0.0, 200.0, 0.5)
    def corner_lift(peak):
        img = np.zeros((n, n, 1), np.float32)
        img[:3, :3] = peak
        return float(ph.glow(img, params)[n - 1, n - 1, 0])
    dim, bright = corner_lift(0.2), corner_lift(4.0)
    assert dim > 0.0
    assert bright > dim * 5.0, f"{dim} vs {bright}"


def test_glow_does_not_clamp():
    """Headroom is carried to highlight_rolloff, not spent here."""
    img = np.ones((9, 9, 1), np.float32)
    assert ph.glow(img, ph.GlowParams(0.0, 2.0, 0.5)).max() > 1.0


def test_glow_threshold_above_everything_is_the_identity():
    img = np.full((15, 15, 1), 0.3, np.float32)
    out = ph.glow(img, ph.GlowParams(2.0, 4.0, 0.5))
    assert np.abs(out - img).max() < 1e-6


@pytest.mark.parametrize(
    "threshold,sigma,amount",
    [(-0.1, 4.0, 0.5), (0.0, 4.0, -0.1), (float("nan"), 4.0, 0.5),
     (0.0, -1.0, 0.5), (0.0, 4.0, float("inf"))],
)
def test_glow_rejects_out_of_domain(grey_f32, threshold, sigma, amount):
    with pytest.raises(ValueError):
        ph.glow(grey_f32, ph.GlowParams(threshold, sigma, amount))


def test_glow_params_repr_and_defaults():
    p = ph.GlowParams()
    assert p.amount == 0.0
    assert p == ph.GlowParams(0.0, 8.0, 0.0)
    assert "amount=0.35" in repr(ph.GlowParams(0.8, 8.0, 0.35))


# ── Type stubs ────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INIT_PYI = _REPO_ROOT / "python" / "phaios_core" / "__init__.pyi"


def test_stub_ships_in_the_installed_package():
    """`__init__.pyi`, `gpu.pyi` and `py.typed` must sit next to the
    installed `phaios_core` package, and the installed `__init__.pyi` must
    be byte-identical to the repo copy. A failure here means a stale
    `maturin develop` or a packaging bug ships a wheel with no type
    information despite this file existing in the repo."""
    errors = stub_contract.check_packaging(ph, _REPO_ROOT)
    assert not errors, "\n".join(errors)


def test_stub_matches_the_runtime_module():
    """Every public name, signature, docstring and class shape in
    `python/phaios_core/__init__.pyi` must match the built `phaios_core`
    module exactly. A failure here means the stub drifted from the code
    it documents — the discrepancy list names exactly what and where."""
    errors = stub_contract.check_stub(_INIT_PYI, ph, top_level=True)
    assert not errors, "\n".join(errors)
