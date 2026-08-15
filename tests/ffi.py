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


@pytest.mark.parametrize("label", [v[0] for v in _layout_variants(np.zeros((4, 4, 1), np.float32))])
def test_all_kernels_accept_any_layout(label, rgb_f32, grey_f32):
    """No kernel may panic on a non-C-contiguous input, whatever its layout."""
    rgb = dict(_layout_variants(rgb_f32))[label]
    grey = dict(_layout_variants(grey_f32))[label]

    for out in (
        ph.luminance_bw(rgb),
        ph.channel_mixer_bw(rgb, 0.3, 0.59, 0.11),
        ph.color_filter_bw(rgb, ph.ColorFilter.Yellow8K2),
    ):
        assert out.shape == rgb.shape[:2] + (1,)
        assert out.flags["C_CONTIGUOUS"], "output must be C-contiguous"

    for out in (
        ph.zone_system(grey, ph.ZoneParams({5: 0.5})),
        ph.local_contrast(grey, ph.GuidedFilterParams(4, 0.01), 0.5),
        ph.encode_srgb(grey),
    ):
        assert out.shape == grey.shape
        assert out.flags["C_CONTIGUOUS"], "output must be C-contiguous"


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
