import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_pad import compute_pad

# Shared 4-strata grid: dz=1, min_layer=[0,1,2,3]. G=0.5, omega=0.77 (R defaults).
_MIN_LAYER = np.array([0.0, 1.0, 2.0, 3.0])
_NRD = np.array([0.2, 0.4, 0.6, 0.8])
_GF = 1 - _NRD
_COMMON = dict(
    min_layer=_MIN_LAYER,
    cos_theta=1.0,
    dz=1.0,
    G=0.5,
    omega=0.77,
    ground_margin=0.0,
)
# Plain Beer-Lambert formula (compute_pad.py's `PAD = -log(Gf) * cos_theta / (G * omega * dz)`).
_BASIC_FORMULA = -np.log(_GF) * _COMMON["cos_theta"] / (_COMMON["G"] * _COMMON["omega"] * _COMMON["dz"])


def _no_veg_gnd_filter(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


# ── cases that fall back to the plain Beer-Lambert formula ──────────────────────────


@pytest.mark.parametrize(
    "cover_h_pad,use_cover,h_abg,height_cover",
    [
        # use_cover=False -> no cover logic at all.
        (np.nan, False, np.array([5.0]), 2.0),
        # cover_h_pad == 0 -> cover method unusable, falls back.
        (0.0, True, np.array([5.0]), 2.0),
        # An upper-stratum NRD (0.6) exactly equals cover_h_pad -> falls back.
        (0.6, True, np.array([5.0]), 2.0),
        # height_cover >= max(h_abg) -> every stratum is below height_cover so
        # cover_h_pad_v is all 1 and the cover formula reduces to the basic one anyway.
        (0.5, True, np.array([3.5]), 5.0),
    ],
    ids=[
        "use_cover_false",
        "cover_zero_fallback",
        "nrd_equals_cover_fallback",
        "height_cover_above_max",
    ],
)
def test_compute_pad_basic_formula_cases(cover_h_pad, use_cover, h_abg, height_cover):
    result = compute_pad(
        NRD=_NRD,
        Gf=_GF,
        h_abg=h_abg,
        veg_gnd=_no_veg_gnd_filter(1),
        height_cover=height_cover,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        **_COMMON,
    )
    np.testing.assert_allclose(result, _BASIC_FORMULA, rtol=1e-7)


# ── cases with distinct expected PAD values ──────────────────────────────────────────


@pytest.mark.parametrize(
    "cover_h_pad,use_cover,h_abg,veg_gnd,ground_margin,expected",
    [
        # Nominal cover-corrected formula: cover_h_pad_v is 1 below height_cover,
        # cover_h_pad (0.9) at/above it.
        (
            0.9,
            True,
            np.array([5.0]),
            np.array([True]),
            0.0,
            np.array([0.57959364, 1.3268198, 2.56818457, 5.13636914]),
        ),
        # Only the min_layer == 0 stratum is rescaled by dz/(dz-ground_margin) = 1/0.9.
        (
            np.nan,
            False,
            np.array([5.0]),
            np.array([True]),
            0.1,
            _BASIC_FORMULA * np.array([1 / 0.9, 1.0, 1.0, 1.0]),
        ),
        # max(h_abg[veg_gnd]) = 1.5 -> min_empty = ceil(1.5/1)*1 = 2.0 -> strata
        # at/above it are forced to 0, even though NRDc would give a nonzero value.
        (
            np.nan,
            False,
            np.array([0.5, 1.5]),
            np.array([True, True]),
            0.0,
            np.array([0.57959364, 1.3268198, 0.0, 0.0]),
        ),
        # No vegetation/ground point at all -> Z_veg_gnd empty -> R's
        # max(numeric(0)) == -Inf -> every stratum is "empty" -> all PAD = 0,
        # without crashing on an empty-array max.
        (np.nan, False, np.array([0.5, 1.5]), np.array([False, False]), 0.0, np.zeros(4)),
    ],
    ids=[
        "cover_corrected_nominal",
        "ground_margin_rescales_bottom_only",
        "zeroes_upper_strata_above_veg_gnd_max",
        "no_veg_gnd_points_zeroes_every_stratum",
    ],
)
def test_compute_pad_value_cases(cover_h_pad, use_cover, h_abg, veg_gnd, ground_margin, expected):
    params = dict(_COMMON, ground_margin=ground_margin)
    result = compute_pad(
        NRD=_NRD,
        Gf=_GF,
        h_abg=h_abg,
        veg_gnd=veg_gnd,
        height_cover=2.0,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        **params,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-7)
