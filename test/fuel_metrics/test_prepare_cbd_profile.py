import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    compute_cbd,
    parse_pad_strata,
    prepare_pad_profile,
    select_default_ladder_threshold,
)

# PAD rasters produced by R (lidarforfuel) for story 3.1: PAD_* bands mixed with others
_REF_PAD_DIR = Path(__file__).resolve().parents[2] / "data/ref_inrae/pad_profile"
_REF_PAD_RASTER_1M = _REF_PAD_DIR / "512500_6854997_pad_1m.tif"  # 186 bands, PAD_1_0 ... PAD_1_59
_REF_PAD_RASTER_05M = _REF_PAD_DIR / "512500_6854997_pad_05m.tif"  # 18 bands, PAD_0.5_0 ... PAD_0.5_1.5


# --- parse_pad_strata (AC-3.2.1, AC-3.2.2) ---------------------------------------------------


def test_parse_pad_strata_keeps_only_pad_bands_sorted_numerically():
    # Numeric sort: PAD_1_10 must come after PAD_1_2 (alphabetical order would put it before)
    names = [f"PAD_1_{k}" for k in (10, 2, 0, 1, 3, 4, 5, 6, 7, 8, 9)]
    names = ["date", "Cover_2", None, *names[:3], "N_1_3", "Ni_1_0", *names[3:]]

    indices, dz, h_bottom = parse_pad_strata(names)

    assert dz == 1.0
    np.testing.assert_array_equal(h_bottom, np.arange(11, dtype=float))
    # Indices point back to the original band positions, in h_bottom order
    assert [names[i] for i in indices] == [f"PAD_1_{k}" for k in range(11)]


def test_parse_pad_strata_decimal_names():
    indices, dz, h_bottom = parse_pad_strata(["PAD_0.5_1.5", "PAD_0.5_0", "PAD_0.5_1", "PAD_0.5_0.5"])

    assert dz == 0.5
    np.testing.assert_array_equal(h_bottom, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(indices, [1, 3, 2, 0])


@pytest.mark.parametrize(
    "raster, dz, expected_names",
    [
        (_REF_PAD_RASTER_1M, 1.0, [f"PAD_1_{k}" for k in range(60)]),
        (_REF_PAD_RASTER_05M, 0.5, ["PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"]),
    ],
)
def test_parse_pad_strata_real_r_raster(raster, dz, expected_names):
    with rasterio.open(raster) as src:
        names = list(src.descriptions)

    indices, parsed_dz, h_bottom = parse_pad_strata(names)

    assert parsed_dz == dz
    np.testing.assert_allclose(h_bottom, dz * np.arange(len(expected_names)))
    assert [names[i] for i in indices] == expected_names


def test_parse_pad_strata_mixed_dz_raises():
    with pytest.raises(ValueError, match="dz"):
        parse_pad_strata(["PAD_1_0", "PAD_0.5_1"])


@pytest.mark.parametrize(
    "names",
    [
        ["PAD_1_0", "PAD_1_1", "PAD_1_1"],  # duplicate stratum
        ["PAD_1_0", "PAD_1_2"],  # gap
        ["PAD_0_0", "PAD_0_1"],  # null thickness
    ],
)
def test_parse_pad_strata_irregular_profile_raises(names):
    with pytest.raises(ValueError):
        parse_pad_strata(names)


def test_parse_pad_strata_no_pad_band_raises():
    with pytest.raises(ValueError, match="PAD_"):
        parse_pad_strata(["Cover_2", "N_1_3", None])


# --- prepare_pad_profile (AC-3.2.3, AC-3.2.4) ------------------------------------------------


def test_prepare_pad_profile_replaces_invalid_values_by_zero(caplog):
    pad = np.array([0.5, np.nan, np.inf, -np.inf, -1.0, -0.2, 2.0])

    with caplog.at_level(logging.DEBUG, logger="lidar_for_fuel.fuel_metrics.prepare_cbd_profile"):
        cleaned, n_corrected = prepare_pad_profile(pad, smooth_pad=False)

    np.testing.assert_array_equal(cleaned, [0.5, 0, 0, 0, 0, 0, 2.0])
    assert n_corrected == 5
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.DEBUG, "PAD profile: 5 non-finite or negative value(s) set to 0")
    ]
    # Input is not modified in place
    assert np.isnan(pad[1])


def test_prepare_pad_profile_valid_profile_unchanged():
    pad = np.array([0.0, 1.2, 4.9])

    cleaned, n_corrected = prepare_pad_profile(pad, smooth_pad=False)

    np.testing.assert_array_equal(cleaned, pad)
    assert n_corrected == 0


def test_prepare_pad_profile_smoothing_partial_window_at_edges():
    cleaned, _ = prepare_pad_profile(np.array([0.0, 3.0, 0.0, 0.0]), smooth_pad=True)

    np.testing.assert_allclose(cleaned, [1.5, 1.0, 1.0, 0.0])


def test_prepare_pad_profile_smoothing_after_cleaning():
    # NaN is set to 0 before smoothing (R order), so it does not propagate
    cleaned, n_corrected = prepare_pad_profile(np.array([np.nan, 3.0, 0.0]), smooth_pad=True)

    np.testing.assert_allclose(cleaned, [1.5, 1.0, 1.5])
    assert n_corrected == 1


def test_prepare_pad_profile_smoothing_bit_identical_to_r():
    # Expected values computed by R: slider::slide_dbl(x, mean, .before = 1, .after = 1).
    # A plain float64 mean differs by 1 ulp on the 2nd and 3rd values (R mean() uses long double).
    pad = np.array(
        [3.1848084366072715, 1.3489335688193516, 0.20486761968097345, 0.08263817764264547, 4.066351196001362]
    )
    expected_r = [2.2668710027133114, 1.5795365417025322, 0.54547978871432345, 1.4512856644416603, 2.0744946868220038]

    cleaned, _ = prepare_pad_profile(pad, smooth_pad=True)

    np.testing.assert_array_equal(cleaned, expected_r)


def test_prepare_pad_profile_smoothing_single_stratum():
    cleaned, _ = prepare_pad_profile(np.array([2.0]), smooth_pad=True)

    np.testing.assert_allclose(cleaned, [2.0])


# --- compute_cbd (AC-3.2.5) -------------------------------------------------------------------


def test_compute_cbd_uses_fma_by_height_limit_and_caps_at_bd_max():
    pad = np.array([1.0, 1.0, 1.0, 5.0])
    h_center = np.array([0.5, 1.5, 2.5, 3.5])

    cbd = compute_cbd(pad, h_center, height_limit=1.0, fma_can=0.3, fma_shrub=0.1, bd_max=1.0)

    np.testing.assert_allclose(cbd, [0.1, 0.3, 0.3, 1.0])


def test_compute_cbd_height_limit_is_strict():
    # A stratum centred exactly on the limit uses fma_shrub (R: zval > limit)
    cbd = compute_cbd(
        np.array([1.0, 1.0]), np.array([1.5, 2.5]), height_limit=1.5, fma_can=0.3, fma_shrub=0.1, bd_max=1.0
    )

    np.testing.assert_allclose(cbd, [0.1, 0.3])


def test_compute_cbd_height_limit_as_cbh2():
    # Same function is reused at step 4 with CBH2 as the limit
    pad = np.ones(6)
    h_center = np.arange(6) + 0.5

    cbd = compute_cbd(pad, h_center, height_limit=3.0, fma_can=0.25, fma_shrub=0.2, bd_max=1.0)

    np.testing.assert_allclose(cbd, [0.2, 0.2, 0.2, 0.25, 0.25, 0.25])


# --- select_default_ladder_threshold (AC-3.2.6) ----------------------------------------------


@pytest.mark.parametrize(
    "thresholds, expected",
    [
        ([0.01, 0.02, 0.05], 0.02),
        # Not an exact tie in double precision: |0.05 - 0.035| < |0.02 - 0.035|, so R
        # (`which.min`, checked with Rscript) returns 0.05 whatever the order
        ([0.01, 0.02, 0.05, 0.1], 0.05),
        ([0.05, 0.02, 0.01, 0.1], 0.05),
        # Exact tie (binary-exact values): first one in the configured order, as R
        ([0.25, 0.5, 0.75, 1.0], 0.5),
        ([0.75, 0.5, 0.25, 1.0], 0.75),
        ([0.02], 0.02),
    ],
)
def test_select_default_ladder_threshold(thresholds, expected):
    assert select_default_ladder_threshold(thresholds) == expected


def test_select_default_ladder_threshold_empty_raises():
    with pytest.raises(ValueError):
        select_default_ladder_threshold([])
