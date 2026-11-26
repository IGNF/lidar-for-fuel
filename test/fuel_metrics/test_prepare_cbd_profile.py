import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    cap_pad,
    compute_cbd,
    prepare_pad_profile,
    read_pad_heights,
    select_default_ladder_threshold,
)

# PAD raster produced by R (lidarforfuel) for story 3.1. Unlike the Python `pad_profile_1m`
# raster (PAD_1_* bands only), it mixes the PAD_1_* bands with Cover_*, N_*, Ni_*... (186 bands)
_REF_PAD_RASTER_1M = Path(__file__).resolve().parents[2] / "data/ref_inrae/pad_profile/512500_6854997_pad_1m.tif"


# --- read_pad_heights: strata heights read from the PAD band names ---------------------------


def test_read_pad_heights_pad_profile_1m():
    dz, h_bottom = read_pad_heights([f"PAD_1_{k}" for k in range(60)])

    assert dz == 1.0
    np.testing.assert_array_equal(h_bottom, np.arange(60, dtype=float))


def test_read_pad_heights_decimal_names():
    dz, h_bottom = read_pad_heights(["PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"])

    assert dz == 0.5
    np.testing.assert_array_equal(h_bottom, [0.0, 0.5, 1.0, 1.5])


def test_read_pad_heights_pad_bands_of_r_raster():
    # Parity tests feed the R PAD raster: they keep only its PAD_* bands, in band order
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        pad_names = [name for name in src.descriptions if name.startswith("PAD_")]

    dz, h_bottom = read_pad_heights(pad_names)

    assert dz == 1.0
    np.testing.assert_array_equal(h_bottom, np.arange(60, dtype=float))


@pytest.mark.parametrize(
    "names, match",
    [
        ([], "no band"),
        (["PAD_1_0", "Cover_2"], "Cover_2"),  # not a PAD band
        (["PAD_1_0", None], "None"),  # unnamed band
        (["PAD_1_0", "PAD_0.5_1"], "dz"),  # mixed thicknesses
        (["PAD_0_0", "PAD_0_1"], "null"),  # null thickness
        (["PAD_1_0", "PAD_1_1", "PAD_1_1"], "contiguous"),  # duplicate stratum
        (["PAD_1_0", "PAD_1_2"], "contiguous"),  # gap
        (["PAD_1_1", "PAD_1_0"], "contiguous"),  # not ascending
    ],
)
def test_read_pad_heights_invalid_names_raise(names, match):
    with pytest.raises(ValueError, match=match):
        read_pad_heights(names)


# --- cap_pad: PAD bands brought back into [0, 5] when the pad_profile_1m raster is read --------


def test_cap_pad_brings_each_band_back_into_0_5(caplog):
    # Raster (bands, rows, cols), as read from pad_profile_1m; same cap as step 2 before the
    # INRAE answer on the PAD cap at 5
    pad = np.array(
        [
            [[7.01, np.nan], [-0.5, 2.0]],
            [[np.inf, -np.inf], [5.0, 0.0]],
            [[np.nan, np.nan], [4.99, 0.1]],
        ],
        dtype=np.float32,
    )

    with caplog.at_level(logging.INFO, logger="lidar_for_fuel.fuel_metrics.prepare_cbd_profile"):
        capped, n_capped = cap_pad(pad, ["PAD_1_0", "PAD_1_1", "PAD_1_2"])

    expected = np.array(
        [[[5.0, np.nan], [0.0, 2.0]], [[5.0, 0.0], [5.0, 0.0]], [[np.nan, np.nan], [4.99, 0.1]]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(capped, expected)  # NaN stays NaN (pixel without data, D2)
    assert capped.dtype == np.float32
    assert n_capped == 4
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "PAD raster: 4 value(s) outside [0, 5] brought back into it (PAD_1_0: 2, PAD_1_1: 2)")
    ]
    # Input is not modified in place
    assert pad[0, 0, 0] == np.float32(7.01)


def test_cap_pad_valid_raster_unchanged_and_not_logged(caplog):
    pad = np.array([[[0.0, np.nan]], [[5.0, 1.2]]], dtype=np.float32)

    with caplog.at_level(logging.INFO, logger="lidar_for_fuel.fuel_metrics.prepare_cbd_profile"):
        capped, n_capped = cap_pad(pad, ["PAD_1_0", "PAD_1_1"])

    np.testing.assert_array_equal(capped, pad)
    assert n_capped == 0
    assert caplog.records == []


def test_cap_pad_custom_pad_max():
    capped, n_capped = cap_pad(np.array([[[3.5, 2.0]]]), ["PAD_1_0"], pad_max=3.0)

    np.testing.assert_array_equal(capped, [[[3.0, 2.0]]])
    assert n_capped == 1

    # A NumPy scalar bound keeps the float32 dtype of the raster
    capped, _ = cap_pad(np.array([[[3.5, 2.0]]], dtype=np.float32), ["PAD_1_0"], pad_max=np.float64(3.0))
    assert capped.dtype == np.float32


def test_cap_pad_band_names_do_not_match_the_bands_raises():
    with pytest.raises(ValueError, match="2 band names for 1 PAD band"):
        cap_pad(np.zeros((1, 2, 2)), ["PAD_1_0", "PAD_1_1"])


def test_cap_pad_leaves_the_r_reference_tile_unchanged():
    # The PAD of the reference tile stays within [0, 5] (max 2.04; the pilot zone max of 7.01 is
    # on other tiles): the cap does not change the parity tests run on this tile
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        indexes = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_1_")]
        pad = src.read(indexes)
        names = [src.descriptions[i - 1] for i in indexes]

    capped, n_capped = cap_pad(pad, names)

    assert n_capped == 0
    np.testing.assert_array_equal(capped, pad)


# --- prepare_pad_profile: invalid values set to 0, optional 3-strata smoothing ---------------


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


# --- compute_cbd: PAD × FMA (canopy or shrub by height), capped at bd_max ---------------------


def test_compute_cbd_uses_fma_height_limit_and_caps_at_bd_max():
    pad = np.array([1.0, 1.0, 1.0, 5.0])
    h_center = np.array([0.5, 1.5, 2.5, 3.5])

    cbd = compute_cbd(pad, h_center, fma_height_limit=1.0, fma_can=0.3, fma_shrub=0.1, bd_max=1.0)

    np.testing.assert_allclose(cbd, [0.1, 0.3, 0.3, 1.0])


def test_compute_cbd_fma_height_limit_is_strict():
    # A stratum centred exactly on the limit uses fma_shrub (R: zval > limit)
    cbd = compute_cbd(
        np.array([1.0, 1.0]), np.array([1.5, 2.5]), fma_height_limit=1.5, fma_can=0.3, fma_shrub=0.1, bd_max=1.0
    )

    np.testing.assert_allclose(cbd, [0.1, 0.3])


def test_compute_cbd_fma_height_limit_as_cbh2():
    # Same function is reused at step 4 with CBH2 as the limit
    pad = np.ones(6)
    h_center = np.arange(6) + 0.5

    cbd = compute_cbd(pad, h_center, fma_height_limit=3.0, fma_can=0.25, fma_shrub=0.2, bd_max=1.0)

    np.testing.assert_allclose(cbd, [0.2, 0.2, 0.2, 0.25, 0.25, 0.25])


# --- select_default_ladder_threshold: threshold closest to the median ------------------------


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
        # Repeated thresholds count in the median (raw list, as R): median 0.015, not 0.02
        ([0.01, 0.01, 0.02, 0.05], 0.01),
        ([0.02], 0.02),
    ],
)
def test_select_default_ladder_threshold(thresholds, expected):
    assert select_default_ladder_threshold(thresholds) == expected


def test_select_default_ladder_threshold_empty_raises():
    with pytest.raises(ValueError):
        select_default_ladder_threshold([])
