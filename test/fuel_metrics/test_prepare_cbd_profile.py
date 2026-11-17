import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    compute_cbd,
    prepare_pad_profile,
    read_fma,
    read_pad_heights,
    select_default_ladder_threshold,
)

# PAD raster produced by R (lidarforfuel) for story 3.1. Unlike the Python `pad_profile_1m`
# raster (PAD_1_* bands only), it mixes the PAD_1_* bands with Cover_*, N_*, Ni_*... (186 bands)
_REF_PAD_RASTER_1M = Path(__file__).resolve().parents[2] / "data/ref_inrae/pad_profile/512500_6854997_pad_1m.tif"


# --- read_pad_heights (AC-3.2.1, AC-3.2.2) ---------------------------------------------------


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


# --- read_fma (AC-3.2.7) -----------------------------------------------------------------------

# PAD grid of the R reference tile 512500_6854997 (10 m, origin on the rounded CosiaFrance grid)
_PAD_TRANSFORM = from_origin(512489.75, 6854996.75, 10.0, 10.0)
_LAMBERT_93 = CRS.from_epsg(2154)


def _write_fma_map(path, values, transform, nodata=None, crs=_LAMBERT_93):
    values = np.asarray(values, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)
    return path


def test_read_fma_scalar_fills_the_pad_grid():
    fma = read_fma(0.2543576, _PAD_TRANSFORM, (2, 3))

    np.testing.assert_array_equal(fma, np.full((2, 3), 0.2543576))


def test_read_fma_map_on_the_pad_grid(tmp_path):
    values = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.05]]
    path = _write_fma_map(tmp_path / "fma.tif", values, _PAD_TRANSFORM)

    fma = read_fma(path, _PAD_TRANSFORM, (2, 3), _LAMBERT_93)

    np.testing.assert_array_equal(fma, np.asarray(values, dtype="float32"))


def test_read_fma_map_window_of_a_larger_map_shifted_by_6_cm(tmp_path):
    # National map on the real CosiaFrance origin (6.1 cm / 5.3 cm from the PAD grid),
    # starting 2 columns left and 1 row above the tile: the nearest FMA pixel is read
    values = np.arange(1, 21, dtype="float32").reshape(4, 5) / 40  # 0.025 .. 0.5
    values[values < 0.05] = 0.05
    fma_transform = from_origin(512489.75 - 20 - 0.0615, 6854996.75 + 10 + 0.0528, 10.0, 10.0)
    path = _write_fma_map(tmp_path / "fma.tif", values, fma_transform)

    fma = read_fma(path, _PAD_TRANSFORM, (2, 3), _LAMBERT_93)

    np.testing.assert_array_equal(fma, values[1:3, 2:5])


def test_read_fma_map_nodata_and_outside_the_map_are_nan(tmp_path):
    # Map covering only the first column of the tile, with a NoData pixel
    path = _write_fma_map(tmp_path / "fma.tif", [[0.2], [-128.0]], _PAD_TRANSFORM, nodata=-128)

    fma = read_fma(path, _PAD_TRANSFORM, (2, 2), _LAMBERT_93)

    np.testing.assert_array_equal(fma, [[np.float32(0.2), np.nan], [np.nan, np.nan]])


@pytest.mark.parametrize("bad_value", [0.04, 0.6])
def test_read_fma_map_value_out_of_range_raises(tmp_path, bad_value):
    path = _write_fma_map(tmp_path / "fma.tif", [[0.2, bad_value]], _PAD_TRANSFORM)

    with pytest.raises(ValueError, match=r"outside \[0.05, 0.5\]"):
        read_fma(path, _PAD_TRANSFORM, (1, 2), _LAMBERT_93)


@pytest.mark.parametrize(
    "fma_transform, crs, match",
    [
        (from_origin(512489.75, 6854996.75, 5.0, 5.0), _LAMBERT_93, "resolution"),
        (from_origin(512489.75 + 3.0, 6854996.75, 10.0, 10.0), _LAMBERT_93, "aligned"),
        (_PAD_TRANSFORM, CRS.from_epsg(4326), "CRS"),
    ],
)
def test_read_fma_map_incompatible_grid_raises(tmp_path, fma_transform, crs, match):
    path = _write_fma_map(tmp_path / "fma.tif", np.full((4, 4), 0.2), fma_transform, crs=crs)

    with pytest.raises(ValueError, match=match):
        read_fma(path, _PAD_TRANSFORM, (2, 2), _LAMBERT_93)


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
