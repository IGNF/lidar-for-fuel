import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from lidar_for_fuel.fuel_metrics.calculate_fuel_metrics import fuel_metrics_core
from lidar_for_fuel.fuel_metrics.export_fuel_metrics import (
    compute_fuel_metrics_raster,
    export_fuel_metrics,
)
from lidar_for_fuel.pad_profile.export_raster import _write_geotiff

_REF_DIR = Path(__file__).resolve().parents[2] / "data/ref_inrae"
# R (lidarforfuel) PAD raster of a 500 m tile (story 3.1) and its fuel metrics computed by
# R Dev 00c654c (data/ref_inrae/fuel_metrics/make_fuel_metrics_00c654c.R)
_REF_PAD_RASTER_1M = _REF_DIR / "pad_profile/512500_6854997_pad_1m.tif"
_REF_FUEL_METRICS = _REF_DIR / "fuel_metrics/512500_6854997_fuel_metrics_00c654c.tif"

_LOGGER = "lidar_for_fuel.fuel_metrics.export_fuel_metrics"

# Parameters of the R reference tile: INRAE config `ffuelmetrics2` section of
# data/ref_inrae/2026-09-28_fuel_metrics_zones-pilotes.yml (FMA given separately)
_FMA = 0.2543576
_PARAMS = dict(
    smooth_pad=False,
    ladder_fuel_bd_thresholds=[0.01, 0.02, 0.05],
    canopy_height_threshold=1.0,
    canopy_bd_thresh_frac=0.3,
    canopy_top_bd_thresholds=[0.0012, 0.01],
    cbh_threshold_ratio=0.333,
    bd_max=1.0,
    m_perrakis=2.5,
)

# Small synthetic tile: 8 strata of 1 m from 0 m, 10 m pixels, Lambert-93
_NAMES = [f"PAD_1_{h}" for h in range(8)]
_H_BOTTOM = np.arange(8.0)
_PROFILE = np.array([0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1])
_TRANSFORM = Affine(10.0, 0.0, 512489.75, 0.0, -10.0, 6854996.75)
_CRS = "EPSG:2154"

# Crown metrics: NaN for a null profile, as R (AC-3.6.7, decision of 24/11/2026)
_CROWN = {
    "CTH_0p0012",
    "CTH_0p0100",
    "CDH",
    "CBH",
    "CBHlow",
    "CanBDmax",
    "CanBDmin",
    "CanBDlowmax",
    "CanBDlowmin",
    "CBDcan",
    "CBD_cbh",
    "CBD_cbh2",
    "CBD_cbhlow",
}


def _tile(rows=2, cols=3, profile=_PROFILE):
    """PAD tile of shape (bands, rows, cols), every pixel with `profile`."""
    return np.repeat(np.asarray(profile, dtype=np.float32)[:, None, None], rows * cols, axis=1).reshape(-1, rows, cols)


def _grid(value, rows=2, cols=3):
    return np.full((rows, cols), value)


def _compute(pad, fma_can=_FMA, fma_shrub=_FMA, names=_NAMES):
    rows, cols = pad.shape[1:]
    return compute_fuel_metrics_raster(pad, names, _grid(fma_can, rows, cols), _grid(fma_shrub, rows, cols), **_PARAMS)


def _core(profile, fma=_FMA):
    return fuel_metrics_core(np.asarray(profile, dtype=float), _H_BOTTOM, 1.0, fma, fma, **_PARAMS)


def _write_pad(path, pad, names=_NAMES, transform=_TRANSFORM, crs=_CRS):
    _write_geotiff(path, dict(zip(names, pad)), transform, crs)
    return path


def _summary(caplog):
    """The end-of-tile log record(s) of the export module."""
    return [(r.levelno, r.getMessage()) for r in caplog.records if r.name == _LOGGER]


# --- File: parity with R on the reference tile (AC-3.6.1, AC-3.6.2, AC-3.6.3) ----------------


def test_export_fuel_metrics_reference_tile_equals_r(tmp_path):
    # AC-3.6.2: only the PAD_1_* bands of the R PAD raster are kept before the call
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        pad_bands = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_1_")]
        names = [src.descriptions[i - 1] for i in pad_bands]
        pad = src.read(pad_bands)
        transform, crs, shape = src.transform, src.crs, src.shape
    pad_path = _write_pad(tmp_path / "in" / "512500_6854997_pad_profile_1m.tif", pad, names, transform, crs)
    assert len(names) == 60

    out = export_fuel_metrics(pad_path, tmp_path / "out", _FMA, _FMA, **_PARAMS)

    assert out == tmp_path / "out" / "512500_6854997_fuel_metrics.tif"
    with rasterio.open(_REF_FUEL_METRICS) as src:
        ref_names = list(src.descriptions)
        ref = src.read().astype(float)
    with rasterio.open(out) as dst:
        # AC-3.6.1: same grid as the input, float32 bands named as fuel_metrics_core, nodata NaN
        assert dst.transform == transform
        assert dst.crs == crs
        assert dst.shape == shape
        assert dst.count == 45
        assert set(dst.dtypes) == {"float32"}
        assert np.isnan(dst.nodata)
        assert list(dst.descriptions) == ref_names
        result = dst.read().astype(float)

    # AC-3.6.3: equal to R except the no-data pixels (D2), all NaN here; the tile PAD stays
    # below 2.04, so the cap on read (AC-3.2.8) changes nothing
    no_data = np.isnan(pad).all(axis=0)
    assert no_data.sum() == 25
    assert np.isnan(result[:, no_data]).all()
    for band, name in enumerate(ref_names):
        np.testing.assert_allclose(
            result[band, ~no_data], ref[band, ~no_data], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=name
        )


# --- Arrays: values and names -----------------------------------------------------------------


def test_compute_fuel_metrics_raster_applies_fuel_metrics_core_to_each_pixel():
    pad = _tile()
    pad[:, 1, 2] = _PROFILE[::-1]

    metrics = _compute(pad)

    expected = _core(_PROFILE)
    assert list(metrics) == list(expected)
    assert all(band.dtype == np.float32 and band.shape == (2, 3) for band in metrics.values())
    reversed_ = _core(_PROFILE[::-1])
    for name, band in metrics.items():
        np.testing.assert_array_equal(band[0, 0], np.float32(expected[name]), err_msg=name)
        np.testing.assert_array_equal(band[1, 2], np.float32(reversed_[name]), err_msg=name)


def test_compute_fuel_metrics_raster_uses_the_fma_of_each_pixel():
    pad = _tile()
    fma = _grid(_FMA)
    fma[0, 1] = 0.1

    metrics = _compute(pad, fma_can=fma, fma_shrub=fma)

    assert metrics["CLFL"][0, 1] == np.float32(_core(_PROFILE, fma=0.1)["CLFL"])
    assert metrics["CLFL"][0, 0] == np.float32(_core(_PROFILE)["CLFL"])


# --- Arrays: edge cases of the I/O matrix (AC-3.6.4, AC-3.6.7, AC-3.6.8) -----------------------


def test_compute_fuel_metrics_raster_no_data_pixels_are_nan_on_all_bands(caplog):
    pad = _tile()
    pad[:, 0, 0] = np.nan  # all-NaN PAD
    fma_can, fma_shrub = _grid(_FMA), _grid(_FMA)
    fma_can[0, 1] = np.nan  # no canopy FMA
    fma_shrub[1, 0] = np.nan  # no shrub FMA

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        metrics = compute_fuel_metrics_raster(pad, _NAMES, fma_can, fma_shrub, **_PARAMS)

    stack = np.stack(list(metrics.values()))
    for row, col in [(0, 0), (0, 1), (1, 0)]:
        assert np.isnan(stack[:, row, col]).all()
    assert np.isfinite(stack[:, 1, 1]).any()
    assert _summary(caplog) == [
        (
            logging.INFO,
            "Fuel metrics: 6 pixel(s) processed; set to NaN on all bands (D2): 1 with an all-NaN PAD, "
            "2 without FMA; 0 partly NaN PAD profile(s) set to 0 (A6); PAD values brought back into [0, 5] "
            "on read: 0 above 5, 0 +inf, 0 negative",
        )
    ]


def test_compute_fuel_metrics_raster_empty_tile_is_all_nan_with_a_warning(caplog):
    pad = np.full((8, 2, 3), np.nan, dtype=np.float32)

    with caplog.at_level(logging.INFO, logger=_LOGGER), warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = _compute(pad)

    assert list(metrics) == list(_core(_PROFILE))
    assert all(band.shape == (2, 3) and np.isnan(band).all() for band in metrics.values())
    assert _summary(caplog) == [
        (
            logging.INFO,
            "Fuel metrics: 6 pixel(s) processed; set to NaN on all bands (D2): 6 with an all-NaN PAD, "
            "0 without FMA; 0 partly NaN PAD profile(s) set to 0 (A6); PAD values brought back into [0, 5] "
            "on read: 0 above 5, 0 +inf, 0 negative",
        ),
        (logging.WARNING, "Fuel metrics: no pixel with data (all-NaN PAD or no FMA), the raster is all NaN"),
    ]


def test_compute_fuel_metrics_raster_tile_without_fma_is_all_nan_with_a_warning(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        metrics = _compute(_tile(), fma_can=np.nan)

    assert all(np.isnan(band).all() for band in metrics.values())
    assert _summary(caplog) == [
        (
            logging.INFO,
            "Fuel metrics: 6 pixel(s) processed; set to NaN on all bands (D2): 0 with an all-NaN PAD, "
            "6 without FMA; 0 partly NaN PAD profile(s) set to 0 (A6); PAD values brought back into [0, 5] "
            "on read: 0 above 5, 0 +inf, 0 negative",
        ),
        (logging.WARNING, "Fuel metrics: no pixel with data (all-NaN PAD or no FMA), the raster is all NaN"),
    ]


def test_compute_fuel_metrics_raster_log_counts_do_not_overlap(caplog):
    pad = _tile()
    pad[:, 0, 0] = np.nan  # all-NaN PAD and no FMA: counted once, as all-NaN
    pad[3, 0, 1] = np.nan  # partly NaN and no FMA: counted as without FMA, not as partly NaN
    pad[3, 1, 1] = np.nan  # partly NaN with FMA: counted as partly NaN
    fma = _grid(_FMA)
    fma[0, 0] = fma[0, 1] = np.nan

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        compute_fuel_metrics_raster(pad, _NAMES, fma, fma, **_PARAMS)

    assert _summary(caplog) == [
        (
            logging.INFO,
            "Fuel metrics: 6 pixel(s) processed; set to NaN on all bands (D2): 1 with an all-NaN PAD, "
            "1 without FMA; 1 partly NaN PAD profile(s) set to 0 (A6); PAD values brought back into [0, 5] "
            "on read: 0 above 5, 0 +inf, 0 negative",
        )
    ]


@pytest.mark.parametrize("shape", [(3,), (1, 3), (3, 2)], ids=["1-D", "one row", "transposed"])
def test_compute_fuel_metrics_raster_fma_grid_of_another_shape_raises(shape):
    with pytest.raises(ValueError, match=r"fma_shrub grid of shape \(.*\) for a PAD grid of shape \(2, 3\)"):
        compute_fuel_metrics_raster(_tile(), _NAMES, _grid(_FMA), np.full(shape, _FMA), **_PARAMS)


@pytest.mark.parametrize(
    "profile",
    [np.zeros(8), [0.4, 0, 0, 0, 0, 0, 0, 0]],
    ids=["null profile", "surface fuel only"],
)
def test_compute_fuel_metrics_raster_bare_ground_follows_r(profile):
    # AC-3.6.7 (decision of 24/11/2026): no exception nor RuntimeWarning; loads, ladder and
    # ProfType finite; the crown metrics stay NaN, as R
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = _compute(_tile(profile=profile))

    nan_names = {name for name, band in metrics.items() if np.isnan(band).all()}
    assert nan_names <= _CROWN
    assert all(np.isfinite(band).all() for name, band in metrics.items() if name not in _CROWN)
    for name in ["CLFL", "FL_k1", "FL_k2", "LFL", "CFL", "FSG", "ProfType"]:
        assert np.isfinite(metrics[name]).all(), name
    if not np.any(profile):
        assert nan_names == _CROWN


def test_compute_fuel_metrics_raster_caps_the_pad_and_counts_each_kind(caplog):
    pad = _tile()
    pad[6, 0, 0] = 7.0  # above 5
    pad[6, 0, 1] = np.inf
    pad[1, 0, 2] = -1.0  # negative
    pad[2, 1, 0] = -np.inf  # negative too
    pad[3, 1, 1] = np.nan  # partly NaN profile (A6)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        metrics = _compute(pad)

    capped = _PROFILE.copy()
    capped[6] = 5.0
    assert metrics["CLFL"][0, 0] == np.float32(_core(capped)["CLFL"])
    assert metrics["CLFL"][0, 1] == np.float32(_core(capped)["CLFL"])
    lowered = _PROFILE.copy()
    lowered[1] = 0.0
    assert metrics["FL_k2"][0, 2] == np.float32(_core(lowered)["FL_k2"]) == 0.0
    assert _summary(caplog) == [
        (
            logging.INFO,
            "Fuel metrics: 6 pixel(s) processed; set to NaN on all bands (D2): 0 with an all-NaN PAD, "
            "0 without FMA; 1 partly NaN PAD profile(s) set to 0 (A6); PAD values brought back into [0, 5] "
            "on read: 1 above 5, 1 +inf, 2 negative",
        )
    ]


# --- File: FMA map, invalid bands (AC-3.6.4, AC-3.6.9) ----------------------------------------


def test_export_fuel_metrics_fma_map_nodata_pixel_is_nan(tmp_path, caplog):
    pad_path = _write_pad(tmp_path / "tile_pad_profile_1m.tif", _tile())
    fma_path = tmp_path / "fma.tif"
    fma = np.full((2, 3), _FMA, dtype=np.float32)
    fma[1, 2] = -9999.0
    profile = dict(driver="GTiff", height=2, width=3, count=1, dtype="float32", crs=_CRS, transform=_TRANSFORM)
    with rasterio.open(fma_path, "w", nodata=-9999.0, **profile) as dst:
        dst.write(fma, 1)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        out = export_fuel_metrics(pad_path, tmp_path / "out", fma_path, fma_path, **_PARAMS)

    assert out.name == "tile_fuel_metrics.tif"
    with rasterio.open(out) as dst:
        result = dst.read()
        assert dst.crs == rasterio.crs.CRS.from_string(_CRS)
        assert dst.transform == _TRANSFORM
    assert np.isnan(result[:, 1, 2]).all()
    # the map and the written PAD are float32
    expected = _core(_PROFILE.astype(np.float32), fma=float(np.float32(_FMA)))["CLFL"]
    assert result[0, 0, 0] == np.float32(expected)
    assert "1 without FMA" in _summary(caplog)[0][1]


@pytest.mark.parametrize(
    "names",
    [["PAD_1_0", "Cover_2"], [""]],
    ids=["band not PAD", "no PAD band"],
)
def test_export_fuel_metrics_invalid_bands_raise_with_the_file_name(tmp_path, names):
    pad = np.zeros((len(names), 2, 3), dtype=np.float32)
    path = tmp_path / "bad_pad_profile_1m.tif"
    _write_geotiff(path, {f"b{i}": band for i, band in enumerate(pad)}, _TRANSFORM, _CRS)
    with rasterio.open(path, "r+") as dst:
        for index, name in enumerate(names, start=1):
            dst.set_band_description(index, name)

    with pytest.raises(ValueError, match="bad_pad_profile_1m.tif"):
        export_fuel_metrics(path, tmp_path / "out", _FMA, _FMA, **_PARAMS)
    assert not (tmp_path / "out").exists()


def test_export_fuel_metrics_r_pad_raster_is_not_an_input(tmp_path):
    # AC-3.6.2: the 186-band R PAD raster is not a production input
    with pytest.raises(ValueError, match="512500_6854997_pad_1m.tif"):
        export_fuel_metrics(_REF_PAD_RASTER_1M, tmp_path, _FMA, _FMA, **_PARAMS)
