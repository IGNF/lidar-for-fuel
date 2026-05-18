<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
from pathlib import Path

import numpy as np
=======
import json
from pathlib import Path

import numpy as np
import pdal
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
from pathlib import Path

import numpy as np
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
import pytest
import rasterio
from rasterio.transform import from_bounds

<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
from lidar_for_fuel.preprocessing.normalize_height_by_dtm import (
    add_h_abg,
    filter_z_by_height,
)

_NODATA_VALUE = -9999.0
_GROUND_Z = 100.0
_DTM_NODATA = -9999.0
<<<<<<< HEAD
=======
from lidar_for_fuel.pretreatment.normalize_height_by_points import add_Zref

_NODATA_VALUE = -9999.0
_GROUND_Z = 100.0  # flat DTM elevation (metres)
_DTM_NODATA = -9999.0  # nodata sentinel used inside the DTM raster

# DTM covers X=[0, 10], Y=[0, 10], 10×10 pixels at 1 m resolution.
# Pixel centre (row=i, col=j): x = 0.5 + j, y = 9.5 - i
_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH = 0.0, 0.0, 10.0, 10.0
_DTM_WIDTH = _DTM_HEIGHT = 10
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

_LAS_DTYPE = np.dtype(
    [
        ("X", np.float64),
        ("Y", np.float64),
        ("Z", np.float64),
        ("Intensity", np.uint16),
        ("ReturnNumber", np.uint8),
        ("NumberOfReturns", np.uint8),
        ("Classification", np.uint8),
    ]
)


<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
def _make_dtm(path: Path, data: np.ndarray, bounds: tuple) -> str:
    """Write a single-band float32 GeoTIFF from a 2-D array."""
    west, south, east, north = bounds
    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
<<<<<<< HEAD
=======
def _make_dtm(path: Path, data: np.ndarray) -> str:
    """Write a single-band float32 GeoTIFF from a 2-D array."""
    transform = from_bounds(_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH, _DTM_WIDTH, _DTM_HEIGHT)
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
        height=height,
        width=width,
=======
        height=_DTM_HEIGHT,
        width=_DTM_WIDTH,
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
        height=height,
        width=width,
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
        count=1,
        dtype="float32",
        transform=transform,
        nodata=_DTM_NODATA,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return str(path)


<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
def _make_points(rows: list) -> np.ndarray:
    """Build a structured point array from a list of (x, y, z) tuples."""
=======
def _make_flat_dtm(path: Path) -> str:
    """Flat DTM: all pixels at _GROUND_Z."""
    return _make_dtm(path, np.full((_DTM_HEIGHT, _DTM_WIDTH), _GROUND_Z))


def _make_pipeline(rows: list) -> pdal.Pipeline:
    """Build an unexecuted pipeline from a list of (x, y, z) tuples."""
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
def _make_points(rows: list) -> np.ndarray:
    """Build a structured point array from a list of (x, y, z) tuples."""
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
    pts = np.zeros(len(rows), dtype=_LAS_DTYPE)
    for i, (x, y, z) in enumerate(rows):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
    return pts


def _pixel_centre(col, row, pixel_size, west, north):
    """Return the geographic (x, y) centre of a pixel given its col/row indices."""
    x = west + (col + 0.5) * pixel_size
    y = north - (row + 0.5) * pixel_size
    return x, y


# ── add_h_abg ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_pixels,pixel_size,west,south",
    [
        (10, 1.0, 0.0, 0.0),  # 1 m, origin at (0, 0)
        (20, 0.5, 0.0, 0.0),  # 0.5 m, origin at (0, 0)
        (10, 1.0, 700000.0, 6400000.0),  # 1 m, realistic Lambert-93 origin
        (20, 0.5, 700000.0, 6400000.0),  # 0.5 m, realistic Lambert-93 origin
    ],
    ids=["1m_origin0", "0.5m_origin0", "1m_offset", "0.5m_offset"],
)
def test_add_zref_flat_dtm(tmp_path, n_pixels, pixel_size, west, south):
    """On a flat DTM, h_abg = Z - Z_sol for a point at a known pixel centre.

    Parametrised over two resolutions and two origins (including a realistic
    Lambert-93 offset) to verify that the rasterio transform is correctly
    applied and not just relative pixel indices.
    """
    north = south + n_pixels * pixel_size
    east = west + n_pixels * pixel_size
    bounds = (west, south, east, north)

    data = np.full((n_pixels, n_pixels), _GROUND_Z)
    dtm = _make_dtm(tmp_path / f"dtm_flat_{n_pixels}_{int(west)}.tif", data, bounds)

    x, y = _pixel_centre(3, 2, pixel_size, west, north)
    result = add_h_abg(_make_points([(x, y, _GROUND_Z + 7.0)]), dtm, nodata_value=_NODATA_VALUE)

    assert isinstance(result, np.ndarray)
    assert "h_abg" in result.dtype.names
    np.testing.assert_allclose(result["h_abg"][0], 7.0, atol=1e-3)


@pytest.mark.parametrize("n_pixels,pixel_size", [(10, 1.0), (20, 0.5)], ids=["1m", "0.5m"])
def test_add_zref_non_flat_dtm(tmp_path, n_pixels, pixel_size):
    """On a non-flat DTM, checks bilinear interpolation, nodata handling, and out-of-extent points.

    DTM: column col_slope at 102 m (rest at 100 m), one nodata pixel at (row=2, col=2).

    Points tested:
    - p1: midpoint between col_slope-1 (100 m) and col_slope (102 m) -> h_abg = 111 - 101 = 10.0
    - p2: centre of the nodata pixel -> h_abg = nodata_value
    - p3: outside DTM extent -> h_abg = nodata_value
    """
    west, south, north = 0.0, 0.0, n_pixels * pixel_size
    bounds = (west, south, n_pixels * pixel_size, north)
    col_slope = 3 * n_pixels // 4  # col 7 (1m) or col 15 (0.5m), far from the nodata pixel
    row_interp = n_pixels // 2  # row 5 (1m) or row 10 (0.5m)

    data = np.full((n_pixels, n_pixels), _GROUND_Z)
    data[:, col_slope] = 102.0
    data[2, 2] = _DTM_NODATA

    dtm = _make_dtm(tmp_path / f"dtm_slope_{n_pixels}.tif", data, bounds)

    # x_mid: midpoint between the centres of col_slope-1 and col_slope
    x_mid = west + col_slope * pixel_size
    _, y_interp = _pixel_centre(0, row_interp, pixel_size, west, north)
    x_nodata, y_nodata = _pixel_centre(2, 2, pixel_size, west, north)

    result = add_h_abg(
        _make_points([(x_mid, y_interp, 111.0), (x_nodata, y_nodata, 110.0), (999.0, 999.0, 110.0)]),
        dtm,
        nodata_value=_NODATA_VALUE,
    )

    np.testing.assert_allclose(result["h_abg"][0], 10.0, atol=1e-3)  # bilinear interpolation
    assert result["h_abg"][1] == _NODATA_VALUE  # nodata
    assert result["h_abg"][2] == _NODATA_VALUE  # outside extent


# ── filter_z_by_height ─────────────────────────────────────────────────────────


def _make_points_with_h_abg(h_abg_values: list) -> np.ndarray:
    """Build a structured array with a h_abg field directly."""
    dtype = np.dtype(_LAS_DTYPE.descr + [("h_abg", np.float64)])
    pts = np.zeros(len(h_abg_values), dtype=dtype)
    for i, z in enumerate(h_abg_values):
        pts[i]["h_abg"] = z
    return pts


@pytest.mark.parametrize(
    "h_abg_values,min_h,max_h,expected",
    [
        ([5.0, 85.0, 50.0], -3, 80, [5.0, 50.0]),  # default bounds: high point removed
        ([-5.0, 5.0], -3, 80, [5.0]),  # default bounds: low point removed
        ([5.0, 15.0, -1.0], -0.5, 10, [5.0]),  # custom bounds: high and low removed
    ],
    ids=["removes_high", "removes_low", "custom_bounds"],
)
def test_filter_z_by_height(h_abg_values, min_h, max_h, expected):
    """Points outside [min_height_filter, height_filter] are removed."""
    result = filter_z_by_height(_make_points_with_h_abg(h_abg_values), min_height_filter=min_h, height_filter=max_h)
    assert len(result) == len(expected)
    np.testing.assert_allclose(sorted(result["h_abg"]), sorted(expected), atol=1e-3)
<<<<<<< HEAD
=======
    return pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[pts])


# Tests


def test_returns_pdal_pipeline(tmp_path):
    """add_Zref returns a pdal.Pipeline."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    pipeline = _make_pipeline([(5.5, 4.5, 110.0)])
    result = add_Zref(pipeline, dtm)
    assert isinstance(result, pdal.Pipeline)


def test_zref_dimension_present(tmp_path):
    """Z_ref is added as an extra dimension in the output pipeline."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    pipeline = _make_pipeline([(5.5, 4.5, 110.0)])
    result = add_Zref(pipeline, dtm)
    result.execute()
    assert "Z_ref" in result.arrays[0].dtype.names


def test_zref_flat_dtm(tmp_path):
    """On a flat DTM at Z=100, Z_ref must equal Z − 100 for each point."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    # Points at known heights above the flat ground
    points = [(5.5, 4.5, 105.0), (2.5, 7.5, 110.0), (8.5, 1.5, 103.0)]
    expected = np.array([5.0, 10.0, 3.0])

    result = add_Zref(_make_pipeline(points), dtm, nodata_value=_NODATA_VALUE)
    result.execute()
    z_ref = result.arrays[0]["Z_ref"]

    np.testing.assert_allclose(z_ref, expected, atol=1e-3)


def test_zref_nodata_pixel(tmp_path):
    """Points on NoData pixels receive Z_ref = nodata_value."""
    data = np.full((_DTM_HEIGHT, _DTM_WIDTH), _GROUND_Z)
    data[5, 5] = _DTM_NODATA  # pixel (row=5, col=5) → centre at x=5.5, y=4.5
    dtm = _make_dtm(tmp_path / "dtm_nodata.tif", data)

    result = add_Zref(_make_pipeline([(5.5, 4.5, 110.0)]), dtm, nodata_value=_NODATA_VALUE)
    result.execute()
    assert result.arrays[0]["Z_ref"][0] == pytest.approx(_NODATA_VALUE)


def test_zref_outside_extent(tmp_path):
    """Points outside the DTM extent receive Z_ref = nodata_value."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    result = add_Zref(_make_pipeline([(999.0, 999.0, 110.0)]), dtm, nodata_value=_NODATA_VALUE)
    result.execute()
    assert result.arrays[0]["Z_ref"][0] == pytest.approx(_NODATA_VALUE)
>>>>>>> 1609350 (add function normalize height):test/pretreatment/test_normalize_height_by_points.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
