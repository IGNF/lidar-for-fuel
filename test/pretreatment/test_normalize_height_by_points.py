import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds

from lidar_for_fuel.pretreatment.normalize_height_by_dtm import add_Zref, filter_z_by_height

_NODATA_VALUE = -9999.0
_GROUND_Z = 100.0
_DTM_NODATA = -9999.0

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


def _make_dtm(path: Path, data: np.ndarray, bounds: tuple) -> str:
    """Write a single-band float32 GeoTIFF from a 2-D array."""
    west, south, east, north = bounds
    height, width = data.shape
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=1, dtype="float32", transform=transform, nodata=_DTM_NODATA,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return str(path)


def _make_points(rows: list) -> np.ndarray:
    """Build a structured point array from a list of (x, y, z) tuples."""
    pts = np.zeros(len(rows), dtype=_LAS_DTYPE)
    for i, (x, y, z) in enumerate(rows):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
    return pts


def _pixel_centre(col, row, pixel_size, west, north):
    """Return the geographic (x, y) centre of a pixel given its col/row indices."""
    x = west + (col + 0.5) * pixel_size
    y = north - (row + 0.5) * pixel_size
    return x, y


# ── add_Zref ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_pixels,pixel_size,west,south", [
    (10, 1.0, 0.0,      0.0),       # 1 m, origin at (0, 0)
    (20, 0.5, 0.0,      0.0),       # 0.5 m, origin at (0, 0)
    (10, 1.0, 700000.0, 6400000.0), # 1 m, realistic Lambert-93 origin
    (20, 0.5, 700000.0, 6400000.0), # 0.5 m, realistic Lambert-93 origin
], ids=["1m_origin0", "0.5m_origin0", "1m_offset", "0.5m_offset"])
def test_add_zref_flat_dtm(tmp_path, n_pixels, pixel_size, west, south):
    """On a flat DTM, Z_ref = Z - Z_sol for a point at a known pixel centre.

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
    result = add_Zref(_make_points([(x, y, _GROUND_Z + 7.0)]), dtm, nodata_value=_NODATA_VALUE)

    assert isinstance(result, np.ndarray)
    assert "Z_ref" in result.dtype.names
    np.testing.assert_allclose(result["Z_ref"][0], 7.0, atol=1e-3)


@pytest.mark.parametrize("n_pixels,pixel_size", [(10, 1.0), (20, 0.5)], ids=["1m", "0.5m"])
def test_add_zref_non_flat_dtm(tmp_path, n_pixels, pixel_size):
    """On a non-flat DTM, checks bilinear interpolation, nodata handling, and out-of-extent points.

    DTM: column col_slope at 102 m (rest at 100 m), one nodata pixel at (row=2, col=2).

    Points tested:
    - p1: midpoint between col_slope-1 (100 m) and col_slope (102 m) -> Z_ref = 111 - 101 = 10.0
    - p2: centre of the nodata pixel -> Z_ref = nodata_value
    - p3: outside DTM extent -> Z_ref = nodata_value
    """
    west, south, north = 0.0, 0.0, n_pixels * pixel_size
    bounds = (west, south, n_pixels * pixel_size, north)
    col_slope = 3 * n_pixels // 4   # col 7 (1m) or col 15 (0.5m), far from the nodata pixel
    row_interp = n_pixels // 2      # row 5 (1m) or row 10 (0.5m)

    data = np.full((n_pixels, n_pixels), _GROUND_Z)
    data[:, col_slope] = 102.0
    data[2, 2] = _DTM_NODATA

    dtm = _make_dtm(tmp_path / f"dtm_slope_{n_pixels}.tif", data, bounds)

    # x_mid: midpoint between the centres of col_slope-1 and col_slope
    x_mid = west + col_slope * pixel_size
    _, y_interp = _pixel_centre(0, row_interp, pixel_size, west, north)
    x_nodata, y_nodata = _pixel_centre(2, 2, pixel_size, west, north)

    result = add_Zref(
        _make_points([(x_mid, y_interp, 111.0), (x_nodata, y_nodata, 110.0), (999.0, 999.0, 110.0)]),
        dtm,
        nodata_value=_NODATA_VALUE,
    )

    np.testing.assert_allclose(result["Z_ref"][0], 10.0, atol=1e-3)  # bilinear interpolation
    assert result["Z_ref"][1] == _NODATA_VALUE                        # nodata
    assert result["Z_ref"][2] == _NODATA_VALUE                        # outside extent


# ── filter_z_by_height ─────────────────────────────────────────────────────────

def _make_points_with_zref(zref_values: list) -> np.ndarray:
    """Build a structured array with a Z_ref field directly."""
    dtype = np.dtype(_LAS_DTYPE.descr + [("Z_ref", np.float64)])
    pts = np.zeros(len(zref_values), dtype=dtype)
    for i, z in enumerate(zref_values):
        pts[i]["Z_ref"] = z
    return pts


@pytest.mark.parametrize("zref_values,min_h,max_h,expected", [
    ([5.0, 85.0, 50.0], -3,   80, [5.0, 50.0]), # default bounds: high point removed
    ([-5.0, 5.0],       -3,   80, [5.0]),        # default bounds: low point removed
    ([5.0, 15.0, -1.0], -0.5, 10, [5.0]),        # custom bounds: high and low removed
], ids=["removes_high", "removes_low", "custom_bounds"])
def test_filter_z_by_height(zref_values, min_h, max_h, expected):
    """Points outside [min_height_filter, height_filter] are removed."""
    result = filter_z_by_height(_make_points_with_zref(zref_values), min_height_filter=min_h, height_filter=max_h)
    assert len(result) == len(expected)
    np.testing.assert_allclose(sorted(result["Z_ref"]), sorted(expected), atol=1e-3)
