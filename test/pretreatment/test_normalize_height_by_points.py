import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds

from lidar_for_fuel.pretreatment.normalize_height_by_points import add_Zref, filter_z_by_height

_NODATA_VALUE = -9999.0
_GROUND_Z = 100.0
_DTM_NODATA = -9999.0

# DTM covers X=[0, 10], Y=[0, 10]
_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH = 0.0, 0.0, 10.0, 10.0

# 1 m resolution → 10×10 pixels ; 0.5 m resolution (LiDAR HD native) → 20×20 pixels
_RESOLUTIONS = pytest.mark.parametrize("n_pixels,pixel_size", [(10, 1.0), (20, 0.5)], ids=["1m", "0.5m"])

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


def _make_dtm(path: Path, data: np.ndarray) -> str:
    """Write a single-band float32 GeoTIFF from a 2-D array."""
    height, width = data.shape
    transform = from_bounds(_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH, width, height)
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


def _pixel_centre(col, row, pixel_size):
    """Return the geographic (x, y) centre of a pixel given its col/row indices."""
    x = (col + 0.5) * pixel_size
    y = _DTM_NORTH - (row + 0.5) * pixel_size
    return x, y


# ── add_Zref ───────────────────────────────────────────────────────────────────

@_RESOLUTIONS
def test_add_zref_flat_dtm(tmp_path, n_pixels, pixel_size):
    """On a flat DTM, Z_ref = Z - Z_sol for each point; output is an ndarray with a Z_ref field.

    Points are placed at the centre of known pixels (col=3, row=2) and (col=8, row=7).
    """
    data = np.full((n_pixels, n_pixels), _GROUND_Z)
    dtm = _make_dtm(tmp_path / f"dtm_flat_{n_pixels}.tif", data)

    x1, y1 = _pixel_centre(3, 2, pixel_size)
    x2, y2 = _pixel_centre(8, 7, pixel_size)
    result = add_Zref(_make_points([(x1, y1, 105.0), (x2, y2, 110.0)]), dtm, nodata_value=_NODATA_VALUE)

    assert isinstance(result, np.ndarray)
    assert "Z_ref" in result.dtype.names
    np.testing.assert_allclose(result["Z_ref"][0], 5.0, atol=1e-3)
    np.testing.assert_allclose(result["Z_ref"][1], 10.0, atol=1e-3)


@_RESOLUTIONS
def test_add_zref_non_flat_dtm(tmp_path, n_pixels, pixel_size):
    """On a non-flat DTM, checks bilinear interpolation, nodata handling, and out-of-extent points.

    DTM: column col_slope at 102 m (rest at 100 m), one nodata pixel at (row=2, col=2).

    Points tested:
    - p1: midpoint between col_slope-1 (100 m) and col_slope (102 m) -> Z_ref = 111 - 101 = 10.0
    - p2: centre of the nodata pixel -> Z_ref = nodata_value
    - p3: outside DTM extent -> Z_ref = nodata_value
    """
    col_slope = 3 * n_pixels // 4   # col 7 (1m) or col 15 (0.5m), far from the nodata pixel
    row_interp = n_pixels // 2      # row 5 (1m) or row 10 (0.5m)

    data = np.full((n_pixels, n_pixels), _GROUND_Z)
    data[:, col_slope] = 102.0
    data[2, 2] = _DTM_NODATA

    dtm = _make_dtm(tmp_path / f"dtm_slope_{n_pixels}.tif", data)

    # x_mid: midpoint between the centres of col_slope-1 and col_slope
    x_mid = col_slope * pixel_size
    _, y_interp = _pixel_centre(0, row_interp, pixel_size)
    x_nodata, y_nodata = _pixel_centre(2, 2, pixel_size)

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


def test_filter_z_removes_high_points():
    """Points with Z_ref > height_filter are removed."""
    # Z_ref = 5.0 (kept), 85.0 (removed with default 80), 50.0 (kept)
    pts = _make_points_with_zref([5.0, 85.0, 50.0])
    result = filter_z_by_height(pts)
    assert len(result) == 2
    np.testing.assert_allclose(sorted(result["Z_ref"]), [5.0, 50.0], atol=1e-3)


def test_filter_z_removes_low_points():
    """Points with Z_ref < min_height_filter are removed."""
    # Z_ref = -5.0 (removed), 5.0 (kept)
    pts = _make_points_with_zref([-5.0, 5.0])
    result = filter_z_by_height(pts)
    assert len(result) == 1
    np.testing.assert_allclose(result["Z_ref"][0], 5.0, atol=1e-3)


def test_filter_z_custom_bounds():
    """Custom min_height_filter and height_filter values are respected."""
    # Z_ref = 5.0 (kept), 15.0 (removed with height_filter=10), -1.0 (removed with min=-0.5)
    pts = _make_points_with_zref([5.0, 15.0, -1.0])
    result = filter_z_by_height(pts, min_height_filter=-0.5, height_filter=10)
    assert len(result) == 1
    np.testing.assert_allclose(result["Z_ref"][0], 5.0, atol=1e-3)
