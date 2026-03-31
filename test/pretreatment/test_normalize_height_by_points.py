import json
from pathlib import Path

import numpy as np
import pdal
import rasterio
from rasterio.transform import from_bounds

from lidar_for_fuel.pretreatment.normalize_height_by_points import add_Zref

_NODATA_VALUE = -9999.0
_GROUND_Z = 100.0  # flat DTM elevation (metres)
_DTM_NODATA = -9999.0  # nodata sentinel used inside the DTM raster

# DTM covers X=[0, 10], Y=[0, 10], 10×10 pixels at 1 m resolution.
# Pixel centre (row=i, col=j): x = 0.5 + j, y = 9.5 - i
_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH = 0.0, 0.0, 10.0, 10.0
_DTM_WIDTH = _DTM_HEIGHT = 10

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
    transform = from_bounds(_DTM_WEST, _DTM_SOUTH, _DTM_EAST, _DTM_NORTH, _DTM_WIDTH, _DTM_HEIGHT)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_DTM_HEIGHT,
        width=_DTM_WIDTH,
        count=1,
        dtype="float32",
        transform=transform,
        nodata=_DTM_NODATA,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return str(path)


def _make_flat_dtm(path: Path) -> str:
    """Flat DTM: all pixels at _GROUND_Z."""
    return _make_dtm(path, np.full((_DTM_HEIGHT, _DTM_WIDTH), _GROUND_Z))


def _make_pipeline(rows: list) -> pdal.Pipeline:
    """Build an unexecuted pipeline from a list of (x, y, z) tuples."""
    pts = np.zeros(len(rows), dtype=_LAS_DTYPE)
    for i, (x, y, z) in enumerate(rows):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
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
    """Points on NoData pixels are removed by the Z_ref filter (Z_ref=nodata_value < -3)."""
    data = np.full((_DTM_HEIGHT, _DTM_WIDTH), _GROUND_Z)
    data[5, 5] = _DTM_NODATA  # pixel (row=5, col=5) → centre at x=5.5, y=4.5
    dtm = _make_dtm(tmp_path / "dtm_nodata.tif", data)

    result = add_Zref(_make_pipeline([(5.5, 4.5, 110.0)]), dtm, nodata_value=_NODATA_VALUE)
    result.execute()
    assert len(result.arrays[0]) == 0


def test_zref_outside_extent(tmp_path):
    """Points outside the DTM extent are removed by the Z_ref filter (Z_ref=nodata_value < -3)."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    result = add_Zref(_make_pipeline([(999.0, 999.0, 110.0)]), dtm, nodata_value=_NODATA_VALUE)
    result.execute()
    assert len(result.arrays[0]) == 0


def test_height_filter_removes_high_points(tmp_path):
    """Points with Z_ref > height_filter are removed."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    # Z_ref = 5.0 (kept), 85.0 (removed with default 80), 50.0 (kept)
    points = [(5.5, 4.5, 105.0), (2.5, 7.5, 185.0), (8.5, 1.5, 150.0)]

    result = add_Zref(_make_pipeline(points), dtm)
    result.execute()
    z_ref = result.arrays[0]["Z_ref"]

    assert len(z_ref) == 2
    np.testing.assert_allclose(sorted(z_ref), [5.0, 50.0], atol=1e-3)


def test_height_filter_removes_low_points(tmp_path):
    """Points with Z_ref < -3 are removed."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    # Z_ref = -5.0 (removed), 5.0 (kept)
    points = [(5.5, 4.5, 95.0), (2.5, 7.5, 105.0)]

    result = add_Zref(_make_pipeline(points), dtm)
    result.execute()
    z_ref = result.arrays[0]["Z_ref"]

    assert len(z_ref) == 1
    np.testing.assert_allclose(z_ref[0], 5.0, atol=1e-3)


def test_height_filter_custom(tmp_path):
    """Custom height_filter value is respected."""
    dtm = _make_flat_dtm(tmp_path / "dtm.tif")
    # Z_ref = 5.0 (kept), 15.0 (removed with height_filter=10)
    points = [(5.5, 4.5, 105.0), (2.5, 7.5, 115.0)]

    result = add_Zref(_make_pipeline(points), dtm, height_filter=10)
    result.execute()
    z_ref = result.arrays[0]["Z_ref"]

    assert len(z_ref) == 1
    np.testing.assert_allclose(z_ref[0], 5.0, atol=1e-3)
