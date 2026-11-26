import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lidar_for_fuel.fuel_metrics.read_fma import read_fma

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


@pytest.mark.parametrize("value", [0.05, 0.5])
def test_read_fma_scalar_on_the_bounds_is_accepted(value):
    np.testing.assert_array_equal(read_fma(value, _PAD_TRANSFORM, (1, 2)), [[value, value]])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, 0.0, -0.1, 0.04, 0.6])
def test_read_fma_scalar_not_finite_or_out_of_range_raises(bad_value):
    # Same range as the FMA map: a NaN value would set the whole tile to NaN without a message
    with pytest.raises(ValueError, match=r"FMA value .* outside \[0.05, 0.5\]"):
        read_fma(bad_value, _PAD_TRANSFORM, (2, 3))


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


@pytest.mark.parametrize("bad_value", [0.04, 0.6, np.inf, -np.inf], ids=["0.04", "0.6", "+inf", "-inf"])
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
