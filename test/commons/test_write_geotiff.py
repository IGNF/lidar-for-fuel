import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from lidar_for_fuel.commons.write_geotiff import write_geotiff

_TRANSFORM = Affine(10.0, 0.0, 512489.75, 0.0, -10.0, 6854996.75)


def test_write_geotiff_writes_named_bands_in_order_on_the_grid(tmp_path):
    bands = {
        "PAD_1_0": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, np.nan]], dtype="float32"),
        "PAD_1_1": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"),
    }
    path = tmp_path / "out.tif"

    write_geotiff(path, bands, _TRANSFORM, "EPSG:2154")

    with rasterio.open(path) as src:
        assert src.count == 2
        assert (src.height, src.width) == (2, 3)
        assert src.dtypes == ("float32", "float32")
        assert src.descriptions == ("PAD_1_0", "PAD_1_1")
        assert src.transform == _TRANSFORM
        assert src.crs == CRS.from_epsg(2154)
        assert np.isnan(src.nodata)
        np.testing.assert_array_equal(src.read(1), bands["PAD_1_0"])
        np.testing.assert_array_equal(src.read(2), bands["PAD_1_1"])


def test_write_geotiff_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "out.tif"

    write_geotiff(path, {"Cover_2": np.zeros((1, 2), dtype="float32")}, _TRANSFORM, CRS.from_epsg(2154))

    with rasterio.open(path) as src:
        assert src.descriptions == ("Cover_2",)
