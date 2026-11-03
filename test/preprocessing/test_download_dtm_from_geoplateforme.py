"""
Integration test for download_dtm.

Requires network access to data.geopf.fr (IGN Géoplateforme).
"""
import os
from pathlib import Path

import rasterio

from lidar_for_fuel.preprocessing.download_dtm_from_geoplateforme import download_dtm

TMP_PATH = Path("./tmp/download_dtm")
SAMPLE_INPUT_DIR = "./data/pointcloud"
SAMPLE_TILENAME = "test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
_LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"


def test_download_dtm():
    """DTM is downloaded, saved as a valid GeoTIFF with varying elevation values."""
    os.makedirs(TMP_PATH, exist_ok=True)
    output_path = download_dtm(SAMPLE_TILENAME, SAMPLE_INPUT_DIR, _LAYER, TMP_PATH)

    # File exists and has the expected name
    assert Path(output_path).exists(), "Output DTM file was not created"
    assert Path(output_path).suffix == ".tif", "Output file should be a GeoTIFF"
    assert Path(output_path).stem == Path(SAMPLE_TILENAME).stem, "Output stem should match input tile stem"

    # File is a readable raster with varying elevation values
    with rasterio.open(output_path) as src:
        assert src.count >= 1, "DTM should have at least one band"
        assert src.crs is not None, "DTM should have a CRS"
        data = src.read(1)

    assert data.size > 0, "DTM raster should not be empty"
    assert data.max() > data.min(), "DTM should contain varying elevation values"


def test_download_dtm_native_grid():
    """DTM is requested on the native LiDAR HD grid (pixel centres on multiples of 0.5 m),
    so the WMS returns native values without resampling and covers the whole tile.

    Tile header bounds: (897000, 897999.99, 6576000.01, 6576999.99).
    """
    os.makedirs(TMP_PATH, exist_ok=True)
    output_path = download_dtm(SAMPLE_TILENAME, SAMPLE_INPUT_DIR, _LAYER, TMP_PATH)

    with rasterio.open(output_path) as src:
        transform = src.transform
        width, height = src.width, src.height

    assert transform.a == 0.5
    assert transform.e == -0.5
    # Origin shifted by half a pixel from the tile corner (897000, 6577000)
    assert transform.c == 896999.75
    assert transform.f == 6577000.25
    assert (width, height) == (2001, 2001)
