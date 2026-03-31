"""
Integration test for download_dtm.

Requires network access to data.geopf.fr (IGN Géoplateforme).
"""
import os
from pathlib import Path

import rasterio

from lidar_for_fuel.pretreatment.download_dtm_from_geoplateforme import download_dtm

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
    assert Path(output_path).stem.startswith(
        Path(SAMPLE_TILENAME).stem
    ), "Output stem should start with input tile stem"

    # File is a readable raster with varying elevation values
    with rasterio.open(output_path) as src:
        assert src.count >= 1, "DTM should have at least one band"
        assert src.crs is not None, "DTM should have a CRS"
        data = src.read(1)

    assert data.size > 0, "DTM raster should not be empty"
    assert data.max() > data.min(), "DTM should contain varying elevation values"
