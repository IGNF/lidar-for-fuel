"""
Integration test for download_dtm.

Requires network access to data.geopf.fr (IGN Géoplateforme).
"""
import os
from pathlib import Path

import rasterio

<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
from lidar_for_fuel.preprocessing.download_dtm_from_geoplateforme import download_dtm

TMP_PATH = Path("./tmp/download_dtm")
SAMPLE_INPUT_DIR = "./data/pointcloud"
SAMPLE_TILENAME = "test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
<<<<<<< HEAD
=======
from lidar_for_fuel.pretreatment.download_dtm_from_geoplateforme import download_dtm

TMP_PATH = Path("./tmp/download_dtm")
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
SAMPLE_LAS = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
>>>>>>> cef44e0 (add function : download DTM from geoservice):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
SAMPLE_INPUT_DIR = "./data/pointcloud"
SAMPLE_TILENAME = "test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
>>>>>>> 0d401a2 (refacto main_pretreatment : change function download_dtm -> add path to DTM):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
_LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"


def test_download_dtm():
    """DTM is downloaded, saved as a valid GeoTIFF with varying elevation values."""
    os.makedirs(TMP_PATH, exist_ok=True)
<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
    output_path = download_dtm(SAMPLE_TILENAME, SAMPLE_INPUT_DIR, _LAYER, TMP_PATH)
=======
    output_path = download_dtm(SAMPLE_LAS, _LAYER, TMP_PATH)
>>>>>>> cef44e0 (add function : download DTM from geoservice):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
    output_path = download_dtm(SAMPLE_TILENAME, SAMPLE_INPUT_DIR, _LAYER, TMP_PATH)
>>>>>>> 0d401a2 (refacto main_pretreatment : change function download_dtm -> add path to DTM):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
    output_path = download_dtm(SAMPLE_TILENAME, SAMPLE_INPUT_DIR, _LAYER, TMP_PATH)
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

    # File exists and has the expected name
    assert Path(output_path).exists(), "Output DTM file was not created"
    assert Path(output_path).suffix == ".tif", "Output file should be a GeoTIFF"
<<<<<<< HEAD
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
<<<<<<< HEAD:test/preprocessing/test_download_dtm_from_geoplateforme.py
    assert Path(output_path).stem == Path(SAMPLE_TILENAME).stem, "Output stem should match input tile stem"
=======
    assert Path(output_path).stem == Path(SAMPLE_LAS).stem, "Output stem should match input tile stem"
>>>>>>> cef44e0 (add function : download DTM from geoservice):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
    assert Path(output_path).stem.startswith(
        Path(SAMPLE_TILENAME).stem
    ), "Output stem should start with input tile stem"
>>>>>>> 0d401a2 (refacto main_pretreatment : change function download_dtm -> add path to DTM):test/pretreatment/test_download_dtm_from_geoplateforme.py
=======
    assert Path(output_path).stem == Path(SAMPLE_TILENAME).stem, "Output stem should match input tile stem"
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

    # File is a readable raster with varying elevation values
    with rasterio.open(output_path) as src:
        assert src.count >= 1, "DTM should have at least one band"
        assert src.crs is not None, "DTM should have a CRS"
        data = src.read(1)

    assert data.size > 0, "DTM raster should not be empty"
    assert data.max() > data.min(), "DTM should contain varying elevation values"
