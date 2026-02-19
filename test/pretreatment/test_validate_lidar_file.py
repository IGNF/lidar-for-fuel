import os
import shutil
from pathlib import Path

import pdal
import pytest

from lidar_for_fuel.pretreatment.validate_lidar_file import check_lidar_file

TMP_PATH = Path("./tmp/check_lidar")
SAMPLE_LAS = "./data/pointcloud/test_data_0000_0000_LA93_IGN69.laz"


def setup_module(module):
    """Clean and recreate tmp directory before tests."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def test_check_lidar_file_return_format_okay():
    """Test function returns valid LasData object."""
    pipeline = check_lidar_file(SAMPLE_LAS, "EPSG:2154")
    assert isinstance(pipeline, pdal.Pipeline)
    arrays = pipeline.arrays
    assert len(arrays) == 1
    assert len(arrays[0]) > 0  # Fichier test a des points
    metadata = pipeline.metadata
    assert isinstance(metadata, dict)


def test_check_lidar_file_unsupported_extension():
    unsupported_path = TMP_PATH / "file.txt"
    unsupported_path.write_text("fake")
    with pytest.raises(ValueError, match="Unsupported extension"):
        check_lidar_file(str(unsupported_path), "EPSG:2154")


def test_check_lidar_file_not_exists():
    with pytest.raises(FileNotFoundError):
        check_lidar_file("nonexistent.laz", "EPSG:2154")
