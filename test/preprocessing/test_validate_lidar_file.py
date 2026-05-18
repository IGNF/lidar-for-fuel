import os
import shutil
from pathlib import Path

import pdal
import pytest

<<<<<<< HEAD
=======
<<<<<<<< HEAD:test/preprocessed/test_validate_lidar_file.py
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
<<<<<<<< HEAD:test/preprocessing/test_validate_lidar_file.py
from lidar_for_fuel.preprocessing.validate_lidar_file import check_lidar_file
========
from lidar_for_fuel.preprocessed.validate_lidar_file import check_lidar_file
>>>>>>>> fc1ae78 (rename pretreatment -> preprocessed):test/preprocessed/test_validate_lidar_file.py
<<<<<<< HEAD
=======
========
from lidar_for_fuel.preprocessing.validate_lidar_file import check_lidar_file
>>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing):test/preprocessing/test_validate_lidar_file.py
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

TMP_PATH = Path("./tmp/check_lidar")
SAMPLE_LAS = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"


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
