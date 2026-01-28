import os
import shutil
from pathlib import Path

import laspy
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
    las = check_lidar_file(SAMPLE_LAS)

    assert isinstance(las, laspy.LasData) is True
    assert len(las.points) > 0
    assert las.header.version == "1.4"


def test_check_lidar_file_empty():
    """Test empty file handling (warning but success)."""
    # Create empty LAS file
    empty_path = TMP_PATH / "empty.las"
    las_empty = laspy.create(point_format=2, file_version="1.2")
    las_empty.write(empty_path)

    las = check_lidar_file(str(empty_path))
    assert isinstance(las, laspy.LasData) is True
    assert len(las.points) == 0


def test_check_lidar_file_unsupported_extension():
    """Test unsupported file extension."""
    bad_path = TMP_PATH / "test.txt"
    bad_path.write_text("fake data")

    with pytest.raises(ValueError, match="Unsupported extension"):
        check_lidar_file(str(bad_path))


def test_check_lidar_file_not_exists():
    """Test non-existent file."""
    fake_path = TMP_PATH / "fake.las"

    with pytest.raises(FileNotFoundError, match="File not found"):
        check_lidar_file(str(fake_path))
