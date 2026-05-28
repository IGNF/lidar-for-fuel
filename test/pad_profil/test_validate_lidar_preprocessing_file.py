"""Tests for pad_profil/validate_lidar_preprocessing_file — validates extra dims h_abg, X_sensor, Y_sensor, Z_sensor."""
import os
import shutil
from pathlib import Path

import laspy
import numpy as np
import pytest

from lidar_for_fuel.pad_profil.validate_lidar_preprocessing_file import check_lidar_file, REQUIRED_EXTRA_DIMS


TMP_PATH = Path("./tmp/check_lidar_pad_profil")

# File produced by the preprocessing pipeline: has all 4 required extra dims.
PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)

# Standard raw file: no extra dims.
RAW_LAS = Path("data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz")


def setup_module(module):
    """Clean and recreate tmp directory before tests."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def _write_las_with_extra_dims(path: Path, extra_dim_names: list[str]) -> None:
    """Write a minimal LAS 1.4 file with the given extra byte dimensions."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dims([laspy.ExtraBytesParams(name=n, type=np.float64) for n in extra_dim_names])
    las = laspy.LasData(header=header)
    n = 5
    las.x = np.zeros(n)
    las.y = np.zeros(n)
    las.z = np.ones(n) * 10.0
    for name in extra_dim_names:
        setattr(las, name, np.zeros(n))
    las.write(str(path))


# ── error cases ───────────────────────────────────────────────────────────────

def test_check_lidar_file_not_exists():
    with pytest.raises(FileNotFoundError):
        check_lidar_file("nonexistent.laz")


def test_check_lidar_file_unsupported_extension():
    unsupported = TMP_PATH / "file.txt"
    unsupported.write_text("fake")
    with pytest.raises(ValueError, match="Unsupported extension"):
        check_lidar_file(str(unsupported))


def test_check_lidar_file_missing_all_extra_dims():
    """A raw LAS file without any extra dims must raise ValueError naming missing dims."""
    if not RAW_LAS.exists():
        pytest.skip(f"Raw data not available: {RAW_LAS}")
    with pytest.raises(ValueError, match="Missing required extra dimensions"):
        check_lidar_file(str(RAW_LAS))


def test_check_lidar_file_missing_some_extra_dims():
    """A file with only a subset of the required extra dims must raise ValueError."""
    partial = TMP_PATH / "partial_extra_dims.laz"
    _write_las_with_extra_dims(partial, ["h_abg", "X_sensor"])  # Y_sensor and Z_sensor missing
    with pytest.raises(ValueError, match="Missing required extra dimensions") as exc_info:
        check_lidar_file(str(partial))
    error_message = str(exc_info.value)
    assert "Y_sensor" in error_message
    assert "Z_sensor" in error_message


