"""Tests for main_pad_profile.pad_profile_one_tile -- grids a tile into pixels."""
from pathlib import Path
from unittest.mock import patch

import laspy
import numpy as np

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile

_EXTRA_DIMS = ("h_abg", "X_sensor", "Y_sensor", "Z_sensor")


def _write_synthetic_las(path: Path, x: np.ndarray, y: np.ndarray, gpstime: np.ndarray) -> None:
    """Write a minimal pre-processed LAS 1.4 file (point format 6 + the 4 extra dims)."""
    n = len(x)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dims([laspy.ExtraBytesParams(name=name, type=np.float64) for name in _EXTRA_DIMS])
    las = laspy.LasData(header=header)
    las.x = x
    las.y = y
    las.z = np.full(n, 100.0)
    las.gps_time = gpstime
    las.return_number = np.ones(n, dtype=np.uint8)
    las.classification = np.full(n, 2, dtype=np.uint8)
    for name in _EXTRA_DIMS:
        setattr(las, name, np.zeros(n))
    las.write(str(path))


def test_pad_profile_one_tile_groups_points_by_pixel(tmp_path):
    """3 points fall in pixel (0,0), 2 points fall in pixel (1,0) on the 10 m grid."""
    x = np.array([0.0, 1.0, 2.0, 12.0, 13.0])
    y = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    gpstime = np.zeros(5)

    input_file = tmp_path / "tile.laz"
    _write_synthetic_las(input_file, x, y, gpstime)

    with patch("lidar_for_fuel.main_pad_profile.pad_metrics_core") as mock_core:
        mock_core.return_value = {"date": 0.0}
        results = pad_profile_one_tile(
            input_filename=str(input_file),
            output_path=str(tmp_path / "out.laz"),
        )

    assert mock_core.call_count == 2
    call_point_counts = sorted(len(call.kwargs["x"]) for call in mock_core.call_args_list)
    assert call_point_counts == [2, 3]
    assert len(results) == 2
    assert all(value == {"date": 0.0} for value in results.values())


def test_pad_profile_one_tile_passes_config_through_to_pad_metrics_core(tmp_path):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    gpstime = np.zeros(2)

    input_file = tmp_path / "tile.laz"
    _write_synthetic_las(input_file, x, y, gpstime)

    with patch("lidar_for_fuel.main_pad_profile.pad_metrics_core") as mock_core:
        mock_core.return_value = None
        pad_profile_one_tile(
            input_filename=str(input_file),
            output_path=str(tmp_path / "out.laz"),
            scanning_angle=False,
            limit_N_points=5,
            limit_flight_agl=400.0,
            deviation_days=7,
            gpstime_ref="2020-01-01 00:00:00",
        )

    assert mock_core.call_count == 1
    kwargs = mock_core.call_args.kwargs
    assert kwargs["scanning_angle"] is False
    assert kwargs["limit_N_points"] == 5
    assert kwargs["limit_flight_agl"] == 400.0
    assert kwargs["deviation_days"] == 7
    assert kwargs["gpstime_ref"] == "2020-01-01 00:00:00"
