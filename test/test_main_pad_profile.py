import logging
import os
from pathlib import Path

import laspy
import numpy as np
import pytest
import rasterio

from lidar_for_fuel.main_pad_profile import process_one_dalle

_TILE_WIDTH = 10  # small override so n_rows == n_cols == 1 at PIXEL_SIZE=10
_N_POINTS = 5


def _write_synthetic_las(path: Path, n: int = _N_POINTS) -> None:
    """Write a small synthetic preprocessed LAS file with all required fields/extra-dims."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="h_abg", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="X_sensor", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="Y_sensor", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="Z_sensor", type=np.float64))
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]

    las = laspy.LasData(header)
    x = 751000.0 + np.linspace(1, 9, n)
    y = 6690000.0 - np.linspace(1, 9, n)
    z = 200.0 + np.arange(n, dtype=np.float64)

    las.x = x
    las.y = y
    las.z = z
    las.gps_time = np.arange(1.0, n + 1.0)
    las.return_number = np.ones(n, dtype=np.uint8)
    las.classification = np.full(n, 2, dtype=np.uint8)
    las.h_abg = np.full(n, 1.5)
    las.X_sensor = x.copy()
    las.Y_sensor = y.copy()
    las.Z_sensor = np.full(n, 1500.0)

    las.write(str(path))


_PRODUCT_NAMES = {
    "pad_profile_1m",
    "pad_sb_0.5m",
    "entering_rays",
    "intercept_ray",
    "class_count",
    "COVER",
}


def test_happy_path_single_file_writes_6_rasters(tmp_path, caplog):
    """Happy path: a single conforming-filename dalle produces 6 COG rasters at the
    dalle's native origin/shape derived from the filename."""
    input_filename = tmp_path / "Semis_2024_0751_6690_LA93_IGN69.laz"
    _write_synthetic_las(input_filename)
    output_dir = tmp_path / "out"

    with caplog.at_level(logging.INFO):
        paths = process_one_dalle(
            input_filename=str(input_filename),
            output_dir=str(output_dir),
            tile_width=_TILE_WIDTH,
            scanning_angle=False,
            use_cover=False,
            limit_N_points=0,
        )

    assert set(paths.keys()) == _PRODUCT_NAMES
    for product, path in paths.items():
        assert os.path.exists(path), f"Missing output for {product}: {path}"
        assert path.endswith(".tif")
        assert "Semis_2024_0751_6690_LA93_IGN69" in os.path.basename(path)

    # Verify origin/shape derived from filename (tile_width=10 -> 1x1 pixel raster).
    with rasterio.open(paths["pad_profile_1m"]) as src:
        assert src.width == 1
        assert src.height == 1
        assert src.transform.c == 751000.0  # origin_x = xs[0]
        assert src.transform.f == 6690000.0  # origin_y = ys[1] (top-left, maxY)

    assert "Processing tile" in caplog.text


def test_directory_mode_processes_each_file_independently(tmp_path):
    """Directory mode: N conforming files in input_dir, no input_filename set,
    each processed independently -> 6*N raster files exist."""
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    filenames = [
        "Semis_2024_0751_6690_LA93_IGN69.laz",
        "Semis_2024_0752_6690_LA93_IGN69.laz",
    ]
    for fname in filenames:
        _write_synthetic_las(input_dir / fname)

    written_paths = []
    for fname in sorted(os.listdir(input_dir)):
        paths = process_one_dalle(
            input_filename=str(input_dir / fname),
            output_dir=str(output_dir),
            tile_width=_TILE_WIDTH,
            scanning_angle=False,
            use_cover=False,
            limit_N_points=0,
        )
        written_paths.extend(paths.values())

    assert len(written_paths) == 6 * len(filenames)
    for path in written_paths:
        assert os.path.exists(path)

    all_tifs = list(output_dir.glob("*.tif"))
    assert len(all_tifs) == 6 * len(filenames)


def test_non_conforming_filename_raises_value_error(tmp_path):
    """A filename that doesn't match parse_filename's expected pattern propagates
    a ValueError from pdaltools (not caught)."""
    input_filename = tmp_path / "not_a_conforming_name.laz"
    _write_synthetic_las(input_filename)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError):
        process_one_dalle(
            input_filename=str(input_filename),
            output_dir=str(output_dir),
            tile_width=_TILE_WIDTH,
        )

    assert not output_dir.exists()


def test_tile_width_not_multiple_of_pixel_size_raises_value_error(tmp_path):
    """tile_width not an exact multiple of PIXEL_SIZE (10) -> ValueError, no files written."""
    input_filename = tmp_path / "Semis_2024_0751_6690_LA93_IGN69.laz"
    _write_synthetic_las(input_filename)
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError):
        process_one_dalle(
            input_filename=str(input_filename),
            output_dir=str(output_dir),
            tile_width=15,  # not a multiple of PIXEL_SIZE=10
        )

    assert not output_dir.exists()


def test_missing_preprocessed_field_raises_value_error(tmp_path):
    """A LAS file missing an expected preprocessed extra-dim (e.g. h_abg) raises a
    clear ValueError instead of a raw KeyError from numpy field access."""
    input_filename = tmp_path / "Semis_2024_0751_6690_LA93_IGN69.laz"

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="X_sensor", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="Y_sensor", type=np.float64))
    header.add_extra_dim(laspy.ExtraBytesParams(name="Z_sensor", type=np.float64))
    # h_abg intentionally omitted.
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]
    las = laspy.LasData(header)
    las.x = np.array([751001.0])
    las.y = np.array([6689999.0])
    las.z = np.array([200.0])
    las.gps_time = np.array([1.0])
    las.return_number = np.array([1], dtype=np.uint8)
    las.classification = np.array([2], dtype=np.uint8)
    las.X_sensor = las.x.copy()
    las.Y_sensor = las.y.copy()
    las.Z_sensor = np.array([1500.0])
    las.write(str(input_filename))

    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="h_abg"):
        process_one_dalle(
            input_filename=str(input_filename),
            output_dir=str(output_dir),
            tile_width=_TILE_WIDTH,
        )

    assert not output_dir.exists()


def test_missing_input_dir_raises_value_error(tmp_path):
    """config.pad_profile.input_dir is None -> ValueError, mirrors main_preprocessing.py."""
    from omegaconf import OmegaConf

    from lidar_for_fuel.main_pad_profile import main

    config = OmegaConf.create(
        {
            "pad_profile": {
                "input_dir": None,
                "input_filename": None,
                "output_dir": str(tmp_path / "out"),
                "G": None,
                "omega": None,
                "scanning_angle": None,
                "use_cover": None,
                "cover_type": None,
                "height_cover": None,
                "limit_N_points": None,
                "limit_flight_agl": None,
            },
            "tile_geometry": {"tile_width": _TILE_WIDTH, "tile_coord_scale": 1000},
            "commons": {"filter_date": {"deviation_days": 14, "gpstime_ref": "2011-09-14 01:46:40"}},
        }
    )

    with pytest.raises(ValueError):
        main.__wrapped__(config)


def test_nonexistent_input_dir_raises_file_not_found_error(tmp_path):
    """config.pad_profile.input_dir is set but doesn't exist -> FileNotFoundError,
    mirrors main_preprocessing.py's existing guard."""
    from omegaconf import OmegaConf

    from lidar_for_fuel.main_pad_profile import main

    config = OmegaConf.create(
        {
            "pad_profile": {
                "input_dir": str(tmp_path / "does_not_exist"),
                "input_filename": None,
                "output_dir": str(tmp_path / "out"),
                "G": None,
                "omega": None,
                "scanning_angle": None,
                "use_cover": None,
                "cover_type": None,
                "height_cover": None,
                "limit_N_points": None,
                "limit_flight_agl": None,
            },
            "tile_geometry": {"tile_width": _TILE_WIDTH, "tile_coord_scale": 1000},
            "commons": {"filter_date": {"deviation_days": 14, "gpstime_ref": "2011-09-14 01:46:40"}},
        }
    )

    with pytest.raises(FileNotFoundError):
        main.__wrapped__(config)


def test_directory_mode_with_no_matching_files_logs_warning(tmp_path, caplog):
    """An empty input_dir (no .las/.laz files) logs a warning and writes nothing."""
    from omegaconf import OmegaConf

    from lidar_for_fuel.main_pad_profile import main

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    config = OmegaConf.create(
        {
            "pad_profile": {
                "input_dir": str(input_dir),
                "input_filename": None,
                "output_dir": str(output_dir),
                "G": None,
                "omega": None,
                "scanning_angle": None,
                "use_cover": None,
                "cover_type": None,
                "height_cover": None,
                "limit_N_points": None,
                "limit_flight_agl": None,
            },
            "tile_geometry": {"tile_width": _TILE_WIDTH, "tile_coord_scale": 1000},
            "commons": {"filter_date": {"deviation_days": 14, "gpstime_ref": "2011-09-14 01:46:40"}},
        }
    )

    with caplog.at_level(logging.WARNING):
        main.__wrapped__(config)

    assert "No" in caplog.text
    assert list(output_dir.glob("*.tif")) == []


def test_missing_output_dir_raises_value_error(tmp_path):
    """config.pad_profile.output_dir is None -> ValueError, mirrors main_preprocessing.py."""
    from omegaconf import OmegaConf

    from lidar_for_fuel.main_pad_profile import main

    config = OmegaConf.create(
        {
            "pad_profile": {
                "input_dir": str(tmp_path),
                "input_filename": None,
                "output_dir": None,
                "G": None,
                "omega": None,
                "scanning_angle": None,
                "use_cover": None,
                "cover_type": None,
                "height_cover": None,
                "limit_N_points": None,
                "limit_flight_agl": None,
            },
            "tile_geometry": {"tile_width": _TILE_WIDTH, "tile_coord_scale": 1000},
            "commons": {"filter_date": {"deviation_days": 14, "gpstime_ref": "2011-09-14 01:46:40"}},
        }
    )

    with pytest.raises(ValueError):
        main.__wrapped__(config)
