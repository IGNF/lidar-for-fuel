from pathlib import Path

import numpy as np
import pandas as pd
import pdal
import pytest
from omegaconf import OmegaConf

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile
from lidar_for_fuel.pad_profile.create_raster import (
    create_raster_from_points,
    points_to_dataframe,
    transform_points_coordinates,
)

_REAL_PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)
_BUFFER_LAS = Path("data/pointcloud/Semis_2022_0691_6484_LA93_IGN69_buffer_10m 1.laz")

_CONFIG = OmegaConf.load("configs/config.yaml")


def test_transform_points_coordinates_applies_vectorized_transformations():
    points = np.array(
        [
            (_CONFIG.pad_profile.create_raster.origin_x, _CONFIG.pad_profile.create_raster.origin_y, 10.0),
            (_CONFIG.pad_profile.create_raster.origin_x + 3.0, _CONFIG.pad_profile.create_raster.origin_y - 4.0, 20.0),
        ],
        dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")],
    )

    original_x = points["X"].copy()
    original_y = points["Y"].copy()
    origin_x = _CONFIG.pad_profile.create_raster.origin_x
    origin_y = _CONFIG.pad_profile.create_raster.origin_y
    resolution_factor = _CONFIG.pad_profile.create_raster.resolution_factor

    points_df = pd.DataFrame(points)

    assert isinstance(points_df, pd.DataFrame)

    df = transform_points_coordinates(
        points_df,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_factor=resolution_factor,
    )
    assert isinstance(df, pd.DataFrame)
    np.testing.assert_allclose(df["X"].to_numpy(), (original_x - origin_x) / resolution_factor)
    np.testing.assert_allclose(df["Y"].to_numpy(), (origin_y - original_y) / resolution_factor)
    assert not np.allclose(df["X"].to_numpy(), original_x)
    assert not np.allclose(df["Y"].to_numpy(), original_y)


def test_create_raster_from_points_synthetic_data():
    """Create a GeoTIFF raster from synthetic point
    cloud with max height per pixel
    """
    output_dir = Path("test/output_test")
    output_dir.mkdir(exist_ok=True)
    output_raster_path = output_dir / "test_raster.tif"
    output_raster_path.unlink(missing_ok=True)

    origin_x = 100.0
    origin_y = 210.5  # north edge = max Y of the synthetic points
    resolution_factor = 10.0

    points = np.array(
        [
            (100.0, 200.0, 5.0),
            (105.0, 200.0, 8.0),
            (105.0, 205.0, 12.0),
            (110.0, 210.0, 15.0),
            (110.0, 210.5, 10.0),
        ],
        dtype=[("X", "f8"), ("Y", "f8"), ("h_abg", "f8")],
    )

    points_df = pd.DataFrame(points)
    transformed_df = transform_points_coordinates(
        points_df,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_factor=resolution_factor,
    )

    raster = create_raster_from_points(
        transformed_df,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_factor=resolution_factor,
        output_path=str(output_raster_path),
        value_column="h_abg",
        aggregation="max",
    )

    assert output_raster_path.exists(), f"Raster file {output_raster_path} was not created."
    assert raster.shape[0] > 0 and raster.shape[1] > 0, "Raster has invalid dimensions."
    assert not np.all(np.isnan(raster)), "Raster contains only NaN values."

    expected_max_per_pixel = (
        transformed_df.assign(
            pixel_x=np.floor(transformed_df["X"]).astype(int),
            pixel_y=np.floor(transformed_df["Y"]).astype(int),
        )
        .groupby(["pixel_y", "pixel_x"])["h_abg"]
        .max()
    )
    for (row, col), expected_val in expected_max_per_pixel.items():
        assert not np.isnan(raster[row, col]), f"Pixel ({row}, {col}) is NaN but should contain {expected_val}"
        np.testing.assert_allclose(
            raster[row, col],
            expected_val,
            err_msg=f"Pixel ({row}, {col}): expected max h_abg={expected_val}, got {raster[row, col]}",
        )


def test_pad_profile_one_tile_real_las_returns_coherent_output_values():
    """Run pad_profile_one_tile on the real pre-treated LAS and assert cos_theta is in [0,1].

    The test is skipped if the LAS file is not present in the workspace.
    """
    real_las = _REAL_PRETRAITED_LAS

    if not real_las.exists():
        pytest.skip(f"Real LAS {real_las} not found in workspace")

    # Lower quality guards so the function returns a numeric value for testing.
    result = pad_profile_one_tile(
        input_filename=str(real_las),
        srid="EPSG:2154",
        keep_classes=[1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67],
        limit_N_points=1,
        limit_flight_agl=0.0,
        deviation_days=36_500,  # ~100 years: wide enough to keep every point in the file
        scanning_angle=True,
        z0=0.0,
        dz=1.0,
        nlayers=60,
        dz_low=0.5,
        nlayers_low=4,
        ground_margin=0.1,
        cover_type="all",
        height_cover=2.0,
        use_cover=True,
        G=0.5,
        omega=0.77,
        keep_values=[2, 3, 4, 5, 9],
    )

    assert isinstance(result, dict)
    cos_theta = result["cos_theta"]
    assert isinstance(cos_theta, (float, int)), "Expected a numeric cos_theta value"
    assert 0.0 <= float(cos_theta) <= 1.0
    pad_keys = [key for key in result if key.startswith("PAD_")]
    assert len(pad_keys) == 60 + 4
    for cover_key in ("Cover_2", "Cover_4", "Cover_6"):
        assert 0.0 <= result[cover_key] <= 1.0


def test_create_raster_preserves_classification_from_buffer_las():
    """Classification value of a point must appear in the raster built from the same file."""
    if not _BUFFER_LAS.exists():
        pytest.skip(f"LAS file not found: {_BUFFER_LAS}")

    output_path = "test/output_test/test_raster_classification.tif"

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=str(_BUFFER_LAS))
    pipeline.execute()
    points = pipeline.arrays[0]

    # Pick a reference point and assert its classification is a known value
    ref_point = points[0]
    ref_classification = int(ref_point["Classification"])
    assert ref_classification >= 0

    origin_x = float(points["X"].min())
    origin_y = float(points["Y"].max())
    resolution_factor = _CONFIG.pad_profile.create_raster.resolution_factor

    points_df = points_to_dataframe(points)
    transformed_df = transform_points_coordinates(
        points_df,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_factor=resolution_factor,
    )

    raster = create_raster_from_points(
        transformed_df,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_factor=resolution_factor,
        output_path=output_path,
        value_column="Classification",
        aggregation="max",
    )

    assert raster.shape[0] > 0 and raster.shape[1] > 0

    expected_pixel_x = int(np.floor((float(ref_point["X"]) - origin_x) / resolution_factor))
    expected_pixel_y = int(np.floor((origin_y - float(ref_point["Y"])) / resolution_factor))
    assert not np.isnan(raster[expected_pixel_y, expected_pixel_x])
    assert int(raster[expected_pixel_y, expected_pixel_x]) >= ref_classification
