import shutil
from pathlib import Path

import pandas as pd
import pytest

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile

TMP_PATH = Path("./tmp/cmain_pad_profile")

# CosiaFrance grid anchor and pixel size (configs/config.yaml: pad_profile.create_raster).
_GLOBAL_ORIGIN_X = 98029.75
_GLOBAL_ORIGIN_Y = 6045536.75
_RESOLUTION_FACTOR = 10.0

# File produced by the preprocessing pipeline: has all 4 required extra dims.
PREPROCESSED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_preprocessed.laz")

# 3x3 grid of real neighboring tiles, used to exercise real buffer merging (as opposed to
# a single isolated tile copied into an empty directory, which has no neighbors to merge in).
BUFFER_DIR = Path("data/buffer")
CENTRAL_TILE = BUFFER_DIR / "Semis_2022_0691_6484_LA93_IGN69_preprocessed.laz"

_PAD_PARAMS = dict(
    keep_classes=[1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67],
    limit_N_points=1,
    limit_flight_agl=0.0,
    deviation_days=36_500,  # ~100 years: wide enough to keep every point, including neighbors'
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


def test_pad_profile_one_tile_real_las_returns_coherent_output_values():
    """Run pad_profile_one_tile on the real pre-treated LAS and assert cos_theta is in [0,1].

    The test is skipped if the LAS file is not present in the workspace.
    """
    real_las = PREPROCESSED_LAS

    if not real_las.exists():
        pytest.skip(f"Real LAS {real_las} not found in workspace")

    # pdaltools' buffer tile-naming parser expects <prefix1>_<prefix2>_<coordX>_<coordY>_<suffix>
    # (e.g. Semis_2024_0751_6690_...). Strip the fixture's leading "test_" segment so the tile
    # coordinates are parsed from the right fields.
    TMP_PATH.mkdir(parents=True, exist_ok=True)
    tile = TMP_PATH / real_las.name.removeprefix("test_")
    shutil.copy(real_las, tile)

    # Lower quality guards so the function returns a numeric value for testing.
    aggregated, origin_pixel, nb_pixels = pad_profile_one_tile(
        input_filename=str(tile),
        input_dir=str(tile.parent),
        buffer_width=100,
        tile_width=1000,
        tile_coord_scale=1000,
        srid="EPSG:2154",
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
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

    assert isinstance(aggregated, pd.DataFrame)
    assert not aggregated.empty
    assert nb_pixels == 100.0  # 1000m tile / 10m pixels
    assert isinstance(origin_pixel, tuple) and len(origin_pixel) == 2

    # cos_theta can be NaN for a pixel with no vegetation/ground point at all (too few
    # points per 10m pixel to guarantee one) -- that's an expected per-pixel data-quality
    # case, not a wiring bug, so only the non-NaN pixels are checked for range.
    cos_theta = aggregated["cos_theta"].dropna()
    assert not cos_theta.empty
    assert cos_theta.between(0.0, 1.0).all()
    pad_columns = [c for c in aggregated.columns if c.startswith("PAD_")]
    assert len(pad_columns) == 60 + 4
    for cover_column in ("Cover_2", "Cover_4", "Cover_6"):
        assert aggregated[cover_column].between(0.0, 1.0).all()


def test_pad_profile_one_tile_buffered_output_covers_more_pixels_than_unbuffered():
    """`compute_pixel_aggregates` windows its output back down to the tile's own
    CosiaFrance footprint, so the buffer no longer inflates the overall point count the
    way the old whole-tile call did (see git history) -- it only fills in the pixels
    that straddle the tile's own boundary (the misalignment between the LiDAR HD tiling
    and the CosiaFrance grid, see test_create_raster.py). Compare buffered vs unbuffered
    (buffer_width=0) runs of the same tile: the buffer must strictly add populated pixels,
    all of them among the pixels the unbuffered run couldn't fill.

    Skipped if the buffer tile set is not present in the workspace.
    """
    if not CENTRAL_TILE.exists():
        pytest.skip(f"Buffer tile set not found in workspace: {CENTRAL_TILE}")

    common_kwargs = dict(
        input_filename=str(CENTRAL_TILE),
        input_dir=str(BUFFER_DIR),
        tile_width=1000,
        tile_coord_scale=1000,
        srid="EPSG:2154",
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        **_PAD_PARAMS,
    )

    aggregated_unbuffered, _, _ = pad_profile_one_tile(buffer_width=0, **common_kwargs)
    aggregated_buffered, _, _ = pad_profile_one_tile(buffer_width=50, **common_kwargs)

    assert isinstance(aggregated_buffered, pd.DataFrame)
    assert not aggregated_buffered.empty
    assert len(aggregated_buffered) > len(aggregated_unbuffered)
    assert set(aggregated_unbuffered.index) <= set(aggregated_buffered.index)
