import shutil
from pathlib import Path

import numpy as np
import pdal
import pytest

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile
from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core

TMP_PATH = Path("./tmp/cmain_pad_profile")

# File produced by the preprocessing pipeline: has all 4 required extra dims.
PREPROCESSED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_preprocessed.laz")

# 3x3 grid of real neighboring tiles, used to exercise real buffer merging (as opposed to
# a single isolated tile copied into an empty directory, which has no neighbors to merge in).
BUFFER_DIR = Path("data/buffer")
CENTRAL_TILE = BUFFER_DIR / "Semis_2022_0691_6484_LA93_IGN69_preprocessed.laz"

# PAD-computation parameters shared between the direct pad_metrics_core call (raw tile,
# no buffer) and pad_profile_one_tile (buffered tile): keeping them identical isolates
# the buffer as the only variable between the two outputs.
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
    result = pad_profile_one_tile(
        input_filename=str(tile),
        input_dir=str(tile.parent),
        buffer_width=100,
        tile_width=1000,
        tile_coord_scale=1000,
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


def _pad_metrics_on_raw_tile(input_filename: str) -> dict[str, float] | None:
    """Compute PAD metrics directly on a tile's own points, bypassing add_buffer
    entirely -- the "no buffer" baseline to compare against pad_profile_one_tile's
    buffered output."""
    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=input_filename, override_srs="EPSG:2154", nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    return pad_metrics_core(
        gpstime=points["GpsTime"].astype(np.float64),
        x=points["X"].astype(np.float64),
        y=points["Y"].astype(np.float64),
        h_abg=points["h_abg"].astype(np.float64),
        z=points["Z"].astype(np.float64),
        return_number=points["ReturnNumber"].astype(np.float64),
        classification=points["Classification"].astype(np.float64),
        x_sensor=points["X_sensor"].astype(np.float64),
        y_sensor=points["Y_sensor"].astype(np.float64),
        z_sensor=points["Z_sensor"].astype(np.float64),
        **_PAD_PARAMS,
    )


def test_pad_profile_one_tile_buffered_output_differs_from_raw_tile():
    """pad_profile_one_tile (which merges in real neighboring tiles via add_buffer)
    must give a different result than pad_metrics_core run on the tile's own raw
    points: the buffered version sees extra points from the 8 real neighbors in
    data/buffer/, so it covers a strictly larger footprint than the raw 1km x 1km tile.

    Skipped if the buffer tile set is not present in the workspace.
    """
    if not CENTRAL_TILE.exists():
        pytest.skip(f"Buffer tile set not found in workspace: {CENTRAL_TILE}")

    result_raw = _pad_metrics_on_raw_tile(str(CENTRAL_TILE))

    result_buffered = pad_profile_one_tile(
        input_filename=str(CENTRAL_TILE),
        input_dir=str(BUFFER_DIR),
        buffer_width=50,
        tile_width=1000,
        tile_coord_scale=1000,
        srid="EPSG:2154",
        **_PAD_PARAMS,
    )

    assert isinstance(result_raw, dict)
    assert isinstance(result_buffered, dict)
    assert result_raw != result_buffered
    # The buffer merges in points from the 8 real neighboring tiles -> strictly more
    # points than the raw 1km x 1km tile alone.
    assert result_buffered["Total"] > result_raw["Total"]
