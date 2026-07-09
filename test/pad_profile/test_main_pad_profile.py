from pathlib import Path

import numpy as np
import pytest

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile

_REAL_PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
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
        limit_N_points=1,
        limit_flight_agl=0.0,
        deviation_days=np.inf,
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
    )

    assert isinstance(result, dict)
    cos_theta = result["cos_theta"]
    assert isinstance(cos_theta, (float, int)), "Expected a numeric cos_theta value"
    assert 0.0 <= float(cos_theta) <= 1.0
    pad_keys = [key for key in result if key.startswith("PAD_")]
    assert len(pad_keys) == 60 + 4
    for cover_key in ("Cover_2", "Cover_4", "Cover_6"):
        assert 0.0 <= result[cover_key] <= 1.0
