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
        ground_margin=0.1,
    )

    cos_theta, ni, n, min_layer, nrd = result
    assert isinstance(cos_theta, (float, int)), "Expected a numeric cos_theta value"
    assert 0.0 <= float(cos_theta) <= 1.0
    assert len(ni) == len(n) == len(min_layer) == len(nrd)
