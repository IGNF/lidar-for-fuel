import numpy as np
import pandas as pd

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core
from lidar_for_fuel.pad_profile.create_raster import (
    build_pad_aggregation,
    compute_pixel_aggregates,
)

_GLOBAL_ORIGIN_X = 98029.75
_GLOBAL_ORIGIN_Y = 6045536.75
_RESOLUTION_FACTOR = 10.0
_TILE_SIZE = 20.0  # 2x2 pixels at resolution_factor=10.0

_PAD_PARAMS = dict(
    scanning_angle=False,  # skips the sensor-geometry guard, no realistic trajectory needed here
    limit_N_points=1,
    limit_flight_agl=0.0,
    deviation_days=36_500,  # ~100 years: wide enough to keep every point
    z0=0.0,
    dz=1.0,
    nlayers=2,
    dz_low=0.5,
    nlayers_low=1,
    ground_margin=0.0,
    cover_type="all",
    height_cover=2.0,
    use_cover=False,
    G=0.5,
    omega=1.0,
    keep_values=[2, 3, 4, 5, 9],
    keep_classes=[2, 3, 4, 5, 9],
)


def _pixel_points_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GpsTime": np.full(n, 100.0),
            "X": np.full(n, _GLOBAL_ORIGIN_X + 1.0),
            "Y": np.full(n, _GLOBAL_ORIGIN_Y - 1.0),
            "h_abg": np.linspace(0.5, 3.0, n),
            "Z": np.full(n, 200.0),
            "ReturnNumber": np.full(n, 1.0),
            "Classification": np.full(n, 4.0),
            "X_sensor": np.zeros(n),
            "Y_sensor": np.zeros(n),
            "Z_sensor": np.zeros(n),
        }
    )


def test_build_pad_aggregation_delegates_to_pad_metrics_core():
    """The wrapper's output for one pixel's group matches calling `pad_metrics_core`
    directly on the same columns as arrays -- it's a pure column-to-array adapter."""
    group = _pixel_points_df(5)

    aggregation = build_pad_aggregation(**_PAD_PARAMS)
    result = aggregation(group)

    expected = pad_metrics_core(
        gpstime=group["GpsTime"].to_numpy(dtype=np.float64),
        x=group["X"].to_numpy(dtype=np.float64),
        y=group["Y"].to_numpy(dtype=np.float64),
        h_abg=group["h_abg"].to_numpy(dtype=np.float64),
        z=group["Z"].to_numpy(dtype=np.float64),
        return_number=group["ReturnNumber"].to_numpy(dtype=np.float64),
        classification=group["Classification"].to_numpy(dtype=np.float64),
        x_sensor=group["X_sensor"].to_numpy(dtype=np.float64),
        y_sensor=group["Y_sensor"].to_numpy(dtype=np.float64),
        z_sensor=group["Z_sensor"].to_numpy(dtype=np.float64),
        **_PAD_PARAMS,
    )

    assert result == expected


def test_build_pad_aggregation_returns_none_below_limit_n_points():
    """The wrapper preserves `pad_metrics_core`'s quality guard: too few points in the
    pixel -> None, so `compute_pixel_aggregates` drops that pixel as NoData."""
    group = _pixel_points_df(1)
    aggregation = build_pad_aggregation(**{**_PAD_PARAMS, "limit_N_points": 5})

    assert aggregation(group) is None


def test_compute_pixel_aggregates_with_pad_aggregation_produces_pad_columns():
    """End-to-end: `build_pad_aggregation` plugged into `compute_pixel_aggregates` yields
    a raster DataFrame with one PAD_* column per stratum, for a single real pixel."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _pixel_points_df(5)

    aggregated, _, _ = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=build_pad_aggregation(**_PAD_PARAMS),
    )

    assert (0, 0) in aggregated.index
    pad_columns = [c for c in aggregated.columns if c.startswith("PAD_")]
    assert len(pad_columns) == _PAD_PARAMS["nlayers"] + _PAD_PARAMS["nlayers_low"]
