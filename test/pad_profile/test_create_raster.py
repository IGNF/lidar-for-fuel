from pathlib import Path

import laspy
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


def test_compute_pixel_aggregates_real_buffered_tile_matches_spatial_histogram():
    """Read the real 0691/6484 tile, including its 30 m buffer, without computing PAD.

    The raw 1 km tile spans X=[691000, 692000), Y=[6483000, 6484000).
    Its CosiaFrance window is shifted: X=[690999.75, 691999.75) and
    Y=[6483006.75, 6484006.75).
    It is offset relative to the raw slab:
        - by 0.25 m to the west;
        - by 6.75 m to the north.
    Expected edges and global indices are fixed independently of the production
    function, not derived from its outputs.
    """
    filename = (
        Path(__file__).resolve().parents[2] / "data/pointcloud/Semis_2022_0691_6484_LA93_IGN69_preprocessed_30m.las"
    )
    assert filename.is_file(), f"Missing versioned LAS fixture: {filename}"

    cloud = laspy.read(filename)
    # Lowercase coordinates apply LAS scale/offset; uppercase X/Y are integers
    # in storage units, not Lambert-93 metres.
    points = pd.DataFrame({"X": np.asarray(cloud.x), "Y": np.asarray(cloud.y), "Z": np.asarray(cloud.z)})
    assert len(points) == cloud.header.point_count
    x, y = points["X"].to_numpy(), points["Y"].to_numpy()
    west, east = 690999.75, 691999.75
    south, north = 6483006.75, 6484006.75

    # Identify the points belonging to the aligned window
    in_window = (x >= west) & (x < east) & (y >= south) & (y < north)
    # Identify the points belonging to the raw kilometer tile.
    in_raw_tile = (x >= 691000) & (x < 692000) & (y >= 6483000) & (y < 6484000)

    # Test the filtering:
    # - presence of points west of the window;
    # - presence of points to the east;
    # - presence of points to the south;
    # - presence of points to the north;
    # - presence of points inside the window but outside the raw tile;
    # - presence of points inside the raw tile but outside the window.
    for outside in [x < west, x >= east, y < south, y >= north]:
        assert outside.any(), "Fixture must contain buffer points beyond each window edge"
    assert (in_window & ~in_raw_tile).any(), "Fixture must populate the window outside the raw tile"
    assert (~in_window & in_raw_tile).any(), "Fixture must populate the raw tile outside the window"

    def count_and_sum_z(group):
        return {"count": len(group), "sum_z": group["Z"].sum()}

    result, origin, nb_pixels = compute_pixel_aggregates(
        points,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=691000.0,
        tile_origin_y=6484000.0,
        tile_size=1000.0,
        resolution_factor=10.0,
        aggregation=count_and_sum_z,
    )

    # Construction of references
    bins = [np.linspace(south, north, 101), np.linspace(west, east, 101)]
    # Histogram : Counts the points in each cell.
    counts, _, _ = np.histogram2d(y[in_window], x[in_window], bins=bins)
    # Histogram : Uses elevations as weights: it sums the Z-values ​​of the points in each cell.
    sums, _, _ = np.histogram2d(y[in_window], x[in_window], bins=bins, weights=points["Z"].to_numpy()[in_window])

    rows, cols = np.nonzero(counts)
    expected = pd.DataFrame(
        {"count": counts[rows, cols].astype(int), "sum_z": sums[rows, cols]},
        index=pd.MultiIndex.from_arrays([43748 + rows, 59297 + cols], names=["pixel_y", "pixel_x"]),
    )

    # / ! \ INFO
    # Global pixel indices (10 m resolution):
    # ix = floor((X - X0) / 10)
    # iy = floor((Y - Y0) / 10) + 1
    #
    # For the raw tile corner (691000, 6484000):
    # ix = floor((691000 - 98029.75) / 10)
    #    = floor(59297.025)
    #    = 59297
    #
    # iy = floor((6484000 - 6045536.75) / 10) + 1
    #    = floor(43846.325) + 1
    #    = 43847
    #
    # Expected origin, in (ix, iy) order: (59297, 43847).
    assert origin == (59297, 43847)  # front line

    assert nb_pixels == 100

    # Same occupied pixels, same columns and expected values
    pd.testing.assert_frame_equal(result, expected, check_index_type=False, rtol=1e-12, atol=1e-8)

    assert result["count"].sum() == np.count_nonzero(in_window)  # conservation of the expected total

    assert 0 < result["count"].sum() < len(points)
