import numpy as np
import pandas as pd
import pytest

from lidar_for_fuel.pad_profile.create_raster import compute_pixel_aggregates

_GLOBAL_ORIGIN_X = 98029.75
_GLOBAL_ORIGIN_Y = 6045536.75
_RESOLUTION_FACTOR = 10.0
_TILE_SIZE = 20.0  # 2x2 pixels at resolution_factor=10.0


def _points_df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def test_compute_pixel_aggregates_returns_dataframe_and_origin_pixel():
    """Points inside the tile window are grouped by pixel and aggregated; `origin_pixel`
    and `nb_pixels` are derived purely from the tile's own corner and size, not from the
    points themselves."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df(
        [
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 5.0},  # pixel (0, 0)
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 9.0},  # pixel (0, 0)
            {"X": tile_origin_x + 15.0, "Y": tile_origin_y + 5.0, "h_abg": 3.0},  # pixel (1, 1)
        ]
    )

    aggregated, origin_pixel, nb_pixels = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=lambda group: group["h_abg"].max(),
    )

    assert isinstance(aggregated, pd.DataFrame)
    assert origin_pixel == (0, 1)
    assert nb_pixels == 2.0
    assert list(aggregated.columns) == ["value"]
    # First two points share pixel (pixel_y=0, pixel_x=0) -> max is kept.
    assert aggregated.loc[(0, 0), "value"] == 9.0
    # Third point lands in the neighbouring pixel (pixel_y=1, pixel_x=1).
    assert aggregated.loc[(1, 1), "value"] == 3.0


def test_compute_pixel_aggregates_excludes_points_outside_tile_window():
    """Points in the buffer (outside the tile's own window) are dropped before
    aggregation, even though they still get a valid pixel index."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df(
        [
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 5.0},  # inside: pixel (0, 0)
            {"X": tile_origin_x - 50.0, "Y": tile_origin_y - 1.0, "h_abg": 99.0},  # buffer, outside
        ]
    )

    aggregated, _, _ = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=lambda group: group["h_abg"].max(),
    )

    assert len(aggregated) == 1
    assert 99.0 not in aggregated["value"].to_numpy()


def test_compute_pixel_aggregates_dict_aggregation_produces_one_column_per_key():
    """When `aggregation` returns a dict, each key becomes its own output column
    (multi-band case, e.g. the PAD profile calculation)."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df(
        [
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 5.0},  # pixel (0, 0)
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 9.0},  # pixel (0, 0)
        ]
    )

    def multi_band(group: pd.DataFrame) -> dict:
        return {"max_h": group["h_abg"].max(), "count": len(group)}

    aggregated, _, _ = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=multi_band,
    )

    assert list(aggregated.columns) == ["max_h", "count"]
    assert aggregated.loc[(0, 0), "max_h"] == 9.0
    assert aggregated.loc[(0, 0), "count"] == 2


def test_compute_pixel_aggregates_drops_pixels_where_aggregation_returns_none():
    """A pixel whose `aggregation` call returns None (quality guard, e.g. too few
    points) is dropped from the output rather than kept as a row of NaNs."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df(
        [
            {"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 5.0},  # pixel (0, 0): 1 point
            {"X": tile_origin_x + 15.0, "Y": tile_origin_y + 5.0, "h_abg": 3.0},  # pixel (1, 1): 1st point
            {"X": tile_origin_x + 15.0, "Y": tile_origin_y + 5.0, "h_abg": 4.0},  # pixel (1, 1): 2nd point
        ]
    )

    def require_two_points(group: pd.DataFrame):
        if len(group) < 2:
            return None
        return group["h_abg"].max()

    aggregated, _, _ = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=require_two_points,
    )

    assert len(aggregated) == 1
    assert (0, 0) not in aggregated.index
    assert aggregated.loc[(1, 1), "value"] == 4.0


def test_compute_pixel_aggregates_raises_when_no_point_in_tile_window():
    """Every point falling outside the tile's window (e.g. only buffer points loaded)
    is a data/config error, not a silently empty result."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df([{"X": tile_origin_x - 50.0, "Y": tile_origin_y - 1.0, "h_abg": 1.0}])

    with pytest.raises(ValueError, match="No point of the tile"):
        compute_pixel_aggregates(
            points_df,
            global_origin_x=_GLOBAL_ORIGIN_X,
            global_origin_y=_GLOBAL_ORIGIN_Y,
            tile_origin_x=tile_origin_x,
            tile_origin_y=tile_origin_y,
            tile_size=_TILE_SIZE,
            resolution_factor=_RESOLUTION_FACTOR,
            aggregation=lambda group: group["h_abg"].max(),
        )


def test_compute_pixel_aggregates_raises_when_aggregation_returns_none_everywhere():
    """If `aggregation` returns None for every pixel, that is surfaced as an error
    rather than an empty DataFrame."""
    tile_origin_x = _GLOBAL_ORIGIN_X
    tile_origin_y = _GLOBAL_ORIGIN_Y
    points_df = _points_df([{"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 1.0}])

    with pytest.raises(ValueError, match="every pixel failed"):
        compute_pixel_aggregates(
            points_df,
            global_origin_x=_GLOBAL_ORIGIN_X,
            global_origin_y=_GLOBAL_ORIGIN_Y,
            tile_origin_x=tile_origin_x,
            tile_origin_y=tile_origin_y,
            tile_size=_TILE_SIZE,
            resolution_factor=_RESOLUTION_FACTOR,
            aggregation=lambda group: None,
        )


def test_compute_pixel_aggregates_origin_pixel_anchored_on_tile_not_global_origin():
    """`origin_pixel` is the CosiaFrance grid corner nearest the tile's own corner, not
    the global grid origin itself -- it shifts by whole pixels as `tile_origin_*` moves."""
    tile_origin_x = _GLOBAL_ORIGIN_X + 35.0  # 3.5 pixels east -> floor to pixel 3
    tile_origin_y = _GLOBAL_ORIGIN_Y - 25.0  # 2.5 pixels south -> pixel index shifts to -2
    points_df = _points_df([{"X": tile_origin_x + 1.0, "Y": tile_origin_y - 1.0, "h_abg": 1.0}])

    _, origin_pixel, nb_pixels = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=tile_origin_x,
        tile_origin_y=tile_origin_y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=lambda group: group["h_abg"].max(),
    )

    assert origin_pixel == (3, -2)
    assert nb_pixels == 2.0


