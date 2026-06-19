import math

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.pad_profile.pad_metrics_core import pad_metrics_core
from lidar_for_fuel.pad_profile.pad_output_grid import (
    PIXEL_SIZE,
    REFERENCE_ORIGIN_X,
    REFERENCE_ORIGIN_Y,
)
from lidar_for_fuel.pad_profile.write_pad_rasters import (
    CLASS_COUNT_VALUES,
    write_pad_rasters,
)

_REF = "2011-09-14 01:46:40"

_CLASS_BAND_NAMES = [f"Class_{c}" for c in CLASS_COUNT_VALUES] + ["Total"]


def _make_points(n_per_pixel, origin_x, origin_y, n_rows, n_cols, classification=2, h_abg=1.0, z_sensor=1000.0):
    """Place `n_per_pixel` synthetic points at the center of every pixel of an n_rows x n_cols dalle."""
    xs, ys = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            cx = origin_x + (col + 0.5) * PIXEL_SIZE
            cy = origin_y - (row + 0.5) * PIXEL_SIZE
            for _ in range(n_per_pixel):
                xs.append(cx)
                ys.append(cy)
    n = len(xs)
    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    return dict(
        gpstime=np.zeros(n, dtype=np.float64),
        x=x,
        y=y,
        h_abg=np.full(n, h_abg, dtype=np.float64),
        z=np.zeros(n, dtype=np.float64),
        return_number=np.ones(n, dtype=np.int64),
        classification=np.full(n, classification, dtype=np.int64),
        x_sensor=x.copy(),
        y_sensor=y.copy(),
        z_sensor=np.full(n, z_sensor, dtype=np.float64),
    )


def test_happy_path_writes_6_cogs_with_correct_values(tmp_path):
    n_rows, n_cols = 2, 2
    points = _make_points(5, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows, n_cols, h_abg=1.0)

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=False,
        use_cover=False,
        limit_N_points=0,
    )

    assert set(paths.keys()) == {
        "pad_profile_1m",
        "pad_sb_0.5m",
        "entering_rays",
        "intercept_ray",
        "class_count",
        "COVER",
    }
    for path in paths.values():
        assert path.endswith(".tif")

    # Direct call on one pixel's points to compare against the raster value.
    direct = pad_metrics_core(
        gpstime=points["gpstime"][:5],
        x=points["x"][:5],
        y=points["y"][:5],
        h_abg=points["h_abg"][:5],
        z=points["z"][:5],
        return_number=points["return_number"][:5],
        classification=points["classification"][:5],
        x_sensor=points["x_sensor"][:5],
        y_sensor=points["y_sensor"][:5],
        z_sensor=points["z_sensor"][:5],
        dz=1.0,
        nlayers=60,
        keep_N=True,
        scanning_angle=False,
        use_cover=False,
        limit_N_points=0,
    )
    assert direct is not None

    with rasterio.open(paths["pad_profile_1m"]) as src:
        band_names = list(src.descriptions)
        assert band_names == [f"PAD_1_{i}" for i in range(60)]
        data = src.read()
        assert data.dtype == np.float32
        assert math.isnan(src.nodata)
        # All 4 pixels have identical points -> identical PAD_1_0 value everywhere.
        expected = min(direct["PAD_1_0"], 5.0)
        assert np.allclose(data[0], expected, atol=1e-5)

    with rasterio.open(paths["entering_rays"]) as src:
        assert list(src.descriptions) == [f"N_1_{i}" for i in range(60)]
        data = src.read()
        assert data.dtype == np.int32
        assert src.nodata == -1
        assert np.all(data[0] == direct["N_1_0"])

    with rasterio.open(paths["intercept_ray"]) as src:
        assert list(src.descriptions) == [f"Ni_1_{i}" for i in range(60)]
        data = src.read()
        assert data.dtype == np.int32
        assert np.all(data[0] == direct["Ni_1_0"])

    with rasterio.open(paths["pad_sb_0.5m"]) as src:
        assert list(src.descriptions) == ["PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"]
        data = src.read()
        assert data.dtype == np.float32

    with rasterio.open(paths["COVER"]) as src:
        assert list(src.descriptions) == ["Cover_2", "Cover_4", "Cover_6"]
        data = src.read()
        assert data.dtype == np.float32

    with rasterio.open(paths["class_count"]) as src:
        assert list(src.descriptions) == _CLASS_BAND_NAMES
        data = src.read()
        assert data.dtype == np.int32
        assert src.nodata == -1
        class_2_idx = _CLASS_BAND_NAMES.index("Class_2")
        total_idx = _CLASS_BAND_NAMES.index("Total")
        assert np.all(data[class_2_idx] == 5)
        assert np.all(data[total_idx] == 5)
        other_classes = [i for i in range(len(_CLASS_BAND_NAMES)) if i not in (class_2_idx, total_idx)]
        for idx in other_classes:
            assert np.all(data[idx] == 0)


def test_transform_and_crs_match_reference_grid(tmp_path):
    n_rows, n_cols = 2, 2
    points = _make_points(3, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows, n_cols)

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=False,
        use_cover=False,
        limit_N_points=0,
    )

    for path in paths.values():
        with rasterio.open(path) as src:
            assert src.crs.to_string() in ("EPSG:2154", "epsg:2154")
            assert src.transform.a == PIXEL_SIZE
            assert src.transform.e == -PIXEL_SIZE
            assert src.transform.c == REFERENCE_ORIGIN_X
            assert src.transform.f == REFERENCE_ORIGIN_Y
            assert src.width == n_cols
            assert src.height == n_rows


def test_empty_pixel_gets_nodata_everywhere(tmp_path):
    n_rows, n_cols = 2, 1
    # Only row=0 has points; row=1 is empty.
    points = _make_points(4, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, 1, 1)

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=False,
        use_cover=False,
        limit_N_points=0,
    )

    with rasterio.open(paths["pad_profile_1m"]) as src:
        data = src.read()
        assert np.all(np.isnan(data[:, 1, 0]))
        assert not np.any(np.isnan(data[:, 0, 0]))

    with rasterio.open(paths["entering_rays"]) as src:
        data = src.read()
        assert np.all(data[:, 1, 0] == -1)

    with rasterio.open(paths["class_count"]) as src:
        data = src.read()
        assert np.all(data[:, 1, 0] == -1)

    with rasterio.open(paths["COVER"]) as src:
        data = src.read()
        assert np.all(np.isnan(data[:, 1, 0]))


def test_quality_guard_pixel_keeps_class_count_but_nodata_pad(tmp_path):
    # A single pixel whose points fail the limit_flight_agl quality guard:
    # flight_agl = z_sensor - z = 100, well below limit_flight_agl=800.
    n_rows, n_cols = 1, 1
    points = _make_points(5, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows, n_cols, z_sensor=100.0)
    points["z"] = np.zeros(len(points["x"]))

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=True,
        limit_flight_agl=800.0,
        use_cover=False,
        limit_N_points=0,
    )

    with rasterio.open(paths["pad_profile_1m"]) as src:
        data = src.read()
        assert np.all(np.isnan(data))

    with rasterio.open(paths["pad_sb_0.5m"]) as src:
        data = src.read()
        assert np.all(np.isnan(data))

    with rasterio.open(paths["entering_rays"]) as src:
        data = src.read()
        assert np.all(data == -1)

    with rasterio.open(paths["COVER"]) as src:
        data = src.read()
        assert np.all(np.isnan(data))

    with rasterio.open(paths["class_count"]) as src:
        data = src.read()
        class_2_idx = _CLASS_BAND_NAMES.index("Class_2")
        total_idx = _CLASS_BAND_NAMES.index("Total")
        assert data[class_2_idx, 0, 0] == 5
        assert data[total_idx, 0, 0] == 5


def test_pad_values_above_cap_are_clipped_to_5(tmp_path):
    n_rows, n_cols = 1, 1
    # Dense vegetation points stacked just above the ground margin to drive PAD_1_0 very high.
    points = _make_points(200, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows, n_cols, h_abg=0.15)

    # Prove the test setup actually drives the *uncapped* metric above 5 before asserting the
    # raster clips it -- otherwise this test would still pass if the clipping logic were removed.
    direct = pad_metrics_core(
        gpstime=points["gpstime"],
        x=points["x"],
        y=points["y"],
        h_abg=points["h_abg"],
        z=points["z"],
        return_number=points["return_number"],
        classification=points["classification"],
        x_sensor=points["x_sensor"],
        y_sensor=points["y_sensor"],
        z_sensor=points["z_sensor"],
        dz=1.0,
        nlayers=60,
        keep_N=True,
        scanning_angle=False,
        use_cover=False,
        G=0.01,
        omega=0.01,
        limit_N_points=0,
    )
    assert direct["PAD_1_0"] > 5.0

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=False,
        use_cover=False,
        G=0.01,
        omega=0.01,
        limit_N_points=0,
    )

    with rasterio.open(paths["pad_profile_1m"]) as src:
        data = src.read()
        assert np.all(data <= 5.0)
        assert data[0, 0, 0] == pytest.approx(5.0)


def test_misaligned_dalle_origin_raises_value_error_and_writes_nothing(tmp_path):
    points = _make_points(2, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, 1, 1)

    with pytest.raises(ValueError):
        write_pad_rasters(
            **points,
            origin_x=REFERENCE_ORIGIN_X + 5.0,
            origin_y=REFERENCE_ORIGIN_Y,
            n_rows=1,
            n_cols=1,
            output_dir=str(tmp_path),
            scanning_angle=False,
            use_cover=False,
            limit_N_points=0,
        )

    assert list(tmp_path.iterdir()) == []


def test_mismatched_array_lengths_raises_value_error(tmp_path):
    points = _make_points(2, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, 1, 1)
    points["classification"] = points["classification"][:-1]  # one element short

    with pytest.raises(ValueError):
        write_pad_rasters(
            **points,
            origin_x=REFERENCE_ORIGIN_X,
            origin_y=REFERENCE_ORIGIN_Y,
            n_rows=1,
            n_cols=1,
            output_dir=str(tmp_path),
            scanning_angle=False,
            use_cover=False,
            limit_N_points=0,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reserved_kwarg", ["dz", "nlayers", "keep_N"])
def test_reserved_pad_metrics_kwarg_raises_value_error(tmp_path, reserved_kwarg):
    points = _make_points(2, REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, 1, 1)

    with pytest.raises(ValueError):
        write_pad_rasters(
            **points,
            origin_x=REFERENCE_ORIGIN_X,
            origin_y=REFERENCE_ORIGIN_Y,
            n_rows=1,
            n_cols=1,
            output_dir=str(tmp_path),
            scanning_angle=False,
            use_cover=False,
            limit_N_points=0,
            **{reserved_kwarg: 2},
        )

    assert list(tmp_path.iterdir()) == []


def test_class_count_uses_filter_gpstime_temporal_subset(tmp_path):
    # 2 points in September (kept), 1 point ~200 days later in April (excluded by season_filter=[9]).
    n_rows, n_cols = 1, 1
    cx = REFERENCE_ORIGIN_X + 5.0
    cy = REFERENCE_ORIGIN_Y - 5.0
    n = 3
    points = dict(
        gpstime=np.array([0.0, 0.0, 200.0 * 86400.0]),
        x=np.full(n, cx),
        y=np.full(n, cy),
        h_abg=np.array([0.5, 0.5, 0.5]),
        z=np.zeros(n),
        return_number=np.ones(n, dtype=np.int64),
        classification=np.array([2, 3, 4]),
        x_sensor=np.full(n, cx),
        y_sensor=np.full(n, cy),
        z_sensor=np.full(n, 1000.0),
    )

    paths = write_pad_rasters(
        **points,
        origin_x=REFERENCE_ORIGIN_X,
        origin_y=REFERENCE_ORIGIN_Y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=str(tmp_path),
        scanning_angle=False,
        use_cover=False,
        limit_N_points=0,
        season_filter=[9],
        gpstime_ref=_REF,
    )

    with rasterio.open(paths["class_count"]) as src:
        data = src.read()
        assert data[_CLASS_BAND_NAMES.index("Class_2"), 0, 0] == 1
        assert data[_CLASS_BAND_NAMES.index("Class_3"), 0, 0] == 1
        # Class_4 point was filtered out by the season filter.
        assert data[_CLASS_BAND_NAMES.index("Class_4"), 0, 0] == 0
        assert data[_CLASS_BAND_NAMES.index("Total"), 0, 0] == 2
