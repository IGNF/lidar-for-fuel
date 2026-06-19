import numpy as np
import pytest
from rasterio.transform import Affine

from lidar_for_fuel.pad_profile.pad_output_grid import (
    PIXEL_SIZE,
    REFERENCE_CRS,
    REFERENCE_ORIGIN_X,
    REFERENCE_ORIGIN_Y,
    bin_points_to_pixels,
    build_dalle_transform,
)


def test_build_dalle_transform_uses_dalle_native_origin():
    # Native LiDAR HD tile origin (round km value), not aligned with the CosiaFrance
    # reference grid -- must NOT raise (recalage is deferred to a separate future step).
    origin_x = 985000.0
    origin_y = 6271000.0

    transform = build_dalle_transform(origin_x, origin_y, n_rows=100, n_cols=100)

    expected = Affine.translation(origin_x, origin_y) * Affine.scale(PIXEL_SIZE, -PIXEL_SIZE)
    assert transform == expected


def test_build_dalle_transform_at_reference_origin_still_works():
    transform = build_dalle_transform(REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows=5, n_cols=5)
    assert transform.a == PIXEL_SIZE
    assert transform.e == -PIXEL_SIZE
    assert transform.c == REFERENCE_ORIGIN_X
    assert transform.f == REFERENCE_ORIGIN_Y


@pytest.mark.parametrize("n_rows,n_cols", [(0, 5), (5, 0), (-1, 5), (5, -1)])
def test_build_dalle_transform_non_positive_shape_raises_value_error(n_rows, n_cols):
    with pytest.raises(ValueError):
        build_dalle_transform(REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows=n_rows, n_cols=n_cols)


def test_bin_points_to_pixels_non_finite_coordinates_map_to_negative_one():
    transform = build_dalle_transform(REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows=3, n_cols=3)
    x = np.array([REFERENCE_ORIGIN_X + 5.0, np.nan, np.inf, -np.inf])
    y = np.array([REFERENCE_ORIGIN_Y - 5.0, REFERENCE_ORIGIN_Y - 5.0, np.nan, REFERENCE_ORIGIN_Y - 5.0])

    rows, cols = bin_points_to_pixels(x, y, transform)

    assert (rows[0], cols[0]) == (0, 0)
    assert rows[1] == -1 and cols[1] == -1
    assert rows[2] == -1 and cols[2] == -1
    assert rows[3] == -1 and cols[3] == -1


def test_bin_points_to_pixels_correctness():
    origin_x = REFERENCE_ORIGIN_X
    origin_y = REFERENCE_ORIGIN_Y
    transform = build_dalle_transform(origin_x, origin_y, n_rows=3, n_cols=3)

    # Pixel (row=0, col=0) covers x in [origin_x, origin_x+10), y in (origin_y-10, origin_y]
    # Pixel (row=1, col=2) covers x in [origin_x+20, origin_x+30), y in (origin_y-20, origin_y-10]
    x = np.array([origin_x + 1.0, origin_x + 25.0, origin_x + 29.9])
    y = np.array([origin_y - 1.0, origin_y - 11.0, origin_y - 19.9])

    rows, cols = bin_points_to_pixels(x, y, transform)

    assert rows.tolist() == [0, 1, 1]
    assert cols.tolist() == [0, 2, 2]


def test_bin_points_to_pixels_returns_integer_dtype():
    transform = build_dalle_transform(REFERENCE_ORIGIN_X, REFERENCE_ORIGIN_Y, n_rows=2, n_cols=2)
    x = np.array([REFERENCE_ORIGIN_X + 5.0])
    y = np.array([REFERENCE_ORIGIN_Y - 5.0])

    rows, cols = bin_points_to_pixels(x, y, transform)

    assert np.issubdtype(rows.dtype, np.integer)
    assert np.issubdtype(cols.dtype, np.integer)


def test_reference_constants():
    assert REFERENCE_ORIGIN_X == pytest.approx(98039.69)
    assert REFERENCE_ORIGIN_Y == pytest.approx(7111486.70)
    assert PIXEL_SIZE == pytest.approx(10.0)
    assert REFERENCE_CRS == "EPSG:2154"
