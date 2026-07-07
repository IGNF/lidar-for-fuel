import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_cover import compute_cover

# h_abg: 8 points, values spread across thresholds -> unambiguous coverage counts at 2/4/6m
_H_ABG = np.array([0.5, 1.5, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0])
_VEG_GND = np.array([True, True, True, True, True, True, False, True])
_RETURN_NUMBER = np.array([1, 2, 1, 1, 1, 2, 1, 1])

# ── cover values for cover_type x use_cover combinations ─────────────────────


@pytest.mark.parametrize(
    "cover_type,use_cover,expected",
    [
        (
            # denom = len(h_abg) = 8; veg_gnd points >2/4/6/height_cover(4)
            "all",
            True,
            (4 / 8, 5 / 8, 4 / 8, 3 / 8),
        ),
        (
            "all",
            False,
            (np.nan, 5 / 8, 4 / 8, 3 / 8),
        ),
        (
            # denom = first returns among all points = 6
            "first",
            True,
            (3 / 6, 4 / 6, 3 / 6, 2 / 6),
        ),
        (
            "first",
            False,
            (np.nan, 4 / 6, 3 / 6, 2 / 6),
        ),
    ],
    ids=[
        "all_use_cover",
        "all_no_use_cover",
        "first_use_cover",
        "first_no_use_cover",
    ],
)
def test_compute_cover(cover_type, use_cover, expected):
    result = compute_cover(
        _H_ABG, _VEG_GND, _RETURN_NUMBER, cover_type=cover_type, height_cover=4.0, use_cover=use_cover
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


# ── invalid cover_type ────────────────────────────────────────────────────────


def test_invalid_cover_type_raises():
    with pytest.raises(ValueError, match="cover_type must be 'all' or 'first'"):
        compute_cover(_H_ABG, _VEG_GND, _RETURN_NUMBER, cover_type="bogus", height_cover=2.0, use_cover=True)


# ── invariants ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cover_type", ["all", "first"])
def test_cover_values_between_zero_and_one(cover_type):
    rng = np.random.default_rng(42)
    h_abg = rng.uniform(0, 20, size=200)
    veg_gnd = rng.integers(0, 2, size=200).astype(bool)
    return_number = rng.integers(1, 4, size=200)

    cover_h_pad, cover_2, cover_4, cover_6 = compute_cover(
        h_abg, veg_gnd, return_number, cover_type=cover_type, height_cover=3.0, use_cover=True
    )
    for value in (cover_h_pad, cover_2, cover_4, cover_6):
        assert 0.0 <= value <= 1.0
