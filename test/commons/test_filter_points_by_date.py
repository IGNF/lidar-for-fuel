import math
import warnings

import numpy as np
import pytest

from lidar_for_fuel.commons.filter_points_by_date import filter_by_date

_SECONDS_PER_DAY = 86_400.0

_N = 9
_DEVIATION_DAYS = 2
_GPSTIME_REF = "2023-01-01 00:00:00"


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_multiday_filters_correct_number_of_points_and_warns(rng):
    gpstime_regular = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    gpstime_extra = (rng.random(_N) + 50) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_regular, gpstime_extra])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    expected = _DEVIATION_DAYS * 2 + _N + 1  # = 14
    assert mask.dtype == bool
    assert mask.shape == gpstime.shape
    assert int(mask.sum()) == expected, f"Expected {expected} points retained, got {int(mask.sum())}"


def test_single_day_no_filtering(rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    assert mask.all()
    assert int(mask.sum()) == n_total


def test_infinite_deviation_returns_all_true(rng):
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY

    mask = filter_by_date(gpstime, deviation_days=math.inf, gpstime_ref=_GPSTIME_REF)

    assert mask.dtype == bool
    assert mask.all()
    assert mask.shape == gpstime.shape


def test_default_deviation_days_is_14():
    """Verify that the default deviation_days is 14."""
    import inspect

    sig = inspect.signature(filter_by_date)
    assert sig.parameters["deviation_days"].default == 14


def test_negative_deviation_raises():
    gpstime = np.arange(1, 10, dtype=np.float64) * _SECONDS_PER_DAY

    with pytest.raises(ValueError, match="deviation_days must be >= 0"):
        filter_by_date(gpstime, deviation_days=-1, gpstime_ref=_GPSTIME_REF)


def test_empty_array_returns_empty_mask():
    gpstime = np.zeros(0, dtype=np.float64)

    mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_window_correctness(rng):
    """Verify the exact retained/excluded GpsTime values around the modal day."""
    GPSTIME_REF = "2023-01-01 00:00:00"
    DEVIATION_DAY = 1

    RETAINED_MIDPOINTS = np.array([388_800.0, 475_200.0, 561_600.0], dtype=np.float64)
    EXCLUDED_MIDPOINTS = np.array(
        [(day + 0.5) * _SECONDS_PER_DAY for day in list(range(0, 4)) + list(range(7, 10))],
        dtype=np.float64,
    )

    n_days = 10
    gpstime_per_day = (np.arange(n_days, dtype=np.float64) + 0.5) * _SECONDS_PER_DAY
    n_extra = 20
    gpstime_modal = (rng.random(n_extra) + 5) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_per_day, gpstime_modal])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask = filter_by_date(gpstime, deviation_days=DEVIATION_DAY, gpstime_ref=GPSTIME_REF)

    retained = gpstime[mask]
    assert len(retained) == len(RETAINED_MIDPOINTS) + n_extra
    for ref_val in RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained
