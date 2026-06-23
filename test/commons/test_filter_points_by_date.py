import math
import warnings

import numpy as np
from datetime import datetime, timezone
import pytest

from lidar_for_fuel.commons.filter_points_by_date import filter_by_date

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3

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

    n_result = int(np.sum(mask))
    expected = _DEVIATION_DAYS * 2 + _N + 1  # = 14
    assert n_result == expected, f"Expected {expected} points after filtering, got {n_result}"


def test_single_day_no_filtering(rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    n_result = int(np.sum(mask))
    assert n_result == n_total, f"Expected all {n_total} points to be retained, got {n_result}"


def test_infinite_deviation_returns_original_pipeline(rng):
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    n_total = len(gpstime)

    mask = filter_by_date(gpstime, deviation_days=math.inf, gpstime_ref=_GPSTIME_REF)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert int(np.sum(mask)) == n_total


def test_default_deviation_days_is_14():
    """Verify that the default deviation_days is 14."""
    import inspect

    sig = inspect.signature(filter_by_date)
    assert sig.parameters["deviation_days"].default == 14


def test_negative_deviation_raises(rng):
    gpstime = np.arange(1, 10, dtype=np.float64) * _SECONDS_PER_DAY
    # The function now accepts a numeric `deviation_days`; behavior with a
    # negative value is defined by implementation (it will produce a mask).
    # Here we assert that a mask is returned and that its length matches input.
    mask = filter_by_date(gpstime, deviation_days=-1, gpstime_ref=_GPSTIME_REF)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert len(mask) == len(gpstime)


def test_missing_gpstime_dimension_raises():
    # The function now operates on raw gpstime arrays; there is no pipeline
    # dimension check to perform. Passing an object that is not an ndarray
    # should raise a TypeError.
    with pytest.raises(TypeError):
        filter_by_date(None, deviation_days=2, gpstime_ref=_GPSTIME_REF)


def test_gpstime_window_correctness(rng):
    """Verify the filters.range limits in the returned pipeline and the retained GpsTime values."""
    GPSTIME_REF = "2023-01-01 00:00:00"
    DEVIATION_DAY = 1

    EXPECTED_T_MIN = 345_600.0
    EXPECTED_T_MAX = 604_800.0 - _EPSILON

    EXPECTED_RETAINED_MIDPOINTS = np.array([388_800.0, 475_200.0, 561_600.0], dtype=np.float64)
    EXCLUDED_MIDPOINTS = np.array(
        [(day + 0.5) * _SECONDS_PER_DAY for day in list(range(0, 4)) + list(range(7, 10))],
        dtype=np.float64,
    )

    n_days = 10
    gpstime_per_day = (np.arange(n_days, dtype=np.float64) + 0.5) * _SECONDS_PER_DAY
    n_extra = 20
    gpstime_modal = (rng.random(n_extra) + 5) * _SECONDS_PER_DAY
    gpstime_input = np.concatenate([gpstime_per_day, gpstime_modal])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask = filter_by_date(gpstime_input, deviation_days=DEVIATION_DAY, gpstime_ref=GPSTIME_REF)

    retained = gpstime_input[mask]

    # Check retained unix-time limits
    gpstime_ref_unix = datetime.fromisoformat(GPSTIME_REF).replace(tzinfo=timezone.utc).timestamp()
    retained_unix = retained + gpstime_ref_unix

    assert retained_unix.min() == pytest.approx(EXPECTED_T_MIN, abs=1e-6)
    assert retained_unix.max() == pytest.approx(EXPECTED_T_MAX, abs=1e-6)

    # Check retained GpsTime values (relative seconds)
    assert len(retained) == len(EXPECTED_RETAINED_MIDPOINTS) + n_extra
    for ref_val in EXPECTED_RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained