import logging
import math
import warnings

import numpy as np
import pytest

from lidar_for_fuel.commons.filter_points_by_date import (
    _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME,
    _GPS_EPOCH,
    filter_by_date,
)

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3

_N = 9
_DEVIATION_DAYS = 2


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_multiday_filters_correct_number_of_points_and_warns(rng):
    gpstime_regular = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    gpstime_extra = (rng.random(_N) + 50) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_regular, gpstime_extra])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS)

    n_result = int(np.sum(mask))
    expected = _DEVIATION_DAYS * 2 + _N + 1  # = 14
    assert n_result == expected, f"Expected {expected} points after filtering, got {n_result}"


def test_single_day_no_filtering(rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = filter_by_date(gpstime, deviation_days=_DEVIATION_DAYS)

    n_result = int(np.sum(mask))
    assert n_result == n_total, f"Expected all {n_total} points to be retained, got {n_result}"


def test_infinite_deviation_returns_original_pipeline(rng):
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    n_total = len(gpstime)

    mask = filter_by_date(gpstime, deviation_days=math.inf)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert int(np.sum(mask)) == n_total


def test_missing_gpstime_dimension_raises():
    with pytest.raises(TypeError):
        filter_by_date(None, deviation_days=2)


def test_gpstime_window_correctness(rng):
    """Verify the filters.range limits in the returned pipeline and the retained GpsTime values."""
    DEVIATION_DAY = 1

    EXPECTED_RETAINED_MIDPOINTS = np.array([388_800.0, 475_200.0, 561_600.0], dtype=np.float64)
    EXCLUDED_MIDPOINTS = np.array(
        [(day + 0.5) * _SECONDS_PER_DAY for day in list(range(0, 4)) + list(range(7, 10))],
        dtype=np.float64,
    )

    # Modal day is day 5 (densest day, see gpstime_modal below). Derive the exact GpsTime
    # window bounds from the same conversion the source uses, instead of hardcoding day*86400
    # boundaries -- the GPS epoch + adjusted-GPS-time offset is not a multiple of one day,
    # so a naive day*86400 boundary does not match the real filtering threshold.
    gpstime_ref_unix = _GPS_EPOCH.timestamp()
    unix_time_modal_day = 5.5 * _SECONDS_PER_DAY + _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime_ref_unix
    modal_day_index = int(unix_time_modal_day // _SECONDS_PER_DAY)
    day_lo = modal_day_index - DEVIATION_DAY
    day_hi = modal_day_index + DEVIATION_DAY
    EXPECTED_T_MIN = (
        max(day_lo, 0) * _SECONDS_PER_DAY - _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME - gpstime_ref_unix - _EPSILON
    )
    EXPECTED_T_MAX = (
        (day_hi + 1) * _SECONDS_PER_DAY - _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME - gpstime_ref_unix + _EPSILON
    )

    n_days = 10
    gpstime_per_day = (np.arange(n_days, dtype=np.float64) + 0.5) * _SECONDS_PER_DAY
    n_extra = 20
    gpstime_modal = (rng.random(n_extra) + 5) * _SECONDS_PER_DAY
    gpstime_input = np.concatenate([gpstime_per_day, gpstime_modal])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask = filter_by_date(gpstime_input, deviation_days=DEVIATION_DAY)

    retained = gpstime_input[mask]

    # Check retained relative-time limits (seconds since gpstime_ref)
    assert np.all(retained >= EXPECTED_T_MIN)
    assert np.all(retained < EXPECTED_T_MAX)

    # Check retained GpsTime values (relative seconds)
    assert len(retained) == len(EXPECTED_RETAINED_MIDPOINTS) + n_extra
    for ref_val in EXPECTED_RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained


def test_logged_gpstime_window_matches_formula(caplog):
    """The debug-logged GpsTime window must be expressed in GpsTime units (LAS adjusted
    GPS time), not raw unix time, with a ±EPSILON margin around the day boundaries."""
    deviation_days = 2
    gpstime = np.full(5, 100.5 * _SECONDS_PER_DAY, dtype=np.float64)

    gpstime_ref_unix = _GPS_EPOCH.timestamp()
    unix_time = gpstime[0] + _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime_ref_unix
    modal_day = int(unix_time // _SECONDS_PER_DAY)
    day_lo = modal_day - deviation_days
    day_hi = modal_day + deviation_days
    expected_t_min = (
        max(day_lo, 0) * _SECONDS_PER_DAY - _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME - gpstime_ref_unix - _EPSILON
    )
    expected_t_max = (
        (day_hi + 1) * _SECONDS_PER_DAY - _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME - gpstime_ref_unix + _EPSILON
    )

    with caplog.at_level(logging.DEBUG, logger="lidar_for_fuel.commons.filter_points_by_date"):
        filter_by_date(gpstime, deviation_days=deviation_days)

    window_records = [r for r in caplog.records if "GpsTime window" in r.msg]
    assert len(window_records) == 1
    logged_modal_day, logged_t_min, logged_t_max, _ = window_records[0].args

    assert logged_modal_day == modal_day
    assert logged_t_min == expected_t_min
    assert logged_t_max == expected_t_max
