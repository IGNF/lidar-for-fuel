import logging
import warnings

import numpy as np
import pytest

from lidar_for_fuel.commons.filter_points_by_date import (
    _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME,
    _GPS_EPOCH,
    filter_by_date,
)

_SECONDS_PER_DAY = 86_400.0
_N = 9



@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_multiday_filters_correct_number_of_points_and_warns(rng):
    gpstime_regular = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    gpstime_extra = (rng.random(_N) + 50) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_regular, gpstime_extra])

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask, modal_time_unix = filter_by_date(gpstime, deviation_days=2)

    n_result = int(np.sum(mask))
    expected = 2 * 2 + _N + 1  # = 14
    assert n_result == expected, f"Expected {expected} points after filtering, got {n_result}"

    # Modal day is day 50 by construction: gpstime_regular[49] == 50 * _SECONDS_PER_DAY,
    # and all 9 gpstime_extra points fall in the same calendar day (see above), giving
    # it 10 points against 1 for every other day.
    modal_date = np.array(
        _GPS_EPOCH
        + np.array(_ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime_regular[49], dtype="timedelta64[s]"),
        dtype="datetime64[D]",
    )
    expected_modal_time_unix = int((modal_date - np.datetime64(0, "s")) / np.timedelta64(1, "s"))
    assert modal_time_unix == expected_modal_time_unix


def test_single_day_no_filtering(rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask, _ = filter_by_date(gpstime, deviation_days=2)

    n_result = int(np.sum(mask))
    assert n_result == n_total, f"Expected all {n_total} points to be retained, got {n_result}"


def test_large2_returns_original_pipeline(rng):
    """A window wide enough to cover the whole span behaves like no filtering."""
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    n_total = len(gpstime)

    mask, modal_time_unix = filter_by_date(gpstime, deviation_days=100)
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    assert int(np.sum(mask)) == n_total
    assert isinstance(modal_time_unix, int)


def test_missing_gpstime_dimension_raises():
    with pytest.raises(TypeError):
        filter_by_date(None, deviation_days=2)


def test_gpstime_window_correctness(rng):
    """Verify the calendar-date window limits and the retained GpsTime values."""
    DEVIATION_DAY = 1

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

    # Modal day is day 5 (densest day, see gpstime_modal above): expected window is
    # [day 4, day 6] inclusive. Convert through the same calendar-date logic as the
    # source instead of hardcoding day*86400 boundaries.
    utcdate = np.array(
        _GPS_EPOCH + np.array(_ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime_input, dtype="timedelta64[s]"),
        dtype="datetime64[D]",
    )
    modal_date = utcdate[5]
    EXPECTED_DATE_MIN = modal_date - np.timedelta64(DEVIATION_DAY, "D")
    EXPECTED_DATE_MAX = modal_date + np.timedelta64(DEVIATION_DAY, "D")
    EXPECTED_MODAL_TIME_UNIX = int((modal_date - np.datetime64(0, "s")) / np.timedelta64(1, "s"))

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        mask, modal_time_unix = filter_by_date(gpstime_input, deviation_days=DEVIATION_DAY)

    retained = gpstime_input[mask]
    retained_dates = utcdate[mask]

    # Check retained calendar-date limits
    assert np.all(retained_dates >= EXPECTED_DATE_MIN)
    assert np.all(retained_dates <= EXPECTED_DATE_MAX)

    # Check the returned modal time matches the modal calendar day, in Unix time
    assert modal_time_unix == pytest.approx(EXPECTED_MODAL_TIME_UNIX)

    # Check retained GpsTime values
    assert len(retained) == len(EXPECTED_RETAINED_MIDPOINTS) + n_extra
    for ref_val in EXPECTED_RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained


def test_logged_date_window_matches_modal_day(caplog):
    """The debug-logged date window must be the modal calendar day ± deviation_days."""
    deviation_days = 2
    gpstime = np.full(5, 100.5 * _SECONDS_PER_DAY, dtype=np.float64)

    utctime = _GPS_EPOCH + np.array(_ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime, dtype="timedelta64[s]")
    modal_date = np.array(utctime, dtype="datetime64[D]")[0]
    window = np.timedelta64(deviation_days, "D")

    with caplog.at_level(logging.DEBUG, logger="lidar_for_fuel.commons.filter_points_by_date"):
        filter_by_date(gpstime, deviation_days=deviation_days)

    window_records = [r for r in caplog.records if "Date window" in r.msg]
    assert len(window_records) == 1
    logged_modal_day, logged_lo, logged_hi, _ = window_records[0].args

    assert logged_modal_day == modal_date
    assert logged_lo == modal_date - window
    assert logged_hi == modal_date + window
