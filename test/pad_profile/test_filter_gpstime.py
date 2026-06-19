from datetime import datetime, timedelta, timezone

import numpy as np

from lidar_for_fuel.pad_profile.filter_gpstime import (
    filter_gpstime,
    gpstime_to_datetime,
    is_in_season,
    is_near_date_mode,
)

_REF = "2011-09-14 01:46:40"
_REF_DT = datetime(2011, 9, 14, 1, 46, 40, tzinfo=timezone.utc)


def _gpstime_at(days_offset: float) -> float:
    return days_offset * 86400.0


def test_gpstime_to_datetime_roundtrip():
    gpstime = np.array([0.0, 86400.0, 3600.5])
    result = gpstime_to_datetime(gpstime, _REF)
    ref_naive = _REF_DT.replace(tzinfo=None)
    expected = np.array(
        [ref_naive, ref_naive + timedelta(days=1), ref_naive + timedelta(hours=1, seconds=0.5)],
        dtype="datetime64[us]",
    )
    assert np.array_equal(result, expected)


def test_is_in_season_full_year_is_noop():
    datetime_arr = gpstime_to_datetime(np.array([_gpstime_at(0), _gpstime_at(100)]), _REF)
    mask = is_in_season(datetime_arr, months=range(1, 13))
    assert mask.tolist() == [True, True]


def test_is_in_season_filters_by_month():
    # _REF is in September (month=9); +200 days lands in April.
    datetime_arr = gpstime_to_datetime(np.array([_gpstime_at(0), _gpstime_at(200)]), _REF)
    mask = is_in_season(datetime_arr, months=[9])
    assert mask.tolist() == [True, False]


def test_is_near_date_mode_infinite_deviation_is_noop():
    datetime_arr = gpstime_to_datetime(np.array([_gpstime_at(0), _gpstime_at(50)]), _REF)
    mask = is_near_date_mode(datetime_arr, deviation_days=np.inf)
    assert mask.tolist() == [True, True]


def test_is_near_date_mode_keeps_only_window_around_modal_day():
    # Day 0 has 3 points (the mode), day 10 has 1 point.
    offsets = [0, 0, 0, 10]
    datetime_arr = gpstime_to_datetime(np.array([_gpstime_at(d) for d in offsets]), _REF)
    mask = is_near_date_mode(datetime_arr, deviation_days=1)
    assert mask.tolist() == [True, True, True, False]


def test_is_near_date_mode_tie_break_uses_first_occurrence():
    # Day 5 and day 0 both have 2 points; day 0 appears first in the array.
    offsets = [0, 5, 0, 5]
    datetime_arr = gpstime_to_datetime(np.array([_gpstime_at(d) for d in offsets]), _REF)
    mask = is_near_date_mode(datetime_arr, deviation_days=0)
    # Modal day must resolve to day 0 (first occurrence) -> only the two day-0 points survive.
    assert mask.tolist() == [True, False, True, False]


def test_filter_gpstime_combines_season_then_local_date_mode():
    # Two points in September (day 0, day 0): mode. One point in April (out of season).
    offsets = [0, 0, 200]
    gpstime = np.array([_gpstime_at(d) for d in offsets])
    mask = filter_gpstime(gpstime, months=[9], deviation_days=0, gpstime_ref=_REF)
    assert mask.tolist() == [True, True, False]


def test_filter_gpstime_all_points_out_of_season_returns_all_false():
    gpstime = np.array([_gpstime_at(200), _gpstime_at(201)])
    mask = filter_gpstime(gpstime, months=[9], deviation_days=np.inf, gpstime_ref=_REF)
    assert mask.tolist() == [False, False]
