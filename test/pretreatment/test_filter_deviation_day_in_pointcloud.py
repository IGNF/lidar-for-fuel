import json
import math
import warnings

import numpy as np
import pdal
import pytest

from lidar_for_fuel.pretreatment.filter_deviation_day import filter_deviation_day

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3

_N = 9
_DEVIATION_DAYS = 2
_GPSTIME_REF = "2023-01-01 00:00:00"


def _make_pipeline(gpstime: np.ndarray, rng: np.random.Generator) -> pdal.Pipeline:
    """Build a minimal executed PDAL Pipeline from a 1-D GpsTime array."""
    n = len(gpstime)
    dtype = np.dtype([("X", np.float64), ("Y", np.float64), ("Z", np.float64), ("GpsTime", np.float64)])
    points = np.zeros(n, dtype=dtype)
    points["X"] = rng.random(n)
    points["Y"] = rng.random(n)
    points["Z"] = rng.random(n)
    points["GpsTime"] = gpstime
    pipeline = pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[points])
    pipeline.execute()
    return pipeline


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def test_multiday_filters_correct_number_of_points_and_warns(rng):
    gpstime_regular = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    gpstime_extra = (rng.random(_N) + 50) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_regular, gpstime_extra])

    pipeline = _make_pipeline(gpstime, rng)

    with pytest.warns(UserWarning, match=r"% of the returns were removed"):
        result_pipeline = filter_deviation_day(pipeline, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    result_pipeline.execute()
    n_result = len(result_pipeline.arrays[0])
    expected = _DEVIATION_DAYS * 2 + _N + 1  # = 14
    assert n_result == expected, f"Expected {expected} points after filtering, got {n_result}"


def test_single_day_no_filtering(rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_pipeline = filter_deviation_day(pipeline, deviation_days=_DEVIATION_DAYS, gpstime_ref=_GPSTIME_REF)

    result_pipeline.execute()
    n_result = len(result_pipeline.arrays[0])
    assert n_result == n_total, f"Expected all {n_total} points to be retained, got {n_result}"


def test_infinite_deviation_returns_original_pipeline(rng):
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)
    n_total = len(pipeline.arrays[0])

    result = filter_deviation_day(pipeline, deviation_days=math.inf, gpstime_ref=_GPSTIME_REF)
    assert result is pipeline
    assert len(result.arrays[0]) == n_total


def test_default_deviation_days_is_14():
    """Verify that the default deviation_days is 14."""
    import inspect

    sig = inspect.signature(filter_deviation_day)
    assert sig.parameters["deviation_days"].default == 14


def test_negative_deviation_raises(rng):
    gpstime = np.arange(1, 10, dtype=np.float64) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)

    with pytest.raises(ValueError, match="deviation_days must be >= 0"):
        filter_deviation_day(pipeline, deviation_days=-1, gpstime_ref=_GPSTIME_REF)


def test_missing_gpstime_dimension_raises():
    n = 10
    dtype = np.dtype([("X", np.float64), ("Y", np.float64), ("Z", np.float64)])
    points = np.zeros(n, dtype=dtype)
    pipeline = pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[points])
    pipeline.execute()

    with pytest.raises(ValueError, match="GpsTime"):
        filter_deviation_day(pipeline, deviation_days=2, gpstime_ref=_GPSTIME_REF)


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
    pipeline = _make_pipeline(gpstime_input, rng)

    with pytest.warns(UserWarning, match=r"% of the returns were removed"):
        result_pipeline = filter_deviation_day(pipeline, deviation_days=DEVIATION_DAY, gpstime_ref=GPSTIME_REF)

    # Check filters.range limits in the PDAL pipeline JSON
    pipeline_spec = json.loads(result_pipeline.pipeline)
    range_stages = [s for s in pipeline_spec["pipeline"] if isinstance(s, dict) and s.get("type") == "filters.range"]
    assert len(range_stages) == 1
    limits_str = range_stages[0]["limits"]
    inner = limits_str[len("GpsTime[") : -1]
    parsed_t_min, parsed_t_max = (float(v) for v in inner.split(":"))

    assert parsed_t_min == pytest.approx(EXPECTED_T_MIN, abs=1e-6)
    assert parsed_t_max == pytest.approx(EXPECTED_T_MAX, abs=1e-6)

    # Check retained GpsTime values
    result_pipeline.execute()
    retained = result_pipeline.arrays[0]["GpsTime"]

    assert len(retained) == len(EXPECTED_RETAINED_MIDPOINTS) + n_extra
    for ref_val in EXPECTED_RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained
