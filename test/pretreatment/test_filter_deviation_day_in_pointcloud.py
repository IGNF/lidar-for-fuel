import json
import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pdal
import pytest

from lidar_for_fuel.pretreatment.filter_deviation_day import (
    compute_gpstime_window,
    filter_points_by_date,
    gpstime_to_day_index,
    modal_day,
)

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3

_N = 9
_DEVIATION_DAYS = 2
_GPSTIME_REF_ISO = "2023-01-01 00:00:00"


def _gpstime_ref_unix(iso: str) -> float:
    """Convert an ISO-8601 UTC string to a UNIX timestamp."""
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return dt.timestamp()


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
def gpstime_ref_unix() -> float:
    return _gpstime_ref_unix(_GPSTIME_REF_ISO)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestGpstimeToDataIndex:
    """
    Reference derivation (ref = 0, i.e. GPS epoch = UNIX epoch)
    ------------------------------------------------------------
    GpsTime  |  UNIX time (s)  |  UNIX day (floor)
    ---------|-----------------|------------------
      0.0    |     0.0         |  0   (midnight day 0)
      0.5d   |  43 200.0       |  0   (noon day 0)
      1.0d   |  86 400.0       |  1   (midnight day 1)
      1.9d   | 164 160.0       |  1   (end of day 1)
      2.0d   | 172 800.0       |  2   (midnight day 2)

    With ref = 1 day (86 400 s):
    GpsTime = 0.5d → UNIX = 1.5d → day index = 1
    GpsTime = -0.5d (before ref epoch) → UNIX = 0.5d → day index = 0
    """

    def test_midday_maps_to_correct_day_zero_ref(self):
        gpstime = np.array([0.5, 1.5, 2.5]) * _SECONDS_PER_DAY
        result = gpstime_to_day_index(gpstime, gpstime_ref_unix=0.0)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_midnight_boundary_floors_to_new_day(self):
        gpstime = np.array([1.0]) * _SECONDS_PER_DAY
        result = gpstime_to_day_index(gpstime, gpstime_ref_unix=0.0)
        np.testing.assert_array_equal(result, [1])

    def test_nonzero_ref_shifts_day_correctly(self):
        ref = 1_673_308_800.0
        gpstime = np.array([0.5 * _SECONDS_PER_DAY])
        result = gpstime_to_day_index(gpstime, gpstime_ref_unix=ref)
        np.testing.assert_array_equal(result, [19_367])

    def test_multiple_points_same_day(self):
        gpstime = np.array([0.0, 1000.0, 43_200.0, 86_399.999])
        result = gpstime_to_day_index(gpstime, gpstime_ref_unix=0.0)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])


class TestModalDay:
    def test_clear_modal_day(self):
        assert modal_day(np.array([10, 10, 10, 11, 12])) == 10

    def test_modal_day_not_first_in_array(self):
        assert modal_day(np.array([5, 5, 6, 6, 6, 7])) == 6

    def test_single_point(self):
        assert modal_day(np.array([42])) == 42

    def test_tie_returns_smallest_day(self):
        assert modal_day(np.array([3, 3, 7, 7])) == 3


class TestComputeGpstimeWindow:
    def test_symmetric_window_zero_ref(self):
        t_min, t_max = compute_gpstime_window(main_day=5, deviation_day=1, gpstime_ref_unix=0.0)
        assert t_min == pytest.approx(345_600.0)
        assert t_max == pytest.approx(604_800.0 - _EPSILON)

    def test_symmetric_window_nonzero_ref(self):
        ref = 1_673_308_800.0
        t_min, t_max = compute_gpstime_window(main_day=19_372, deviation_day=1, gpstime_ref_unix=ref)
        assert t_min == pytest.approx(345_600.0)
        assert t_max == pytest.approx(604_800.0 - _EPSILON)

    def test_zero_deviation_covers_only_modal_day(self):
        t_min, t_max = compute_gpstime_window(main_day=5, deviation_day=0, gpstime_ref_unix=0.0)
        assert t_min == pytest.approx(5 * _SECONDS_PER_DAY)
        assert t_max == pytest.approx(6 * _SECONDS_PER_DAY - _EPSILON)

    def test_tmax_strictly_excludes_next_day_midnight(self):
        t_min, t_max = compute_gpstime_window(main_day=5, deviation_day=2, gpstime_ref_unix=0.0)
        day_hi_plus_one_midnight = (5 + 2 + 1) * _SECONDS_PER_DAY
        assert t_max == pytest.approx(day_hi_plus_one_midnight - _EPSILON)

    def test_day_lo_clamps_to_zero(self):
        t_min, _ = compute_gpstime_window(main_day=1, deviation_day=5, gpstime_ref_unix=0.0)
        assert t_min == pytest.approx(0.0)


def test_multiday_filters_correct_number_of_points_and_warns(gpstime_ref_unix, rng):
    gpstime_regular = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    gpstime_extra = (rng.random(_N) + 50) * _SECONDS_PER_DAY
    gpstime = np.concatenate([gpstime_regular, gpstime_extra])

    pipeline = _make_pipeline(gpstime, rng)

    with pytest.warns(UserWarning, match=r"% of the returns were removed"):
        result_pipeline = filter_points_by_date(
            pipeline,
            deviation_day=_DEVIATION_DAYS,
            gpstime_ref_unix=gpstime_ref_unix,
        )

    result_pipeline.execute()
    n_result = len(result_pipeline.arrays[0])
    expected = _DEVIATION_DAYS * 2 + _N + 1  # = 14
    assert n_result == expected, f"Expected {expected} points after filtering, got {n_result}"


def test_single_day_no_filtering(gpstime_ref_unix, rng):
    n_total = 100 + _N
    gpstime = rng.random(n_total) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result_pipeline = filter_points_by_date(
            pipeline,
            deviation_day=_DEVIATION_DAYS,
            gpstime_ref_unix=gpstime_ref_unix,
        )

    result_pipeline.execute()
    n_result = len(result_pipeline.arrays[0])
    assert n_result == n_total, f"Expected all {n_total} points to be retained, got {n_result}"


def test_infinite_deviation_returns_all_points(gpstime_ref_unix, rng):
    gpstime = np.arange(1, 101, dtype=np.float64) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)
    n_total = len(pipeline.arrays[0])

    result = filter_points_by_date(pipeline, deviation_day=math.inf, gpstime_ref_unix=gpstime_ref_unix)
    assert result is pipeline
    assert len(result.arrays[0]) == n_total


def test_negative_deviation_raises(gpstime_ref_unix, rng):
    gpstime = np.arange(1, 10, dtype=np.float64) * _SECONDS_PER_DAY
    pipeline = _make_pipeline(gpstime, rng)

    with pytest.raises(ValueError, match="deviation_day must be >= 0"):
        filter_points_by_date(pipeline, deviation_day=-1, gpstime_ref_unix=gpstime_ref_unix)


def test_missing_gpstime_dimension_raises(gpstime_ref_unix):
    n = 10
    dtype = np.dtype([("X", np.float64), ("Y", np.float64), ("Z", np.float64)])
    points = np.zeros(n, dtype=dtype)
    pipeline = pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[points])
    pipeline.execute()

    with pytest.raises(ValueError, match="GpsTime"):
        filter_points_by_date(pipeline, deviation_day=2, gpstime_ref_unix=gpstime_ref_unix)


def test_gpstime_window_correctness(rng):
    """
    Verify two things simultaneously:
      A) the ``filters.range`` limits in the returned PDAL pipeline JSON
         match the expected t_min / t_max computed by hand;
      B) the GpsTime values of the deterministic retained points match
         a hardcoded reference list exactly.

    """
    GPSTIME_REF_UNIX = 1_673_308_800.0
    DEVIATION_DAY = 1

    EXPECTED_T_MIN = 345_600.0
    EXPECTED_T_MAX = 604_800.0 - _EPSILON

    EXPECTED_RETAINED_MIDPOINTS = np.array([388_800.0, 475_200.0, 561_600.0], dtype=np.float64)
    EXCLUDED_MIDPOINTS = np.array(
        [(day + 0.5) * _SECONDS_PER_DAY for day in list(range(0, 4)) + list(range(7, 10))],
        dtype=np.float64,
    )

    # Build input
    n_days = 10
    gpstime_per_day = (np.arange(n_days, dtype=np.float64) + 0.5) * _SECONDS_PER_DAY
    n_extra = 20
    gpstime_modal = (rng.random(n_extra) + 5) * _SECONDS_PER_DAY
    gpstime_input = np.concatenate([gpstime_per_day, gpstime_modal])
    pipeline = _make_pipeline(gpstime_input, rng)

    with pytest.warns(UserWarning, match=r"% of the returns were removed"):
        result_pipeline = filter_points_by_date(
            pipeline, deviation_day=DEVIATION_DAY, gpstime_ref_unix=GPSTIME_REF_UNIX
        )

    # --- Check A: PDAL pipeline limits ---
    pipeline_spec = json.loads(result_pipeline.pipeline)
    range_stages = [s for s in pipeline_spec["pipeline"] if isinstance(s, dict) and s.get("type") == "filters.range"]
    assert len(range_stages) == 1, (
        f"Expected exactly one filters.range stage, found {len(range_stages)}.\n"
        f"Full pipeline: {json.dumps(pipeline_spec, indent=2)}"
    )
    limits_str = range_stages[0]["limits"]
    prefix_len = len("GpsTime[")
    inner = limits_str[prefix_len:-1]
    parsed_t_min, parsed_t_max = (float(v) for v in inner.split(":"))

    assert parsed_t_min == pytest.approx(
        EXPECTED_T_MIN, abs=1e-6
    ), f"t_min: got {parsed_t_min}, expected {EXPECTED_T_MIN}"
    assert parsed_t_max == pytest.approx(
        EXPECTED_T_MAX, abs=1e-6
    ), f"t_max: got {parsed_t_max}, expected {EXPECTED_T_MAX}"

    # --- Check B: retained GpsTime values vs hardcoded reference ---
    result_pipeline.execute()
    retained = result_pipeline.arrays[0]["GpsTime"]

    assert (
        len(retained) == len(EXPECTED_RETAINED_MIDPOINTS) + n_extra
    ), f"Expected {len(EXPECTED_RETAINED_MIDPOINTS) + n_extra} points, got {len(retained)}"
    for ref_val in EXPECTED_RETAINED_MIDPOINTS:
        assert np.sum(retained == ref_val) == 1, f"GpsTime={ref_val} should appear exactly once in retained points."
    for excl_val in EXCLUDED_MIDPOINTS:
        assert excl_val not in retained, f"GpsTime={excl_val} (excluded day) must not appear in retained points."
