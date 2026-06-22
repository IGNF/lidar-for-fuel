import numpy as np
import pytest

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core

_SECONDS_PER_DAY = 86_400.0
_GPSTIME_REF = "2023-01-01 00:00:00"


def _points(n: int, gpstime: np.ndarray) -> dict:
    zeros = np.zeros(n, dtype=np.float64)
    return dict(
        gpstime=gpstime,
        x=zeros.copy(),
        y=zeros.copy(),
        h_abg=zeros.copy(),
        z=zeros.copy(),
        return_number=zeros.copy(),
        classification=zeros.copy(),
        x_sensor=zeros.copy(),
        y_sensor=zeros.copy(),
        z_sensor=np.full(n, 1000.0, dtype=np.float64),  # nonzero flight AGL, avoids 0/0 in compute_cos_theta
    )


def test_date_filter_removes_out_of_window_points_then_limit_n_points_guard_returns_none():
    # 3 points on the modal day, 2 points 10 days away -> excluded by deviation_days=1.
    gpstime = np.array(
        [0.5, 0.6, 0.7, 10.5, 10.6], dtype=np.float64
    ) * _SECONDS_PER_DAY
    points = _points(5, gpstime)

    with pytest.warns(UserWarning, match=r"%\) of the returns were removed"):
        result = pad_metrics_core(
            **points,
            limit_N_points=4,
            deviation_days=1,
            gpstime_ref=_GPSTIME_REF,
        )

    assert result is None


def test_points_within_window_pass_limit_n_points_guard(caplog):
    gpstime = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64) * _SECONDS_PER_DAY
    points = _points(4, gpstime)

    pad_metrics_core(
        **points,
        limit_N_points=4,
        deviation_days=1,
        gpstime_ref=_GPSTIME_REF,
    )

    assert "limit_N_points" not in caplog.text
