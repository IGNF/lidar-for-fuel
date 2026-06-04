"""Tests for compute_Nz_U — vertical component of the unit pulse vector."""
from pathlib import Path

import laspy
import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_Nz_U import compute_Nz_U

_PREPROCESSING_LAS= Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_points(n: int, flight_agl: float = 1500.0, horizontal_offset: float = 0.0):
    """Return synthetic arrays for n points.

    All points lie at (0, 0) on the ground; the sensor is placed at
    (horizontal_offset, 0, flight_agl) relative to each point.
    """
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    h_abg = np.zeros(n, dtype=np.float64)
    x_sensor = x + horizontal_offset
    y_sensor = np.zeros(n, dtype=np.float64)
    z_sensor = h_abg + flight_agl
    return x, y, h_abg, x_sensor, y_sensor, z_sensor


# ── unit tests ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n, flight_agl, horizontal_offset, scanning_angle, limit_flight_agl, check", [
    pytest.param(
        10, 500.0, 0.0, False, 800.0,
        lambda r: r == 1.0,
        id="scanning_angle_false",
    ),
    pytest.param(
        10, 500.0, 0.0, True, 800.0,
        lambda r: r is None,
        id="low_agl_returns_none",
    ),
    pytest.param(
        50, 1500.0, 0.0, True, 800.0,
        lambda r: r is not None and r.shape == (50,) and np.allclose(r, 1.0, atol=1e-10),
        id="vertical_pulses",
    ),
    pytest.param(
        100, 1000.0, 1000.0, True, 800.0,
        lambda r: r is not None and np.allclose(r, 1.0 / np.sqrt(2), atol=1e-10),
        id="oblique_45deg",
    ),
    pytest.param(
        37, 1500.0, 0.0, True, 800.0,
        lambda r: r is not None and isinstance(r, np.ndarray) and r.shape == (37,),
        id="output_shape",
    ),
    pytest.param(
        100, 1500.0, 500.0, True, 800.0,
        lambda r: r is not None and bool(np.all(r > 0.0) and np.all(r <= 1.0)),
        id="values_in_0_1",
    ),
])
def test_compute_Nz_U(n, flight_agl, horizontal_offset, scanning_angle, limit_flight_agl, check):
    x, y, h_abg, x_sensor, y_sensor, z_sensor = _make_points(n, flight_agl, horizontal_offset)
    result = compute_Nz_U(x, y, h_abg, x_sensor, y_sensor, z_sensor, scanning_angle, limit_flight_agl)
    assert check(result)


# ── integration test on real preprocessed data ───────────────────────────────

def test_compute_Nz_U_on_real_data():
    """On real preprocessed data, all Nz_U values must be in (0, 1]."""
    if not _PREPROCESSING_LAS.exists():
        pytest.skip(f"Real data not available: {_PREPROCESSING_LAS}")

    las = laspy.read(str(_PREPROCESSING_LAS))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    h_abg = np.asarray(las["h_abg"], dtype=np.float64)
    x_sensor = np.asarray(las["X_sensor"], dtype=np.float64)
    y_sensor = np.asarray(las["Y_sensor"], dtype=np.float64)
    z_sensor = np.asarray(las["Z_sensor"], dtype=np.float64)

    result = compute_Nz_U(x, y, h_abg, x_sensor, y_sensor, z_sensor)

    assert result is not None, "Nz_U should not be None for valid real data."
    assert result.shape == x.shape
    assert np.all(result > 0.0) and np.all(result <= 1.0), (
        f"Nz_U values out of (0, 1]: min={result.min():.4f}, max={result.max():.4f}"
    )
