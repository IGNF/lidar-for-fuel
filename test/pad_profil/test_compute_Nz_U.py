"""Tests for compute_Nz_U — vertical component of the unit pulse vector."""
from pathlib import Path

import laspy
import numpy as np
import pytest

from lidar_for_fuel.pad_profil.compute_Nz_U import Nz_U

_PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_points(n: int, flight_agl: float = 1500.0, horizontal_offset: float = 0.0):
    """Return synthetic arrays for n points.

    All points lie at (0, 0) on the ground; the sensor is placed at
    (horizontal_offset, 0, flight_agl) relative to each point.
    """
    X = np.zeros(n, dtype=np.float64)
    Y = np.zeros(n, dtype=np.float64)
    h_abg = np.zeros(n, dtype=np.float64)
    X_sensor = X + horizontal_offset
    Y_sensor = np.zeros(n, dtype=np.float64)
    Z_sensor = h_abg + flight_agl
    return X, Y, h_abg, X_sensor, Y_sensor, Z_sensor


# ── unit tests ────────────────────────────────────────────────────────────────

def test_compute_Nz_U_scanning_angle_false_returns_one():
    """Disabling scanning_angle must always return 1.0 regardless of geometry."""
    X, Y, h_abg, E, N, Elev = _make_points(10, flight_agl=500.0)
    assert Nz_U(X, Y, h_abg, E, N, Elev, scanning_angle=False) == 1.0


def test_compute_Nz_U_low_flight_height_returns_none():
    """Mean flight height below limit_flight_agl must return None."""
    X, Y, h_abg, E, N, Elev = _make_points(10, flight_agl=500.0)
    result = Nz_U(X, Y, h_abg, E, N, Elev, limit_flight_agl=800.0)
    assert result is None


def test_compute_Nz_U_vertical_pulses_is_one():
    """Sensor directly above each point (horizontal_offset=0) → Nz_U = 1 for all points."""
    X, Y, h_abg, E, N, Elev = _make_points(50, flight_agl=1500.0, horizontal_offset=0.0)
    result = Nz_U(X, Y, h_abg, E, N, Elev)
    assert result is not None
    assert result.shape == (50,)
    np.testing.assert_almost_equal(result, 1.0, decimal=10)


def test_compute_Nz_U_oblique_angle_45_degrees():
    """flight_agl == horizontal_dist → scan angle = 45° → Nz_U = 1/sqrt(2)."""
    X, Y, h_abg, E, N, Elev = _make_points(
        100, flight_agl=1000.0, horizontal_offset=1000.0
    )
    result = Nz_U(X, Y, h_abg, E, N, Elev, limit_flight_agl=800.0)
    assert result is not None
    np.testing.assert_almost_equal(result, 1.0 / np.sqrt(2), decimal=10)


def test_compute_Nz_U_output_shape():
    """compute_Nz_U must return an ndarray with the same length as the input."""
    n = 37
    X, Y, h_abg, E, N, Elev = _make_points(n, flight_agl=1500.0)
    result = Nz_U(X, Y, h_abg, E, N, Elev)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (n,)


def test_compute_Nz_U_values_in_0_1():
    """All Nz_U values must lie in (0, 1] for valid geometry."""
    X, Y, h_abg, E, N, Elev = _make_points(100, flight_agl=1500.0, horizontal_offset=500.0)
    result = Nz_U(X, Y, h_abg, E, N, Elev)
    assert result is not None
    assert np.all(result > 0.0)
    assert np.all(result <= 1.0)


# ── integration test on real preprocessed data ───────────────────────────────

def test_compute_Nz_U_on_real_data():
    """On real preprocessed data, all Nz_U values must be in (0, 1]."""
    if not _PRETRAITED_LAS.exists():
        pytest.skip(f"Real data not available: {_PRETRAITED_LAS}")

    las = laspy.read(str(_PRETRAITED_LAS))
    X = np.asarray(las.x, dtype=np.float64)
    Y = np.asarray(las.y, dtype=np.float64)
    h_abg = np.asarray(las["h_abg"], dtype=np.float64)
    X_sensor = np.asarray(las["X_sensor"], dtype=np.float64)
    Y_sensor = np.asarray(las["Y_sensor"], dtype=np.float64)
    Z_sensor = np.asarray(las["Z_sensor"], dtype=np.float64)

    result = Nz_U(X, Y, h_abg, X_sensor, Y_sensor, Z_sensor)

    assert result is not None, "Nz_U should not be None for valid real data."
    assert result.shape == X.shape
    assert np.all(result > 0.0) and np.all(result <= 1.0), (
        f"Nz_U values out of (0, 1]: min={result.min():.4f}, max={result.max():.4f}"
    )
