"""Tests for compute_cos_theta (US-6 — angle de scan réel du LiDAR)."""
from pathlib import Path

import laspy
import numpy as np
import pytest

from lidar_for_fuel.pad_profil.compute_scan_angle import compute_cos_theta

_PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_veg_points(n: int, flight_agl: float = 1500.0, horizontal_offset: float = 0.0):
    """Return synthetic arrays for n vegetation-class (5) points.

    All points lie at (0, 0) on the ground; the sensor is placed at
    (horizontal_offset, 0, flight_agl) relative to each point.
    """
    X = np.zeros(n, dtype=np.float64)
    Y = np.zeros(n, dtype=np.float64)
    Zref = np.zeros(n, dtype=np.float64)
    Easting = X + horizontal_offset
    Northing = np.zeros(n, dtype=np.float64)
    Elevation = Zref + flight_agl
    Classification = np.full(n, 5, dtype=np.uint8)
    return X, Y, Zref, Easting, Northing, Elevation, Classification


# ── unit tests ────────────────────────────────────────────────────────────────

def test_scanning_angle_false_returns_one():
    """Disabling scanning_angle must always return 1.0 regardless of geometry."""
    X, Y, Zref, E, N, Elev, C = _make_veg_points(10, flight_agl=500.0)
    assert compute_cos_theta(X, Y, Zref, E, N, Elev, C, scanning_angle=False) == 1.0


def test_low_flight_height_returns_none():
    """Mean flight height below limit_flight_agl must return None."""
    X, Y, Zref, E, N, Elev, C = _make_veg_points(10, flight_agl=500.0)
    result = compute_cos_theta(X, Y, Zref, E, N, Elev, C, limit_flight_agl=800.0)
    assert result is None


def test_vertical_pulses_cos_theta_is_one():
    """Sensor directly above each point (horizontal_offset=0) → cos_theta = 1."""
    X, Y, Zref, E, N, Elev, C = _make_veg_points(50, flight_agl=1500.0, horizontal_offset=0.0)
    result = compute_cos_theta(X, Y, Zref, E, N, Elev, C)
    assert result is not None
    np.testing.assert_almost_equal(result, 1.0, decimal=10)


def test_known_oblique_angle_45_degrees():
    """flight_agl == horizontal_dist → scan angle = 45° → cos_theta = 1/sqrt(2)."""
    X, Y, Zref, E, N, Elev, C = _make_veg_points(
        100, flight_agl=1000.0, horizontal_offset=1000.0
    )
    result = compute_cos_theta(X, Y, Zref, E, N, Elev, C, limit_flight_agl=800.0)
    assert result is not None
    np.testing.assert_almost_equal(result, 1.0 / np.sqrt(2), decimal=10)


def test_only_veg_gnd_points_used():
    """Non-vegetation/ground points (class 1) must not influence cos_theta."""
    n = 100
    X = np.zeros(n, dtype=np.float64)
    Y = np.zeros(n, dtype=np.float64)
    Zref = np.zeros(n, dtype=np.float64)
    Elevation = Zref + 1500.0
    # First half: vegetation (class 5), sensor directly above → Nz_U = 1
    # Second half: noise (class 1), sensor at 45° → Nz_U = 1/sqrt(2) if counted
    Easting = np.where(np.arange(n) < n // 2, X, X + 1000.0)
    Northing = np.zeros(n, dtype=np.float64)
    Classification = np.where(np.arange(n) < n // 2, 5, 1).astype(np.uint8)

    result = compute_cos_theta(X, Y, Zref, Easting, Northing, Elevation, Classification)
    assert result is not None
    np.testing.assert_almost_equal(result, 1.0, decimal=10)


def test_ground_class_9_included():
    """Ground points (class 9) must be included in the cos_theta average."""
    n = 50
    X = np.zeros(n, dtype=np.float64)
    Y = np.zeros(n, dtype=np.float64)
    Zref = np.zeros(n, dtype=np.float64)
    Easting = np.zeros(n, dtype=np.float64)
    Northing = np.zeros(n, dtype=np.float64)
    Elevation = Zref + 1500.0
    Classification = np.full(n, 9, dtype=np.uint8)  # ground only

    result = compute_cos_theta(X, Y, Zref, Easting, Northing, Elevation, Classification)
    assert result is not None
    np.testing.assert_almost_equal(result, 1.0, decimal=10)


# ── integration test on real preprocessed data ───────────────────────────────

def test_cos_theta_on_real_data():
    """On real preprocessed data, cos_theta must be in (0, 1]."""
    if not _PRETRAITED_LAS.exists():
        pytest.skip(f"Real data not available: {_PRETRAITED_LAS}")

    las = laspy.read(str(_PRETRAITED_LAS))
    X = np.asarray(las.x, dtype=np.float64)
    Y = np.asarray(las.y, dtype=np.float64)
    Zref = np.asarray(las["Zref"], dtype=np.float64)
    Easting = np.asarray(las["Easting"], dtype=np.float64)
    Northing = np.asarray(las["Northing"], dtype=np.float64)
    Elevation = np.asarray(las["Elevation"], dtype=np.float64)
    Classification = np.asarray(las.classification, dtype=np.uint8)

    result = compute_cos_theta(X, Y, Zref, Easting, Northing, Elevation, Classification)

    assert result is not None, "cos_theta should not be None for valid real data."
    assert 0.0 < result <= 1.0, f"cos_theta expected in (0, 1], got {result:.4f}."
