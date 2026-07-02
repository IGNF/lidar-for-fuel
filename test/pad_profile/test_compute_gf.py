import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_gf import compute_gf

# ── Gf values for various NRD inputs ─────────────────────────────────────────


@pytest.mark.parametrize(
    "NRD,expected",
    [
        (
            # Intermediate values
            np.array([0.4, 0.5]),
            np.array([0.6, 0.5]),
        ),
        (
            # Laplace-corrected boundary values (never exactly 0 or 1 in practice)
            np.array([1 / 7, 6 / 7]),
            np.array([6 / 7, 1 / 7]),
        ),
        (
            # Exact boundaries: Gf = 1 - NRD holds regardless of how NRD was produced
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
        ),
    ],
    ids=[
        "intermediate",
        "laplace_corrected",
        "exact_boundaries",
    ],
)
def test_compute_gf(NRD, expected):
    result = compute_gf(NRD)
    np.testing.assert_allclose(result, expected)


# ── invariants ────────────────────────────────────────────────────────────────


def test_output_shape_matches_input():
    NRD = np.array([0.1, 0.2, 0.3, 0.4])
    result = compute_gf(NRD)
    assert result.shape == NRD.shape


def test_gf_plus_nrd_equals_one():
    rng = np.random.default_rng(42)
    NRD = rng.uniform(0, 1, size=100)
    result = compute_gf(NRD)
    np.testing.assert_allclose(result + NRD, np.ones_like(NRD))
