import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_nrd import compute_nrd

# ── NRD values for various Ni/N combinations ──────────────────────────────────


@pytest.mark.parametrize(
    "Ni,N,expected",
    [
        (
            # Intermediate values: NRD in (0, 1) -> no Laplace correction applied
            np.array([2, 3], dtype=float),
            np.array([5, 6], dtype=float),
            np.array([2 / 5, 3 / 6]),
        ),
        (
            # NRD=0 (Ni=0, N>0): Laplace correction -> (0+1) / (N+2)
            np.array([0], dtype=float),
            np.array([5], dtype=float),
            np.array([1 / 7]),
        ),
        (
            # NRD=1 (Ni=N, N>0): Laplace correction -> (N+1) / (N+2)
            np.array([5], dtype=float),
            np.array([5], dtype=float),
            np.array([6 / 7]),
        ),
        (
            # Ni=0, N=0: 0/0 -> NaN -> set to 0 -> Laplace correction -> 1/2
            np.array([0], dtype=float),
            np.array([0], dtype=float),
            np.array([0.5]),
        ),
        (
            # Mixed: covers NaN, NRD=0, intermediate, and NRD=1 in one array
            np.array([0, 0, 2, 5], dtype=float),
            np.array([0, 5, 5, 5], dtype=float),
            np.array([0.5, 1 / 7, 0.4, 6 / 7]),
        ),
    ],
    ids=[
        "intermediate_no_correction",
        "nrd_zero_laplace_correction",
        "nrd_one_laplace_correction",
        "nan_case_ni0_n0",
        "mixed_all_cases",
    ],
)
def test_compute_nrd(Ni, N, expected):
    result = compute_nrd(Ni, N)
    np.testing.assert_allclose(result, expected)


# ── invariants ────────────────────────────────────────────────────────────────


def test_output_strictly_between_zero_and_one():
    rng = np.random.default_rng(42)
    N = rng.integers(0, 20, size=100).astype(float)
    Ni = np.array([rng.integers(0, int(n) + 1) for n in N], dtype=float)
    result = compute_nrd(Ni, N)
    assert np.all(result > 0)
    assert np.all(result < 1)


def test_no_nan_in_output():
    Ni = np.array([0, 0, 3], dtype=float)
    N = np.array([0, 5, 5], dtype=float)
    result = compute_nrd(Ni, N)
    assert not np.any(np.isnan(result))
