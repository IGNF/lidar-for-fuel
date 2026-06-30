import numpy as np
import pytest

from lidar_for_fuel.pad_profile.compute_ni_n import compute_ni_n

# ── ni/n/min_layer for various points + params ──────────────────────────────────


@pytest.mark.parametrize(
    "h_abg,veg_gnd,kwargs,expected_ni,expected_n,expected_min_layer",
    [
        (
            # breaks=[-inf,0.1,1,2,3]: -0.5,-0.5,0.05 below-ground (dropped);
            # 0.5,0.9 -> stratum[0,1] (veg_gnd: 0.9); 1.5,1.8,2.0 -> stratum[1,2]
            # (veg_gnd: 1.5,2.0); 2.5 -> stratum[2,3] (veg_gnd: 2.5); 5.0 excluded.
            np.array([-0.5, -0.5, 0.05, 0.5, 0.9, 1.5, 1.8, 2.0, 2.5, 5.0]),
            np.array([False, False, False, False, True, True, False, True, True, False]),
            dict(z0=0.0, dz=1.0, nlayers=3),
            [1, 2, 1],
            [5, 8, 9],
            [0.0, 1.0, 2.0],
        ),
        (
            # nonzero z0, ground_margin=0
            np.array([2.5, 3.5, 4.5]),
            np.array([True, True, False]),
            dict(z0=2.0, dz=1.0, nlayers=3, ground_margin=0.0),
            [1, 1, 0],
            [1, 2, 3],
            [2.0, 3.0, 4.0],
        ),
        (
            # ground_margin=0: 0.0 falls in the below-ground stratum (right-closed
            # at 0.0), dropped from ni -- but still counts toward n (see below).
            np.array([0.0, 0.5]),
            np.array([True, True]),
            dict(z0=0.0, dz=1.0, nlayers=2, ground_margin=0.0),
            [1, 0],
            [2, 2],
            [0.0, 1.0],
        ),
        (
            # empty veg_gnd mask -> ni all zero, n still derived from all h_abg
            np.array([0.5, 1.5, 2.5]),
            np.zeros(3, dtype=bool),
            dict(z0=0.0, dz=1.0, nlayers=3),
            [0, 0, 0],
            [1, 2, 3],
            [0.0, 1.0, 2.0],
        ),
        (
            # points below the ground margin never enter the kept strata directly,
            # but a ray ending there still passed through every stratum above on
            # its way down -- so it contributes to n (cumulative), not to ni.
            np.array([-1.0, 0.0, 0.05]),
            np.array([True, True, True]),
            dict(z0=0.0, dz=1.0, nlayers=3),
            [0, 0, 0],
            [3, 3, 3],
            [0.0, 1.0, 2.0],
        ),
        (
            # points above the top stratum are excluded without crashing
            np.array([0.5, 100.0]),
            np.array([True, True]),
            dict(z0=0.0, dz=1.0, nlayers=3),
            [1, 0, 0],
            [1, 1, 1],
            [0.0, 1.0, 2.0],
        ),
    ],
    ids=[
        "nominal_hand_traced",
        "custom_z0_dz_nlayers",
        "ground_margin_zero",
        "empty_veg_gnd_nonzero_n",
        "below_ground_still_counts_in_n",
        "above_top_excluded_no_crash",
    ],
)
def test_ni_n_min_layer_for_scenario(h_abg, veg_gnd, kwargs, expected_ni, expected_n, expected_min_layer):
    ni, n, min_layer = compute_ni_n(h_abg, veg_gnd, **kwargs)
    np.testing.assert_array_equal(ni, expected_ni)
    np.testing.assert_array_equal(n, expected_n)
    np.testing.assert_array_equal(min_layer, expected_min_layer)


# ── output length, explicit nlayers vs nlayers=None ──────────────────────────────


@pytest.mark.parametrize(
    "h_abg,nlayers,expected_n_strata",
    [
        (np.zeros(10), 5, 5),  # explicit nlayers
        (np.array([0.5, 5.3]), None, 6),  # nlayers=None -> ceil(5.3 / 1) = 6
    ],
    ids=["explicit_nlayers", "nlayers_none"],
)
def test_output_length_matches_n_strata(h_abg, nlayers, expected_n_strata):
    veg_gnd = np.zeros(len(h_abg), dtype=bool)
    ni, n, min_layer = compute_ni_n(h_abg, veg_gnd, nlayers=nlayers)
    assert len(ni) == len(n) == len(min_layer) == expected_n_strata


# ── invariants ────────────────────────────────────────────────────────────────────


def test_n_is_non_decreasing():
    h_abg = np.array([0.5, 0.9, 1.5, 1.8, 2.5, 2.9])
    veg_gnd = np.zeros(6, dtype=bool)
    _, n, _ = compute_ni_n(h_abg, veg_gnd, z0=0.0, dz=1.0, nlayers=3)
    assert np.all(np.diff(n) >= 0)


def test_ni_does_not_exceed_n_per_stratum():
    rng = np.random.default_rng(0)
    h_abg = rng.uniform(-1, 5, size=200)
    veg_gnd = rng.random(200) < 0.5
    ni, n, _ = compute_ni_n(h_abg, veg_gnd, z0=0.0, dz=1.0, nlayers=5)
    assert np.all(ni <= n)


def test_output_dtypes():
    h_abg = np.array([0.5, 1.5])
    veg_gnd = np.array([True, False])
    ni, n, min_layer = compute_ni_n(h_abg, veg_gnd, nlayers=3)
    assert np.issubdtype(ni.dtype, np.integer)
    assert np.issubdtype(n.dtype, np.integer)
    assert min_layer.dtype == np.float64
