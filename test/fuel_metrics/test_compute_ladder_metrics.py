import numpy as np

from lidar_for_fuel.fuel_metrics.compute_ladder_metrics import compute_ladder_metrics

_LADDER_PREFIXES = ["LadderBot", "LadderTop", "LadderLoad", "SFLeq0", "SFLeq_intot", "ProfType", "CFL", "FSG"]

# Hand-written profiles: CBD given directly, 1 m strata starting at 0 m, CHT = 1 m, M = 2.5
_HAND_PARAMS = dict(canopy_height_threshold=1.0, ladder_fuel_bd_thresholds=[0.02], m_perrakis=2.5)


def _ladder(cbd, cbh2, dz=1.0, **params):
    """Ladder metrics of a CBD profile of `dz` strata starting at 0 m (h_center = dz/2, 3·dz/2, ...)."""
    cbd = np.asarray(cbd, dtype=float)
    h_center = dz * np.arange(cbd.size) + dz / 2

    return compute_ladder_metrics(cbd, h_center, dz, cbh2, **{**_HAND_PARAMS, **params})


def _assert_ladder(metrics, suffix, expected):
    """Compare the 8 ladder metrics of one threshold to the hand-computed values."""
    for prefix, value in zip(_LADDER_PREFIXES, expected):
        name = f"{prefix}_{suffix}"
        np.testing.assert_allclose(metrics[name], value, rtol=1e-12, err_msg=name)


# --- Hand-written profiles, one per ladder fuel case ------------------------------------------
# Expected values, in the order of _LADDER_PREFIXES, written as the formulas of step 8.


def test_ladder_continuous_from_surface_gives_type_3():
    # h_center: 0.5   1.5   2.5   3.5   4.5  5.5  6.5  7.5
    cbd = [0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1]
    # ladder = strata 1.5 to 3.5 (CBD >= 0.02, 1 < h <= CBH2 = 5); bottom 1 m, top 4 m
    load = 0.05 + 0.03 + 0.03
    expected = [
        1.0,
        4.0,
        load,
        2.5 * load * 5 / (5 - 4 / 2),
        2.5 * (0.05 * 5 / (5 - 1.5) + 0.03 * 5 / (5 - 2.5) + 0.03 * 5 / (5 - 3.5)),
        3,
        0.0 + 0.2 + 0.3 + 0.1,
        5 - 4 + 0.5,
    ]

    _assert_ladder(_ladder(cbd, cbh2=5.0), "0p020", expected)


def test_ladder_suspended_gives_type_4():
    # h_center: 0.5   1.5  2.5  3.5   4.5   5.5  6.5  7.5
    cbd = [0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.3, 0.2]
    # ladder = strata 3.5 and 4.5, above a gap: bottom 3 m > CHT; the load starts at CHT
    load = 0.0 + 0.0 + 0.05 + 0.05
    expected = [
        3.0,
        5.0,
        load,
        2.5 * load * 5 / (5 - 5 / 2),
        2.5 * (0.05 * 5 / (5 - 3.5) + 0.05 * 5 / (5 - 4.5)),
        4,
        0.0 + 0.3 + 0.2,
        5 - 5 + 0.5,
    ]

    _assert_ladder(_ladder(cbd, cbh2=5.0), "0p020", expected)


def test_ladder_interrupted_keeps_the_first_block_only():
    # h_center: 0.5  1.5   2.5  3.5   4.5  5.5
    cbd = [0.0, 0.05, 0.0, 0.05, 0.0, 0.3]
    # strata 1.5 and 3.5 reach the threshold, but the block stops at 2.5: top 2 m
    expected = [
        1.0,
        2.0,
        0.05,
        2.5 * 0.05 * 4 / (4 - 2 / 2),
        2.5 * 0.05 * 4 / (4 - 1.5),
        3,
        0.0 + 0.05 + 0.0 + 0.3,
        4 - 2 + 0.5,
    ]

    _assert_ladder(_ladder(cbd, cbh2=4.0), "0p020", expected)


def test_ladder_without_canopy_gives_type_1():
    # CBH2 fell back on CHT: the search range 1 < h <= CBH2 is empty
    cbd = [0.05, 0.05, 0.03, 0.0]
    expected = [1.0, 1.0, 0.0, 0.0, 0.0, 1, 0.05 + 0.03 + 0.0, 1 - 1 + 0.5]

    _assert_ladder(_ladder(cbd, cbh2=1.0), "0p020", expected)


def test_ladder_without_ladder_starts_at_cht():
    # CHT = 2 m, CBH2 fell back on it; the only stratum of 1 < h <= 2 is below the threshold:
    # bottom and top stay at CHT (not at the 1 m bottom of the search)
    cbd = [0.05, 0.01, 0.05, 0.0]
    expected = [2.0, 2.0, 0.0, 0.0, 0.0, 1, 0.05 + 0.0, 2 - 2 + 0.5]

    _assert_ladder(_ladder(cbd, cbh2=2.0, canopy_height_threshold=2.0), "0p020", expected)


def test_ladder_canopy_without_ladder_gives_type_2():
    # h_center: 0.5   1.5   2.5  3.5  4.5  5.5  6.5
    cbd = [0.05, 0.01, 0.01, 0.0, 0.0, 0.3, 0.2]
    # no stratum >= 0.02 between 1 m and CBH2 = 5 m
    expected = [1.0, 1.0, 0.0, 0.0, 0.0, 2, 0.01 + 0.01 + 0.0 + 0.0 + 0.3 + 0.2, 5 - 1 + 0.5]

    _assert_ladder(_ladder(cbd, cbh2=5.0), "0p020", expected)


def test_ladder_search_starts_at_1_m_whatever_cht():
    # CHT = 2 m: the ladder is still searched above 1 m (R hard-coded `zval > 1`, kept by
    # INRAE), so it starts at 1 m (type 3, bottom <= CHT), but the load only counts h > CHT.
    # h_center: 0.5  1.5   2.5   3.5   4.5  5.5
    cbd = [0.0, 0.05, 0.05, 0.05, 0.0, 0.3]
    load = 0.05 + 0.05
    expected = [
        1.0,
        4.0,
        load,
        2.5 * load * 4 / (4 - 4 / 2),
        2.5 * (0.05 * 4 / (4 - 2.5) + 0.05 * 4 / (4 - 3.5)),
        3,
        0.0 + 0.3,
        4 - 4 + 0.5,
    ]

    _assert_ladder(_ladder(cbd, cbh2=4.0, canopy_height_threshold=2.0), "0p020", expected)


def test_ladder_half_metre_strata():
    # h_center: 0.25  0.75  1.25  1.75  2.25  2.75  3.25
    cbd = [0.05, 0.05, 0.04, 0.04, 0.0, 0.3, 0.3]
    # ladder = strata 1.25 and 1.75 (h > 1, h <= CBH2 = 2.5): bottom 1 m, top 2 m
    load = (0.04 + 0.04) * 0.5
    expected = [
        1.0,
        2.0,
        load,
        2.5 * load * 2.5 / (2.5 - 2 / 2),
        2.5 * (0.04 * 2.5 / (2.5 - 1.25) + 0.04 * 2.5 / (2.5 - 1.75)) * 0.5,
        3,
        (0.0 + 0.3 + 0.3) * 0.5,
        2.5 - 2 + 0.25,
    ]

    _assert_ladder(_ladder(cbd, cbh2=2.5, dz=0.5), "0p020", expected)


def test_ladder_one_pass_per_threshold():
    # Same profile as type 3: each threshold has its own ladder
    cbd = [0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1]

    metrics = _ladder(cbd, cbh2=5.0, ladder_fuel_bd_thresholds=[0.02, 0.05])

    assert metrics["LadderTop_0p020"] == 4.0
    assert metrics["LadderTop_0p050"] == 2.0
    assert metrics["ProfType_0p050"] == 3


# --- Names and order --------------------------------------------------------------------------


def test_ladder_names_follow_the_sorted_distinct_thresholds():
    # R: thresholds <- sort(unique(LadderFuelBDthresholds)), then 8 metrics per threshold
    metrics = _ladder([0.0, 0.05, 0.3], cbh2=2.0, ladder_fuel_bd_thresholds=[0.1, 0.01, 0.05, 0.01])

    assert list(metrics) == [
        f"{prefix}_{suffix}" for suffix in ["0p010", "0p050", "0p100"] for prefix in _LADDER_PREFIXES
    ]
    assert all(isinstance(value, float) for value in metrics.values())
