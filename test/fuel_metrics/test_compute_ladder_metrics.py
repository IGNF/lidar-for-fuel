import numpy as np
import pytest

from lidar_for_fuel.fuel_metrics.compute_ladder_metrics import compute_ladder_metrics

_LADDER_PREFIXES = ["LadderBot", "LadderTop", "LadderLoad", "SFLeq0", "SFLeq_intot", "ProfType", "CFL", "FSG"]

# Hand-written profiles: CBD given directly, 1 m strata starting at 0 m, CHT = 1 m, M = 2.5
_HAND_PARAMS = dict(canopy_height_threshold=1.0, ladder_fuel_bd_thresholds=[0.02], m_perrakis=2.5)


def _ladder(cbd, cbh2, dz=1.0, **params):
    """Ladder metrics of a CBD profile of `dz` strata starting at 0 m (h_center = dz/2, 3·dz/2, ...)."""
    cbd = np.asarray(cbd, dtype=float)
    h_center = dz * np.arange(cbd.size) + dz / 2

    return compute_ladder_metrics(cbd, h_center, dz, cbh2, **{**_HAND_PARAMS, **params})


# --- Hand-written profiles, one per ladder fuel case ------------------------------------------
# Each case: CBD profile, CBH2, parameters overriding _HAND_PARAMS (and dz), then the expected
# values of the 0.02 threshold, in the order of _LADDER_PREFIXES, written as the formulas of
# step 8.

_HAND_CASES = [
    pytest.param(
        # h_center: 0.5   1.5   2.5   3.5   4.5  5.5  6.5  7.5
        [0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1],
        5.0,
        {},
        # ladder = strata 1.5 to 3.5 (CBD >= 0.02, 1 < h <= CBH2 = 5); bottom 1 m, top 4 m
        [
            1.0,
            4.0,
            0.05 + 0.03 + 0.03,
            2.5 * (0.05 + 0.03 + 0.03) * 5 / (5 - 4 / 2),
            2.5 * (0.05 * 5 / (5 - 1.5) + 0.03 * 5 / (5 - 2.5) + 0.03 * 5 / (5 - 3.5)),
            3,
            0.0 + 0.2 + 0.3 + 0.1,
            5 - 4 + 0.5,
        ],
        id="continuous_from_surface_gives_type_3",
    ),
    pytest.param(
        # h_center: 0.5   1.5  2.5  3.5   4.5   5.5  6.5  7.5
        [0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.3, 0.2],
        5.0,
        {},
        # ladder = strata 3.5 and 4.5, above a gap: bottom 3 m > CHT; the load starts at CHT
        [
            3.0,
            5.0,
            0.0 + 0.0 + 0.05 + 0.05,
            2.5 * (0.0 + 0.0 + 0.05 + 0.05) * 5 / (5 - 5 / 2),
            2.5 * (0.05 * 5 / (5 - 3.5) + 0.05 * 5 / (5 - 4.5)),
            4,
            0.0 + 0.3 + 0.2,
            5 - 5 + 0.5,
        ],
        id="suspended_gives_type_4",
    ),
    pytest.param(
        # h_center: 0.5  1.5   2.5  3.5   4.5  5.5
        [0.0, 0.05, 0.0, 0.05, 0.0, 0.3],
        4.0,
        {},
        # strata 1.5 and 3.5 reach the threshold, but the block stops at 2.5: top 2 m
        [
            1.0,
            2.0,
            0.05,
            2.5 * 0.05 * 4 / (4 - 2 / 2),
            2.5 * 0.05 * 4 / (4 - 1.5),
            3,
            0.0 + 0.05 + 0.0 + 0.3,
            4 - 2 + 0.5,
        ],
        id="interrupted_keeps_the_first_block_only",
    ),
    pytest.param(
        # CBH2 fell back on CHT: the search range 1 < h <= CBH2 is empty
        [0.05, 0.05, 0.03, 0.0],
        1.0,
        {},
        [1.0, 1.0, 0.0, 0.0, 0.0, 1, 0.05 + 0.03 + 0.0, 1 - 1 + 0.5],
        id="without_canopy_gives_type_1",
    ),
    pytest.param(
        # CHT = 2 m, CBH2 fell back on it; the only stratum of 1 < h <= 2 is below the threshold:
        # bottom and top stay at CHT (not at the 1 m bottom of the search)
        [0.05, 0.01, 0.05, 0.0],
        2.0,
        {"canopy_height_threshold": 2.0},
        [2.0, 2.0, 0.0, 0.0, 0.0, 1, 0.05 + 0.0, 2 - 2 + 0.5],
        id="without_ladder_starts_at_cht",
    ),
    pytest.param(
        # h_center: 0.5   1.5   2.5  3.5  4.5  5.5  6.5
        [0.05, 0.01, 0.01, 0.0, 0.0, 0.3, 0.2],
        5.0,
        {},
        # no stratum >= 0.02 between 1 m and CBH2 = 5 m
        [1.0, 1.0, 0.0, 0.0, 0.0, 2, 0.01 + 0.01 + 0.0 + 0.0 + 0.3 + 0.2, 5 - 1 + 0.5],
        id="canopy_without_ladder_gives_type_2",
    ),
    pytest.param(
        # CHT = 2 m: the ladder is still searched above 1 m (R hard-coded `zval > 1`, kept by
        # INRAE), so it starts at 1 m (type 3, bottom <= CHT), but the load only counts h > CHT.
        # h_center: 0.5  1.5   2.5   3.5   4.5  5.5
        [0.0, 0.05, 0.05, 0.05, 0.0, 0.3],
        4.0,
        {"canopy_height_threshold": 2.0},
        [
            1.0,
            4.0,
            0.05 + 0.05,
            2.5 * (0.05 + 0.05) * 4 / (4 - 4 / 2),
            2.5 * (0.05 * 4 / (4 - 2.5) + 0.05 * 4 / (4 - 3.5)),
            3,
            0.0 + 0.3,
            4 - 4 + 0.5,
        ],
        id="search_starts_at_1_m_whatever_cht",
    ),
    pytest.param(
        # h_center: 0.25  0.75  1.25  1.75  2.25  2.75  3.25
        [0.05, 0.05, 0.04, 0.04, 0.0, 0.3, 0.3],
        2.5,
        {"dz": 0.5},
        # ladder = strata 1.25 and 1.75 (h > 1, h <= CBH2 = 2.5): bottom 1 m, top 2 m
        [
            1.0,
            2.0,
            (0.04 + 0.04) * 0.5,
            2.5 * (0.04 + 0.04) * 0.5 * 2.5 / (2.5 - 2 / 2),
            2.5 * (0.04 * 2.5 / (2.5 - 1.25) + 0.04 * 2.5 / (2.5 - 1.75)) * 0.5,
            3,
            (0.0 + 0.3 + 0.3) * 0.5,
            2.5 - 2 + 0.25,
        ],
        id="half_metre_strata",
    ),
]


@pytest.mark.parametrize("cbd, cbh2, params, expected", _HAND_CASES)
def test_ladder_hand_profile(cbd, cbh2, params, expected):
    metrics = _ladder(cbd, cbh2, **params)

    for prefix, value in zip(_LADDER_PREFIXES, expected, strict=True):
        name = f"{prefix}_0p020"
        np.testing.assert_allclose(metrics[name], value, rtol=1e-12, err_msg=name)


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
