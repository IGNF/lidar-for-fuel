from pathlib import Path

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.fuel_metrics.calculate_fuel_metrics import fuel_metrics_core
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import read_pad_heights

_REF_DIR = Path(__file__).resolve().parents[2] / "data/ref_inrae"
_REF_PAD_RASTER_1M = _REF_DIR / "pad_profile/512500_6854997_pad_1m.tif"
_REF_FUEL_METRICS = _REF_DIR / "fuel_metrics/512500_6854997_fuel_metrics.tif"

# Parameters of the R reference tile: INRAE config `ffuelmetrics2` section of
# data/ref_inrae/2026-09-28_fuel_metrics_zones-pilotes.yml
_INRAE_PARAMS = dict(
    fma_can=0.2543576,
    fma_shrub=0.2543576,
    smooth_pad=False,
    ladder_fuel_bd_thresholds=[0.01, 0.02, 0.05],
    canopy_height_threshold=1.0,
    canopy_bd_thresh_frac=0.3,
    canopy_top_bd_thresholds=[0.0012, 0.01],
    cbh_threshold_ratio=0.333,
    bd_max=1.0,
    m_perrakis=2.5,
)

# Hand-written profile, 1 m strata from 0 m, with distinct FMA so that the loads and the
# ladder use the CBD recomputed with CBH2 (shrub FMA up to CBH2 = 5 m). The ratio stays 1/3,
# the value of the R computations below (with 0.333, the 2.5 m stratum would move to 3A).
# h_center: 0.5  1.5   2.5   3.5   4.5  5.5  6.5  7.5
_HAND_PAD = [0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1]
_HAND_PARAMS = {**_INRAE_PARAMS, "fma_can": 1.0, "fma_shrub": 0.5, "cbh_threshold_ratio": 1 / 3}

# R ffuelmetrics2(PADval = _HAND_PAD, FMAcan = 1, FMAshrub = 0.5, zval = 0:7, dz = 1,
# CanopyBDTreshFrac = 0.3, CBHThresholdRatio = 1/3), computed with Rscript. Ladder
# continuous from the surface (type 3) at 0.01 and 0.02, canopy without ladder (type 2) at 0.05.
_HAND_R_METRICS = {
    "CLFL": 0.655,
    "FL_k1": 0.025,
    "FL_k2": 0.025,
    "LadderBot_0p010": 1.0,
    "LadderTop_0p010": 4.0,
    "LadderLoad_0p010": 0.055,
    "SFLeq0_0p010": 0.22916666666666666,
    "SFLeq_intot_0p010": 0.28928571428571426,
    "ProfType_0p010": 3.0,
    "CFL_0p010": 0.6,
    "FSG_0p010": 1.5,
    "LadderBot_0p020": 1.0,
    "LadderTop_0p020": 2.0,
    "LadderLoad_0p020": 0.025,
    "SFLeq0_0p020": 0.078125,
    "SFLeq_intot_0p020": 0.089285714285714274,
    "ProfType_0p020": 3.0,
    "CFL_0p020": 0.63,
    "FSG_0p020": 3.5,
    "LadderBot_0p050": 1.0,
    "LadderTop_0p050": 1.0,
    "LadderLoad_0p050": 0.0,
    "SFLeq0_0p050": 0.0,
    "SFLeq_intot_0p050": 0.0,
    "ProfType_0p050": 2.0,
    "CFL_0p050": 0.655,
    "FSG_0p050": 4.5,
    "CTH_0p0012": 7.5,
    "CTH_0p0100": 7.5,
    "CDH": 7.5,
    "CBH": 5.0,
    "CBH2": 5.0,
    "FSG": 3.5,
    "CBHlow": np.nan,
    "CanBDmax": 0.3,
    "CanBDmin": 0.0,
    "CanBDlowmax": 0.05,
    "CanBDlowmin": np.nan,
    "CBDcan": 0.2,
    "CBD_cbh": 0.2,  # stratum centred on 5.5 m (CBH = CBH2 = 5 m)
    "CBD_cbh2": 0.2,
    "CBD_cbhlow": np.nan,
    "LFL": 0.025,
    "CFL": 0.63,
    "ProfType": 3.0,
}


def _core(pad, dz=1.0, **params):
    """Fuel metrics of a profile of `dz` strata starting at 0 m."""
    pad = np.asarray(pad, dtype=float)
    h_bottom = dz * np.arange(pad.size)

    return fuel_metrics_core(pad, h_bottom, dz, **{**_HAND_PARAMS, **params})


def _reference_band_names():
    with rasterio.open(_REF_FUEL_METRICS) as src:
        return list(src.descriptions)


# --- Names and order --------------------------------------------------------------------------


def test_fuel_metrics_names_and_order_are_those_of_r():
    metrics = _core(_HAND_PAD)

    assert list(metrics) == _reference_band_names()
    assert len(metrics) == 45
    assert all(isinstance(value, float) for value in metrics.values())


def test_fuel_metrics_band_count_follows_the_ladder_thresholds():
    metrics = _core(_HAND_PAD, ladder_fuel_bd_thresholds=[0.01, 0.02, 0.05, 0.1])

    # 19 + 8 × 4 + 2 bands; the 4th ladder block comes right after the 3rd one
    assert len(metrics) == 53
    names = list(metrics)
    assert names[27:35] == [
        "LadderBot_0p100",
        "LadderTop_0p100",
        "LadderLoad_0p100",
        "SFLeq0_0p100",
        "SFLeq_intot_0p100",
        "ProfType_0p100",
        "CFL_0p100",
        "FSG_0p100",
    ]
    assert names[35] == "CTH_0p0012"


# --- Values -----------------------------------------------------------------------------------


def test_fuel_metrics_hand_profile_computed_by_r():
    metrics = _core(_HAND_PAD)

    assert list(metrics) == list(_HAND_R_METRICS)
    for name, expected in _HAND_R_METRICS.items():
        np.testing.assert_allclose(metrics[name], expected, rtol=1e-12, equal_nan=True, err_msg=name)


def test_fuel_metrics_loads_use_the_cbd_recomputed_with_cbh2():
    metrics = _core(_HAND_PAD)

    # CBH2 = 5 m: shrub FMA (0.5) up to 5 m, canopy FMA (1) above
    assert metrics["CBH2"] == 5.0
    np.testing.assert_allclose(metrics["CLFL"], (0.05 + 0.03 + 0.03 + 0.0) * 0.5 + 0.2 + 0.3 + 0.1, rtol=1e-12)
    np.testing.assert_allclose(metrics["FL_k1"], 0.05 * 0.5, rtol=1e-12)
    np.testing.assert_allclose(metrics["FL_k2"], 0.05 * 0.5, rtol=1e-12)


def test_fuel_metrics_canopy_load_starts_at_cht():
    # CHT = 2 m (R gives CLFL = 0.63, CBH2 = 5): the stratum at 1.5 m is not counted
    metrics = _core(_HAND_PAD, canopy_height_threshold=2.0)

    assert metrics["CBH2"] == 5.0
    np.testing.assert_allclose(metrics["CLFL"], (0.03 + 0.03 + 0.0) * 0.5 + 0.2 + 0.3 + 0.1, rtol=1e-12)


def test_fuel_metrics_load_excludes_the_stratum_centred_on_cht():
    # CHT = 0.5 m, centre of the first stratum: `h > CHT` excludes it from CLFL and from the
    # ladder load; the ladder found above 1 m is suspended (type 4). Values computed by R:
    # call of _HAND_R_METRICS with CanopyHeightThreshold = 0.5.
    metrics = _core(_HAND_PAD, canopy_height_threshold=0.5)

    expected = {
        "CBH2": 5.0,
        "CLFL": 0.655,
        "LadderBot_0p020": 1.0,  # above CHT: suspended
        "LadderTop_0p020": 2.0,
        "LadderLoad_0p020": 0.025,
        "SFLeq0_0p020": 0.078125,
        "SFLeq_intot_0p020": 0.0892857142857143,
        "ProfType_0p020": 4.0,
        "CFL_0p020": 0.63,
        "FSG_0p020": 3.5,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(metrics[name], value, rtol=1e-12, err_msg=name)


@pytest.mark.parametrize(
    "params, expected",
    [
        pytest.param(
            {"smooth_pad": True},
            {
                "CLFL": 0.693333333333333,
                "FL_k2": 0.0216666666666667,
                "CBH2": 4.0,
                "CanBDmax": 0.2,
                "CanBDmin": 0.02,
                "CBDcan": 0.160833333333333,
                "LFL": 0.0216666666666667,
                "CFL": 0.671666666666667,
            },
            id="smooth_pad",
        ),
        pytest.param(
            # CBD of the stratum at 6.5 m (0.3) capped at 0.2
            {"bd_max": 0.2},
            {"CLFL": 0.555, "CanBDmax": 0.2, "CBDcan": 0.166666666666667, "CFL": 0.53},
            id="bd_max",
        ),
        pytest.param(
            # the lower canopy search goes up to 0.8 × CTH = 6 m: a CBHlow is found
            {"cbh_threshold_ratio": 0.8},
            {"CBHlow": 5.0, "CanBDlowmax": 0.2, "CanBDlowmin": 0.0},
            id="cbh_threshold_ratio",
        ),
        pytest.param(
            {"canopy_bd_thresh_frac": 0.8},
            {"CBH2": 6.0, "CLFL": 0.555, "CFL": 0.53, "FSG": 4.5},
            id="canopy_bd_thresh_frac",
        ),
        pytest.param(
            {"m_perrakis": 3.0},
            {"SFLeq0_0p020": 0.09375, "SFLeq_intot_0p020": 0.10714285714285714},
            id="m_perrakis",
        ),
    ],
)
def test_fuel_metrics_profile_parameters_are_passed_on(params, expected):
    # R ffuelmetrics2(PADval = _HAND_PAD, FMAcan = 1, FMAshrub = 0.5, zval = 0:7, dz = 1,
    # CanopyBDTreshFrac = 0.3, CBHThresholdRatio = 1/3, M = 2.5), with the parameter of the
    # case instead (smooth_pad, bdmax, CBHThresholdRatio, CanopyBDTreshFrac or M), computed
    # with Rscript
    metrics = _core(_HAND_PAD, **params)

    for name, value in expected.items():
        np.testing.assert_allclose(metrics[name], value, rtol=1e-12, err_msg=name)


def test_fuel_metrics_single_stratum_gives_nan_second_load():
    # R: CBDval[2] is NA with one stratum
    metrics = _core([0.05])

    assert metrics["FL_k1"] == 0.05 * 0.5
    assert np.isnan(metrics["FL_k2"])


@pytest.mark.parametrize(
    "pad, thresholds, suffix, r_lfl",
    [
        pytest.param(_HAND_PAD, [0.01, 0.02, 0.05], "0p020", 0.025, id="median_threshold"),
        # 0.02 and 0.05 are both 0.015 from the median 0.035: 0.05 is closer in double
        # precision (PR #48); the 0.05 ladder differs from the 3 others (0.01 and 0.02 are equal)
        pytest.param(
            [0.05, 0.3, 0.15, 0.08, 0.0, 0.4, 0.5, 0.2],
            [0.1, 0.02, 0.01, 0.05],
            "0p050",
            0.225,
            id="even_number_of_thresholds",
        ),
        # 0.5 and 0.75 are exactly 0.125 from the median 0.625: the first of the list given
        # wins, not the first of the sorted thresholds
        pytest.param(
            [0.05, 1.8, 1.2, 0.6, 0.0, 3, 3, 1],
            [0.75, 0.5, 0.25, 1],
            "0p750",
            0.9,
            id="exact_tie_keeps_the_order_given",
        ),
    ],
)
def test_fuel_metrics_defaults_are_those_of_the_threshold_closest_to_the_median(pad, thresholds, suffix, r_lfl):
    # r_lfl: LFL computed by R (call of _HAND_R_METRICS with PADval = pad and these thresholds)
    metrics = _core(pad, ladder_fuel_bd_thresholds=thresholds)

    np.testing.assert_allclose(metrics["LFL"], r_lfl, rtol=1e-12)
    # LFL, CFL and FSG of the default threshold differ from those of the other thresholds
    for default, prefix in [("LFL", "LadderLoad"), ("CFL", "CFL"), ("FSG", "FSG")]:
        others = [
            value for name, value in metrics.items() if name.startswith(f"{prefix}_") and name != f"{prefix}_{suffix}"
        ]
        assert metrics[default] not in others, default
        assert metrics[default] == metrics[f"{prefix}_{suffix}"], default
    assert metrics["ProfType"] == metrics[f"ProfType_{suffix}"]


def test_fuel_metrics_invalid_pad_values_are_set_to_0():
    # Same as R: NaN, inf and negative values of a partly valid profile count as 0
    pad = np.array(_HAND_PAD)
    invalid = pad.copy()
    invalid[[1, 4, 7]] = [np.nan, -1.0, np.inf]
    zeroed = pad.copy()
    zeroed[[1, 4, 7]] = 0.0

    np.testing.assert_equal(_core(invalid), _core(zeroed))


@pytest.mark.parametrize("thresholds", [[0.01, 0.02, 0.05], [0.01, 0.02, 0.05, 0.1]])
def test_fuel_metrics_all_nan_profile_gives_nan_on_all_metrics(thresholds):
    # Difference from R: R turns a no-data pixel into a null profile (zeros, ProfType 1),
    # which reads as "no fuel" on the maps
    metrics = _core([np.nan] * 8, ladder_fuel_bd_thresholds=thresholds)

    assert list(metrics) == list(_core(_HAND_PAD, ladder_fuel_bd_thresholds=thresholds))
    assert all(isinstance(value, float) and np.isnan(value) for value in metrics.values())


@pytest.mark.parametrize("fma", ["fma_can", "fma_shrub"])
def test_fuel_metrics_no_fma_gives_nan_on_all_metrics(fma):
    # A pixel without FMA (NaN, i.e. NoData or outside the FMA map) has no vegetation
    # information, so all its metrics are NaN, as a no-data PAD
    metrics = _core(_HAND_PAD, **{fma: np.nan})

    assert list(metrics) == list(_core(_HAND_PAD))
    assert all(isinstance(value, float) and np.isnan(value) for value in metrics.values())


def test_fuel_metrics_one_finite_pad_value_is_enough_to_compute_the_metrics():
    # Only an all-NaN profile has no data: the NaN of a partly valid profile count as 0
    pad = np.full(8, np.nan)
    pad[5] = 0.2
    zeroed = np.where(np.isnan(pad), 0.0, pad)

    metrics = _core(pad)

    np.testing.assert_equal(metrics, _core(zeroed))
    assert metrics["CLFL"] > 0.0


# --- Parity with R ----------------------------------------------------------------------------


def test_fuel_metrics_parity_with_r():
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        pad_bands = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_")]
        pad_names = [src.descriptions[i - 1] for i in pad_bands]
        pad = src.read(pad_bands).astype(float)
    with rasterio.open(_REF_FUEL_METRICS) as src:
        names = list(src.descriptions)
        ref = src.read().astype(float)

    dz, h_bottom = read_pad_heights(pad_names)

    _, height, width = pad.shape
    result = np.full_like(ref, np.nan)
    for row in range(height):
        for col in range(width):
            metrics = fuel_metrics_core(pad[:, row, col], h_bottom, dz, **_INRAE_PARAMS)
            assert list(metrics) == names
            result[:, row, col] = list(metrics.values())

    # the tile has the 4 profile types for each ladder threshold
    for name in names:
        if name.startswith("ProfType_"):
            assert set(np.unique(ref[names.index(name)])) == {1.0, 2.0, 3.0, 4.0}, name

    # Difference from R: no-data pixels are NaN on all bands, where R gives the metrics of a
    # null profile; they are excluded from the parity check below. The tile has no partly NaN
    # pixel: that case is covered by the hand profiles.
    no_data = np.isnan(pad).all(axis=0)
    assert no_data.any()
    assert (ref[names.index("CLFL"), no_data] == 0.0).all()
    assert (ref[names.index("CBH2"), no_data] == _INRAE_PARAMS["canopy_height_threshold"]).all()
    assert (ref[names.index("ProfType"), no_data] == 1.0).all()
    assert np.isnan(result[:, no_data]).all()

    for band, name in enumerate(names):
        np.testing.assert_allclose(
            result[band, ~no_data], ref[band, ~no_data], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=name
        )
