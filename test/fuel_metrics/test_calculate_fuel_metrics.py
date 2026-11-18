from pathlib import Path

import numpy as np
import rasterio

from lidar_for_fuel.fuel_metrics.calculate_fuel_metrics import fuel_metrics_core
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import read_pad_heights

_REF_DIR = Path(__file__).resolve().parents[2] / "data/ref_inrae"
# R (lidarforfuel) PAD raster and fuel metrics of the same 500 m tile (story 3.1)
_REF_PAD_RASTER_1M = _REF_DIR / "pad_profile/512500_6854997_pad_1m.tif"
_REF_FUEL_METRICS = _REF_DIR / "fuel_metrics/512500_6854997_fuel_metrics.tif"

# Parameters of the R reference tile: INRAE config `ffuelmetrics2` section of
# data/ref_inrae/2026-09-28_fuel_metrics_zones-pilotes.yml, except the ratio:
# R ffuelmetrics2Rast does not pass CBHThresholdRatio on and uses 1/3, not 0.333 (A3)
_INRAE_PARAMS = dict(
    fma_can=0.2543576,
    fma_shrub=0.2543576,
    smooth_pad=False,
    ladder_fuel_bd_thresholds=[0.01, 0.02, 0.05],
    canopy_height_threshold=1.0,
    canopy_bd_thresh_frac=0.3,
    canopy_top_bd_thresholds=[0.0012, 0.01],
    cbh_threshold_ratio=1 / 3,
    bd_max=1.0,
    m_perrakis=2.5,
)

# Hand-written profile, 1 m strata from 0 m, with distinct FMA so that the loads and the
# ladder use the CBD recomputed with CBH2 (shrub FMA up to CBH2 = 5 m).
# h_center: 0.5  1.5   2.5   3.5   4.5  5.5  6.5  7.5
_HAND_PAD = [0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1]
_HAND_PARAMS = {**_INRAE_PARAMS, "fma_can": 1.0, "fma_shrub": 0.5}

# R ffuelmetrics2(PADval = _HAND_PAD, FMAcan = 1, FMAshrub = 0.5, zval = 0:7, dz = 1,
# CanopyBDTreshFrac = 0.3, CBHThresholdRatio = 1/3), computed with Rscript (Dev 8fcccd2
# equivalent). One ladder of each kind: type 3 at 0.01 and 0.02, type 2 at 0.05.
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
    "CBD_cbh": np.nan,
    "CBD_cbh2": np.nan,
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


def test_fuel_metrics_single_stratum_gives_nan_second_load():
    # R: CBDval[2] is NA with one stratum
    metrics = _core([0.05])

    assert metrics["FL_k1"] == 0.05 * 0.5
    assert np.isnan(metrics["FL_k2"])


def test_fuel_metrics_defaults_are_those_of_the_median_threshold():
    metrics = _core(_HAND_PAD)

    assert metrics["LFL"] == metrics["LadderLoad_0p020"]
    assert metrics["CFL"] == metrics["CFL_0p020"]
    assert metrics["FSG"] == metrics["FSG_0p020"]
    assert metrics["ProfType"] == metrics["ProfType_0p020"]


def test_fuel_metrics_defaults_with_an_even_number_of_thresholds():
    # [0.01, 0.02, 0.05, 0.1] -> 0.05 (closest to the median in double precision, PR #48)
    metrics = _core(_HAND_PAD, ladder_fuel_bd_thresholds=[0.1, 0.02, 0.01, 0.05])

    assert metrics["LFL"] == metrics["LadderLoad_0p050"]
    assert metrics["CFL"] == metrics["CFL_0p050"]
    assert metrics["FSG"] == metrics["FSG_0p050"]
    assert metrics["ProfType"] == metrics["ProfType_0p050"]


def test_fuel_metrics_all_nan_profile_gives_the_zeros_of_r():
    # Strict parity: R turns a no-data pixel into a null profile (A5, corrected in story 3.5)
    metrics = _core([np.nan] * 8)

    assert metrics["CLFL"] == metrics["FL_k1"] == metrics["LFL"] == metrics["CFL"] == 0.0
    assert metrics["CBH2"] == 1.0
    assert metrics["ProfType"] == 1.0
    assert np.isnan(metrics["CTH_0p0100"])


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

    # the tile has no-data pixels (R zeros), the 4 profile types for each ladder threshold,
    # and the CBD_cbh* bands are all NaN in R (A1)
    assert np.isnan(pad).all(axis=0).any()
    for name in names:
        if name.startswith("ProfType_"):
            assert set(np.unique(ref[names.index(name)])) == {1.0, 2.0, 3.0, 4.0}, name
    for band, name in enumerate(names):
        np.testing.assert_allclose(result[band], ref[band], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=name)
