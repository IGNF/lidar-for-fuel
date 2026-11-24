from pathlib import Path

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.fuel_metrics.compute_canopy_metrics import compute_canopy_metrics
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    prepare_pad_profile,
    read_pad_heights,
)

_REF_DIR = Path(__file__).resolve().parents[2] / "data/ref_inrae"
# R (lidarforfuel) PAD raster of a 500 m tile (story 3.1) and its fuel metrics computed by
# R Dev 00c654c (data/ref_inrae/fuel_metrics/make_fuel_metrics_00c654c.R)
_REF_PAD_RASTER_1M = _REF_DIR / "pad_profile/512500_6854997_pad_1m.tif"
_REF_FUEL_METRICS = _REF_DIR / "fuel_metrics/512500_6854997_fuel_metrics_00c654c.tif"

_CANOPY_BANDS = [
    "CTH_0p0012",
    "CTH_0p0100",
    "CDH",
    "CBH",
    "CBH2",
    "CBHlow",
    "CanBDmax",
    "CanBDmin",
    "CanBDlowmax",
    "CanBDlowmin",
    "CBDcan",
    "CBD_cbh",
    "CBD_cbh2",
    "CBD_cbhlow",
]

# Hand-written profiles: FMA = 1 and bd_max = 1, so CBD = PAD (values below 1)
_HAND_PARAMS = dict(
    fma_can=1.0,
    fma_shrub=1.0,
    bd_max=1.0,
    canopy_height_threshold=1.0,
    canopy_bd_thresh_frac=0.3,
    canopy_top_bd_thresholds=[0.0012, 0.01],
    cbh_threshold_ratio=1 / 3,
)

# Parameters of the R reference tile: INRAE config `ffuelmetrics2` section of
# data/ref_inrae/2026-09-28_fuel_metrics_zones-pilotes.yml
_INRAE_PARAMS = dict(
    fma_can=0.2543576,
    fma_shrub=0.2543576,
    bd_max=1.0,
    canopy_height_threshold=1.0,
    canopy_bd_thresh_frac=0.3,
    canopy_top_bd_thresholds=[0.0012, 0.01],
    cbh_threshold_ratio=0.333,
)


def _canopy(pad, dz=1.0, **params):
    """Canopy metrics of a profile of `dz` strata starting at 0 m (h_center = dz/2, 3·dz/2, ...)."""
    pad = np.asarray(pad, dtype=float)
    h_center = dz * np.arange(pad.size) + dz / 2

    return compute_canopy_metrics(pad, h_center, dz, **{**_HAND_PARAMS, **params})


def _assert_metrics(metrics, expected):
    """Compare the canopy metrics to the hand-computed values (NaN equal to NaN)."""
    assert list(metrics) == _CANOPY_BANDS
    for name in _CANOPY_BANDS:
        np.testing.assert_allclose(metrics[name], expected[name], rtol=1e-12, equal_nan=True, err_msg=name)


# --- Hand-written profiles, one per branch of the algorithm -----------------------------------

# h_center: 0.5 1.5   2.5   3.5  4.5  5.5  6.5  7.5  8.5  9.5  10.5   11.5
_PROFILE_3A = [0.05, 0.05, 0.02, 0.0, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.005, 0.001]


# Each case: PAD profile, parameters overriding _HAND_PARAMS (and dz), expected metrics.
_HAND_CASES = [
    pytest.param(
        # CTH = 9.5; 3A searches 3.17 < h <= 9.5: max 0.5 at 7.5, min 0 at 3.5 below it,
        # S = 0.3 * 0.5 = 0.15. CBH = lowest stratum >= S above 3.5 (4.5) -> 4.0;
        # CBH2: going down from 7.5, the first stratum below S is 5.5 -> 6.5 - 0.5 = 6.0.
        # 3B searches 1 < h <= 3.17: max 0.05 at 1.5, nothing between 1 m and 1.5 m -> no min.
        _PROFILE_3A,
        {},
        {
            "CTH_0p0012": 10.5,
            "CTH_0p0100": 9.5,
            "CDH": 9.5,  # highest stratum >= 0.1 * 0.5
            "CBH": 4.0,
            "CBH2": 6.0,
            "CBHlow": np.nan,
            "CanBDmax": 0.5,
            "CanBDmin": 0.0,
            "CanBDlowmax": 0.05,
            "CanBDlowmin": np.nan,
            "CBDcan": np.mean([0.3, 0.5, 0.4, 0.2]),  # 6.5 to 9.5
            "CBD_cbh": 0.2,
            "CBD_cbh2": 0.3,
            "CBD_cbhlow": np.nan,
        },
        id="3a_normal",
    ),
    pytest.param(
        # Flat profile: the 3A min (1.5-2.5 m) equals the 3A max -> no CBH, but CanBDmax and
        # CanBDmin are still given. 3B has no stratum between 1 m and its max (1.5 m) -> no CBHlow.
        [0.2] * 10,
        {},
        {
            "CTH_0p0012": 9.5,
            "CTH_0p0100": 9.5,
            "CDH": 9.5,
            "CBH": np.nan,
            "CBH2": 1.0,  # canopy_height_threshold
            "CBHlow": np.nan,
            "CanBDmax": 0.2,
            "CanBDmin": 0.2,
            "CanBDlowmax": 0.2,
            "CanBDlowmin": np.nan,
            "CBDcan": 0.2,  # 1.5 to 9.5
            "CBD_cbh": np.nan,
            "CBD_cbh2": np.nan,
            "CBD_cbhlow": np.nan,
        },
        id="3a_min_equal_max_gives_no_cbh_and_falls_back_on_cht",
    ),
    pytest.param(
        # CTH = 15.5. 3A (5.17 < h <= 15.5) is sparse (0.1) while 1.5-4.5 m is denser: CanBDmin
        # (0.2) > CanBDmax (0.1) -> no CBH. 3B (1 < h <= 5.17): max 0.6, first at 3.5 (tie with
        # 4.5), min 0.2 at 1.5, S = 0.7 * 0.2 + 0.3 * 0.6 = 0.32 -> CBHlow = 3.5 - 0.5 = 3.0.
        [0.0, 0.2, 0.3, 0.6, 0.6] + [0.1] * 11,
        {},
        {
            "CTH_0p0012": 15.5,
            "CTH_0p0100": 15.5,
            "CDH": 15.5,
            "CBH": np.nan,
            "CBH2": 3.0,  # = CBHlow
            "CBHlow": 3.0,
            "CanBDmax": 0.1,
            "CanBDmin": 0.2,
            "CanBDlowmax": 0.6,
            "CanBDlowmin": 0.2,
            "CBDcan": np.mean([0.6, 0.6] + [0.1] * 11),  # 3.5 to 15.5
            "CBD_cbh": np.nan,
            "CBD_cbh2": np.nan,
            "CBD_cbhlow": 0.6,
        },
        id="falls_back_on_cbhlow",
    ),
    pytest.param(
        # Only the first stratum is dense: CTH = 0.5 is below 1 m, no canopy search at all.
        # CDH falls back on CTH; no stratum between CBH2 (1 m) and 0.5 m -> CBDcan = 0.
        [0.5, 0.0, 0.0, 0.0],
        {},
        {
            "CTH_0p0012": 0.5,
            "CTH_0p0100": 0.5,
            "CDH": 0.5,
            "CBH": np.nan,
            "CBH2": 1.0,
            "CBHlow": np.nan,
            "CanBDmax": np.nan,
            "CanBDmin": np.nan,
            "CanBDlowmax": np.nan,
            "CanBDlowmin": np.nan,
            "CBDcan": 0.0,
            "CBD_cbh": np.nan,
            "CBD_cbh2": np.nan,
            "CBD_cbhlow": np.nan,
        },
        id="surface_only_falls_back_on_cht",
    ),
    pytest.param(
        # 3A profile with dz = 0.5 (h_center = 0.25 ... 5.75): bottoms are h - dz / 2, not h - 0.5.
        # Expected values computed by R ffuelmetrics2 (zval = 0, 0.5, ..., dz = 0.5), 17/11/2026.
        _PROFILE_3A,
        {"dz": 0.5},
        {
            "CTH_0p0012": 5.25,
            "CTH_0p0100": 4.75,
            "CDH": 4.75,
            "CBH": 2.0,  # stratum centred on 2.25 m
            "CBH2": 3.0,  # stratum centred on 3.25 m
            "CBHlow": np.nan,
            "CanBDmax": 0.5,
            "CanBDmin": 0.0,
            "CanBDlowmax": 0.02,
            "CanBDlowmin": np.nan,
            "CBDcan": 0.34999999999999998,
            "CBD_cbh": 0.2,
            "CBD_cbh2": 0.3,
            "CBD_cbhlow": np.nan,
        },
        id="half_metre_strata",
    ),
    pytest.param(
        # No stratum dense enough for the 0.01 canopy top (CBD < 0.01): CTH_0p0100 and CDH are
        # NaN, 3A finds nothing and CBH2 falls back on CHT; the 0.0012 canopy top still exists
        [0.005] * 6 + [0.0] * 4,
        {},
        {
            "CTH_0p0012": 5.5,
            "CTH_0p0100": np.nan,
            "CDH": np.nan,
            "CBH": np.nan,
            "CBH2": 1.0,
            "CBHlow": np.nan,
            "CanBDmax": np.nan,
            "CanBDmin": np.nan,
            "CanBDlowmax": np.nan,
            "CanBDlowmin": np.nan,
            "CBDcan": np.nan,
            "CBD_cbh": np.nan,
            "CBD_cbh2": np.nan,
            "CBD_cbhlow": np.nan,
        },
        id="no_stratum_above_0p01",
    ),
]


@pytest.mark.parametrize("pad, params, expected", _HAND_CASES)
def test_canopy_metrics_hand_profile(pad, params, expected):
    metrics, _ = _canopy(pad, **params)

    _assert_metrics(metrics, expected)


def test_canopy_metrics_null_profile():
    metrics, cbd = _canopy([0.0] * 10)

    _assert_metrics(
        metrics,
        {
            "CTH_0p0012": np.nan,
            "CTH_0p0100": np.nan,
            "CDH": np.nan,
            "CBH": np.nan,
            "CBH2": 1.0,
            "CBHlow": np.nan,
            "CanBDmax": np.nan,
            "CanBDmin": np.nan,
            "CanBDlowmax": np.nan,
            "CanBDlowmin": np.nan,
            "CBDcan": np.nan,  # R any(NA): no canopy top, but strata above CBH2
            "CBD_cbh": np.nan,
            "CBD_cbh2": np.nan,
            "CBD_cbhlow": np.nan,
        },
    )
    np.testing.assert_array_equal(cbd, np.zeros(10))


def test_canopy_metrics_max_on_several_strata_takes_the_first():
    # 3A max 0.5 at 6.5 and at 8.5, with a gap (0.05) between them. With the first one,
    # CBH2 is searched below 6.5 (gap at 5.5 -> 6.0); with the last one it would be 8.0.
    metrics, _ = _canopy([0.05, 0.05, 0.02, 0.0, 0.2, 0.1, 0.5, 0.05, 0.5, 0.2, 0.005, 0.001])

    assert metrics["CanBDmax"] == 0.5
    assert metrics["CBH"] == 4.0
    assert metrics["CBH2"] == 6.0


def test_canopy_metrics_frac_0_gives_cbh_without_cbh2():
    # frac = 0: S = CanBDmin, so no stratum is < S on the way down and 3A finds CBH (4.0) but
    # no CBH2; CBH2 falls back on CHT, below CBH (as R, checked with Rscript on 17/11/2026)
    metrics, _ = _canopy(_PROFILE_3A, canopy_bd_thresh_frac=0.0)

    assert metrics["CBH"] == 4.0
    assert metrics["CBH2"] == 1.0
    np.testing.assert_allclose(metrics["CBDcan"], 0.19666666666666668, rtol=1e-12)  # 1.5 to 9.5
    # R 00c654c: CBD_cbh at the CBH stratum; no CBD_cbh2, as CBH2 does not come from 3A
    assert metrics["CBD_cbh"] == 0.2
    assert np.isnan(metrics["CBD_cbh2"])


# --- No stratum dense enough for the canopy top (CBD < 0.01) -----------------------------------
# The profile with a 0.0012 canopy top only is the hand case `no_stratum_above_0p01` above.


def test_canopy_metrics_no_canopy_top_and_no_stratum_above_cbh2():
    # R any(h >= CBH2 & h <= NA) is FALSE when no stratum reaches CBH2 (1 m): CBDcan = 0, not NaN
    metrics, _ = _canopy([0.0])

    assert np.isnan(metrics["CTH_0p0100"])
    assert np.isnan(metrics["CDH"])
    assert metrics["CBDcan"] == 0.0


# --- Canopy dominant height (step 5) -----------------------------------------------------------


def test_canopy_metrics_cdh_below_cth():
    # CanBDmax = 0.5: the strata at 8.5 and 9.5 m reach 0.01 (CTH = 9.5) but not 0.1 * 0.5
    metrics, _ = _canopy([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.03, 0.02])

    assert metrics["CTH_0p0100"] == 9.5
    assert metrics["CBH2"] == 4.0
    assert metrics["CDH"] == 7.5
    np.testing.assert_allclose(metrics["CBDcan"], 0.5)  # 4.5 to 7.5


def test_canopy_metrics_cdh_capped_by_cth():
    # CanBDmax = 0.05: 0.1 * CanBDmax is below 0.01, so the 8.5 m stratum (0.006) gives
    # CDH = 8.5, above CTH_0p0100 = 6.5 -> capped to 6.5
    metrics, _ = _canopy([0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.005, 0.006])

    assert metrics["CTH_0p0100"] == 6.5
    assert metrics["CTH_0p0012"] == 8.5
    assert metrics["CDH"] == 6.5
    np.testing.assert_allclose(metrics["CBDcan"], 0.05)  # 4.5 to 6.5


# --- Search bounds compared exactly as in R (CTH × ratio) ---------------------------------------


def test_canopy_metrics_3a_excludes_the_stratum_at_cth_times_ratio():
    # CTH = 4.5 and ratio = 1/3: CTH * ratio = 1.5, a stratum centre. 3A keeps h > 1.5 (as R):
    # the dense 1.5 m stratum (0.8) is left to 3B. Were it included, CanBDmax would be 0.8.
    metrics, _ = _canopy([0.0, 0.8, 0.0, 0.3, 0.2])

    assert metrics["CTH_0p0100"] == 4.5
    assert metrics["CanBDmax"] == 0.3
    assert metrics["CanBDmin"] == 0.0
    assert metrics["CBH"] == 3.0
    assert metrics["CBH2"] == 3.0
    assert metrics["CanBDlowmax"] == 0.8  # 1 < h <= 1.5


# --- CBD recomputed with CBH2 (step 4) and thresholds --------------------------------------------


def test_canopy_metrics_cbd_recomputed_with_cbh2():
    # Step 1 uses the canopy FMA above 1 m, step 4 above CBH2 (6.0 m): the strata from
    # 1.5 to 5.5 m switch to the shrub FMA. The metrics are those of the 3A profile.
    pad = np.asarray(_PROFILE_3A)
    metrics, cbd = _canopy(pad, fma_can=1.0, fma_shrub=0.5)

    assert metrics["CBH2"] == 6.0
    assert metrics["CDH"] == 9.5
    np.testing.assert_allclose(metrics["CBDcan"], np.mean([0.3, 0.5, 0.4, 0.2]))
    np.testing.assert_array_equal(cbd, np.concatenate([pad[:6] * 0.5, pad[6:]]))


def test_canopy_metrics_distinct_fma_values_computed_by_r():
    # The reference tile has fma_can == fma_shrub, so steps 1 and 4 (FMA switch at CHT, then
    # at CBH2) are checked here against R: values printed with "%.17g" by Rscript, R
    # lidarforfuel `ffuelmetrics2(PADval = pad, FMAcan = 0.3, FMAshrub = 0.15, zval = 0:15,
    # dz = 1, CanopyBDTreshFrac = 0.3, CBHThresholdRatio = 1/3, bdmax = 1)`, 17/11/2026, and
    # CBD_cbh* with R Dev 00c654c, 23/11/2026. As R, CBD_cbh* are read in the step 1 CBD
    # (canopy FMA above CHT): 0.3 x 0.3 at the CBH stratum, not 0.3 x 0.15 (step 4 CBD).
    pad = np.array([0.2, 0.1, 0.3, 0.6, 0.6, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.4, 0.4, 0.4, 0.4])
    metrics, cbd = _canopy(pad, fma_can=0.3, fma_shrub=0.15)

    _assert_metrics(
        metrics,
        {
            "CTH_0p0012": 15.5,
            "CTH_0p0100": 15.5,
            "CDH": 15.5,
            "CBH": 2.0,
            "CBH2": 6.0,
            "CBHlow": 2.0,
            "CanBDmax": 0.12,
            "CanBDmin": 0.029999999999999999,
            "CanBDlowmax": 0.17999999999999999,
            "CanBDlowmin": 0.029999999999999999,
            "CBDcan": 0.111,
            "CBD_cbh": 0.089999999999999997,  # stratum centred on 2.5 m
            "CBD_cbh2": 0.12,  # 6.5 m
            "CBD_cbhlow": 0.089999999999999997,  # 2.5 m
        },
    )
    # Step 4: shrub FMA up to CBH2 = 6 m (centres 0.5 to 5.5), canopy FMA above
    np.testing.assert_array_equal(cbd, np.concatenate([pad[:6] * 0.15, pad[6:] * 0.3]))


def test_canopy_metrics_cth_names_follow_the_configured_thresholds():
    # Names formatted as R ("%0.4f"), in the configured order; CTH is the one of the largest threshold
    metrics, _ = _canopy(_PROFILE_3A, canopy_top_bd_thresholds=[0.05, 0.0012])

    assert list(metrics)[:2] == ["CTH_0p0500", "CTH_0p0012"]
    assert metrics["CTH_0p0500"] == 9.5
    assert metrics["CTH_0p0012"] == 10.5
    assert metrics["CDH"] == 9.5  # min(CDH, CTH_0p0500)


# --- Parity with R on the reference tile --------------------------------------------------------


def test_canopy_metrics_parity_with_r():
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        pad_bands = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_")]
        pad_names = [src.descriptions[i - 1] for i in pad_bands]
        pad = src.read(pad_bands).astype(float)
    with rasterio.open(_REF_FUEL_METRICS) as src:
        ref = {name: src.read(i).astype(float) for i, name in enumerate(src.descriptions, start=1)}

    dz, h_bottom = read_pad_heights(pad_names)
    h_center = h_bottom + dz / 2

    _, height, width = pad.shape
    result = {name: np.full((height, width), np.nan) for name in _CANOPY_BANDS}
    for row in range(height):
        for col in range(width):
            profile, _ = prepare_pad_profile(pad[:, row, col], smooth_pad=False)
            metrics, _ = compute_canopy_metrics(profile, h_center, dz, **_INRAE_PARAMS)
            for name in _CANOPY_BANDS:
                result[name][row, col] = metrics[name]

    for name in _CANOPY_BANDS:
        np.testing.assert_allclose(result[name], ref[name], rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=name)


@pytest.mark.parametrize("name", ["CBH", "CBHlow", "CanBDmin", "CanBDlowmin", "CBD_cbh", "CBD_cbh2", "CBD_cbhlow"])
def test_reference_tile_exercises_the_band(name):
    # Guard on the parity test: the reference tile has both valid and NaN values for the
    # optional canopy metrics, so the parity test covers the branches that set them.
    with rasterio.open(_REF_FUEL_METRICS) as src:
        band = src.read(src.descriptions.index(name) + 1)

    assert np.isfinite(band).any() and np.isnan(band).any()
