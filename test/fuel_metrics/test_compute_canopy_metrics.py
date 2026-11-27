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
_REF_PAD_RASTER_1M = _REF_DIR / "pad_profile/512500_6854997_pad_1m.tif"
# 45 bands computed by R lidarforfuel
_REF_FUEL_METRICS = _REF_DIR / "fuel_metrics/512500_6854997_fuel_metrics.tif"


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


def _assert_metrics(metrics, expected, rtol=1e-12, atol=0.0):
    """Compare the canopy metrics to the expected values (NaN equal to NaN)."""
    assert list(metrics) == _CANOPY_BANDS
    for name in _CANOPY_BANDS:
        np.testing.assert_allclose(metrics[name], expected[name], rtol=rtol, atol=atol, equal_nan=True, err_msg=name)


# --- R reference tile, pixel by pixel ------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_tile():
    """PAD profiles and R canopy metrics of the reference tile, as NumPy arrays.

    Returns `pad` (strata, rows, cols), `h_center`, `dz` and `ref` (canopy band, rows, cols),
    the bands of `ref` in `_CANOPY_BANDS` order.
    """
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        pad_bands = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_")]
        pad_names = [src.descriptions[i - 1] for i in pad_bands]
        pad = src.read(pad_bands).astype(float)
    with rasterio.open(_REF_FUEL_METRICS) as src:
        ref = src.read([src.descriptions.index(name) + 1 for name in _CANOPY_BANDS]).astype(float)

    dz, h_bottom = read_pad_heights(pad_names)

    return pad, h_bottom + dz / 2, dz, ref


# Pixels (row, col) of the reference tile, one per branch of the algorithm found on it. The tile
# has no pixel whose CBH2 falls back on CBHlow.
_REF_PIXELS = [
    # CTH 19.5 m: CBH (6 m) and CBH2 (9 m) found in 3A, CBHlow (2 m) in 3B -> the 3 CBD_cbh*.
    # Sensitive to canopy_bd_thresh_frac (CBH and CBH2 move with frac = 0.1)
    pytest.param(1, 25, id="cbh_cbh2_and_cbhlow"),
    # CTH 4.5 m: CBH = CBH2 = 2 m, no CBHlow. CTH × 0.333 = 1.4985 puts the 1.5 m stratum in 3A
    # (with 1/3, CanBDlowmax would not be NaN)
    pytest.param(0, 1, id="cbh_without_cbhlow"),
    # CBH = CBH2 = 2 m, CDH (3.5 m) below CTH (4.5 m)
    pytest.param(0, 5, id="cbh_and_cdh_below_cth"),
    # 3A max on its lowest stratum (1.5 m): no CanBDmin, no crown base, CBH2 falls back on CHT
    pytest.param(0, 11, id="no_crown_base_falls_back_on_cht"),
    # No crown base, CDH (2.5 m) below CTH (3.5 m)
    pytest.param(0, 21, id="no_crown_base_and_cdh_below_cth"),
    # No stratum reaches 0.01: CTH_0p0100, CDH and CBDcan NaN, only CTH_0p0012 (0.5 m)
    pytest.param(1, 22, id="no_canopy_top"),
    # PAD NaN on every stratum: prepared into a null profile, CBH2 = CHT as R
    pytest.param(0, 0, id="empty_pixel"),
]


@pytest.mark.parametrize("row, col", _REF_PIXELS)
def test_canopy_metrics_reference_pixel_matches_r(ref_tile, row, col):
    pad, h_center, dz, ref = ref_tile
    profile, _ = prepare_pad_profile(pad[:, row, col], smooth_pad=False)

    metrics, _ = compute_canopy_metrics(profile, h_center, dz, **_INRAE_PARAMS)

    # R writes float32: same tolerance as the parity of the 45 bands
    _assert_metrics(metrics, dict(zip(_CANOPY_BANDS, ref[:, row, col])), rtol=1e-5, atol=1e-6)


# --- Hand-written profiles, for the cases the reference tile does not cover ---------------------

# h_center: 0.5 1.5   2.5   3.5  4.5  5.5  6.5  7.5  8.5  9.5  10.5   11.5
_PROFILE_3A = [0.05, 0.05, 0.02, 0.0, 0.2, 0.1, 0.3, 0.5, 0.4, 0.2, 0.005, 0.001]


def test_canopy_metrics_half_metre_strata():
    # 3A profile with dz = 0.5 (h_center = 0.25 ... 5.75): bottoms are h - dz / 2, not h - 0.5.
    # Expected values computed by R ffuelmetrics2 (zval = 0, 0.5, ..., dz = 0.5).
    metrics, _ = _canopy(_PROFILE_3A, dz=0.5)

    _assert_metrics(
        metrics,
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
    )


def test_canopy_metrics_frac_0_gives_cbh_without_cbh2():
    # frac = 0: S = CanBDmin, so no stratum is < S on the way down and 3A finds CBH (4.0) but
    # no CBH2; CBH2 falls back on CHT, below CBH (as R, checked with Rscript)
    metrics, _ = _canopy(_PROFILE_3A, canopy_bd_thresh_frac=0.0)

    assert metrics["CBH"] == 4.0
    assert metrics["CBH2"] == 1.0
    np.testing.assert_allclose(metrics["CBDcan"], 0.19666666666666668, rtol=1e-12)  # 1.5 to 9.5
    # As R: CBD_cbh at the CBH stratum; no CBD_cbh2, as CBH2 does not come from 3A
    assert metrics["CBD_cbh"] == 0.2
    assert np.isnan(metrics["CBD_cbh2"])


def test_canopy_metrics_distinct_fma_values_computed_by_r():
    # The reference tile has fma_can == fma_shrub, so steps 1 and 4 (FMA switch at CHT, then
    # at CBH2) are checked here against R: values printed with "%.17g" by Rscript, R
    # lidarforfuel `ffuelmetrics2(PADval = pad, FMAcan = 0.3, FMAshrub = 0.15, zval = 0:15,
    # dz = 1, CanopyBDTreshFrac = 0.3, CBHThresholdRatio = 1/3, bdmax = 1)`. As R, CBD_cbh*
    # are read in the step 1 CBD
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
