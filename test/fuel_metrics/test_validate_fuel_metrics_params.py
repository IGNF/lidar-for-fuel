import numpy as np
import pytest

from lidar_for_fuel.fuel_metrics.validate_fuel_metrics_params import (
    validate_fuel_metrics_params,
)

# The `fuel_metrics` section of configs/config.yaml, plus `pad_max`
_PARAMS = dict(
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
    pad_max=5.0,
)


def test_validate_fuel_metrics_params_default_config_passes():
    validate_fuel_metrics_params(**_PARAMS)


def test_validate_fuel_metrics_params_does_not_check_the_fma():
    # The FMA (value or map path) is checked by `read_fma`
    validate_fuel_metrics_params(**{**_PARAMS, "fma_can": np.nan, "fma_shrub": "fma.tif"})


@pytest.mark.parametrize("key", ["ladder_fuel_bd_thresholds", "canopy_top_bd_thresholds"])
@pytest.mark.parametrize(
    "thresholds",
    [[], None, 0.02, [0.01, np.nan], [0.01, np.inf], [0.01, 0.0], [-0.01, 0.02], [0.01, None], [0.01, "a"]],
    ids=["empty", "null", "not a list", "NaN", "inf", "zero", "negative", "null value", "string"],
)
def test_validate_fuel_metrics_params_invalid_thresholds_raise_with_the_key(key, thresholds):
    with pytest.raises(ValueError, match=key):
        validate_fuel_metrics_params(**{**_PARAMS, key: thresholds})


@pytest.mark.parametrize(
    "key, thresholds",
    [
        ("canopy_top_bd_thresholds", [0.01, 0.01004]),  # both CTH_0p0100
        ("canopy_top_bd_thresholds", [0.0012, 0.0012]),
        ("ladder_fuel_bd_thresholds", [0.0101, 0.0102]),  # both 0p010
        ("ladder_fuel_bd_thresholds", [0.02, 0.01, 0.02]),
    ],
    ids=["CTH same name", "CTH duplicate", "ladder same name", "ladder duplicate"],
)
def test_validate_fuel_metrics_params_thresholds_with_the_same_band_name_raise(key, thresholds):
    with pytest.raises(ValueError, match=rf"{key}.*same band name"):
        validate_fuel_metrics_params(**{**_PARAMS, key: thresholds})


@pytest.mark.parametrize(
    "key, thresholds",
    [("canopy_top_bd_thresholds", [0.01, 0.0101]), ("ladder_fuel_bd_thresholds", [0.010, 0.011])],
    ids=["CTH", "ladder"],
)
def test_validate_fuel_metrics_params_close_thresholds_with_distinct_names_pass(key, thresholds):
    validate_fuel_metrics_params(**{**_PARAMS, key: thresholds})


@pytest.mark.parametrize("key", ["bd_max", "pad_max"])
@pytest.mark.parametrize("value", [0, -1.0, np.nan, np.inf, None, "5"], ids=["0", "-1", "NaN", "inf", "null", "str"])
def test_validate_fuel_metrics_params_invalid_bound_raises_with_the_key(key, value):
    with pytest.raises(ValueError, match=key):
        validate_fuel_metrics_params(**{**_PARAMS, key: value})
