"""
Check of the `fuel_metrics` section of the configuration, run once by `main_fuel_metrics` before
the first tile: an abnormal value (duplicated thresholds, null or infinite bound) would otherwise
give wrong fuel metrics maps without any error. The FMA is checked by `read_fma`.
"""

import math
from collections.abc import Callable, Sequence
from numbers import Real

from lidar_for_fuel.fuel_metrics.compute_canopy_metrics import _cth_name
from lidar_for_fuel.fuel_metrics.compute_ladder_metrics import ladder_suffix


def validate_fuel_metrics_params(
    ladder_fuel_bd_thresholds: Sequence[float],
    canopy_top_bd_thresholds: Sequence[float],
    bd_max: float,
    pad_max: float,
    **other_params,
) -> None:
    """Check the parameters of the fuel metrics that would give wrong maps without an error.

    - `ladder_fuel_bd_thresholds` and `canopy_top_bd_thresholds`: a non-empty list of finite
      numbers > 0, without two thresholds of the same band name (`0p010` for 0.0101 and
      0.0102, `CTH_0p0100` for 0.01 and 0.01004), which would merge two blocks of bands;
    - `bd_max` and `pad_max`: a finite number > 0 (`null` refused: there is no switch to
      remove the PAD cap).

    Args:
        ladder_fuel_bd_thresholds (Sequence[float]): CBD thresholds of the ladder fuel (kg/m³).
        canopy_top_bd_thresholds (Sequence[float]): CBD thresholds of the canopy top (kg/m³).
        bd_max (float): Maximum CBD value (kg/m³).
        pad_max (float): Upper bound of the PAD (m²/m³) applied on read.
        **other_params: The other parameters of the `fuel_metrics` section, not checked here
            (the FMA is checked by `read_fma`), so that the whole section can be passed.

    Raises:
        ValueError: If a parameter is invalid; the message names its key.
    """
    _check_thresholds("ladder_fuel_bd_thresholds", ladder_fuel_bd_thresholds, ladder_suffix)
    _check_thresholds("canopy_top_bd_thresholds", canopy_top_bd_thresholds, _cth_name)
    _check_positive("bd_max", bd_max)
    _check_positive("pad_max", pad_max)


def _check_positive(key: str, value: object) -> None:
    """Raise a ValueError naming `key` if `value` is not a finite number > 0."""
    if isinstance(value, bool) or not isinstance(value, Real) or not (math.isfinite(value) and value > 0):
        raise ValueError(f"{key} must be a finite number > 0, got {value!r}.")


def _check_thresholds(key: str, thresholds: object, band_name: Callable[[float], str]) -> None:
    """Raise a ValueError naming `key` if `thresholds` is not a non-empty list of finite numbers
    > 0 with one band name (`band_name`) per threshold."""
    if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence) or not thresholds:
        raise ValueError(f"{key} must be a non-empty list of thresholds, got {thresholds!r}.")
    for threshold in thresholds:
        _check_positive(f"{key} value", threshold)

    names = [band_name(threshold) for threshold in thresholds]
    if len(set(names)) < len(names):
        raise ValueError(f"{key} has thresholds with the same band name: {list(thresholds)} -> {names}.")
