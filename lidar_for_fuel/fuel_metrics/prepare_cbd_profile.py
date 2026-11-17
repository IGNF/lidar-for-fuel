"""
Input layer of the fuel metrics computation, ported from R `ffuelmetrics2`:
- read the strata heights from the band names of the `pad_profile_1m` raster;
- prepare a PAD profile and turn it into a CBD profile;
- pick the default ladder fuel threshold (rule of step, needed from the configuration onwards).
"""

import logging
import re
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_PAD_BAND_PATTERN = re.compile(r"^PAD_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


def read_pad_heights(band_names: Sequence[str | None]) -> tuple[float, np.ndarray]:
    """Read the strata of the `{stem}_pad_profile_1m.tif` raster from its band names.

    Every band is a `PAD_{dz}_{h_bottom}` stratum, written in ascending height order by
    step 2 (`export_raster`). The bands are used in that order (the profile is not re-sorted),
    so the names must describe a regular profile: same thickness, contiguous, ascending.

    Args:
        band_names (Sequence[str | None]): Band names (descriptions) of the PAD raster,
            in band order.

    Returns:
        tuple[float, np.ndarray]:
            - dz: thickness of the strata (m);
            - h_bottom: height above ground of the bottom of each stratum (m), in band order.

    Raises:
        ValueError: If there is no band, if a band is not named `PAD_{dz}_{h_bottom}`, or if
            the strata do not have one non-null thickness or are not contiguous and ascending.
    """
    if not band_names:
        raise ValueError("The PAD raster has no band.")

    strata = []
    for name in band_names:
        match = _PAD_BAND_PATTERN.match(name) if name else None
        if match is None:
            raise ValueError(f"Band {name!r} is not a PAD_{{dz}}_{{h_bottom}} band.")
        strata.append((float(match.group(1)), float(match.group(2))))

    thicknesses = sorted({dz for dz, _ in strata})
    if len(thicknesses) > 1:
        raise ValueError(f"PAD bands have different strata thicknesses (dz): {thicknesses}.")
    dz = thicknesses[0]
    if dz <= 0:
        raise ValueError(f"PAD bands have a null strata thickness (dz = {dz}).")

    h_bottom = np.array([h for _, h in strata])
    if not np.allclose(h_bottom, h_bottom[0] + dz * np.arange(len(h_bottom))):
        raise ValueError(f"PAD strata are not contiguous and ascending (dz = {dz}): h_bottom = {h_bottom.tolist()}.")

    return dz, h_bottom


def prepare_pad_profile(pad: np.ndarray, smooth_pad: bool) -> tuple[np.ndarray, int]:
    """Clean a vertical PAD profile and optionally smooth it, as R `ffuelmetrics2` does.

    Non-finite (NaN, inf) and negative values are set to 0, then, if `smooth_pad`, each
    value is replaced by the mean of itself and its 2 neighbours. At the edges the mean
    is taken over the 2 available values (R `slider::slide_dbl(mean, .before = 1, .after = 1)`).

    Args:
        pad (np.ndarray): 1-D PAD profile (m²/m³), one value per stratum, ordered by height.
        smooth_pad (bool): Whether to apply the 3-strata moving average.

    Returns:
        tuple[np.ndarray, int]: The prepared profile (new array, the input is not modified)
            and the number of values set to 0.
    """
    pad = np.asarray(pad, dtype=float)
    invalid = ~np.isfinite(pad) | (pad < 0)
    n_corrected = int(invalid.sum())

    cleaned = np.where(invalid, 0.0, pad)
    if n_corrected:
        logger.debug("PAD profile: %d non-finite or negative value(s) set to 0", n_corrected)

    if smooth_pad and cleaned.size > 1:
        cleaned = _moving_mean_3(cleaned)

    return cleaned, n_corrected


def _moving_mean_3(values: np.ndarray) -> np.ndarray:
    """Centred moving mean over 3 values (2 at the edges), bit-identical to R `mean()`.

    R `mean()` sums in long double, divides by n, then adds a correction pass
    `sum(x - mean) / n` before rounding to double. Reproducing it (instead of a plain
    float64 mean, which differs by 1 ulp on some values) keeps the strict parity with R
    that the threshold comparisons of the next steps rely on. `np.longdouble` is 80-bit
    extended precision on Linux x86-64, like R's long double.

    Args:
        values (np.ndarray): 1-D array of at least 2 values.

    Returns:
        np.ndarray: Smoothed values (float64), same length.
    """
    x = values.astype(np.longdouble)
    left = np.concatenate(([np.nan], x[:-1])).astype(np.longdouble)
    right = np.concatenate((x[1:], [np.nan])).astype(np.longdouble)
    has_left, has_right = ~np.isnan(left), ~np.isnan(right)
    count = (1 + has_left + has_right).astype(np.longdouble)

    # Same summation order as R: left, centre, right
    total = np.where(has_left, left, 0) + x
    total = total + np.where(has_right, right, 0)
    mean = total / count

    correction = np.where(has_left, left - mean, 0) + (x - mean)
    correction = correction + np.where(has_right, right - mean, 0)

    return (mean + correction / count).astype(np.float64)


def compute_cbd(
    pad: np.ndarray,
    h_center: np.ndarray,
    height_limit: float,
    fma_can: float,
    fma_shrub: float,
    bd_max: float,
) -> np.ndarray:
    """Convert a PAD profile into a CBD (Canopy Bulk Density) profile.

    `CBD = min(PAD × FMA, bd_max)`, with `fma_can` for the strata whose centre is strictly
    above `height_limit` and `fma_shrub` for the others. The limit is the canopy height
    threshold at step 1 of the algorithm and CBH2 when the CBD is recomputed at step 4.

    Args:
        pad (np.ndarray): Prepared 1-D PAD profile (m²/m³), see `prepare_pad_profile`.
        h_center (np.ndarray): Height above ground of the centre of each stratum (m),
            i.e. `h_bottom + dz / 2`.
        height_limit (float): Height (m) above which the canopy FMA is used.
        fma_can (float): Fuel mass area of the canopy fuel (kg/m²).
        fma_shrub (float): Fuel mass area of the shrub fuel (kg/m²).
        bd_max (float): Maximum CBD value (kg/m³).

    Returns:
        np.ndarray: CBD profile (kg/m³), same shape as `pad`.
    """
    fma = np.where(np.asarray(h_center) > height_limit, fma_can, fma_shrub)

    return np.minimum(np.asarray(pad) * fma, bd_max)


def select_default_ladder_threshold(thresholds: Sequence[float]) -> float:
    """Pick the ladder fuel threshold used for the default `LFL`, `CFL`, `FSG` and `ProfType` bands.

    Rule of R PR #48 (`thresholds[which.min(abs(thresholds - median(thresholds)))]`): the
    threshold closest to the median. With an even number of thresholds, two of them are
    usually *not* at exactly the same distance in double precision, so rounding decides
    (e.g. `[0.01, 0.02, 0.05, 0.1]` gives 0.05, as in R). Only on an exact tie is the first
    one **in the configured order** kept (the list is not sorted).

    Args:
        thresholds (Sequence[float]): Ladder fuel CBD thresholds, as configured (kg/m³).

    Returns:
        float: The default threshold.

    Raises:
        ValueError: If `thresholds` is empty.
    """
    values = np.asarray(thresholds, dtype=float)
    if values.size == 0:
        raise ValueError("ladder_fuel_bd_thresholds must contain at least one threshold.")

    return float(values[np.argmin(np.abs(values - np.median(values)))])
