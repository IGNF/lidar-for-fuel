"""
Input layer of the fuel metrics computation, ported from R `ffuelmetrics2`:
- read the strata heights from the band names of the `pad_profile_1m` raster;
- bring the PAD bands of that raster back into [0, 5] (cap removed from step 2 after the INRAE
  answer on the PAD cap at 5);
- read the fuel mass area (FMA), either a configured value or the FMA map, on the PAD grid;
- prepare a PAD profile and turn it into a CBD profile;
- pick the default ladder fuel threshold (rule of step, needed from the configuration onwards).
"""

import logging
import os
import re
from collections.abc import Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

logger = logging.getLogger(__name__)

_PAD_BAND_PATTERN = re.compile(r"^PAD_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")

# PAD range (m²/m³) of the INRAE PAD specification ("0-5, capped at 5"). Step 2 writes the raw PAD,
# as R `pad_metrics` (INRAE answer on the PAD cap at 5); the cap is applied when step 3 reads the
# PAD raster (`cap_pad`)
PAD_MIN = 0.0
PAD_MAX = 5.0

# Valid range of the FMA map values (kg/m²), from the FMA map specification
FMA_MAP_MIN = 0.05
FMA_MAP_MAX = 0.5
# Largest shift (fraction of a pixel) tolerated between the PAD grid and the FMA map grid.
# Covers the 6 cm offset between the rounded CosiaFrance origin of the PAD rasters
# (98039.75, 7111486.75) and the real one of the national maps (98039.6885, 7111486.6972).
_FMA_GRID_TOLERANCE = 0.1


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


def cap_pad(pad: np.ndarray, band_names: Sequence[str], pad_max: float = PAD_MAX) -> tuple[np.ndarray, int]:
    """Bring the PAD bands of the `{stem}_pad_profile_1m.tif` raster back into [0, `pad_max`].

    Since the INRAE answer on the PAD cap at 5 (23/11/2026; cap removed from step 2 on
    24/11/2026), step 2 writes the raw PAD, as R `pad_metrics`: it can go above 5 (up to 7.01
    on the pilot zone), or be inf. This post-processing applies, when step 3 reads the raster,
    the cap that step 2 applied before (`np.clip(pad, 0, 5)`):

    - values above `pad_max` (+inf included) become `pad_max`;
    - negative values (-inf included) become 0;
    - NaN stays NaN, so a pixel without data stays one (D2).

    R does not cap. With the configured FMA (0.254) and `smooth_pad = False` the metrics do not
    change, as `5 × FMA ≥ bd_max`. They can with an FMA below 0.2 (FMA map), with smoothing,
    or where the PAD is +inf (5 instead of 0, A6).

    Args:
        pad (np.ndarray): PAD raster (m²/m³), of shape (bands, rows, cols), in band order.
        band_names (Sequence[str]): Names of the PAD bands, in band order (for the log).
        pad_max (float): Upper bound of the PAD (m²/m³). Default `PAD_MAX` (5).

    Returns:
        tuple[np.ndarray, int]: The capped raster (new array, same dtype for a float raster;
            the input is not modified) and the number of values brought back into the range, also logged
            in info, band by band.

    Raises:
        ValueError: If there are not as many band names as PAD bands.
    """
    pad = np.asarray(pad)
    if len(band_names) != pad.shape[0]:
        raise ValueError(f"{len(band_names)} band names for {pad.shape[0]} PAD band(s).")

    out_of_range = (pad < PAD_MIN) | (pad > pad_max)
    n_by_band = out_of_range.reshape(pad.shape[0], -1).sum(axis=1)
    n_capped = int(n_by_band.sum())
    if n_capped:
        by_band = ", ".join(f"{name}: {n}" for name, n in zip(band_names, n_by_band) if n)
        logger.info(
            "PAD raster: %d value(s) outside [%g, %g] brought back into it (%s)",
            n_capped,
            PAD_MIN,
            pad_max,
            by_band,
        )

    return np.clip(pad, PAD_MIN, pad_max).astype(pad.dtype, copy=False), n_capped


def read_fma(
    fma: float | str | os.PathLike,
    transform: Affine,
    shape: tuple[int, int],
    crs: CRS | None = None,
) -> np.ndarray:
    """Get the fuel mass area (FMA, kg/m²) of every pixel of the PAD grid.

    The FMA converts the PAD into a bulk density (`CBD = PAD × FMA`). It depends on the
    dominant species, so it is either a single configured value or the national FMA map
    (single-band raster, 10 m, values in [0.05, 0.5] kg/m²). As in R `ffuelmetrics2`,
    `fma_can` and `fma_shrub` each accept one or the other; the same map can be given to both.

    - **Value**: the grid is filled with it (current behaviour, no check here).
    - **Map path**: the window of the map covering the PAD grid is read, each PAD pixel
      taking the FMA pixel whose centre is the nearest. The two grids must have the same
      resolution and be aligned within `_FMA_GRID_TOLERANCE` of a pixel (the 6 cm offset of
      the CosiaFrance origin is tolerated). NoData pixels and pixels outside the map are NaN.

    Args:
        fma (float | str | os.PathLike): FMA value (kg/m²) or path to the FMA map.
        transform (Affine): Affine transform of the PAD raster.
        shape (tuple[int, int]): (height, width) of the PAD raster.
        crs (CRS | None): CRS of the PAD raster, compared with the map CRS when both are EPSG codes.

    Returns:
        np.ndarray: FMA (kg/m²) of every PAD pixel, float64, of shape `shape`.

    Raises:
        ValueError: If the map has another CRS or resolution than the PAD grid, if it is not
            aligned on it, or if one of its values in the window is outside [0.05, 0.5].
    """
    if not isinstance(fma, (str, os.PathLike)):
        return np.full(shape, float(fma))

    height, width = shape
    with rasterio.open(fma) as src:
        if crs is not None and src.crs is not None and None not in (crs.to_epsg(), src.crs.to_epsg()):
            if crs.to_epsg() != src.crs.to_epsg():
                raise ValueError(f"FMA map {fma}: CRS EPSG:{src.crs.to_epsg()} differs from PAD EPSG:{crs.to_epsg()}.")
        if not np.allclose((src.transform.a, src.transform.e), (transform.a, transform.e)):
            raise ValueError(f"FMA map {fma}: resolution {src.res} differs from PAD ({transform.a}, {-transform.e}).")

        # Position of the PAD grid in the map grid, in pixels, and nearest whole pixel
        col_offset = (transform.c - src.transform.c) / src.transform.a
        row_offset = (transform.f - src.transform.f) / src.transform.e
        col0, row0 = round(col_offset), round(row_offset)
        shift = max(abs(col_offset - col0), abs(row_offset - row0))
        if shift > _FMA_GRID_TOLERANCE:
            raise ValueError(
                f"FMA map {fma}: grid not aligned on the PAD grid (shift of {shift:.2f} pixel, "
                f"tolerance {_FMA_GRID_TOLERANCE})."
            )

        # Part of the PAD grid covered by the map; the rest stays NaN
        values = np.full(shape, np.nan)
        top, left = max(row0, 0), max(col0, 0)
        bottom, right = min(row0 + height, src.height), min(col0 + width, src.width)
        if bottom > top and right > left:
            window = Window(left, top, right - left, bottom - top)
            data = src.read(1, window=window, masked=True).astype(float).filled(np.nan)
            values[top - row0 : bottom - row0, left - col0 : right - col0] = data

    valid = values[np.isfinite(values)]
    out_of_range = (valid < FMA_MAP_MIN) | (valid > FMA_MAP_MAX)
    if out_of_range.any():
        raise ValueError(
            f"FMA map {fma}: {int(out_of_range.sum())} value(s) outside [{FMA_MAP_MIN}, {FMA_MAP_MAX}] kg/m² "
            f"(min {valid.min()}, max {valid.max()})."
        )

    return values


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
    """Smooth a PAD profile: each stratum becomes the mean of itself and its 2 neighbours.

    Port of R `slider::slide_dbl(x, mean, .before = 1, .after = 1)`, called by
    `ffuelmetrics2` when `smooth_pad = TRUE`. Only used when `smooth_pad` is true; with the
    current INRAE config (`smooth_pad: false`) it is never called.

    **What is computed.** The window is centred on each stratum. At the edges, it only
    holds the 2 available values (no division by 3):

    - first stratum: mean(x[0], x[1]);
    - middle strata: mean(x[i-1], x[i], x[i+1]);
    - last stratum: mean(x[n-2], x[n-1]).

    Example: `[0, 3, 0, 0]` -> `[1.5, 1, 1, 0]`.

    **Why not a plain `(a + b + c) / 3`.** R `mean()` does not compute that way. In C, it:

    1. sums the values in *long double* (80-bit extended precision, vs 64 bits for float64);
    2. divides by n;
    3. adds a correction pass: `sum(x - mean) / n`, which would be 0 with exact arithmetic,
       so it is the rounding error of steps 1-2, added back;
    4. rounds the result to double.

    A plain float64 mean differs from R by 1 ulp (~4e-16) on about 1 value out of 4
    (measured against `slider::slide_dbl` on random profiles: 16/60 values with
    `pandas.rolling`, 0/192 with this function). It looks negligible, but the next steps
    compare the CBD to thresholds with strict operators (`CBD >= threshold` for CTH and
    ladder fuel, `CBD[k-1] < S` for CBH2). A value sitting exactly on a threshold can
    then fall on the other side, while the INRAE expects a zero difference with R.
    So this function reproduces the 4 steps above, vectorized over the whole profile.

    **Limits.**

    - `np.longdouble` is 80-bit only on Linux x86-64 (the dev and production platform,
      same as R's long double). On Windows or ARM macOS it is a plain float64: the result
      stays correct to 1 ulp, but is no longer bit-identical to R.
    - NaN is used below as the "no neighbour" marker at the edges. This is safe because
      `prepare_pad_profile` sets every NaN of the PAD to 0 before smoothing.

    Args:
        values (np.ndarray): Cleaned 1-D PAD profile (no NaN), at least 2 values.

    Returns:
        np.ndarray: Smoothed profile (float64), same length.
    """
    x = values.astype(np.longdouble)

    # Neighbours of each stratum: shift the profile by one stratum down / up.
    # NaN marks the missing neighbour (below the first stratum, above the last one).
    left = np.concatenate(([np.nan], x[:-1])).astype(np.longdouble)
    right = np.concatenate((x[1:], [np.nan])).astype(np.longdouble)
    has_left, has_right = ~np.isnan(left), ~np.isnan(right)
    count = (1 + has_left + has_right).astype(np.longdouble)  # window size: 2 at the edges, 3 elsewhere

    # Steps 1-2: sum in long double, in R's order (left, centre, right), then divide.
    total = np.where(has_left, left, 0) + x
    total = total + np.where(has_right, right, 0)
    mean = total / count

    # Step 3: correction pass, sum of the deviations to the mean = rounding error of steps 1-2.
    correction = np.where(has_left, left - mean, 0) + (x - mean)
    correction = correction + np.where(has_right, right - mean, 0)

    # Step 4: add the correction back and round to double.
    return (mean + correction / count).astype(np.float64)


def compute_cbd(
    pad: np.ndarray,
    h_center: np.ndarray,
    fma_height_limit: float,
    fma_can: float,
    fma_shrub: float,
    bd_max: float,
) -> np.ndarray:
    """Convert a PAD profile into a CBD (Canopy Bulk Density) profile.

    `CBD = min(PAD × FMA, bd_max)`, with `fma_can` for the strata whose centre is strictly
    above `fma_height_limit` and `fma_shrub` for the others. The limit is the canopy height
    threshold at step 1 of the algorithm and CBH2 when the CBD is recomputed at step 4.

    Args:
        pad (np.ndarray): Prepared 1-D PAD profile (m²/m³), see `prepare_pad_profile`.
        h_center (np.ndarray): Height above ground of the centre of each stratum (m),
            i.e. `h_bottom + dz / 2`.
        fma_height_limit (float): Height (m) above which the canopy FMA is used.
        fma_can (float): Fuel mass area of the canopy fuel (kg/m²) for this pixel, see `read_fma`.
        fma_shrub (float): Fuel mass area of the shrub fuel (kg/m²) for this pixel, see `read_fma`.
        bd_max (float): Maximum CBD value (kg/m³).

    Returns:
        np.ndarray: CBD profile (kg/m³), same shape as `pad`.
    """
    fma = np.where(np.asarray(h_center) > fma_height_limit, fma_can, fma_shrub)

    return np.minimum(np.asarray(pad) * fma, bd_max)


def select_default_ladder_threshold(thresholds: Sequence[float]) -> float:
    """Pick the ladder fuel threshold used for the default `LFL`, `CFL`, `FSG` and `ProfType` bands.

    The threshold closest to the median, as R `ffuelmetrics2` does
    (`thresholds[which.min(abs(thresholds - median(thresholds)))]`). With an even number of
    thresholds, two of them are usually *not* at exactly the same distance in double
    precision, so rounding decides (e.g. `[0.01, 0.02, 0.05, 0.1]` gives 0.05, as in R).
    Only on an exact tie is the first one **in the configured order** kept (the list is
    not sorted).

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
