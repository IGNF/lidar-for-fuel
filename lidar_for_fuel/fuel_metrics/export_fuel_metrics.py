"""
Fuel metrics of a whole tile, from its `{stem}_pad_profile_1m.tif` PAD raster to the
`{stem}_fuel_metrics.tif` GeoTIFF:
- `compute_fuel_metrics_raster`: on arrays, caps the PAD (`cap_pad`), applies
  `fuel_metrics_core` pixel by pixel and logs what was corrected;
- `export_fuel_metrics`: on files, reads the PAD raster and the FMA (value or map), calls the
  former and writes the GeoTIFF on the grid of the PAD raster (`write_geotiff`).
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio

from lidar_for_fuel.commons.write_geotiff import write_geotiff
from lidar_for_fuel.fuel_metrics.calculate_fuel_metrics import fuel_metrics_core
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    PAD_MAX,
    PAD_MIN,
    cap_pad,
    read_pad_heights,
)
from lidar_for_fuel.fuel_metrics.read_fma import read_fma

logger = logging.getLogger(__name__)

# Suffix of the step 2 PAD raster, removed from the input name to build the output one
_PAD_RASTER_SUFFIX = "_pad_profile_1m"


def compute_fuel_metrics_raster(
    pad: np.ndarray,
    band_names: Sequence[str | None],
    fma_can: np.ndarray,
    fma_shrub: np.ndarray,
    smooth_pad: bool,
    ladder_fuel_bd_thresholds: Sequence[float],
    canopy_height_threshold: float,
    canopy_bd_thresh_frac: float,
    canopy_top_bd_thresholds: Sequence[float],
    cbh_threshold_ratio: float,
    bd_max: float,
    m_perrakis: float,
    pad_max: float = PAD_MAX,
) -> dict[str, np.ndarray]:
    """Compute the fuel metrics of every pixel of a PAD raster.

    The strata heights are read from the band names (`read_pad_heights`), the PAD is brought
    back into [0, `pad_max`] (`cap_pad`), then `fuel_metrics_core` is applied to
    the profile of each pixel, with the FMA of that pixel. A pixel with no data (all-NaN PAD,
    or no FMA) is NaN on all the bands (handled by `fuel_metrics_core`). The computation
    never stops on an atypical tile; at the end of the tile, one info log gives:

    - the number of pixels processed (all the pixels of the tile);
    - the pixels set to NaN: those with an all-NaN PAD, and those with a PAD but no FMA
      (`fma_can` or `fma_shrub` NaN), so that the two counts do not overlap;
    - the partly NaN PAD profiles of the pixels with data, whose NaN are set to 0;
    - the PAD values brought back into [0, `pad_max`] by `cap_pad`, split into finite values
      above `pad_max`, +inf (CBD min(`pad_max` × FMA, `bd_max`) instead of 0 in R) and
      negative values (-inf included).

    A warning is also logged when no pixel has data (water, edge of the survey): the raster
    is then all NaN.

    Args:
        pad (np.ndarray): Raw PAD raster (m²/m³), of shape (bands, rows, cols), NaN for no data.
        band_names (Sequence[str]): Names of the PAD bands (`PAD_{dz}_{h_bottom}`), in band order.
        fma_can (np.ndarray): Fuel mass area of the canopy fuel (kg/m²) of each pixel, of
            shape (rows, cols), NaN for no FMA (see `read_fma`).
        fma_shrub (np.ndarray): Fuel mass area of the shrub fuel (kg/m²) of each pixel, of
            shape (rows, cols), NaN for no FMA (see `read_fma`).
        smooth_pad (bool): Whether to smooth the PAD profile over 3 strata.
        ladder_fuel_bd_thresholds (Sequence[float]): CBD thresholds of the ladder fuel (kg/m³).
        canopy_height_threshold (float): Height (m) separating surface and canopy fuel (CHT).
        canopy_bd_thresh_frac (float): Fraction between CanBDmin and CanBDmax giving the
            crown base threshold.
        canopy_top_bd_thresholds (Sequence[float]): CBD thresholds of the canopy top (kg/m³).
        cbh_threshold_ratio (float): Fraction of CTH separating the upper and lower crown
            base searches.
        bd_max (float): Maximum CBD value (kg/m³).
        m_perrakis (float): M factor of the equivalent surface fuel load (Perrakis 2025).
        pad_max (float): Upper bound of the PAD (m²/m³) applied on read. Default `PAD_MAX` (5).

    Returns:
        dict[str, np.ndarray]: Metric name -> (rows, cols) float32 band, in the order and with
            the names of `fuel_metrics_core`.

    Raises:
        ValueError: If a band is not a `PAD_{dz}_{h_bottom}` band or there is no band
            (`read_pad_heights`), or if a FMA grid does not have the shape of the PAD grid.
    """
    dz, h_bottom = read_pad_heights(band_names)
    pad = np.asarray(pad)
    _, rows, cols = pad.shape
    fma_can, fma_shrub = np.asarray(fma_can, dtype=float), np.asarray(fma_shrub, dtype=float)
    for name, fma in (("fma_can", fma_can), ("fma_shrub", fma_shrub)):
        if fma.shape != (rows, cols):
            raise ValueError(f"{name} grid of shape {fma.shape} for a PAD grid of shape {(rows, cols)}.")

    # Counted before the cap, which gives a single count for the three kinds
    n_above = int((np.isfinite(pad) & (pad > pad_max)).sum())
    n_pos_inf = int(np.isposinf(pad).sum())
    n_negative = int((pad < PAD_MIN).sum())
    pad, _ = cap_pad(pad, band_names, pad_max=pad_max)

    all_nan = np.isnan(pad).all(axis=0)
    no_fma = ~all_nan & (np.isnan(fma_can) | np.isnan(fma_shrub))
    partly_nan = np.isnan(pad).any(axis=0) & ~all_nan & ~no_fma

    result = None
    for row in range(rows):
        for col in range(cols):
            metrics = fuel_metrics_core(
                pad[:, row, col],
                h_bottom,
                dz,
                fma_can=float(fma_can[row, col]),
                fma_shrub=float(fma_shrub[row, col]),
                smooth_pad=smooth_pad,
                ladder_fuel_bd_thresholds=ladder_fuel_bd_thresholds,
                canopy_height_threshold=canopy_height_threshold,
                canopy_bd_thresh_frac=canopy_bd_thresh_frac,
                canopy_top_bd_thresholds=canopy_top_bd_thresholds,
                cbh_threshold_ratio=cbh_threshold_ratio,
                bd_max=bd_max,
                m_perrakis=m_perrakis,
            )
            if result is None:
                names = list(metrics)
                result = np.full((len(names), rows, cols), np.nan, dtype=np.float32)
            result[:, row, col] = list(metrics.values())

    n_all_nan, n_no_fma = int(all_nan.sum()), int(no_fma.sum())
    logger.info(
        "Fuel metrics: %d pixel(s) processed; set to NaN on all bands: %d with an all-NaN PAD, "
        "%d without FMA; %d partly NaN PAD profile(s) set to 0; PAD values brought back into "
        "[%g, %g] on read: %d above %g, %d +inf, %d negative",
        rows * cols,
        n_all_nan,
        n_no_fma,
        int(partly_nan.sum()),
        PAD_MIN,
        pad_max,
        n_above,
        pad_max,
        n_pos_inf,
        n_negative,
    )
    if n_all_nan + n_no_fma == rows * cols:
        logger.warning("Fuel metrics: no pixel with data (all-NaN PAD or no FMA), the raster is all NaN")

    return dict(zip(names, result))


def export_fuel_metrics(
    pad_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    fma_can: float | str | os.PathLike,
    fma_shrub: float | str | os.PathLike,
    smooth_pad: bool,
    ladder_fuel_bd_thresholds: Sequence[float],
    canopy_height_threshold: float,
    canopy_bd_thresh_frac: float,
    canopy_top_bd_thresholds: Sequence[float],
    cbh_threshold_ratio: float,
    bd_max: float,
    m_perrakis: float,
    pad_max: float = PAD_MAX,
) -> Path:
    """Compute the fuel metrics of a tile from its PAD raster and write them as a GeoTIFF.

    The input is the `{stem}_pad_profile_1m.tif` raster of step 2, whose bands are all
    `PAD_{dz}_{h_bottom}` bands. The output `{stem}_fuel_metrics.tif` (the `_pad_profile_1m`
    suffix is removed from the input name, if present) is written in `output_dir` on the same
    grid (transform, CRS, size), with one float32 band per metric, nodata NaN, named and
    ordered as `fuel_metrics_core` (see `compute_fuel_metrics_raster`).

    Args:
        pad_path (str | os.PathLike): Path to the `{stem}_pad_profile_1m.tif` PAD raster.
        output_dir (str | os.PathLike): Directory the fuel metrics GeoTIFF is written into.
        fma_can (float | str | os.PathLike): Fuel mass area of the canopy fuel (kg/m²): a
            value, or the path to the FMA map (see `read_fma`).
        fma_shrub (float | str | os.PathLike): Fuel mass area of the shrub fuel (kg/m²): a
            value, or the path to the FMA map (see `read_fma`).
        smooth_pad (bool): Whether to smooth the PAD profile over 3 strata.
        ladder_fuel_bd_thresholds (Sequence[float]): CBD thresholds of the ladder fuel (kg/m³).
        canopy_height_threshold (float): Height (m) separating surface and canopy fuel (CHT).
        canopy_bd_thresh_frac (float): Fraction between CanBDmin and CanBDmax giving the
            crown base threshold.
        canopy_top_bd_thresholds (Sequence[float]): CBD thresholds of the canopy top (kg/m³).
        cbh_threshold_ratio (float): Fraction of CTH separating the upper and lower crown
            base searches.
        bd_max (float): Maximum CBD value (kg/m³).
        m_perrakis (float): M factor of the equivalent surface fuel load (Perrakis 2025).
        pad_max (float): Upper bound of the PAD (m²/m³) applied on read. Default `PAD_MAX` (5).

    Returns:
        Path: Path of the fuel metrics GeoTIFF written.

    Raises:
        ValueError: If a band of the PAD raster is not a `PAD_{dz}_{h_bottom}` band, if a FMA
            value is outside [0.05, 0.5], or if the FMA map does not fit the PAD grid (CRS,
            resolution, alignment) or has values outside [0.05, 0.5] on it. The message gives
            the PAD raster name, and the FMA parameter (`fma_can` or `fma_shrub`) for a FMA error.
    """
    pad_path = Path(pad_path)
    with rasterio.open(pad_path) as src:
        band_names = list(src.descriptions)
        # Input error, not an atypical tile: checked before reading the data
        try:
            read_pad_heights(band_names)
        except ValueError as error:
            raise ValueError(f"PAD raster {pad_path}: {error}") from error
        pad = src.read(masked=True).astype(float).filled(np.nan)
        transform, crs, shape = src.transform, src.crs, src.shape

    # A FMA map is the same for every tile: the error names the tile and the parameter
    fma = {}
    for key, value in (("fma_can", fma_can), ("fma_shrub", fma_shrub)):
        try:
            fma[key] = read_fma(value, transform, shape, crs)
        except ValueError as error:
            raise ValueError(f"PAD raster {pad_path}: {key}: {error}") from error

    metrics = compute_fuel_metrics_raster(
        pad,
        band_names,
        fma["fma_can"],
        fma["fma_shrub"],
        smooth_pad=smooth_pad,
        ladder_fuel_bd_thresholds=ladder_fuel_bd_thresholds,
        canopy_height_threshold=canopy_height_threshold,
        canopy_bd_thresh_frac=canopy_bd_thresh_frac,
        canopy_top_bd_thresholds=canopy_top_bd_thresholds,
        cbh_threshold_ratio=cbh_threshold_ratio,
        bd_max=bd_max,
        m_perrakis=m_perrakis,
        pad_max=pad_max,
    )

    stem = pad_path.stem.removesuffix(_PAD_RASTER_SUFFIX)
    out_path = Path(output_dir) / f"{stem}_fuel_metrics.tif"
    # The CRS is passed as read: the R reference one is a LOCAL_CS without EPSG code
    write_geotiff(out_path, metrics, transform, crs)
    logger.info("Wrote fuel metrics (%d bands) to %s", len(metrics), out_path)

    return out_path
