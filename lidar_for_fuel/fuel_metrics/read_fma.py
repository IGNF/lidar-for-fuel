"""
Read the fuel mass area (FMA, kg/m²) of the fuel metrics computation on the PAD grid, either a
configured value or the national FMA map.
"""
import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

# Valid range of the FMA map values (kg/m²), from the FMA map specification
FMA_MAP_MIN = 0.05
FMA_MAP_MAX = 0.5
# Largest shift (fraction of a pixel) tolerated between the PAD grid and the FMA map grid.
# Covers the 6 cm offset between the rounded CosiaFrance origin of the PAD rasters
# (98039.75, 7111486.75) and the real one of the national maps (98039.6885, 7111486.6972).
_FMA_GRID_TOLERANCE = 0.1


def read_fma(
    fma: float | str | os.PathLike,
    transform: Affine,
    shape: tuple[int, int],
    crs: CRS | None = None,
) -> np.ndarray:
    """Get the fuel mass area (FMA, kg/m²) of every pixel of the PAD grid.

    The FMA converts the PAD into a bulk density (`CBD = PAD × FMA`). It depends on the
    dominant species, so it is either a single configured value or the national FMA map
    (single-band raster, 10 m, values in [0.05, 0.5] kg/m²).

    - **Value**: the grid is filled with it. It must be in [0.05, 0.5], as the map values
      (a NaN or inf value is refused).
    - **Map path**: the window of the map covering the PAD grid is read, each PAD pixel
      taking the FMA pixel whose centre is the nearest. The two grids must have the same
      resolution and be aligned within `_FMA_GRID_TOLERANCE` of a pixel (the 6 cm offset of
      the CosiaFrance origin is tolerated). NoData pixels and pixels outside the map are NaN;
      every other value (±inf included) must be in [0.05, 0.5].

    Args:
        fma (float | str | os.PathLike): FMA value (kg/m²) or path to the FMA map.
        transform (Affine): Affine transform of the PAD raster.
        shape (tuple[int, int]): (height, width) of the PAD raster.
        crs (CRS | None): CRS of the PAD raster, compared with the map CRS when both are EPSG codes.

    Returns:
        np.ndarray: FMA (kg/m²) of every PAD pixel, float64, of shape `shape`.

    Raises:
        ValueError: If the value is outside [0.05, 0.5] (NaN and inf included), if the map has
            another CRS or resolution than the PAD grid, if it is not aligned on it, or if one
            of its values in the window is outside [0.05, 0.5].
    """
    if not isinstance(fma, (str, os.PathLike)):
        value = float(fma)
        if not FMA_MAP_MIN <= value <= FMA_MAP_MAX:  # also false for NaN
            raise ValueError(f"FMA value {value} outside [{FMA_MAP_MIN}, {FMA_MAP_MAX}] kg/m².")
        return np.full(shape, value)

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

    # NaN is no FMA; ±inf is a value, outside the range
    valid = values[~np.isnan(values)]
    out_of_range = (valid < FMA_MAP_MIN) | (valid > FMA_MAP_MAX)
    if out_of_range.any():
        raise ValueError(
            f"FMA map {fma}: {int(out_of_range.sum())} value(s) outside [{FMA_MAP_MIN}, {FMA_MAP_MAX}] kg/m² "
            f"(min {valid.min()}, max {valid.max()})."
        )

    return values
