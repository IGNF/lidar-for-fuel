"""
CosiaFrance reference grid constants and per-dalle transform/binning helpers.

PAD output rasters must align to the FMA `CosiaFrance_V16.tif` reference grid
(EPSG:2154, 10 m pixels, France-wide). A dalle's own raster origin must be
congruent with this global grid, i.e. its offset from the reference origin
must be an exact multiple of the pixel size; otherwise points cannot be
binned into a grid-aligned raster and a `ValueError` is raised.
"""

import numpy as np
from rasterio.transform import Affine

REFERENCE_ORIGIN_X = 98039.69
REFERENCE_ORIGIN_Y = 7111486.70
PIXEL_SIZE = 10.0
REFERENCE_CRS = "EPSG:2154"

_ALIGNMENT_TOLERANCE = 1e-6


def _is_aligned(offset: float, pixel_size: float, tolerance: float = _ALIGNMENT_TOLERANCE) -> bool:
    """Return True if `offset` is an exact multiple of `pixel_size`, within float tolerance."""
    remainder = offset % pixel_size
    return remainder <= tolerance or (pixel_size - remainder) <= tolerance


def build_dalle_transform(origin_x: float, origin_y: float, n_rows: int, n_cols: int) -> Affine:
    """Build/validate the rasterio Affine transform for one dalle's output raster.

    The dalle's origin (top-left corner) must be grid-aligned with the CosiaFrance
    reference grid: `(origin_x - REFERENCE_ORIGIN_X)` and `(REFERENCE_ORIGIN_Y - origin_y)`
    must both be exact multiples of `PIXEL_SIZE` (within float tolerance).

    Args:
        origin_x (float): Dalle raster origin, top-left X (m, EPSG:2154).
        origin_y (float): Dalle raster origin, top-left Y (m, EPSG:2154).
        n_rows (int): Output raster height (pixels). Must be a positive integer.
        n_cols (int): Output raster width (pixels). Must be a positive integer.

    Returns:
        Affine: Rasterio affine transform mapping (col, row) -> (x, y), with
            10 m pixels, north-up (negative Y pixel size).

    Raises:
        ValueError: If `origin_x`/`origin_y` are not grid-aligned with the
            CosiaFrance reference grid, or if `n_rows`/`n_cols` are not
            positive integers.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows and n_cols must be positive integers, got n_rows={n_rows}, n_cols={n_cols}")

    offset_x = origin_x - REFERENCE_ORIGIN_X
    offset_y = REFERENCE_ORIGIN_Y - origin_y

    if not _is_aligned(offset_x, PIXEL_SIZE) or not _is_aligned(offset_y, PIXEL_SIZE):
        raise ValueError(
            "Misaligned dalle origin: offset from the CosiaFrance reference grid origin "
            f"(X={REFERENCE_ORIGIN_X}, Y={REFERENCE_ORIGIN_Y}) must be an exact multiple of "
            f"{PIXEL_SIZE} m. Got origin_x={origin_x}, origin_y={origin_y} "
            f"(offset_x={offset_x}, offset_y={offset_y})."
        )

    return Affine.translation(origin_x, origin_y) * Affine.scale(PIXEL_SIZE, -PIXEL_SIZE)


def bin_points_to_pixels(x: np.ndarray, y: np.ndarray, transform: Affine) -> tuple[np.ndarray, np.ndarray]:
    """Bin point x/y coordinates into integer (row, col) pixel indices.

    Args:
        x (np.ndarray): Point easting (m, EPSG:2154).
        y (np.ndarray): Point northing (m, EPSG:2154).
        transform (Affine): Raster transform, as built by `build_dalle_transform`.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(rows, cols)`, integer arrays the same
            shape as `x`/`y`. Points outside the raster extent are not filtered
            here; callers must bound-check against the raster shape. Non-finite
            (`NaN`/`inf`) coordinates are mapped to `(-1, -1)`, which callers'
            bound-checks will naturally treat as out-of-bounds.
    """
    inv_transform = ~transform
    cols_f, rows_f = inv_transform * (np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))

    finite = np.isfinite(rows_f) & np.isfinite(cols_f)
    rows = np.full(rows_f.shape, -1, dtype=np.int64)
    cols = np.full(cols_f.shape, -1, dtype=np.int64)
    rows[finite] = np.floor(rows_f[finite]).astype(np.int64)
    cols[finite] = np.floor(cols_f[finite]).astype(np.int64)
    return rows, cols
