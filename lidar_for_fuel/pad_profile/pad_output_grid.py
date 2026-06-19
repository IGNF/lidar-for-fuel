"""
Per-dalle transform/binning helpers, plus the CosiaFrance reference grid constants.

PAD output rasters are written using the dalle's own native origin (e.g. derived
from the IGN LiDAR HD tiling convention) at the project's standard 10 m / EPSG:2154
grid. Recalage onto the FMA `CosiaFrance_V16.tif` national reference grid
(France-wide, same resolution/CRS but a different, sub-pixel-offset origin) is
deferred to a separate future step and is NOT enforced here: real LiDAR HD dalle
origins (round km values) are not grid-aligned with the CosiaFrance origin to
within a multiple of the pixel size, so requiring that alignment at this stage
would reject all real input data.
"""

import numpy as np
from rasterio.transform import Affine

# CosiaFrance national reference grid -- kept for the future recalage step, not
# used by build_dalle_transform (see module docstring).
REFERENCE_ORIGIN_X = 98039.69
REFERENCE_ORIGIN_Y = 7111486.70
PIXEL_SIZE = 10.0
REFERENCE_CRS = "EPSG:2154"


def build_dalle_transform(origin_x: float, origin_y: float, n_rows: int, n_cols: int) -> Affine:
    """Build the rasterio Affine transform for one dalle's output raster.

    Uses the dalle's own native origin (top-left corner) -- no alignment check
    against the CosiaFrance reference grid (see module docstring).

    Args:
        origin_x (float): Dalle raster origin, top-left X (m, EPSG:2154).
        origin_y (float): Dalle raster origin, top-left Y (m, EPSG:2154).
        n_rows (int): Output raster height (pixels). Must be a positive integer.
        n_cols (int): Output raster width (pixels). Must be a positive integer.

    Returns:
        Affine: Rasterio affine transform mapping (col, row) -> (x, y), with
            10 m pixels, north-up (negative Y pixel size).

    Raises:
        ValueError: If `n_rows`/`n_cols` are not positive integers.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"n_rows and n_cols must be positive integers, got n_rows={n_rows}, n_cols={n_cols}")

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
