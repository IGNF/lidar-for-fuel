"""
Compute the vertical height above ground (Z_ref = Z - Z_sol) for each LiDAR point
using a pre-computed DTM raster (GeoTIFF).

Interpolation strategy:
    - Bilinear interpolation (RegularGridInterpolator, method='linear'):
      the grid is defined in geographic coordinates (pixel centres computed from the
      rasterio transform); LiDAR points are queried directly in that same space,
      without converting to pixel row/col indices.
    - NoData fallback: points on NoData pixels or outside the DTM extent
      receive Z_ref = nodata_value (default: -9999).
"""
import logging

import numpy as np
import rasterio
from numpy.lib import recfunctions as rfn
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def filter_z_by_height(
    points: np.ndarray,
    min_height_filter: float = -3,
    height_filter: float = 80,
) -> np.ndarray:
    """
    Remove points too low (<-3) or too high (>height_filter m).

    Args:
        points (np.ndarray): Structured array of LiDAR points with fields X, Y, Z.
        min_height_filter (float): Minimum height (in metres) to remove noise points below the ground. 
                                    Points with Z_ref > height_filter are removed. Default: -3.
        height_filter (float): Height limit (in metres) to remove noise points above
            the canopy. Points with Z_ref > height_filter are removed. Default: 80.

    Returns:
        np.ndarray: Filtered structured array with Z_ref (float64) and filtered added as an
            extra field.
    """
    # Remove points too low (<-3) or too high (>height_filter m)
    mask = (points["Z_ref"] >= min_height_filter) & (points["Z_ref"] <= height_filter)
    n_removed = int((~mask).sum())
    if n_removed:
        logger.debug("%d points removed by Z_ref filter [%s, %s]", n_removed, min_height_filter, height_filter)
    points = points[mask]

    return points


def add_Zref(
    points: np.ndarray,
    dtm_path: str,
    nodata_value: float = -9999,
) -> np.ndarray:
    """
    Add Z_ref = Z - Z_sol to each LiDAR point using bilinear interpolation on a DTM.

    Ground elevation Z_sol is sampled from the DTM raster at each point's (X, Y)
    position using bilinear interpolation (map_coordinates, order=1). Points that
    fall on NoData pixels or outside the DTM extent receive Z_ref = nodata_value.


    Args:
        points (np.ndarray): Structured array of LiDAR points with fields X, Y, Z.
        dtm_path (str): Path to the DTM GeoTIFF (single band, elevation in metres,
            resolution 0.5 m, EPSG:2154).
        nodata_value (float): Value assigned to Z_ref for points on NoData DTM pixels
            or outside the DTM extent. Default: -9999 (from config dtm.nodata_value).

    Returns:
        np.ndarray: Filtered structured array with Z_ref (float64) added as an
            extra field.
    """
    # Load DTM raster and build coordinate arrays for pixel centres
    with rasterio.open(dtm_path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        height, width = data.shape
        # rasterio stores the transform at pixel corners (top-left edge).
        # +0.5 converts corners to pixel centres — where faceraster interpolated the DTM Z values.
        # Without it, RegularGridInterpolator would be anchored on pixel edges, not centres.
        xs = transform.c + (np.arange(width) + 0.5) * transform.a
        ys = transform.f + (np.arange(height) + 0.5) * transform.e  # e < 0 → Y decreasing

    # Replace nodata with NaN
    if nodata is not None:
        data[data == nodata] = np.nan

    # Bilinear interpolation directly on geographic coordinates.
    # RegularGridInterpolator requires increasing axes, so Y is flipped.
    interp = RegularGridInterpolator(
        (ys[::-1], xs), data[::-1],
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    z_ground = interp(np.column_stack([points["Y"], points["X"]]))

    # Compute vertical height above ground
    z_ref = (points["Z"] - z_ground).astype(np.float64)

    # Assign nodata_value to points on NoData DTM pixels or outside DTM extent
    nan_mask = np.isnan(z_ref)
    if np.any(nan_mask):
        logger.debug(
            "%d points on nodata or outside DTM extent — Z_ref set to %s",
            int(nan_mask.sum()),
            nodata_value,
        )
        z_ref[nan_mask] = nodata_value
    logger.debug("Z_ref — min: %.2f m, max: %.2f m, mean: %.2f m", z_ref.min(), z_ref.max(), z_ref.mean())

    # Append Z_ref as extra dimension
    points_with_zref = rfn.append_fields(points, "Z_ref", z_ref, dtypes=np.float64, usemask=False)

    return points_with_zref
