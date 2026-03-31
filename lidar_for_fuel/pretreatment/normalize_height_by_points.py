"""
Compute the vertical height above ground (Z_ref = Z - Z_sol) for each LiDAR point
using a pre-computed DTM raster (GeoTIFF).

Interpolation strategy:
    - Bilinear interpolation (map_coordinates, order=1):
      converts (X, Y) world coordinates to fractional pixel indices, then
      weights the 4 surrounding DTM pixels in a single vectorised C call.
    - NoData fallback: points on NoData pixels or outside the DTM extent
      receive Z_ref = nodata_value (default: -9999).
"""
import json
import logging

import numpy as np
import pdal
import rasterio
from numpy.lib import recfunctions as rfn
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


def add_Zref(
    input_pipeline: pdal.Pipeline,
    dtm_path: str,
    nodata_value: float = -9999,
) -> pdal.Pipeline:
    """
    Add Z_ref = Z - Z_sol to each LiDAR point using bilinear interpolation on a DTM.

    Ground elevation Z_sol is sampled from the DTM raster at each point's (X, Y)
    position using bilinear interpolation (map_coordinates, order=1). Points that
    fall on NoData pixels or outside the DTM extent receive Z_ref = nodata_value.

    Args:
        input_pipeline (pdal.Pipeline): PDAL pipeline (may be unexecuted).
        dtm_path (str): Path to the DTM GeoTIFF (single band, elevation in metres,
            resolution 0.5 m, EPSG:2154).
        nodata_value (float): Value assigned to Z_ref for points on NoData DTM pixels
            or outside the DTM extent. Default: -9999 (from config dtm.nodata_value).

    Returns:
        pdal.Pipeline: Unexecuted pipeline with Z_ref (float64) added as an
            extra dimension.

    Raises:
        ValueError: If the pipeline produces no arrays.
    """
    # Execute pipeline and extract point array
    pipeline = input_pipeline if isinstance(input_pipeline, pdal.Pipeline) else pdal.Pipeline() | input_pipeline
    pipeline.execute()

    arrays = pipeline.arrays
    if not arrays:
        raise ValueError("No arrays produced by the pipeline.")
    points = arrays[0]

    # Load DTM raster
    with rasterio.open(dtm_path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform

    # Replace nodata with NaN
    if nodata is not None:
        data[data == nodata] = np.nan

    # Convert world coordinates (X, Y) to fractional pixel indices (row, col)
    # Pixel centre (row=i, col=j): x = c + (j+0.5)*a  ;  y = f + (i+0.5)*e  (e < 0)
    row_idx = (points["Y"] - transform.f) / transform.e - 0.5
    col_idx = (points["X"] - transform.c) / transform.a - 0.5

    # Bilinear interpolation — cval=NaN for points outside DTM extent
    z_ground = map_coordinates(
        data,
        [row_idx, col_idx],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )

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

    return pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[points_with_zref])
