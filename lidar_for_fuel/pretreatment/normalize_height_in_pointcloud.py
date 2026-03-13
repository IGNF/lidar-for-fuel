"""
Calculate normalized height in pointcloud, i.e. height above ground.
"""
import json
import logging
from typing import List

import numpy as np
import pdal
from numpy.lib import recfunctions as rfn
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

logger = logging.getLogger(__name__)


def normalize_height(
    input_pipeline: pdal.Pipeline,
    output_file: str,
    filter_dimension: str,
    filter_values: List[int],
    height_filter: float = 60.0,
    min_height: float = -3.0,
) -> None:
    """
    Normalize heights using a Delaunay TIN built from ground points, and write result.

    Equivalent to the R pipeline:
        normalize_height(algorithm = tin())
        filter_poi((Classification <= 5 | Classification == 9) & Z < Height_filter)

    Steps:
        1. Execute the input pipeline.
        2. Build a Delaunay TIN from all points with dtm_marker == 1.
        3. Interpolate ground elevation for every point. Points outside the TIN convex
           hull are handled by nearest-neighbour extrapolation.
        4. Compute Z_ref = Z - interpolated_ground_Z.
        5. Filter points by filter_dimension/filter_values.
        6. Remove points outside [min_height, height_filter].
        7. Write output LAS/LAZ with Z_ref as an extra dimension.

    Args:
        input_pipeline (pdal.Pipeline): PDAL Pipeline object (may be unexecuted).
        output_file (str): Output LAS/LAZ path.
        filter_dimension (str): Dimension name used to filter output points.
        filter_values (List[int]): Values to keep along filter_dimension.
        height_filter (float): Upper height threshold in metres (default: 60 m).
        min_height (float): Lower height threshold in metres (default: -3 m).

    Raises:
        ValueError: If the pipeline produces no arrays, or if there are fewer than
            3 ground points to build a TIN.
    """
    # Execute the pipeline on all points
    pipeline = input_pipeline if isinstance(input_pipeline, pdal.Pipeline) else pdal.Pipeline() | input_pipeline
    pipeline.execute()

    arrays = pipeline.arrays
    if not arrays:
        raise ValueError("No arrays produced by the pipeline.")
    points = arrays[0]

    # Select ground points for DTM computation: all points with dtm_marker == 1
    ground_mask = points["dtm_marker"] == 1
    logger.debug("%d ground points with dtm_marker=1", int(ground_mask.sum()))
    ground_points = points[ground_mask]

    if len(ground_points) < 3:
        raise ValueError(f"Not enough ground points to build a TIN (found {len(ground_points)}, need at least 3).")

    # Build Delaunay TIN interpolator from ground points (X, Y) → Z
    # LinearNDInterpolator: builds a Delaunay triangulation from ground points.
    # For each point (X, Y), finds the triangle it falls in and computes Z by
    # barycentric interpolation between the 3 vertices. Equivalent to lidR::tin().
    # Returns NaN for points outside the convex hull of ground points.
    #
    # NearestNDInterpolator: fallback for points outside the convex hull (e.g. at
    # tile edges where vegetation points exceed the ground point extent).
    # Returns the Z of the nearest ground point.
    ground_xy = np.column_stack([ground_points["X"], ground_points["Y"]])
    ground_z = ground_points["Z"]
    lin_interp = LinearNDInterpolator(ground_xy, ground_z)
    nn_interp = NearestNDInterpolator(ground_xy, ground_z)

    # Interpolate ground elevation for all points
    all_xy = np.column_stack([points["X"], points["Y"]])
    ground_z_interp = lin_interp(all_xy)

    # Fill NaN (outside convex hull) with nearest-neighbour extrapolation
    nan_mask = np.isnan(ground_z_interp)
    if np.any(nan_mask):
        logger.debug("%d points outside TIN convex hull — using nearest-neighbour extrapolation", int(nan_mask.sum()))
        ground_z_interp[nan_mask] = nn_interp(all_xy[nan_mask])

    # Compute height above ground
    hag = points["Z"] - ground_z_interp

    # Filter points by filter_dimension/filter_values
    if filter_dimension and filter_values:
        dim_mask = np.isin(points[filter_dimension], filter_values)
        points = points[dim_mask]
        hag = hag[dim_mask]
        logger.debug("Dimension filter on %s %s: %d points kept", filter_dimension, filter_values, len(points))

    # Filter by height bounds [min_height, height_filter]
    valid = (hag >= min_height) & (hag <= height_filter)
    pct_removed = (1 - valid.sum() / len(points)) * 100
    logger.debug("Height filter [%.1f, %.1f]: %.1f%% points removed", min_height, height_filter, pct_removed)

    points_out = points[valid]
    hag_out = hag[valid].astype(np.float64)

    # Add Z_ref dimension (= HeightAboveGround)
    points_with_zref = rfn.append_fields(points_out, "Z_ref", hag_out, dtypes=np.float64, usemask=False)

    # Write output
    writer_pipeline = pdal.Pipeline(
        json.dumps(
            {
                "pipeline": [
                    {
                        "type": "writers.las",
                        "filename": str(output_file),
                        "extra_dims": "all",
                        "forward": "all",
                        "minor_version": "4",
                    }
                ]
            }
        ),
        arrays=[points_with_zref],
    )
    writer_pipeline.execute()
