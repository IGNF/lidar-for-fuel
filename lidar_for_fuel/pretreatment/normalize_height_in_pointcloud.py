"""
Calculate normalized height in pointcloud, i.e. height above ground.
"""
import logging
from typing import List

import pdal

logger = logging.getLogger(__name__)


def normalize_height(
    input_pipeline: pdal.Pipeline,
    output_file: str,
    filter_dimension: str,
    filter_values: List[int],
    height_filter: float = 60.0,
) -> None:
    """
    Normalize heights using TIN on ground points where dtm_marker = 1,
    but calculate HeightAboveGround for all points.

    Steps:
        1. Keep only points classified as 1-5 or 9.
        2. Split pipeline: use points with dtm_marker = 1 for TIN.
        3. Normalize height for all points using the TIN.
        4. Remove points with HeightAboveGround < height_filter.

    Args:
        input_pipeline (pdal.Pipeline): Executed PDAL Pipeline object.
        output_file (str): Output LAS/LAZ path.
        filter_dimension (str): Name of the dimension along which to filter input points.
        filter_values (List[int]): Values to keep for input points along filter_dimension.
        height_filter (float): Value to remove points too high (60 m default)

    """
    # Read pdal.Pipeline
    pipeline = input_pipeline

    # Filter points by classification
    if filter_dimension and filter_values:
        pipeline |= pdal.Filter.range(limits=",".join(f"{filter_dimension}[{v}:{v}]" for v in filter_values))

    # Normalize height
    pipeline |= pdal.Filter.hag_delaunay(allow_extrapolation=True)

    # Remove points HeightAboveGround < height filter
    pipeline |= pdal.Filter.range(limits=f"HeightAboveGround[:{height_filter}]")

    # Copy HAG → Z_ref
    pipeline |= pdal.Filter.ferry(dimensions="HeightAboveGround=>Z_ref")

    # Writer output
    pipeline |= pdal.Writer.las(filename=output_file, extra_dims="all", forward="all", minor_version="4")

    pipeline.execute()
