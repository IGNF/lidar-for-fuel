"""
Calculate normalized height in pointcloud, i.e. height above ground.
"""
import logging
from typing import List

import pdal

logger = logging.getLogger(__name__)


def normalize_height(
    input_file: str,
    output_file: str,
    spatial_ref: str,
    filter_dimension: str,
    filter_values: List[int],
    height_filter: float = 60.0,
) -> None:
    """
    Pre-filtering and Normalize heights using TIN on ground points.

    Stepas are :
        1. Keep only points classified as 1-5 or 9.
        2. Normalyze height
        43 Remove points Z < height filter

    Args:
        input_file: Input LAS/LAZ path.
        output_file: Output LAS/LAZ path.
        spatial_ref (str): spatial reference to use when reading las file.
        filter_dimension (str): Name of the dimension along which to filter input points
        (keep empty to disable input filter).
        filter_values (List[int]): Values to keep for input points along filter_dimension.
        height_filter: Value to remove points too high (60 m default)

    """
    # Read with pdal
    pipeline = pdal.Reader.las(filename=input_file, override_srs=spatial_ref, nosrs=True)

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
