"""
Filter points by filter_dimension/filter_values.
"""
import logging
from typing import List

import pdal

logger = logging.getLogger(__name__)


def filter_by_dimension_values(
    input_pipeline: pdal.Pipeline,
    filter_dimension: str,
    filter_values: List[int],
) -> pdal.Pipeline:
    """
    Filter points by filter_dimension/filter_values.

    Args:
        input_pipeline (pdal.Pipeline): PDAL Pipeline object (may be unexecuted).
        filter_dimension (str): Dimension name used to filter output points.
        filter_values (List[int]): Values to keep along filter_dimension.

    Returns:
        pdal.Pipeline: Updated PDAL pipeline (not yet executed).

    Raises:
        ValueError: If filter_values is empty.
    """
    if not filter_values:
        raise ValueError(
            "`filter_values` cannot be empty — " "no selected value would result in an empty point cloud."
        )

    pipeline = input_pipeline | pdal.Filter.range(
        limits=",".join(f"{filter_dimension}[{v}:{v}]" for v in filter_values)
    )

    logger.info(
        "filter_by_dimension_values: filter %s = %s added to pipeline (not yet executed).",
        filter_dimension,
        sorted(filter_values),
    )

    return pipeline
