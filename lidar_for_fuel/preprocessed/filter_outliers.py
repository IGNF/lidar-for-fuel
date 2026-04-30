"""
Detect and remove outliers with statistical method
"""
import logging

import pdal

logger = logging.getLogger(__name__)


def remove_outliers(
    input_pipeline: pdal.Pipeline,
    mean_k: int = 5,
    multiplier: float = 10.0,
) -> pdal.Pipeline:
    """
    Identify and remove outliers: for each point, the average distance to its k nearest neighbors is computed.
    If this distance exceeds a statistical threshold, the point is considered isolated.

    Args:
        input_pipeline (pdal.Pipeline): Executed PDAL Pipeline object.
        mean_k (int) : Mean number of neighbors (statistical method only from PDAL). (Default = 5)
        multiplier (float): Standard deviation threshold (statistical method only). (Default = 10.0)

    Returns:
        pdal.Pipeline: Updated PDAL pipeline (not yet executed).

    """
    pipeline = input_pipeline | pdal.Filter.outlier(
        method="statistical",
        mean_k=mean_k,
        multiplier=multiplier,
    )

    pipeline |= pdal.Filter.range(limits="Classification![7:7]")

    return pipeline
