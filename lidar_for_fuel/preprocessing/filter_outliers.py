"""
Classify statistical outliers using height above ground without removing points.
"""
import logging

import pdal
from numpy.lib import recfunctions as rfn

logger = logging.getLogger(__name__)


def detect_outliers(
    input_pipeline: pdal.Pipeline,
    mean_k: int = 5,
    multiplier: float = 10.0,
) -> pdal.Pipeline:
    """
    Classify outliers using distances in (X, Y, h_abg), preserving all points.

    For each point, compute the average distance to its k nearest neighbors.
    Points exceeding the statistical threshold are classified as 18.
    Save the original altitude in Zref, temporarily replace Z with h_abg for
    detection, then restore Z. The h_abg dimension is required and unchanged.
    The pipeline is executed and the temporary Zref dimension is dropped, so it is
    not exported.

    Args:
        input_pipeline (pdal.Pipeline): Pipeline containing normalized heights in h_abg.
        mean_k (int) : Mean number of neighbors (statistical method only from PDAL). (Default = 5)
        multiplier (float): Standard deviation threshold (statistical method only). (Default = 10.0)

    Returns:
        pdal.Pipeline: Pipeline holding the classified points, without Zref.

    """
    pipeline = input_pipeline | pdal.Filter.ferry(dimensions="Z=>Zref")
    pipeline |= pdal.Filter.ferry(dimensions="h_abg=>Z")
    pipeline |= pdal.Filter.outlier(
        method="statistical",
        mean_k=mean_k,
        multiplier=multiplier,
    )

    pipeline |= pdal.Filter.assign(value="Classification=18 WHERE Classification==7")
    pipeline |= pdal.Filter.ferry(dimensions="Zref=>Z")
    pipeline.execute()

    # PDAL has no filter to remove a dimension: drop Zref from the computed points.
    points = rfn.drop_fields(pipeline.arrays[0], "Zref", usemask=False)

    return pdal.Pipeline(arrays=[points])
