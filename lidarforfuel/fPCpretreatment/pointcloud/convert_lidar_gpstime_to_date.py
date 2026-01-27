import logging
from datetime import datetime, timedelta
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def convert_gpstime_to_time(
    las: "laspy.LasData",
    start_date: Union[str, datetime],
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> np.ndarray:
    """
    Convert LAS/LAZ gps_time values to absolute datetimes.

    Parameters
    ----------
    las : laspy.LasData
        LAS/LAZ point cloud object, expected to provide a `gps_time` attribute.
    start_date : str or datetime.datetime
        Reference date-time used as origin for the GPS time values.
        If a string is provided, it must match `date_format`.
    date_format : str, optional
        Format string used to parse `start_date` when it is a string.
        Default is "%Y-%m-%d %H:%M:%S".

    Returns
    -------
    numpy.ndarray
        Array of `datetime.datetime` objects corresponding to each point.

    Raises
    ------
    TypeError
        If `las` is None or `start_date` is neither a string nor a datetime.
    ValueError
        If `start_date` is a string that cannot be parsed with `date_format`.
    AttributeError
        If `las` does not provide a `points` attribute.
    """
    # Basic validation
    if las is None:
        raise TypeError("`las` must be a valid laspy.LasData object")

    if not hasattr(las, "points"):
        raise AttributeError("`las` object must have a 'points' attribute")

    # Normalize start_date to datetime
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, date_format)
    elif not isinstance(start_date, datetime):
        raise TypeError("`start_date` must be a string or a datetime.datetime instance")

    # Get gps_time if available, otherwise dummy zeros
    if hasattr(las, "gps_time"):
        gpstime = las.gps_time
    else:
        logger.warning("No 'gps_time' attribute found; using zeros array.")
        gpstime = np.zeros(len(las.points))

    # Convert gpstime offsets (in seconds) to absolute datetimes
    new_date = np.array(
        [start_date + timedelta(seconds=float(t)) for t in gpstime]
    )
    logger.info("Converted gps_time to absolute datetimes for %s points", len(new_date))
    
    return new_date
