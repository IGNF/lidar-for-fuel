"""Keep points within a ±deviation_days window around the most densely sampled acquisition day."""

import json
import logging                                         
import math                                            
import warnings                                        
from datetime import datetime, timezone   

import numpy as np
import pdal

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3  # 1 ms — smaller than any realistic GpsTime resolution


def filter_deviation_day(
    pipeline: pdal.Pipeline,
    deviation_days: int | float = 14,
    gpstime_ref: str = "2011-09-14 01:46:40",
) -> pdal.Pipeline:
    """Filter a LiDAR point cloud keeping only points acquired within ±deviation_days
    around the most densely sampled calendar day.

    Equivalent to the R function ``filter_date_mode()`` from lidR.

    Args:
        pipeline (pdal.Pipeline): Executed PDAL Pipeline object.
        deviation_days (int | float): Half-width of the retention window in days.
            Pass ``math.inf`` to skip filtering entirely. Default: 14.
        gpstime_ref (str): ISO-8601 UTC string of the GPS time reference epoch.
            Default: "2011-09-14 01:46:40".

    Returns:
        pdal.Pipeline: A new, configured-but-not-yet-executed PDAL Pipeline restricted
        to the selected time window, or the original pipeline unchanged if
        ``deviation_days`` is infinite.

    Raises:
        ValueError: If the pipeline has no arrays, lacks a ``GpsTime`` dimension,
            or if ``deviation_days`` is negative.
    """
    if not math.isinf(deviation_days) and deviation_days < 0:
        raise ValueError(f"deviation_days must be >= 0 or math.inf, got {deviation_days!r}")

    arrays = pipeline.arrays
    if not arrays:
        raise ValueError("No arrays produced by the pipeline.")

    points = arrays[0]

    if "GpsTime" not in points.dtype.names:
        raise ValueError("Point cloud does not contain a 'GpsTime' dimension.")

    if math.isinf(deviation_days):
        logger.debug("deviation_days is Inf — no filtering applied.")
        return pipeline

    gpstime_ref_unix = datetime.fromisoformat(gpstime_ref).replace(tzinfo=timezone.utc).timestamp()
    n_total = len(points)

    # Convert GpsTime to absolute UNIX time and floor to calendar day
    unix_time = points["GpsTime"] + gpstime_ref_unix
    day_index = np.floor(unix_time / _SECONDS_PER_DAY).astype(np.int64)

    # Find the modal day (most abundantly sampled)
    unique_days, counts = np.unique(day_index, return_counts=True)
    modal_day = int(unique_days[counts.argmax()])

    # Compute the GpsTime filter window [t_min, t_max]
    day_lo = modal_day - int(deviation_days)
    day_hi = modal_day + int(deviation_days)
    t_min = max(day_lo, 0) * _SECONDS_PER_DAY - gpstime_ref_unix
    t_max = (day_hi + 1) * _SECONDS_PER_DAY - gpstime_ref_unix - _EPSILON

    # Warn about removed points
    n_retained = int(
        np.sum((unix_time >= max(day_lo, 0) * _SECONDS_PER_DAY) & (unix_time < (day_hi + 1) * _SECONDS_PER_DAY))
    )
    pct_removed = (1 - n_retained / n_total) * 100
    if pct_removed > 0:
        warnings.warn(
            f"Careful {round(pct_removed)} % of the returns were removed because they had a "
            f"deviation of days around the most abundant date greater than your threshold "
            f"({deviation_days} days).",
            UserWarning,
            stacklevel=2,
        )

    logger.debug(
        "Modal day: %d | GpsTime window [%.1f, %.1f] | %.1f%% points removed",
        modal_day,
        t_min,
        t_max,
        pct_removed,
    )

    filter_json = {"pipeline": [{"type": "filters.range", "limits": f"GpsTime[{t_min}:{t_max}]"}]}
    return pdal.Pipeline(json.dumps(filter_json), arrays=[points])
