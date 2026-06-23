"""Keep points within a ±deviation_days window around the most densely sampled acquisition day."""
import logging
import math
import warnings
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3  # 1 ms — smaller than any realistic GpsTime resolution


def filter_by_date(
    gpstime: np.ndarray,
    deviation_days: int | float = 14,
    gpstime_ref: str = "2011-09-14 01:46:40",
) -> np.ndarray:
    """Filter a LiDAR point cloud keeping only points acquired within ±deviation_days
    around the most densely sampled calendar day.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        deviation_days (int | float): Half-width of the retention window in days.
            Pass ``math.inf`` to skip filtering entirely. Default: 14.
        gpstime_ref (str): ISO-8601 UTC string of the GPS time reference epoch.
            Default: "2011-09-14 01:46:40".

    Returns:
        np.ndarray: Boolean mask, same shape as ``gpstime``, True for retained points.

    Raises:
        ValueError: If the pipeline has no arrays, lacks a ``GpsTime`` dimension,
            or if ``deviation_days`` is negative.
    """
    if math.isinf(deviation_days):
        logger.debug("deviation_days is Inf — no filtering applied.")
        return np.ones_like(gpstime, dtype=bool)

    # Convert GPStime -> calendar day
    gpstime_ref_unix = datetime.fromisoformat(gpstime_ref).replace(tzinfo=timezone.utc).timestamp()
    n_total = len(gpstime)
    unix_time = gpstime + gpstime_ref_unix
    day_index = np.floor(unix_time / _SECONDS_PER_DAY).astype(np.int64)

    # Find the modal day
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
    n_removed = n_total - n_retained
    pct_removed = n_removed / n_total * 100
    if n_removed > 0:
        warnings.warn(
            f"Careful {n_removed} / {n_total} ({pct_removed:.1f} %) of the returns were removed "
            f"because they had a deviation of days around the most abundant date greater than your "
            f"threshold ({deviation_days} days).",
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
    
    # Resulting mask: True for points within the window, False for points outside
    mask = (unix_time >= max(day_lo, 0) * _SECONDS_PER_DAY) & (unix_time < (day_hi + 1) * _SECONDS_PER_DAY)

    return mask