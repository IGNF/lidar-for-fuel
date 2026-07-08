"""Keep points within a ±deviation_days window around the most densely sampled acquisition day."""
import logging
import math
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# See Las 1.4 specification to get information about conversion from las adjusted gps time to standard gps time
_ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME = 1e9

# GPS epoch (fixed): 1980-01-06 00:00:00 UTC
_GPS_EPOCH = np.datetime64("1980-01-06T00:00")


def filter_by_date(
    gpstime: np.ndarray,
    deviation_days: float = np.inf,
) -> tuple[np.ndarray, float]:
    """Filter a LiDAR point cloud keeping only points acquired within ±deviation_days
    around the most densely sampled calendar day.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        deviation_days (float): Max deviation in days around the local modal acquisition date.
                                `inf` = no filter.
                                Default `inf`.
        Note: the GPS epoch is fixed to 1980-01-06 00:00:00 UTC.

    Returns:
        tuple[np.ndarray, float]: (mask, modal_gpstime).
            mask: Boolean mask, same shape as ``gpstime``, True for retained points.
            modal_gpstime: GPS time (same units as ``gpstime``, at 00:00 UTC) of the
                most densely sampled calendar day -- the center of the
                ±deviation_days temporal window.

    Raises:
        ValueError: If the pipeline has no arrays, lacks a ``GpsTime`` dimension,
            or if ``deviation_days`` is negative.
    """
    # Convert GPStime -> calendar day using fixed GPS epoch
    # Approximation: GPS time is a continuous SI-second count with no leap-second
    # adjustments, while unix/UTC time absorbs them. As of today the accumulated
    # offset is ~18 s, negligible for day-level bucketing except right at a
    # midnight boundary.
    utctime = _GPS_EPOCH + np.array(_ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME + gpstime, dtype="timedelta64[s]")
    utcdate = np.array(utctime, dtype="datetime64[D]")

    unique_days, counts = np.unique(utcdate, return_counts=True)
    modal_day = unique_days[counts.argmax()]
    modal_gpstime = float(
        (modal_day - _GPS_EPOCH) / np.timedelta64(1, "s") - _ADJUSTED_GPS_TIME_TO_STANDARD_GPS_TIME
    )

    if math.isinf(deviation_days):
        logger.debug("deviation_days is Inf — no filtering applied.")
        return np.ones_like(gpstime, dtype=bool), modal_gpstime

    # Resulting mask: True for points within the window, False for points outside
    window = np.timedelta64(int(deviation_days), "D")
    retained_mask = np.logical_and(utcdate >= modal_day - window, utcdate <= modal_day + window)

    n_retained = np.sum(retained_mask)
    n_removed = len(retained_mask) - n_retained

    n_total = len(gpstime)
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
        "Modal day: %s | Date window [%s, %s] | %.1f%% points removed",
        modal_day,
        modal_day - window,
        modal_day + window,
        pct_removed,
    )

    return retained_mask, modal_gpstime
