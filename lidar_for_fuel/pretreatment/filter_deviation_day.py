"""
Keep points within a ±deviation_day window around the most densely sampled acquisition day.
"""
import json
import logging
import math
import warnings

import numpy as np
import pdal

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0
_EPSILON = 1e-3  # 1 ms — smaller than any realistic GpsTime resolution


def gpstime_to_day_index(gpstime: np.ndarray, gpstime_ref_unix: float) -> np.ndarray:
    """Convert a GpsTime array (relative seconds) to integer UTC day indices.

    Each value is floored to midnight UTC, expressed as *days since the
    UNIX epoch* (1970-01-01).

    Args:
        gpstime (np.ndarray): 1-D array of relative GPS timestamps (seconds).
        gpstime_ref_unix (float): UNIX timestamp of the GPS reference epoch.

    Returns:
        np.ndarray: Integer array of shape ``(n,)`` with UNIX day indices.

    """
    unix_time = gpstime + gpstime_ref_unix
    return np.floor(unix_time / _SECONDS_PER_DAY).astype(np.int64)


def modal_day(day_index: np.ndarray) -> int:
    """Return the UNIX day index of the most densely sampled calendar day.

    Args:
        day_index (np.ndarray): Integer array of UNIX day indices (one per point).

    Returns:
        int: UNIX day index of the mode.

    """
    unique_days, counts = np.unique(day_index, return_counts=True)
    return int(unique_days[counts.argmax()])


def compute_gpstime_window(
    main_day: int,
    deviation_day: int | float,
    gpstime_ref_unix: float,
) -> tuple[float, float]:
    """Compute the GpsTime [t_min, t_max] filter window.

    The window covers all calendar days in
    ``[main_day - deviation_day, main_day + deviation_day]``.
    ``t_max`` is nudged inward by ``_EPSILON`` so that a point landing
    exactly on midnight of ``day_hi + 1`` is excluded.

    Args:
        main_day (int): UNIX day index of the modal acquisition day.
        deviation_day (int | float): Half-width of the window in days.
        gpstime_ref_unix (float): UNIX timestamp of the GPS reference epoch.

    Returns:
        tuple[float, float]: ``(t_min, t_max)`` in GpsTime space
    """
    day_lo = main_day - int(deviation_day)
    day_hi = main_day + int(deviation_day)
    t_min = max(day_lo, 0) * _SECONDS_PER_DAY - gpstime_ref_unix
    t_max = (day_hi + 1) * _SECONDS_PER_DAY - gpstime_ref_unix - _EPSILON
    return t_min, t_max


def filter_points_by_date(
    input_pipeline: pdal.Pipeline,
    deviation_day: int | float,
    gpstime_ref_unix: float,
) -> pdal.Pipeline:
    """
    Filter a LiDAR point cloud, keeping only points acquired within
    [main_day - deviation_day, main_day + deviation_day], where *main_day*
    is the calendar day that contains the most points.

    Args:
        input_pipeline (pdal.Pipeline): Executed PDAL Pipeline object.
        deviation_day  (int | float): Number of days around the main acquisition day.
                                    Pass ``math.inf`` to skip filtering entirely.
        gpstime_ref_unix (float): UNIX timestamp (seconds since 1970-01-01 UTC)

    Returns:
        pdal.Pipeline: A new, configured-but-not-yet-executed PDAL Pipeline restricted to
        the selected time window.

    Raises:
        ValueError: If the pipeline produced no arrays, if the point cloud
            lacks a ``GpsTime`` dimension, or if ``deviation_day`` is negative.
    """
    if not math.isinf(deviation_day) and deviation_day < 0:
        raise ValueError(f"deviation_day must be >= 0 or math.inf, got {deviation_day!r}")

    arrays = input_pipeline.arrays
    if not arrays:
        raise ValueError("No arrays produced by the pipeline.")

    points = arrays[0]

    if "GpsTime" not in points.dtype.names:
        raise ValueError("Point cloud does not contain a 'GpsTime' dimension.")

    if math.isinf(deviation_day):
        logger.debug("deviation_day is Inf — no filtering applied.")
        return input_pipeline

    n_total = len(points)

    day_index = gpstime_to_day_index(points["GpsTime"], gpstime_ref_unix)
    main_day_idx = modal_day(day_index)
    t_min, t_max = compute_gpstime_window(main_day_idx, deviation_day, gpstime_ref_unix)

    unix_time = points["GpsTime"] + gpstime_ref_unix
    day_lo = main_day_idx - int(deviation_day)
    day_hi = main_day_idx + int(deviation_day)
    n_retained = int(
        np.sum((unix_time >= max(day_lo, 0) * _SECONDS_PER_DAY) & (unix_time < (day_hi + 1) * _SECONDS_PER_DAY))
    )
    pct_removed = (1 - n_retained / n_total) * 100
    if pct_removed > 0:
        warnings.warn(
            f"Careful {round(pct_removed)} % of the returns were removed because they had a "
            f"deviation of days around the most abundant date greater than your threshold "
            f"({deviation_day} days).",
            UserWarning,
            stacklevel=2,
        )

    unique_days = np.unique(day_index)
    logger.debug(
        "Main day index: %d | GpsTime window [%.1f, %.1f) | %d/%d days retained | %.1f %% points removed",
        main_day_idx,
        t_min,
        t_max,
        int(np.sum((unique_days >= day_lo) & (unique_days <= day_hi))),
        len(unique_days),
        pct_removed,
    )

    filter_pipeline_json = {"pipeline": [{"type": "filters.range", "limits": f"GpsTime[{t_min}:{t_max}]"}]}
    return pdal.Pipeline(json.dumps(filter_pipeline_json), arrays=[points])
