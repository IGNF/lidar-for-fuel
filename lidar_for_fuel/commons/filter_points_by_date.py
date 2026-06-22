"""Keep points within a ±deviation_days window around the most densely sampled acquisition day."""

import logging
import math
import warnings
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0
_DEFAULT_GPSTIME_REF = "2011-09-14 01:46:40"


def filter_by_date(
    gpstime: np.ndarray,
    deviation_days: int | float = 14,
    gpstime_ref: str = _DEFAULT_GPSTIME_REF,
) -> np.ndarray:
    """Boolean mask keeping only points acquired within ±deviation_days around the
    most densely sampled calendar day.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        deviation_days (int | float): Half-width of the retention window in days.
            Pass ``math.inf`` to skip filtering entirely (mask is all True). Default: 14.
        gpstime_ref (str): ISO-8601 UTC string of the GPS time reference epoch.
            Default: "2011-09-14 01:46:40".

    Returns:
        np.ndarray: Boolean mask, same shape as ``gpstime``, True for retained points.

    Raises:
        ValueError: If ``deviation_days`` is negative.
    """
    if not math.isinf(deviation_days) and deviation_days < 0:
        raise ValueError(f"deviation_days must be >= 0 or math.inf, got {deviation_days!r}")

    n_total = gpstime.shape[0]
    if n_total == 0:
        return np.zeros(0, dtype=bool)

    if math.isinf(deviation_days):
        logger.debug("deviation_days is Inf — no filtering applied.")
        return np.ones(n_total, dtype=bool)

    # Convert GpsTime -> calendar day
    gpstime_ref_unix = datetime.fromisoformat(gpstime_ref).replace(tzinfo=timezone.utc).timestamp()
    unix_time = gpstime + gpstime_ref_unix
    day_index = np.floor(unix_time / _SECONDS_PER_DAY).astype(np.int64)

    # Find the modal day
    unique_days, counts = np.unique(day_index, return_counts=True)
    modal_day = int(unique_days[counts.argmax()])

    # Keep points within the [day_lo, day_hi] window around the modal day
    day_lo = modal_day - int(deviation_days)
    day_hi = modal_day + int(deviation_days)
    mask = (day_index >= day_lo) & (day_index <= day_hi)

    n_removed = n_total - int(mask.sum())
    if n_removed > 0:
        pct_removed = n_removed / n_total * 100
        warnings.warn(
            f"Careful {n_removed} / {n_total} ({pct_removed:.1f} %) of the returns were removed "
            f"because they had a deviation of days around the most abundant date greater than your "
            f"threshold ({deviation_days} days).",
            UserWarning,
            stacklevel=2,
        )

    logger.debug(
        "Modal day: %d | window [%d, %d] days | %d/%d points removed",
        modal_day,
        day_lo,
        day_hi,
        n_removed,
        n_total,
    )

    return mask
