# """PDAL Python filter for deviation day filtering"""
import json
import logging
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


def filter_deviation_day(ins, outs, pdalargs):
    """
    Filter points within deviation_day around most frequent acquisition day.

    Keeps flights within deviation_day of most abundant date.

    Args:
        ins: Input point cloud dimensions from PDAL
        outs: Output point cloud dimensions to populate
        pdalargs: JSON string MUST contain:
            - deviation_day: int, max days around mode
            - gpstime_ref: str, GPS time reference (YYYY-MM-DD HH:MM:SS)

    Returns:
        bool: True if successful
    """
    args = json.loads(pdalargs)

    # Parameters
    try:
        deviation_days = int(args["deviation_day"])
        gpstime_ref = str(args["gpstime_ref"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"filter_deviation_day: missing/invalid pdalargs. "
            f"Need 'deviation_day'(int) and 'gpstime_ref'(str). Got: {e}"
        )

    if deviation_days is None or not np.isfinite(deviation_days):
        # No filtering requested
        for dim_name in ins:
            outs[dim_name] = ins[dim_name]
        return True

    # Convert GPS time → date objects
    ref_dt = datetime.strptime(gpstime_ref, "%Y-%m-%d %H:%M:%S")
    gpstime = ins["GpsTime"] if "GpsTime" in ins else ins["gps_time"]
    new_date = [(ref_dt + timedelta(seconds=float(t))).date() for t in gpstime]

    # Histogram dates
    dates_only = np.array(new_date)
    unique_dates, counts = np.unique(dates_only, return_counts=True)

    if len(unique_dates) <= deviation_days:
        # Dates already close - keep all
        for dim_name in ins:
            outs[dim_name] = ins[dim_name]
        return True

    # Find mode date
    max_idx = np.argmax(counts)
    max_count_date = unique_dates[max_idx]

    # Good dates: mode ± deviation_days
    date_diffs = np.abs([(d - max_count_date).days for d in unique_dates])
    good_date_indices = np.where(date_diffs <= deviation_days)[0]
    good_dates = unique_dates[good_date_indices]

    # Filter mask
    keep_mask = np.isin(dates_only, good_dates)
    n_total, n_kept = len(keep_mask), np.sum(keep_mask)
    percentage_removed = (1 - n_kept / n_total) * 100

    logger.info(
        "filter_deviation_day: %.1f%% supprimés (> %d jours autour %s)",
        percentage_removed,
        deviation_days,
        max_count_date,
    )

    # Apply to all dimensions
    for dim_name in ins:
        outs[dim_name] = ins[dim_name][keep_mask]

    return True
