"""
Per-pixel temporal filter for PAD computation.

Ports R/filters.R (`gpstime_to_datetime`, `is_in_season`, `is_near_date_mode`,
`filter_gpstime`). Unlike `commons/filter_points_by_date.py` (dalle-wide PDAL
pre-filter), this module operates on a single pixel's own points: the modal
acquisition date used by the deviation-days filter is computed locally on
that pixel's data, not on the whole tile.
"""

from typing import Iterable

import numpy as np

_DEFAULT_GPSTIME_REF = "2011-09-14 01:46:40"


def gpstime_to_datetime(gpstime: np.ndarray, gpstime_ref: str = _DEFAULT_GPSTIME_REF) -> np.ndarray:
    """Convert GPS time (seconds since `gpstime_ref`) to `datetime64[us]`.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        gpstime_ref (str): UTC reference datetime ("YYYY-MM-DD HH:MM:SS") for gpstime=0.

    Returns:
        np.ndarray: `datetime64[us]` array, same shape as `gpstime`.
    """
    ref = np.datetime64(gpstime_ref.replace(" ", "T"), "us")
    micros = np.round(np.asarray(gpstime, dtype=np.float64) * 1e6).astype(np.int64)
    return ref + micros.astype("timedelta64[us]")


def is_in_season(datetime_arr: np.ndarray, months: Iterable[int] = range(1, 13)) -> np.ndarray:
    """Return True for points whose calendar month is in `months`.

    Args:
        datetime_arr (np.ndarray): `datetime64` array.
        months (Iterable[int]): Calendar months (1-12) defining the season. Default: all year.

    Returns:
        np.ndarray: Boolean mask, same shape as `datetime_arr`.
    """
    months = sorted(months)
    if months == list(range(1, 13)):
        return np.ones(len(datetime_arr), dtype=bool)
    month_values = datetime_arr.astype("datetime64[M]").astype(np.int64) % 12 + 1
    return np.isin(month_values, months)


def _mode_first_occurrence(values: np.ndarray) -> np.generic:
    """Most frequent value, ties broken by first-occurrence order (mirrors R's `.mode`)."""
    unique_vals, first_idx, counts = np.unique(values, return_index=True, return_counts=True)
    order = np.argsort(first_idx)
    return unique_vals[order][np.argmax(counts[order])]


def is_near_date_mode(datetime_arr: np.ndarray, deviation_days: float = np.inf) -> np.ndarray:
    """Return True for points within `deviation_days` of the local modal calendar day.

    Args:
        datetime_arr (np.ndarray): `datetime64` array.
        deviation_days (float): Half-width of the retention window in days. `inf` = no filter.

    Returns:
        np.ndarray: Boolean mask, same shape as `datetime_arr`.
    """
    if np.isinf(deviation_days):
        return np.ones(len(datetime_arr), dtype=bool)
    dates = datetime_arr.astype("datetime64[D]")
    modal_date = _mode_first_occurrence(dates)
    diff_days = np.abs((dates - modal_date).astype(np.int64))
    return diff_days <= deviation_days


def filter_gpstime(
    gpstime: np.ndarray,
    months: Iterable[int] = range(1, 13),
    deviation_days: float = np.inf,
    gpstime_ref: str = _DEFAULT_GPSTIME_REF,
) -> np.ndarray:
    """Boolean mask for points within the season and around the local date mode.

    Mirrors R's `filter_gpstime`: season filter is applied first; the date-mode
    window is then computed only on the season-filtered subset.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        months (Iterable[int]): Calendar months (1-12) defining the season. Default: all year.
        deviation_days (float): Half-width of the retention window in days around the
            local modal acquisition day. `inf` = no filter. Default: `inf`.
        gpstime_ref (str): UTC reference datetime for gpstime=0.

    Returns:
        np.ndarray: Boolean mask, same shape as `gpstime`.
    """
    datetime_arr = gpstime_to_datetime(gpstime, gpstime_ref)
    season_mask = is_in_season(datetime_arr, months)

    valid = np.zeros(len(gpstime), dtype=bool)
    if np.any(season_mask):
        valid[season_mask] = is_near_date_mode(datetime_arr[season_mask], deviation_days)
    return valid
