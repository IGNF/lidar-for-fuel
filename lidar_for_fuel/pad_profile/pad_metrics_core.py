"""
Compute PAD (Plant Area Density) metrics for a single pixel/plot of LiDAR points.

Pure-numpy port of R/pad_metrics.R `.pad_metrics()`. Operates on one group of
points (no file I/O, no multi-pixel batching). Field naming follows the
repo's preprocessing convention, not the R/doc names:

    h_abg              <-> R's Z      (normalized height above ground)
    Z                  <-> R's Zref   (raw, unnormalized elevation)
    X_sensor/Y_sensor/Z_sensor <-> R's Easting/Northing/Elevation
"""

import logging
from typing import Iterable

import numpy as np

from lidar_for_fuel.pad_profile.build_vertical_strata import build_vertical_strata
from lidar_for_fuel.pad_profile.filter_gpstime import filter_gpstime

logger = logging.getLogger(__name__)

_DEFAULT_GPSTIME_REF = "2011-09-14 01:46:40"


def _format_num(value: float) -> str:
    """Format a number the way R's `paste()` does: drop a trailing `.0`."""
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


def _bin_counts(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Count values per stratum, mirroring R's `cut(values, breaks, right=TRUE) |> table()`.

    Returns one count per interval `(breaks[i-1], breaks[i]]`, length `len(breaks) - 1`,
    including empty strata (zero count) and excluding out-of-range values.
    """
    idx = np.digitize(values, breaks, right=True)
    counts = np.bincount(idx, minlength=len(breaks))
    return counts[1 : len(breaks)]


def pad_metrics_core(
    gpstime: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    h_abg: np.ndarray,
    z: np.ndarray,
    return_number: np.ndarray,
    classification: np.ndarray,
    x_sensor: np.ndarray,
    y_sensor: np.ndarray,
    z_sensor: np.ndarray,
    z0: float = 0.0,
    dz: float = 1.0,
    nlayers: int | None = 60,
    ground_margin: float = 0.1,
    G: float = 0.5,
    omega: float = 0.77,
    scanning_angle: bool = True,
    cover_type: str = "all",
    height_cover: float = 2.0,
    use_cover: bool = True,
    limit_N_points: int = 0,
    limit_flight_agl: float = 800.0,
    keep_N: bool = False,
    season_filter: Iterable[int] = range(1, 13),
    deviation_days: float = np.inf,
    gpstime_ref: str = _DEFAULT_GPSTIME_REF,
) -> dict[str, float] | None:
    """Compute PAD metrics for one pixel/plot of LiDAR points.

    Args:
        gpstime (np.ndarray): GPS time in seconds.
        x (np.ndarray): Point easting.
        y (np.ndarray): Point northing.
        h_abg (np.ndarray): Normalized height above ground (R's `Z`).
        z (np.ndarray): Raw, unnormalized elevation (R's `Zref`).
        return_number (np.ndarray): Return number within the pulse.
        classification (np.ndarray): LAS classification code.
        x_sensor (np.ndarray): Sensor easting at acquisition time (R's `Easting`).
        y_sensor (np.ndarray): Sensor northing (R's `Northing`).
        z_sensor (np.ndarray): Sensor altitude (R's `Elevation`).
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness (m). Default 1.
        nlayers (int | None): Number of strata. If None, derived from `max(h_abg)`. Default 60.
        ground_margin (float): Margin above `z0` excluded from the first stratum (m). Default 0.1.
        G (float): Leaf projection ratio. Default 0.5.
        omega (float): Clumping factor. Default 0.77.
        scanning_angle (bool): If True, estimate cos(theta) from the trajectory. Default True.
        cover_type (str): "all" or "first". Default "all".
        height_cover (float): Height threshold (m) for canopy cover estimation. Default 2.
        use_cover (bool): Use the cover-normalised PAD correction. Default True.
        limit_N_points (int): Minimum point count (after temporal filter) to compute metrics.
            Default 0.
        limit_flight_agl (float): Minimum acceptable mean flight height above ground (m).
            Default 800.
        keep_N (bool): If True, include raw `Ni_*`/`N_*` counts per stratum. Default False.
        season_filter (Iterable[int]): Calendar months (1-12) to keep. Default: all year.
        deviation_days (float): Max deviation in days around the local modal acquisition
            date. `inf` = no filter. Default `inf`.
        gpstime_ref (str): UTC reference datetime for gpstime=0.

    Returns:
        dict[str, float] | None: PAD metrics (`date`, `Cover_h_pad`, `Cover_2`, `Cover_4`,
        `Cover_6`, `cos_theta`, `PAD_{dz}_{z_bottom}` per stratum, plus `Ni_*`/`N_*` when
        `keep_N=True`), or None if the pixel fails a quality guard.
    """
    valid = filter_gpstime(gpstime, months=season_filter, deviation_days=deviation_days, gpstime_ref=gpstime_ref)
    gpstime = gpstime[valid]
    x = x[valid]
    y = y[valid]
    h_abg = h_abg[valid]
    z = z[valid]
    return_number = return_number[valid]
    classification = classification[valid]
    x_sensor = x_sensor[valid]
    y_sensor = y_sensor[valid]
    z_sensor = z_sensor[valid]

    if len(h_abg) < limit_N_points:
        logger.warning("NULL return: the number of points < limit_N_points. Check the point cloud.")
        return None

    veg_gnd = (classification <= 5) | (classification == 9)
    h_abg_veg_gnd = h_abg[veg_gnd]

    if scanning_angle:
        flight_agl = z_sensor - z
        mean_agl = float(np.nanmean(flight_agl))  # R: mean(flight_agl, na.rm = TRUE)
        if mean_agl < limit_flight_agl:
            logger.warning(
                "NULL return: limit_flight_agl below the threshold (%.1f < %.1f). "
                "Check your trajectory and avoid using scanning_angle mode if the trajectory is uncertain.",
                mean_agl,
                limit_flight_agl,
            )
            return None
        norm_u = np.sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2 + flight_agl**2)
        nz_u = flight_agl / norm_u
    else:
        nz_u = np.ones_like(h_abg)

    breaks, min_layer = build_vertical_strata(
        z0=z0, dz=dz, nlayers=nlayers, ground_margin=ground_margin, z_values=h_abg
    )

    ni_full = _bin_counts(h_abg_veg_gnd, breaks)
    n_full = np.cumsum(_bin_counts(h_abg, breaks))

    ni = ni_full[1:]
    n = n_full[1:]
    min_layer = min_layer[1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        nrd = np.where(n == 0, 0.0, ni / n)

    i_nrdc = (nrd == 0) | (nrd == 1)
    nrd = np.where(i_nrdc, (ni + 1) / (n + 2), nrd)

    gf = 1 - nrd
    cos_theta = float(np.mean(np.abs(nz_u[veg_gnd])))

    cover_h_pad = np.nan
    n_total = len(h_abg)
    if cover_type == "first":
        first_returns = return_number == 1
        fr_veg_gnd = first_returns[veg_gnd]
        n_f = int(np.sum(first_returns))
        if use_cover:
            cover_h_pad = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > height_cover]) / n_f)
        cover_2 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 2]) / n_f)
        cover_4 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 4]) / n_f)
        cover_6 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 6]) / n_f)
    elif cover_type == "all":
        if use_cover:
            cover_h_pad = float(np.sum(h_abg_veg_gnd > height_cover) / n_total)
        cover_2 = float(np.sum(h_abg_veg_gnd > 2) / n_total)
        cover_4 = float(np.sum(h_abg_veg_gnd > 4) / n_total)
        cover_6 = float(np.sum(h_abg_veg_gnd > 6) / n_total)
    else:
        raise ValueError("cover_type must be 'all' or 'first'")

    if use_cover:
        max_h_abg = np.max(h_abg) if h_abg.size else -np.inf  # R: max(numeric(0)) == -Inf, no crash
        if height_cover >= max_h_abg:
            logger.warning("height_cover > maximum vegetation height")
        if cover_h_pad == 0:
            logger.warning("Cover method not used in PAD computation as Cover_h_pad = 0")
            pad = -np.log(gf) * cos_theta / (G * omega * dz)
        elif np.any(nrd[min_layer >= height_cover] == cover_h_pad):
            logger.warning("Found NRD values equal to Cover_h_pad: not using Cover method.")
            pad = -np.log(gf) * cos_theta / (G * omega * dz)
        else:
            cover_h_pad_v = np.where(min_layer < height_cover, 1.0, cover_h_pad)
            pad = -np.log(1 - nrd / cover_h_pad_v) * cover_h_pad_v * cos_theta / (G * omega * dz)
    else:
        pad = -np.log(gf) * cos_theta / (G * omega * dz)

    if ground_margin > 0:
        pad = np.where(min_layer == 0, pad * dz / (dz - ground_margin), pad)

    if h_abg_veg_gnd.size:
        min_empty = np.ceil(np.max(h_abg_veg_gnd) / dz) * dz
    else:
        min_empty = -np.inf
    pad = np.where(min_layer >= min_empty, 0.0, pad)

    z_names = [f"{_format_num(dz)}_{_format_num(layer)}" for layer in min_layer]
    output: dict[str, float] = {
        "date": float(np.mean(gpstime)),
        "Cover_h_pad": cover_h_pad,
        "Cover_2": cover_2,
        "Cover_4": cover_4,
        "Cover_6": cover_6,
        "cos_theta": cos_theta,
    }
    output.update({f"PAD_{name}": float(value) for name, value in zip(z_names, pad)})
    if keep_N:
        output.update({f"Ni_{name}": int(value) for name, value in zip(z_names, ni)})
        output.update({f"N_{name}": int(value) for name, value in zip(z_names, n)})

    return output
