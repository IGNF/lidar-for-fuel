"""
Compute PAD (Plant Area Density) metrics for a single pixel/plot of LiDAR points.

"""

import logging

import numpy as np

from lidar_for_fuel.commons.filter_points_by_date import filter_by_date
from lidar_for_fuel.pad_profile.compute_cos_theta import compute_cos_theta
from lidar_for_fuel.pad_profile.compute_ni_n import compute_ni_n

logger = logging.getLogger(__name__)

_KEEP_VALUES = [1, 2, 3, 4, 5, 9]  # Classes to keep


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
    scanning_angle: bool,
    limit_N_points: int,
    limit_flight_agl: float,
    deviation_days: float,
    z0: float,
    dz: float,
    nlayers: int | None,
    ground_margin: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
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
        scanning_angle (bool): If True, estimate cos(theta) from the trajectory.
                               If False, returns 1.0 (vertical pulses assumed, no correction).
                               Default True.
        limit_N_points (int): Minimum number of point in the pixel/plot for computing profiles & metrics.
                            Default 400 points.
        limit_flight_agl (float):  Limit flight height above ground in m.
                                   If the distance between the flight height and the ground and (Elevation - Zref)
                                   is lower than `limit_flight_agl`, NULL is returned.
                                   Default 800 meters.
        deviation_days (float): Max deviation in days around the local modal acquisition date.
                                `inf` = no filter.
                                Default `inf`.
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness (m). Default 1.
        nlayers (int | None): Number of strata above z0. If None, derived from
            `max(h_abg)`. Default 60.
        ground_margin (float): Margin above `z0` excluded from the first stratum
            (m). Default 0.1.

    Returns:
        tuple[float, np.ndarray, np.ndarray, np.ndarray] | None: `(cos_theta, ni, n,
        min_layer)`, or None if a quality guard fails.
            cos_theta: scan angle factor (1.0 if `scanning_angle=False`).
            ni: vegetation/ground hit count per stratum.
            n: cumulative entering-ray count per stratum (non-decreasing). A
                point below the ground margin still counts toward every
                stratum's `n` via the cumulative sum, since a ray ending there
                had to pass through every stratum above it on the way down.
            min_layer: lower height bound of each stratum (m), pre-ground-margin-shift.
    """
    # # Step 1:
    # Filter points by ±deviation_days around the most densely sampled calendar day.
    valid = filter_by_date(gpstime, deviation_days=deviation_days)

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

    # # Step 2:
    # Check that there is a minimum number of points
    if len(h_abg) < limit_N_points:
        logger.warning("NULL return: the number of points < limit_N_points. Check the point cloud.")
        return None

    # # Step 3:
    # Keep only points with classes unclassified, ground, vegetations and water
    veg_ground_points = np.isin(classification, _KEEP_VALUES)

    # # Step 4:
    # Calculate "cos_theta"
    cos_theta, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_ground_points,
        scanning_angle=scanning_angle,
    )
    if scanning_angle and mean_flight_agl < limit_flight_agl:
        logger.warning(
            "NULL return: limit_flight_agl below the threshold (%.1f < %.1f). "
            "Check your trajectory and avoid using scanning_angle mode if the trajectory is uncertain.",
            mean_flight_agl,
            limit_flight_agl,
        )
        return None

    # # Step 5:
    # Create a sequence to make strata
    # Then, get number of returns intercepted and "pulses" entering in each strata
    Ni, N, min_layer = compute_ni_n(h_abg, veg_ground_points, z0, dz, nlayers, ground_margin)

    return cos_theta, Ni, N, min_layer
