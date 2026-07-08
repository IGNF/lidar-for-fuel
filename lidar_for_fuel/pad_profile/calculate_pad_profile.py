"""
Compute PAD (Plant Area Density) metrics for a single pixel/plot of LiDAR points.

"""

import logging

import numpy as np

from lidar_for_fuel.commons.filter_points_by_date import filter_by_date
from lidar_for_fuel.pad_profile.compute_cos_theta import compute_cos_theta
from lidar_for_fuel.pad_profile.compute_cover import compute_cover
from lidar_for_fuel.pad_profile.compute_gf import compute_gf
from lidar_for_fuel.pad_profile.compute_ni_n import compute_ni_n
from lidar_for_fuel.pad_profile.compute_nrd import compute_nrd
from lidar_for_fuel.pad_profile.compute_pad import compute_pad

logger = logging.getLogger(__name__)

_KEEP_VALUES = [1, 2, 3, 4, 5, 9]  # Classes to keep


def _format_num(value: float) -> str:
    """Format a number the way R's `paste()` does: drop a trailing `.0`."""
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


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
    cover_type: str,
    height_cover: float,
    use_cover: bool,
    G: float,
    omega: float,
    keep_N: bool,
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
        cover_type (str): Either "first" (cover estimated from first returns
            only) or "all" (cover estimated from all returns).
        height_cover (float): Height threshold (m) used for `cover_h_pad`.
        use_cover (bool): If False, `cover_h_pad` is `NaN`; the 2/4/6 m
            cover fractions are always computed, and the PAD formula falls
            back to the plain Beer-Lambert form for every stratum (no cover
            correction).
        G (float): Leaf projection ratio. Default 0.5.
        omega (float): Clumping factor. Default 1.
        keep_N (bool): If True, include `Ni_{dz}_{min_layer}`/`N_{dz}_{min_layer}`
            per stratum in the output. Default False.

    Returns:
        dict[str, float] | None: `None` if a quality guard fails, otherwise a dict
        containing named-list output exactly:
            date: GPS time of the modal acquisition day for the points in the
                pixel/plot -- the center of the ±deviation_days temporal window.
            Cover_h_pad: canopy cover fraction above `height_cover`, or `NaN` if `use_cover=False`.
            Cover_2: canopy cover fraction above 2m.
            Cover_4: canopy cover fraction above 4m.
            Cover_6: canopy cover fraction above 6m.
            cos_theta: scan angle factor (1.0 if `scanning_angle=False`).
            PAD_{dz}_{min_layer}: Plant Area Density for that stratum, cover-corrected
                when possible. One key per stratum.
            Ni_{dz}_{min_layer}, N_{dz}_{min_layer}: vegetation/ground hit count and
                cumulative entering-ray count for that stratum. Only present when
                `keep_N=True`.
    """
    # # Step 1:
    # Filter points by ±deviation_days around the most densely sampled calendar day.
    valid, modal_gpstime = filter_by_date(gpstime, deviation_days=deviation_days)

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

    # # Step 6:
    # Compute the fractions of incoming rays intercepted for each "NRD" stratum.
    NRD = compute_nrd(Ni, N)

    # # Step 7:
    # the probability that a ray crosses the stratum without being intercepted: Gap fraction
    Gf = compute_gf(NRD)

    # # Step 8:
    # Compute the canopy cover fraction above height_cover, 2m, 4m and 6m.
    cover_h_pad, cover_2, cover_4, cover_6 = compute_cover(
        h_abg=h_abg,
        veg_gnd=veg_ground_points,
        return_number=return_number,
        cover_type=cover_type,
        height_cover=height_cover,
        use_cover=use_cover,
    )

    # # Step 9:
    # Compute PAD per stratum from the Gap Fraction, with an optional cover correction.
    PAD = compute_pad(
        NRD=NRD,
        Gf=Gf,
        min_layer=min_layer,
        cos_theta=cos_theta,
        h_abg=h_abg,
        veg_gnd=veg_ground_points,
        height_cover=height_cover,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        ground_margin=ground_margin,
        dz=dz,
        G=G,
        omega=omega,
    )

    # # Step 10:
    # Assemble the final output dict.
    z_names = [f"{_format_num(dz)}_{_format_num(layer)}" for layer in min_layer]
    output: dict[str, float] = {
        "date": modal_gpstime, 
        # Predominant acquisition date for the points contained within a pixel; 
        # this date corresponds to the center of the time window.
        "Cover_h_pad": cover_h_pad,
        "Cover_2": cover_2,
        "Cover_4": cover_4,
        "Cover_6": cover_6,
        "cos_theta": cos_theta,
    }
    output.update({f"PAD_{name}": float(value) for name, value in zip(z_names, PAD)})
    if keep_N:
        output.update({f"Ni_{name}": int(value) for name, value in zip(z_names, Ni)})
        output.update({f"N_{name}": int(value) for name, value in zip(z_names, N)})

    return output
