"""
Compute PAD (Plant Area Density) metrics for a single pixel/plot of LiDAR points.

"""

import logging

import numpy as np

from lidar_for_fuel.commons.compute_class_counts import compute_class_counts
from lidar_for_fuel.commons.filter_points_by_date import filter_by_date
from lidar_for_fuel.pad_profile.compute_cos_theta import compute_cos_theta
from lidar_for_fuel.pad_profile.compute_cover import compute_cover
from lidar_for_fuel.pad_profile.compute_gf import compute_gf
from lidar_for_fuel.pad_profile.compute_ni_n import compute_ni_n
from lidar_for_fuel.pad_profile.compute_nrd import compute_nrd
from lidar_for_fuel.pad_profile.compute_pad import compute_pad

logger = logging.getLogger(__name__)


def _format_num(value: float) -> str:
    """Format a number the way R's `paste()` does: drop a trailing `.0`."""
    rounded = round(float(value), 6)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


def _compute_pad_profile_for_resolution(
    h_abg: np.ndarray,
    veg_gnd: np.ndarray,
    z0: float,
    dz: float,
    nlayers: int | None,
    ground_margin: float,
    cos_theta: float,
    height_cover: float,
    cover_h_pad: float,
    use_cover: bool,
    G: float,
    omega: float,
    include_ni_n: bool,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Compute one `PAD_{dz}_{min_layer}` stratum set, optionally with `N_*`/`Ni_*`.

    Returns:
        tuple[dict[str, float], dict[str, int], dict[str, int]]: `(pad, n, ni)`.
        `n`/`ni` are empty when `include_ni_n=False`.
    """
    # Step 1 : Create a sequence to make strata
    # Then, get number of returns intercepted and "pulses" entering in each strata
    Ni, N, min_layer = compute_ni_n(h_abg, veg_gnd, z0, dz, nlayers, ground_margin)

    # Step 2: Compute the fractions of incoming rays intercepted for each "NRD" stratum.
    NRD = compute_nrd(Ni, N)

    # Step 3: The probability that a ray crosses the stratum without being intercepted: Gap fraction
    Gf = compute_gf(NRD)

    # Step 4 : Compute PAD per stratum from the Gap Fraction, with an optional cover correction.
    PAD = compute_pad(
        NRD=NRD,
        Gf=Gf,
        min_layer=min_layer,
        cos_theta=cos_theta,
        h_abg=h_abg,
        veg_gnd=veg_gnd,
        height_cover=height_cover,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        ground_margin=ground_margin,
        dz=dz,
        G=G,
        omega=omega,
    )

    z_names = [f"{_format_num(dz)}_{_format_num(layer)}" for layer in min_layer]
    pad = {f"PAD_{name}": float(value) for name, value in zip(z_names, PAD)}
    if include_ni_n:
        n = {f"N_{name}": int(value) for name, value in zip(z_names, N)}
        ni = {f"Ni_{name}": int(value) for name, value in zip(z_names, Ni)}
    else:
        n, ni = {}, {}

    return pad, n, ni


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
    deviation_days: int,
    z0: float,
    dz: float,
    nlayers: int | None,
    dz_low: float,
    nlayers_low: int | None,
    ground_margin: float,
    cover_type: str,
    height_cover: float,
    use_cover: bool,
    G: float,
    omega: float,
    keep_values: list,
    keep_classes: list,
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
        deviation_days (int): Max deviation in days around the local modal acquisition date.
                                Default 0 (only the modal calendar day is retained).
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness of the main PAD profile (m). Default 1.
        nlayers (int | None): Number of strata above z0 for the main PAD
            profile. If None, derived from `max(h_abg)`. Default 60.
        dz_low (float): Stratum thickness of the low-strata PAD band (m).
            Default 0.5.
        nlayers_low (int | None): Number of strata above z0 for the
            low-strata PAD band. If None, derived from `max(h_abg)`. Default 4.
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
        keep_values (list): Classes to keep for counting Ni. Default: [2, 3, 4, 5, 9].
        keep_classes(list): Classes to keep for counting points. Default: [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67].


    Returns:
        dict[str, float] | None: `None` if a quality guard fails, otherwise a dict
        containing named-list output, in this exact channel order:
            PAD_{dz_low}_{min_layer}: Plant Area Density for that stratum of the
                low-strata band. One key per stratum.
            PAD_{dz}_{min_layer}: Plant Area Density for that stratum of the main
                profile, cover-corrected when possible. One key per stratum.
            Class_{code}: point count for each tracked LAS classification code,
                after the temporal filter, before the vegetation/ground subsetting.
            Total: point count of any classification, after the temporal filter.
            N_{dz}_{min_layer}, Ni_{dz}_{min_layer}: cumulative entering-ray count
                and vegetation/ground hit count for that stratum of the main profile.
            Cover_h_pad: canopy cover fraction above `height_cover`, or `NaN` if `use_cover=False`.
            Cover_2: canopy cover fraction above 2m.
            Cover_4: canopy cover fraction above 4m.
            Cover_6: canopy cover fraction above 6m.
            cos_theta: scan angle factor (1.0 if `scanning_angle=False`).
            Date_maj: Unix time (seconds) of the modal acquisition day for the
                points in the pixel/plot -- the center of the ±deviation_days
                temporal window.
            Date_min, Date_max: Unix time (seconds) of the lower/upper bound of
                that ±deviation_days temporal window (`Date_maj` -/+ `deviation_days`).
    """
    # # Step 1:
    # Filter points by ±deviation_days around the most densely sampled calendar day.
    valid, modal_time_unix = filter_by_date(gpstime, deviation_days=deviation_days)

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
    veg_ground_points = np.isin(classification, keep_values)

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
    # Compute the canopy cover fraction above height_cover, 2m, 4m and 6m.
    cover_h_pad, cover_2, cover_4, cover_6 = compute_cover(
        h_abg=h_abg,
        veg_gnd=veg_ground_points,
        return_number=return_number,
        cover_type=cover_type,
        height_cover=height_cover,
        use_cover=use_cover,
    )

    # # Step 6:
    # Count points per LAS classification code (post temporal filter, pre
    # vegetation/ground subsetting), plus the pixel total.
    class_counts = compute_class_counts(classification, keep_classes)

    # # Step 7:
    # Compute PAD per stratum from the Gap Fraction, with an optional cover
    # correction. Both calls share the same filtered points/cos_theta/cover_h_pad
    # (computed once above) so the two resolutions stay consistent with each other.

    # Low-strata band (dz_low/nlayers_low, e.g. 0.5 m x 4 = 0-2 m): only the
    # PAD_0.5_* channels are needed (output-list sections 4/5 only cover the
    # 1 m profile below), so N/Ni are discarded here.
    pad_low, _, _ = _compute_pad_profile_for_resolution(
        h_abg=h_abg,
        veg_gnd=veg_ground_points,
        z0=z0,
        dz=dz_low,  # strata grid for THIS resolution only
        nlayers=nlayers_low,
        ground_margin=ground_margin,
        cos_theta=cos_theta,  # shared across both resolutions
        height_cover=height_cover,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        G=G,
        omega=omega,
        include_ni_n=False,
    )
    # Main profile (dz/nlayers, e.g. 1 m x 60 = 0-60 m): also returns the
    # N_1_*/Ni_1_* stratum counts alongside PAD_1_*.
    pad_main, n_main, ni_main = _compute_pad_profile_for_resolution(
        h_abg=h_abg,
        veg_gnd=veg_ground_points,
        z0=z0,
        dz=dz,  # strata grid for THIS resolution only
        nlayers=nlayers,
        ground_margin=ground_margin,
        cos_theta=cos_theta,  # shared across both resolutions
        height_cover=height_cover,
        cover_h_pad=cover_h_pad,
        use_cover=use_cover,
        G=G,
        omega=omega,
        include_ni_n=True,
    )

    # # Step 8:
    # Assemble the final output dict, in the exact channel order of the
    # PAD output-list spec: PAD (low-strata, then main profile), Class_*/Total,
    # N_*, Ni_*, Cover_*, cos_theta, Date_*.
    output: dict[str, float] = pad_low | pad_main | class_counts | n_main | ni_main
    output["Cover_h_pad"] = cover_h_pad
    output["Cover_2"] = cover_2
    output["Cover_4"] = cover_4
    output["Cover_6"] = cover_6
    output["cos_theta"] = cos_theta
    modal_date = np.datetime64(modal_time_unix, "s")
    date_min = modal_date - np.timedelta64(deviation_days, "D")
    date_max = modal_date + np.timedelta64(deviation_days, "D")

    output["Date_maj"] = modal_time_unix
    output["Date_min"] = int((date_min - np.datetime64(0, "s")) / np.timedelta64(1, "s"))
    output["Date_max"] = int((date_max - np.datetime64(0, "s")) / np.timedelta64(1, "s"))

    return output
