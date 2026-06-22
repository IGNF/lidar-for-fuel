"""
Compute the LiDAR scan angle factor (`cos_theta`) for PAD computation.

When `scanning_angle=TRUE`,`cos(theta)` is estimated from the sensor-to-point trajectory.
When `scanning_angle=FALSE`,`cos(theta)` is fixed to 1 (no correction).
"""

import numpy as np


def compute_cos_theta(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_sensor: np.ndarray,
    y_sensor: np.ndarray,
    z_sensor: np.ndarray,
    veg_gnd: np.ndarray,
    scanning_angle: bool,
) -> tuple[float, float | None]:
    """Compute `cos_theta` and the mean flight height above ground.

    Args:
        x (np.ndarray): Point easting.
        y (np.ndarray): Point northing.
        z (np.ndarray): Raw, unnormalized elevation (R's `Zref`).
        x_sensor (np.ndarray): Sensor easting at acquisition time.
        y_sensor (np.ndarray): Sensor northing.
        z_sensor (np.ndarray): Sensor altitude.
        veg_gnd (np.ndarray): Boolean mask, True for vegetation/ground points.
        scanning_angle (bool): If False, `cos_theta=1.0` and `mean_flight_agl` is not
            computed (mirrors R's documented behaviour, not the R reference bug --
            see Design Notes in spec-pad-metrics-core.md).

    Returns:
        tuple[float, float | None]: `(cos_theta, mean_flight_agl)`. `mean_flight_agl`
        is `None` when `scanning_angle=False`; otherwise it ignores NaNs
        for the caller's own `limit_flight_agl` quality guard.
    """
    if not scanning_angle:
        return 1.0, None

    # Calculate flight height above ground in m. 
    # If the distance between the flight height and the ground
    flight_agl = z_sensor - z

    # Calculate mean flight height above ground in m.
    mean_flight_agl = float(np.nanmean(flight_agl))

    norm_u = np.sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2 + flight_agl**2)

    # Calculate component of  vector U (plane -> point). 
    nz_u = flight_agl / norm_u

    # Calculate "cos theta"
    cos_theta = float(np.mean(np.abs(nz_u[veg_gnd])))

    return cos_theta, mean_flight_agl
