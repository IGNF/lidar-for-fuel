"""
Compute Nz_U, the vertical component of the unit pulse vector (plane → point).

    flight_agl = Elevation - h_abg                     (sensor height above ground)
    norm_u     = ||(x-x_sensor, y-y_sensor, agl)||      (pulse vector norm)
    Nz_U       = flight_agl / norm_U  =  cos(θ)       (θ = angle from nadir)

Nz_U is used downstream as cos_theta to correct PAD for the scanning angle.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_Nz_U(
    x: np.ndarray,
    y: np.ndarray,
    h_abg: np.ndarray,
    x_sensor: np.ndarray,
    y_sensor: np.ndarray,
    z_sensor: np.ndarray,
    scanning_angle: bool = True,
    limit_flight_agl: float = 800.0,
) -> np.ndarray | None:
    """Compute the vertical component of the unit pulse vector (Nz_U = cos θ).

    Nz_U is the cosine of the scanning angle from nadir. It is used as
    ``cos_theta`` in the PAD calculation to correct for oblique pulses.

    Args:
        x (np.ndarray): Point easting coordinate.
        y (np.ndarray): Point northing coordinate.
        h_abg (np.ndarray): Ground elevation at each point (m).
        x_sensor (np.ndarray): Sensor easting at the point's acquisition time (m).
        y_sensor (np.ndarray): Sensor northing at the point's acquisition time (m).
        z_sensor (np.ndarray): Sensor altitude at the point's acquisition time (m).
        scanning_angle (bool): If False, returns 1.0 (vertical pulses assumed, no correction).
        limit_flight_agl (float): Minimum acceptable mean sensor height above ground (m).
            Below this threshold the trajectory is considered aberrant and
            None is returned with a warning.

    Returns:
        Nz_U (np.ndarray): vertical component of the unit pulse vector (array),
        or 1.0 if ``scanning_angle`` is False,
        or None if the trajectory is aberrant.
    """
    if not scanning_angle:
        return 1.0

    flight_agl = z_sensor - h_abg
    mean_agl = float(np.mean(flight_agl))

    if mean_agl < limit_flight_agl:
        logger.warning(
            "NULL return: mean flight AGL (%.1f m) is below the threshold (%.1f m). "
            "Check your trajectory and avoid using scanning_angle if it is uncertain.",
            mean_agl,
            limit_flight_agl,
        )
        return None

    norm_u = np.sqrt((x - x_sensor) ** 2 + (y - y_sensor) ** 2 + flight_agl ** 2)
    nz_u = flight_agl / norm_u
    return nz_u
