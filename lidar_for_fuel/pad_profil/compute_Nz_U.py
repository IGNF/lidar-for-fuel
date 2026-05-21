"""
Compute Nz_U, the vertical component of the unit pulse vector (plane → point).

    flight_agl = Elevation - Zref                     (sensor height above ground)
    norm_U     = ||(X-Easting, Y-Northing, agl)||     (pulse vector norm)
    Nz_U       = flight_agl / norm_U  =  cos(θ)       (θ = angle from nadir)

Nz_U is used downstream as cos_theta to correct PAD for the scanning angle.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def Nz_U(
    X: np.ndarray,
    Y: np.ndarray,
    Zref: np.ndarray,
    Easting: np.ndarray,
    Northing: np.ndarray,
    Elevation: np.ndarray,
    scanning_angle: bool = True,
    limit_flight_agl: float = 800.0,
) -> np.ndarray | None:
    """Compute the vertical component of the unit pulse vector (Nz_U = cos θ).

    Nz_U is the cosine of the scanning angle from nadir. It is used as
    ``cos_theta`` in the PAD calculation to correct for oblique pulses.

    Args:
        X: Point easting coordinate (Lambert-93, m).
        Y: Point northing coordinate (Lambert-93, m).
        Zref: Ground elevation at each point (from DTM, field ``Z_ref``, m).
        Easting: Sensor easting at the point's acquisition time (m).
        Northing: Sensor northing at the point's acquisition time (m).
        Elevation: Sensor altitude at the point's acquisition time (m).
        scanning_angle: If False, returns 1.0 (vertical pulses assumed, no correction).
        limit_flight_agl: Minimum acceptable mean sensor height above ground (m).
            Below this threshold the trajectory is considered aberrant and
            None is returned with a warning.

    Returns:
        Nz_U (= cos θ): vertical component of the unit pulse vector (array),
        or 1.0 if ``scanning_angle`` is False,
        or None if the trajectory is aberrant.
    """
    if not scanning_angle:
        return 1.0

    flight_agl = Elevation - Zref
    mean_agl = float(np.mean(flight_agl))

    if mean_agl < limit_flight_agl:
        logger.warning(
            "NULL return: mean flight AGL (%.1f m) is below the threshold (%.1f m). "
            "Check your trajectory and avoid using scanning_angle if it is uncertain.",
            mean_agl,
            limit_flight_agl,
        )
        return None

    norm_U = np.sqrt((X - Easting) ** 2 + (Y - Northing) ** 2 + flight_agl ** 2)
    Nz_U = flight_agl / norm_U
    return Nz_U
