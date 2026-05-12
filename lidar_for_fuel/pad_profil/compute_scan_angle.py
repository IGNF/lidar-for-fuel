"""
Compute the scan angle correction factor "cos_theta" for PAD calculation.

The "cos_theta" factor corrects for the obliqueness of LiDAR pulses:
    flight_agl = Elevation - Zref                     (sensor height above ground)
    norm_U     = ||(X-Easting, Y-Northing, agl)||     (pulse vector norm)
    Nz_U       = flight_agl / norm_U                  (vertical component, normalised)
    cos_theta  = mean(|Nz_U|) over vegetation + ground points

Reference: Pimont et al. (2018); R source: pad_metrics.R L128-141.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def compute_cos_theta(
    X: np.ndarray,
    Y: np.ndarray,
    Zref: np.ndarray,
    Easting: np.ndarray,
    Northing: np.ndarray,
    Elevation: np.ndarray,
    Classification: np.ndarray,
    scanning_angle: bool = True,
    limit_flight_agl: float = 800.0,
) -> float | None:
    """Compute the mean cosine of the scan zenith angle ("cos_theta").

    Args:
        X: Point easting coordinate (Lambert-93, m).
        Y: Point northing coordinate (Lambert-93, m).
        Zref: Ground elevation at each point (from DTM, field ``Z_ref``, m).
        Easting: Sensor easting at the point's acquisition time (m).
        Northing: Sensor northing at the point's acquisition time (m).
        Elevation: Sensor altitude at the point's acquisition time (m).
        Classification: LiDAR classification code for each point.
        scanning_angle: If False, returns 1.0 (vertical pulses assumed).
        limit_flight_agl: Minimum acceptable mean sensor height above ground (m).
            Below this threshold the trajectory is considered aberrant:
            None is returned with a warning.

    Returns:
        cos_theta in (0, 1] averaged over vegetation and ground points,
        or None if the flight height guard fails.
    """
    if not scanning_angle:
        return 1.0

    flight_agl = Elevation - Zref
    mean_agl = float(np.mean(flight_agl))

    if mean_agl < limit_flight_agl:
        logger.warning(
            "cos_theta: mean flight height above ground (%.1f m) is below "
            "limit_flight_agl (%.1f m). "
            "Pixel set to NaN. Check trajectory or disable scanning_angle.",
            mean_agl,
            limit_flight_agl,
        )
        return None

    norm_U = np.sqrt((X - Easting) ** 2 + (Y - Northing) ** 2 + flight_agl ** 2)
    Nz_U = flight_agl / norm_U

    veg_gnd_mask = (Classification >= 5) | (Classification == 9)
    return float(np.mean(np.abs(Nz_U[veg_gnd_mask])))
