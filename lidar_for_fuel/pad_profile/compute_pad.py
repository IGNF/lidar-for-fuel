"""
Compute the Plant Area Density (PAD) per vertical stratum.

"""

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def _max_or_neg_inf(values: np.ndarray) -> float:
    """`max()`, or `-inf` for an empty array (mirrors R's `max(numeric(0)) == -Inf`)."""
    return float(np.max(values)) if len(values) else -np.inf


def compute_pad(
    NRD: np.ndarray,
    Gf: np.ndarray,
    min_layer: np.ndarray,
    cos_theta: float,
    h_abg: np.ndarray,
    veg_gnd: np.ndarray,
    height_cover: float,
    cover_h_pad: float,
    use_cover: bool,
    ground_margin: float,
    dz: float,
    G: float,
    omega: float,
) -> np.ndarray:
    """Compute PAD per stratum from the Gap Fraction, with an optional cover correction.

    Args:
        NRD (np.ndarray): fractions of incoming rays intercepted for each stratum.
        Gf (np.ndarray): Gap Fraction per stratum, `Gf = 1 - NRD`.
        min_layer (np.ndarray): lower height bound of each stratum (m).
        cos_theta (float): scan angle factor.
        h_abg (np.ndarray): Normalized height above ground of every point (R's `Z`),
            used only for the `height_cover >= max(Z)` warning.
        veg_gnd (np.ndarray): Boolean mask, True for vegetation/ground points.
        height_cover (float): Height threshold (m) used for `cover_h_pad`.
        cover_h_pad (float): canopy cover fraction above `height_cover`.
        use_cover (bool): If False, skip the cover correction and use the plain
            Beer-Lambert formula for every stratum.
        ground_margin (float): Margin above `z0` excluded from the first stratum (m).
        dz (float): Stratum thickness (m).
        G (float): Leaf projection ratio.
        omega (float): Clumping factor.

    Returns:
        np.ndarray: PAD per stratum.
    """
    h_abg_veg_gnd = h_abg[veg_gnd]

    if use_cover:
        if height_cover >= _max_or_neg_inf(h_abg):
            logger.warning("height_cover > maximum vegetation height")

        if cover_h_pad == 0:
            logger.warning("Cover method not used in PAD computation as Cover_h_pad = 0")
            PAD = -np.log(Gf) * cos_theta / (G * omega * dz)
        elif np.any(NRD[min_layer >= height_cover] == cover_h_pad):
            logger.warning("Found NRD values equal to Cover_h_pad: not using Cover method.")
            PAD = -np.log(Gf) * cos_theta / (G * omega * dz)
        else:
            cover_h_pad_v = np.full(len(min_layer), cover_h_pad)
            cover_h_pad_v[min_layer < height_cover] = 1.0
            PAD = -np.log(1 - NRD / cover_h_pad_v) * cover_h_pad_v * cos_theta / (G * omega * dz)
    else:
        PAD = -np.log(Gf) * cos_theta / (G * omega * dz)

    if ground_margin > 0:
        # Regularize PAD on the bottom stratum, making the hypothesis of the same
        # PAD below the ground margin as above.
        PAD[min_layer == 0] = PAD[min_layer == 0] * dz / (dz - ground_margin)

    # Set PAD to 0 for upper strata with no vegetation/ground points.
    max_h_abg_veg_gnd = _max_or_neg_inf(h_abg_veg_gnd)
    min_empty = -np.inf if math.isinf(max_h_abg_veg_gnd) else math.ceil(max_h_abg_veg_gnd / dz) * dz
    PAD[min_layer >= min_empty] = 0.0

    return PAD
