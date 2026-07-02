"""
Calculate the probability that a ray crosses the stratum without being intercepted: "Gap Fraction".

"""
import numpy as np


def compute_gf(
    NRD: np.ndarray,
) -> np.ndarray:
    """Calculate the probability that a ray crosses the stratum without being intercepted: "Gap Fraction".

    Args:
        NRD (np.ndarray):  NRD values, corrected for boundary cases (NRD=0 or NRD=1)
            per equations 23-24 of Pimont et al. 2018.

    Returns:
        np.ndarray: Gf values.
    """
    # Compute Gap Fraction estimation
    Gf = 1 - NRD

    return Gf
