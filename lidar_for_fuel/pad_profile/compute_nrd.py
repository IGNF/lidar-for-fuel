"""
Calculate the fractions of incoming rays intercepted for each "NRD" stratum.

"""
import numpy as np


def compute_nrd(
    Ni: np.ndarray,
    N: np.ndarray,
) -> np.ndarray:
    """Calculate the fractions of incoming rays intercepted for each "NRD" stratum.

    Args:
        Ni (np.ndarray): each of length `nlayers`, vegetation/ground hit count per stratum.
        N (np.ndarray): each of length `nlayers`, cumulative entering-ray count per stratum (non-decreasing).

    Returns:
        np.ndarray: NRD values, corrected for boundary cases (NRD=0 or NRD=1)
            per equations 23-24 of Pimont et al. 2018.
    """
    # General case:
    # Compute the fraction of intercepted rays per stratum.
    # /!\ np.errstate(invalid="ignore") suppresses the RuntimeWarning triggered by 0/0 (when N=0 and Ni=0).
    with np.errstate(invalid="ignore"):
        nrd = Ni / N

    # When N=0 then Ni=0 (no incoming rays → no interception):
    # 0/0 yields NaN, replaced by 0 (null intercepted fraction).
    nrd[np.isnan(nrd)] = 0.0

    # Boundary cases NRD=0 or NRD=1:
    # Laplace Bayesian correction (Pimont et al. 2018, eq. 23-24).
    # These extreme values make the PAD computation unstable.
    # => (Ni+1)/(N+2) pulls the extremes slightly toward the center without affecting intermediate values.
    i_nrdc = (nrd == 0) | (nrd == 1)
    nrd[i_nrdc] = (Ni[i_nrdc] + 1) / (N[i_nrdc] + 2)

    return nrd
