"""
Count vegetation/ground hits (Ni) and entering rays (N) per vertical stratum.

"""

import numpy as np

from lidar_for_fuel.pad_profile.build_vertical_strata import build_vertical_strata


def _bin_counts(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """Count values per stratum.

    One count per interval `(breaks[i-1], breaks[i]]`, length `len(breaks) - 1`,
    including empty strata (zero count) and excluding out-of-range values --
    matching `table()`'s silent drop of the `NA` factor level `cut()` assigns
    to values outside `breaks`.
    """
    idx = np.digitize(values, breaks, right=True)
    counts = np.bincount(idx, minlength=len(breaks))
    return counts[1 : len(breaks)]


def compute_ni_n(
    h_abg: np.ndarray,
    veg_gnd: np.ndarray,
    z0: float,
    dz: float,
    nlayers: int | None,
    ground_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count vegetation/ground hits (Ni) and entering rays (N) per vertical stratum.

    Args:
        h_abg (np.ndarray): Normalized height above ground (R's `Z`).
        veg_gnd (np.ndarray): Boolean mask, True for vegetation/ground points.
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness (m). Default 1.
        nlayers (int | None): Number of strata above z0. If None, derived from
            `max(h_abg)`. Default 60.
        ground_margin (float): Margin above `z0` excluded from the first stratum
            (m). Default 0.1.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `(ni, n, min_layer)`, each of
        length `nlayers` (the below-ground stratum is dropped, matching R's
        `Ni[-1]`/`N[-1]`/`min_layer[-1]`):
            ni: vegetation/ground hit count per stratum.
            n: cumulative entering-ray count per stratum (non-decreasing). A
                point below the ground margin still counts toward every
                stratum's `n` via the cumulative sum, since a ray ending there
                had to pass through every stratum above it on the way down.
            min_layer: lower height bound of each stratum (m), pre-ground-margin-shift.
    """
    breaks, min_layer = build_vertical_strata(z0=z0, dz=dz, nlayers=nlayers, ground_margin=ground_margin, h_abg=h_abg)

    ni_full = _bin_counts(h_abg[veg_gnd], breaks)
    n_full = np.cumsum(_bin_counts(h_abg, breaks))

    return ni_full[1:], n_full[1:], min_layer[1:]
