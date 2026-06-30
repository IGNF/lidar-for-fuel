"""
Build the vertical strata grid used to bin LiDAR point heights for PAD computation.

"""
import math

import numpy as np


def build_vertical_strata(
    z0: float = 0.0,
    dz: float = 1.0,
    nlayers: int | None = 60,
    ground_margin: float = 0.1,
    h_abg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the break sequence used to bin point heights into PAD strata.

    Args:
        z0 (float): Minimum height of the first layer in meters. Default 0 meters.
        dz (float): Height of a layer in meters. Default 1 meters.
        nlayers (int | None): Number of strata above z0. If None, derived from
            `h_abg`: z_max_pad is the smallest multiple of dz that is >= max(h_abg).
            Default 60.
        ground_margin (float): Margin between the ground (Z=0) and the minimum height of the first layer in meters.
            This margin is used to remove the points near the ground from the PAD computation.
            Default 0.1 meters.
        h_abg (np.ndarray | None): Normalized height above ground, required
            only when `nlayers` is None. Ignored otherwise. (R's `Z`).

    Returns:
        tuple[np.ndarray, np.ndarray]: `(breaks, min_layer)`.
        `breaks` starts with `-np.inf` and ends with `z_max_pad`; the break at
        exact height 0 is shifted up by `ground_margin`.
        `min_layer` is `breaks[:-1]` captured *before* the "theoretical" ground-margin shift
        (so it still reports the unshifted stratum lower bound); its first
        element is `-np.inf` (the below-ground stratum, dropped by the caller).

    Raises:
        ValueError: If `nlayers` is None and `h_abg` is not provided or is empty.
    """
    # Determine the top of the grid (z_max_pad), the highest stratum edge.
    if nlayers is None:
        if h_abg is None or len(h_abg) == 0:
            raise ValueError("h_abg must be a non-empty array when nlayers is None")
        # No fixed number of strata was given: derive it from the data, rounding
        # the highest point up to the next multiple of dz so no point is cut off.
        z_max_pad = math.ceil(float(np.max(h_abg)) / dz) * dz
    else:
        z_max_pad = z0 + dz * nlayers

    # Build the regularly-spaced stratum edges [z0, z0+dz, ..., z_max_pad].
    n_steps = round((z_max_pad - z0) / dz)
    seq = z0 + np.arange(n_steps + 1, dtype=np.float64) * dz

    # Prepend -inf as a safety net:
    # any point below z0 (below ground) falls into this first, fictitious stratum.
    breaks = np.empty(len(seq) + 1, dtype=np.float64)
    breaks[0] = -np.inf
    breaks[1:] = seq

    # Capture each stratum's lower bound *before* the ground-margin shift
    # below, by dropping the last edge.
    min_layer = breaks[:-1].copy()

    # Push the ground-level edge up by ground_margin, so points within [0, ground_margin]
    # fall into the dropped stratum instead of the first real one.
    breaks[breaks == 0.0] += ground_margin

    return breaks, min_layer
