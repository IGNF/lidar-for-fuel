"""
Build the vertical strata grid for PAD computation.

Ports the strata-building section of R/pad_metrics.R (lines 143-153):
    if (is.null(nlayers)) z_max_pad <- plyr::round_any(max(Z), dz, ceiling)
    else                  z_max_pad <- z0 + dz * nlayers
    breaks    <- c(-Inf, seq(z0, z_max_pad, dz))
    min_layer <- breaks[-length(breaks)]
    breaks[breaks == 0] <- breaks[breaks == 0] + ground_margin
"""

import math

import numpy as np


def build_vertical_strata(
    z0: float = 0.0,
    dz: float = 1.0,
    nlayers: int | None = 60,
    ground_margin: float = 0.1,
    z_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the break sequence used to bin points into PAD strata.

    Args:
        z0 (float): Bottom height of the first real stratum (m). Default 0.
        dz (float): Stratum thickness (m). Default 1.
        nlayers (int | None): Number of strata above z0. If None, derived from
            `z_values`: the smallest multiple of `dz` that is >= max(z_values). Default 60.
        ground_margin (float): Offset added to the break at height 0 (m), excluding points
            very close to the ground from PAD computation. Default 0.1.
        z_values (np.ndarray | None): Point heights, required only when `nlayers` is None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            breaks (np.ndarray): Bin edges, starting with `-np.inf`, length `nlayers + 2`.
                The edge at height 0 is shifted up by `ground_margin`.
            min_layer (np.ndarray): Lower bound of each stratum *before* the ground-margin
                shift, length `nlayers + 1`. First element is `-np.inf` (the below-z0
                stratum, to be dropped by the caller after Ni/N counting).

    Raises:
        ValueError: If `nlayers` is None and `z_values` is empty or not provided.
    """
    if nlayers is None:
        if z_values is None or len(z_values) == 0:
            raise ValueError("z_values must be a non-empty array when nlayers is None")
        z_max_pad = math.ceil(float(np.max(z_values)) / dz) * dz
    else:
        z_max_pad = z0 + dz * nlayers

    n_steps = round((z_max_pad - z0) / dz)
    seq = z0 + np.arange(n_steps + 1, dtype=np.float64) * dz

    breaks = np.empty(len(seq) + 1, dtype=np.float64)
    breaks[0] = -np.inf
    breaks[1:] = seq

    min_layer = breaks[:-1].copy()

    ground_mask = breaks == 0.0
    breaks[ground_mask] += ground_margin

    return breaks, min_layer
