"""
Compute the canopy cover fraction at several height thresholds.

"""
import numpy as np


def compute_cover(
    h_abg: np.ndarray,
    veg_gnd: np.ndarray,
    return_number: np.ndarray,
    cover_type: str,
    height_cover: float,
    use_cover: bool,
) -> tuple[float, float, float, float]:
    """Compute the canopy cover fraction above `height_cover`, 2m, 4m and 6m.

    Args:
        h_abg (np.ndarray): Normalized height above ground (R's `Z`).
        veg_gnd (np.ndarray): Boolean mask, True for vegetation/ground points.
        return_number (np.ndarray): Return number within the pulse.
        cover_type (str): Either "first" (cover estimated from first returns
            only) or "all" (cover estimated from all returns).
        height_cover (float): Height threshold (m) used for `cover_h_pad`.
        use_cover (bool): If False, `cover_h_pad` is `NaN` (R's `NA`); the
            other three thresholds (2/4/6 m) are always computed.

    Returns:
        tuple[float, float, float, float]: `(cover_h_pad, cover_2, cover_4, cover_6)`.
            cover_h_pad: cover fraction above `height_cover`, or `NaN` if `use_cover=False`.
            cover_2: cover fraction above 2m.
            cover_4: cover fraction above 4m.
            cover_6: cover fraction above 6m.

    Raises:
        ValueError: If `cover_type` is neither "first" nor "all".
    """
    h_abg_veg_gnd = h_abg[veg_gnd]

    if cover_type == "first":
        # Cover among first returns only: numerator restricted to veg/ground
        # points that are also first returns, denominator = all first returns.
        first_returns = return_number == 1
        fr_veg_gnd = first_returns[veg_gnd]
        n_denom = np.sum(first_returns)
        if use_cover:
            cover_h_pad = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > height_cover]) / n_denom)
        else:
            cover_h_pad = np.nan
        cover_2 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 2]) / n_denom)
        cover_4 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 4]) / n_denom)
        cover_6 = float(np.sum(fr_veg_gnd[h_abg_veg_gnd > 6]) / n_denom)
    elif cover_type == "all":
        # Cover among all returns: numerator = veg/ground points above threshold,
        # denominator = all points (post date-filter), not just veg/ground.
        n_denom = len(h_abg)
        if use_cover:
            cover_h_pad = float(np.sum(h_abg_veg_gnd > height_cover) / n_denom)
        else:
            cover_h_pad = np.nan
        cover_2 = float(np.sum(h_abg_veg_gnd > 2) / n_denom)
        cover_4 = float(np.sum(h_abg_veg_gnd > 4) / n_denom)
        cover_6 = float(np.sum(h_abg_veg_gnd > 6) / n_denom)
    else:
        raise ValueError("cover_type must be 'all' or 'first'")

    return cover_h_pad, cover_2, cover_4, cover_6
