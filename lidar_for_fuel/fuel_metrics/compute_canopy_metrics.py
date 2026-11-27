"""
Canopy metrics of a vertical profile :
- CBD profile with the canopy height threshold as the canopy / shrub limit;
- canopy top height (CTH) for each canopy top threshold;
- crown base heights: CBH and CBH2 in the upper canopy, CBHlow in the lower canopy, and the
  CBD at these heights;
- CBD profile recomputed with CBH2 as the canopy / shrub limit;
- canopy dominant height (CDH) and mean canopy bulk density (CBDcan).

"""

from collections.abc import Sequence

import numpy as np

from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import compute_cbd


def compute_canopy_metrics(
    pad: np.ndarray,
    h_center: np.ndarray,
    dz: float,
    fma_can: float,
    fma_shrub: float,
    bd_max: float,
    canopy_height_threshold: float,
    canopy_bd_thresh_frac: float,
    canopy_top_bd_thresholds: Sequence[float],
    cbh_threshold_ratio: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute the canopy metrics of one vertical profile.

    Steps (h is the stratum centre, CHT the canopy height threshold, CBD the bulk density):
    1. CBD with the canopy FMA above CHT (`compute_cbd`).
    2. `CTH_t` = highest h with CBD >= t, for each threshold t; CTH = the one of the largest t.
    3. Crown base, only if CTH exists:

       - **3A, upper canopy**, CHT and CTH × ratio < h <= CTH: `CanBDmax` and the first stratum
         reaching it; `CanBDmin` below it (h > CHT). If CanBDmin < CanBDmax, threshold
         `S = CanBDmin × (1 − frac) + frac × CanBDmax`, `CBH` = bottom of the lowest stratum
         >= S between them. Only if CBH is found, `CBH2` = bottom of the stratum above the
         first one < S met going down from the max to just above the min.
       - **3B, lower canopy**, CHT < h <= CTH × ratio: same search for `CanBDlowmax`,
         `CanBDlowmin` and `CBHlow`, without the walk down to a continuous base.
       - **3C**: CBH2 falls back on CBHlow, then on CHT.

       `CanBD*` are given even when no crown base is found. `CBD_cbh`, `CBD_cbh2` and
       `CBD_cbhlow` are the CBD of step 1 (not the one recomputed with CBH2) of the stratum
       whose bottom is CBH, CBH2 or CBHlow, as R: `CBD_cbh2` only for a CBH2
       found in 3A, NaN when CBH2 falls back on CBHlow or CHT.
    4. CBD recomputed from the PAD with the canopy FMA above CBH2.
    5. `CDH` = highest h with CBD >= 0.1 × CanBDmax (CTH if none), then `min(CDH, CTH)`.
    6. `CBDcan` = mean CBD of the strata with CBH2 <= h <= min(CDH, CTH), 0 if none.


    Args:
        pad (np.ndarray): Prepared 1-D PAD profile (m²/m³), see `prepare_pad_profile`.
        h_center (np.ndarray): Height above ground of the centre of each stratum (m),
            ascending, i.e. `h_bottom + dz / 2`.
        dz (float): Thickness of the strata (m).
        fma_can (float): Fuel mass area of the canopy fuel (kg/m²) for this pixel.
        fma_shrub (float): Fuel mass area of the shrub fuel (kg/m²) for this pixel.
        bd_max (float): Maximum CBD value (kg/m³).
        canopy_height_threshold (float): Height (m) separating surface and canopy fuel (CHT).
        canopy_bd_thresh_frac (float): Fraction between CanBDmin and CanBDmax giving the
            crown base threshold S.
        canopy_top_bd_thresholds (Sequence[float]): CBD thresholds of the canopy top (kg/m³).
        cbh_threshold_ratio (float): Fraction of CTH separating the upper (3A) and lower (3B)
            crown base searches.

    Returns:
        tuple[dict[str, float], np.ndarray]:
            - the metrics, keyed by their R band name: `CTH_{t:0.4f}` (one per threshold,
              in the configured order, `.` written `p`), `CDH`, `CBH`, `CBH2`, `CBHlow`,
              `CanBDmax`, `CanBDmin`, `CanBDlowmax`, `CanBDlowmin`, `CBDcan`, `CBD_cbh`,
              `CBD_cbh2`, `CBD_cbhlow`;
            - the CBD profile recomputed with CBH2 (step 4), used by the ladder fuel metrics.
    """
    pad = np.asarray(pad, dtype=float)
    h = np.asarray(h_center, dtype=float)
    cht = float(canopy_height_threshold)  # CBH2 falls back on it: always return a float

    # 1. CBD with CHT as the canopy / shrub limit
    cbd = compute_cbd(pad, h, cht, fma_can, fma_shrub, bd_max)

    # 2. Canopy top heights; the reference CTH is the one of the largest threshold
    cth_by_name = {}
    for threshold in canopy_top_bd_thresholds:
        cth_by_name[_cth_name(threshold)] = _highest(h, cbd >= threshold)
    cth = cth_by_name[_cth_name(max(canopy_top_bd_thresholds))]

    # 3. Crown base heights. With a NaN CTH, both search ranges are empty (`h <= nan` is
    # False, as `h <= NA` in R); the guard only makes this case explicit.
    upper = lower = (np.nan, np.nan, np.nan, np.nan)
    if not np.isnan(cth):
        # 3A. Upper canopy
        upper = _crown_base(
            cbd,
            h,
            dz,
            (h > max(cht, cth * cbh_threshold_ratio)) & (h <= cth),
            cht,
            canopy_bd_thresh_frac,
            upper_canopy=True,
        )
        # 3B. Lower canopy
        lower = _crown_base(
            cbd,
            h,
            dz,
            (h > cht) & (h <= cth * cbh_threshold_ratio),
            cht,
            canopy_bd_thresh_frac,
            upper_canopy=False,
        )
    can_bd_max, can_bd_min, cbh, cbh2 = upper
    can_bd_low_max, can_bd_low_min, cbh_low, _ = lower  # no continuous base (CBH2) in 3B

    # CBD at the crown base heights, in the step 1 CBD and before the CBH2 fallbacks (as R)
    cbd_cbh = _cbd_at_bottom(cbd, h, cbh, dz)
    cbd_cbh2 = _cbd_at_bottom(cbd, h, cbh2, dz)
    cbd_cbh_low = _cbd_at_bottom(cbd, h, cbh_low, dz)

    # 3C. Fallbacks of CBH2
    if np.isnan(cbh2):
        cbh2 = cbh_low
    if np.isnan(cbh2):
        cbh2 = cht

    # 4. CBD recomputed from the PAD, with CBH2 as the canopy / shrub limit
    cbd = compute_cbd(pad, h, cbh2, fma_can, fma_shrub, bd_max)

    # 5. Canopy dominant height, capped by CTH (NaN if CTH is NaN, as R `min(x, NA)`)
    cdh = _highest(h, cbd >= 0.1 * can_bd_max) if not np.isnan(can_bd_max) else np.nan
    if np.isnan(cdh):
        cdh = cth
    cdh = float(np.minimum(cdh, cth))

    # 6. Mean canopy bulk density between CBH2 and min(CDH, CTH), i.e. CDH, already capped.
    # np.mean may differ from R `mean` (long double) by 1 ulp in float64, invisible in float32.
    if np.isnan(cdh):
        # R `any(h >= CBH2 & h <= NA)`: NA if a stratum is above CBH2, FALSE (-> 0) otherwise
        cbd_can = np.nan if (h >= cbh2).any() else 0.0
    else:
        in_canopy = (h >= cbh2) & (h <= cdh)
        cbd_can = float(np.mean(cbd[in_canopy])) if in_canopy.any() else 0.0

    metrics = {
        **cth_by_name,
        "CDH": cdh,
        "CBH": cbh,
        "CBH2": cbh2,
        "CBHlow": cbh_low,
        "CanBDmax": can_bd_max,
        "CanBDmin": can_bd_min,
        "CanBDlowmax": can_bd_low_max,
        "CanBDlowmin": can_bd_low_min,
        "CBDcan": cbd_can,
        "CBD_cbh": cbd_cbh,
        "CBD_cbh2": cbd_cbh2,
        "CBD_cbhlow": cbd_cbh_low,
    }

    return metrics, cbd


def _cth_name(threshold: float) -> str:
    """Band name of a canopy top height: `sprintf("%0.4f")` with `.` written `p` (0.01 -> CTH_0p0100)."""
    return "CTH_" + f"{threshold:0.4f}".replace(".", "p")


def _highest(h: np.ndarray, mask: np.ndarray) -> float:
    """Highest height selected by `mask`, NaN if none (R `if (any(m)) max(h[m]) else NA`)."""
    return float(h[mask].max()) if mask.any() else np.nan


def _cbd_at_bottom(cbd: np.ndarray, h: np.ndarray, h_bottom: float, dz: float) -> float:
    """CBD of the first stratum whose bottom is `h_bottom`, NaN if none or if `h_bottom` is NaN.

    `CBDval[zval == CBH + (0.5 * dz)][1]`, with the same exact comparison of the centre.
    """
    match = np.flatnonzero(h == h_bottom + 0.5 * dz)
    return float(cbd[match[0]]) if match.size else np.nan


def _crown_base(
    cbd: np.ndarray,
    h: np.ndarray,
    dz: float,
    search: np.ndarray,
    cht: float,
    frac: float,
    upper_canopy: bool,
) -> tuple[float, float, float, float]:
    """Search a crown base in a height range of the CBD profile (steps 3A and 3B).

    - `cbd_max` = max CBD of the range; `i_max` = first stratum of the range reaching it.
    - `cbd_min` = min CBD between CHT and `i_max` (h > CHT, h < h[i_max]); `i_min` = first
      stratum reaching it.
    - Threshold `S = cbd_min × (1 − frac) + frac × cbd_max`, then
      `cbh` = bottom of the lowest stratum with CBD >= S and max(CHT, h[i_min]) < h <= h[i_max].
    - Upper canopy (3A) only, as in R:
      - the crown base is only searched if `cbd_min < cbd_max`. R omits this test in 3B,
        where it always holds: the strata below `i_max` are then all in the search range,
        below the first maximum;
      - `cbh2` = going down from `i_max` to `i_min + 1`, bottom of the first stratum k whose
        lower neighbour has CBD < S, only computed when `cbh` is found. It can stay NaN even
        then: with `frac = 0`, S = `cbd_min` and no CBD is < S (CBH2 then falls back on
        CBHlow or CHT, possibly below CBH, as in R).

    Args:
        cbd (np.ndarray): CBD profile (kg/m³), no NaN.
        h (np.ndarray): Height of the stratum centres (m), ascending.
        dz (float): Thickness of the strata (m).
        search (np.ndarray): Boolean mask of the strata where the max is searched.
        cht (float): Canopy height threshold (m).
        frac (float): Fraction between `cbd_min` and `cbd_max` giving the threshold S.
        upper_canopy (bool): Upper canopy search (3A): `min < max` test and CBH2.

    Returns:
        tuple[float, float, float, float]: `cbd_max`, `cbd_min`, `cbh`, `cbh2` (always NaN
            in the lower canopy), NaN when not found.
    """
    cbd_max = cbd_min = cbh = cbh2 = np.nan

    in_range = np.flatnonzero(search)
    if in_range.size == 0:
        return cbd_max, cbd_min, cbh, cbh2
    cbd_max = float(cbd[in_range].max())
    i_max = in_range[np.flatnonzero(cbd[in_range] == cbd_max)[0]]

    below = np.flatnonzero((h > cht) & (h < h[i_max]))
    if below.size == 0:
        return cbd_max, cbd_min, cbh, cbh2
    cbd_min = float(cbd[below].min())
    if upper_canopy and not cbd_min < cbd_max:
        return cbd_max, cbd_min, cbh, cbh2
    i_min = below[np.flatnonzero(cbd[below] == cbd_min)[0]]

    threshold = cbd_min * (1 - frac) + frac * cbd_max
    above = (cbd >= threshold) & (h > max(cht, h[i_min])) & (h <= h[i_max])
    if not above.any():
        return cbd_max, cbd_min, cbh, cbh2
    cbh = float(h[above].min()) - 0.5 * dz

    if upper_canopy:
        for k in range(i_max, i_min, -1):
            if cbd[k - 1] < threshold:
                cbh2 = float(h[k]) - 0.5 * dz
                break

    return cbd_max, cbd_min, cbh, cbh2
