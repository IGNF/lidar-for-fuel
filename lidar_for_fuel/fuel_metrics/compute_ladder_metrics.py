"""
Ladder fuel metrics of a vertical profile:
for each ladder fuel threshold, bottom and top of the ladder fuel, its load, the equivalent
surface fuel load (Perrakis 2025), the profile type, the canopy fuel load above the ladder
and the fuel strata gap.

"""

from collections.abc import Sequence

import numpy as np

# Bottom of the ladder fuel search (m). Hard-coded `zval > 1`, kept on purpose to
# characterise the low stratum: independent of the canopy height threshold.
LADDER_MIN_HEIGHT = 1.0


def compute_ladder_metrics(
    cbd: np.ndarray,
    h_center: np.ndarray,
    dz: float,
    cbh2: float,
    canopy_height_threshold: float,
    ladder_fuel_bd_thresholds: Sequence[float],
    m_perrakis: float,
) -> dict[str, float]:
    """Compute the ladder fuel metrics of one vertical profile.

    For each threshold t of `sorted(set(ladder_fuel_bd_thresholds))` (h is the stratum
    centre, CHT the canopy height threshold, M `m_perrakis`):

    - ladder strata: CBD >= t and `LADDER_MIN_HEIGHT` < h <= CBH2. Only the first continuous
      block counts: `LadderBot` = bottom of its first stratum, `LadderTop` = top of its last.
    - `ProfType` = 3 if `LadderBot` <= CHT (ladder continuous from the surface), 4 otherwise
      (suspended ladder). As the search starts above 1 m, `LadderBot` > 1 − dz / 2: a ladder
      is always of type 4 when CHT <= 1 − dz / 2, e.g. CHT = 0.5 m with 1 m strata.
    - `LadderLoad` = Σ CBD × dz of the strata with CHT < h <= `LadderTop`;
      `SFLeq0` = M × LadderLoad × CBH2 / (CBH2 − LadderTop / 2) and
      `SFLeq_intot` = M × Σ (CBD × CBH2 / (CBH2 − h)) × dz on the same strata.
    - Without ladder strata: `LadderBot` = `LadderTop` = CHT, loads 0, `ProfType` = 1 if
      CBH2 fell back on CHT (neither CBH2 nor CBHlow found), 2 otherwise (canopy without
      ladder).
    - In all cases: `CFL` = Σ CBD × dz above `LadderTop`, `FSG` = CBH2 − LadderTop + dz / 2.

    Args:
        cbd (np.ndarray): CBD profile (kg/m³) recomputed with CBH2, see `compute_canopy_metrics`.
        h_center (np.ndarray): Height above ground of the centre of each stratum (m), ascending.
        dz (float): Thickness of the strata (m).
        cbh2 (float): Continuous crown base height (m), see `compute_canopy_metrics`. When it
            fell back on CHT, it must be exactly `canopy_height_threshold` (as returned by
            `compute_canopy_metrics`): a recomputed or float32 value gives `ProfType` 2, not 1.
        canopy_height_threshold (float): Height (m) separating surface and canopy fuel (CHT).
        ladder_fuel_bd_thresholds (Sequence[float]): CBD thresholds of the ladder fuel (kg/m³).
        m_perrakis (float): M factor of the equivalent surface fuel load (Perrakis 2025).

    Returns:
        dict[str, float]: For each threshold, in ascending order, `LadderBot_s`, `LadderTop_s`,
            `LadderLoad_s`, `SFLeq0_s`, `SFLeq_intot_s`, `ProfType_s`, `CFL_s` and `FSG_s`,
            where s is the R band suffix `sprintf("%0.3f")` with `.` written `p` (0.02 -> 0p020).
    """
    cbd = np.asarray(cbd, dtype=float)
    h = np.asarray(h_center, dtype=float)
    cht = float(canopy_height_threshold)
    cbh2 = np.float64(cbh2)  # numpy division: inf / NaN as R instead of ZeroDivisionError

    metrics = {}
    for threshold in sorted(set(ladder_fuel_bd_thresholds)):
        suffix = ladder_suffix(threshold)
        values = _ladder_for_threshold(cbd, h, dz, cbh2, cht, threshold, m_perrakis)
        for prefix, value in values.items():
            metrics[f"{prefix}_{suffix}"] = value

    return metrics


def ladder_suffix(threshold: float) -> str:
    """R band suffix of a ladder fuel threshold: `sprintf("%0.3f")` with `.` written `p` (0.02 -> 0p020)."""
    return f"{threshold:0.3f}".replace(".", "p")


def _ladder_for_threshold(
    cbd: np.ndarray,
    h: np.ndarray,
    dz: float,
    cbh2: np.float64,
    cht: float,
    threshold: float,
    m_perrakis: float,
) -> dict[str, float]:
    """Ladder fuel metrics for one threshold (R `.ladder_properties_for_threshold`).

    Returns:
        dict[str, float]: `LadderBot`, `LadderTop`, `LadderLoad`, `SFLeq0`, `SFLeq_intot`,
            `ProfType`, `CFL` and `FSG`, in this order (the R order).
    """
    ladder_bot = ladder_top = cht
    load = sfl_eq0 = sfl_eq_int = 0.0
    # Written as `CBH2 == CanopyHeightThreshold`. CBH2 is CHT itself when it fell back on it;
    # otherwise it is the CBH2 or the CBHlow found, the bottom of a stratum above the CanBDmin
    # stratum, whose centre is above CHT: strictly above CHT + dz / 2. The exact comparison is
    # then safe whatever dz
    prof_type = 1 if cbh2 == cht else 2

    above = (cbd >= threshold) & (h > LADDER_MIN_HEIGHT) & (h <= cbh2)
    in_ladder = np.flatnonzero(above)
    if in_ladder.size:
        first = last = in_ladder[0]
        # R: while (j <= n && isTRUE(above[j])) j <- j + 1  -> first continuous block only
        while last + 1 < above.size and above[last + 1]:
            last += 1
        ladder_bot = float(h[first]) - 0.5 * dz
        ladder_top = float(h[last]) + 0.5 * dz

        prof_type = 3 if ladder_bot <= cht else 4

        k = (h > cht) & (h <= ladder_top)
        load = float(np.sum(cbd[k])) * dz
        with np.errstate(divide="ignore", invalid="ignore"):
            sfl_eq0 = float(m_perrakis * load * cbh2 / (cbh2 - 0.5 * ladder_top))
            sfl_eq_int = float(m_perrakis * np.sum(cbd[k] * cbh2 / (cbh2 - h[k])) * dz)

    return {
        "LadderBot": ladder_bot,
        "LadderTop": ladder_top,
        "LadderLoad": load,
        "SFLeq0": sfl_eq0,
        "SFLeq_intot": sfl_eq_int,
        "ProfType": float(prof_type),
        "CFL": float(np.sum(cbd[h > ladder_top])) * dz,
        "FSG": float(cbh2 - ladder_top + 0.5 * dz),
    }
