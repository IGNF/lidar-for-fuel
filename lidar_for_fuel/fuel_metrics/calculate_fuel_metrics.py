"""
Fuel metrics of a vertical PAD profile, ported from R `ffuelmetrics2`: profile preparation,
canopy metrics, fuel loads, ladder fuel metrics, default ladder metrics and assembly of the
outputs in the R order and with the R band names (45 with the default configuration).

This is a port of R that keeps its known defects, except the assumed divergence agreed with
INRAE. `fuel_metrics_core` lists the divergence and the defects kept.
"""

from collections.abc import Sequence

import numpy as np

from lidar_for_fuel.fuel_metrics.compute_canopy_metrics import compute_canopy_metrics
from lidar_for_fuel.fuel_metrics.compute_ladder_metrics import (
    compute_ladder_metrics,
    ladder_suffix,
)
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import (
    prepare_pad_profile,
    select_default_ladder_threshold,
)


def fuel_metrics_core(
    pad: np.ndarray,
    h_bottom: np.ndarray,
    dz: float,
    fma_can: float,
    fma_shrub: float,
    smooth_pad: bool,
    ladder_fuel_bd_thresholds: Sequence[float],
    canopy_height_threshold: float,
    canopy_bd_thresh_frac: float,
    canopy_top_bd_thresholds: Sequence[float],
    cbh_threshold_ratio: float,
    bd_max: float,
    m_perrakis: float,
) -> dict[str, float]:
    """Compute all the fuel metrics of one vertical PAD profile, as R `ffuelmetrics2` does.

    The parameters after `dz` are named as the `fuel_metrics` section of the configuration.

    Steps (h is the stratum centre `h_bottom + dz / 2`, CHT the canopy height threshold):

    1. PAD prepared (`prepare_pad_profile`), then canopy metrics and CBD recomputed with CBH2
       (`compute_canopy_metrics`, steps 1 to 6).
    2. Fuel loads on that CBD (step 7): `CLFL` = Σ CBD × dz above CHT; `FL_k1`, `FL_k2` =
       CBD × dz of the first and second strata, whatever their height (as R, the first
       stratum is assumed to start at the ground); `FL_k2` is NaN with a single stratum.
    3. Ladder fuel metrics for each threshold (`compute_ladder_metrics`, step 8).
    4. Default `LFL`, `CFL`, `FSG` and `ProfType`: those of the threshold closest to the
       median (`select_default_ladder_threshold`, step 9).

    Assumed divergence from R:

    - a profile with no data (all PAD values NaN) gives NaN for all the metrics (D2). R gives
      the metrics of a null profile (zeros, `CBH2` = CHT, `ProfType` = 1, A5), which would
      show areas without measurement as areas without fuel.

    Otherwise, strict parity with R, including its defects:

    - NaN, inf and negative PAD values of a partly valid profile are set to 0, with a debug
      log only; their count is not returned (A6, counted in story 3.6);
    - `CBD_cbh`, `CBD_cbh2` and `CBD_cbhlow` are always NaN (A1).

    The parameters are used as given. The R rasters (`ffuelmetrics2Rast`) pass on neither
    `smooth_pad` nor `CBHThresholdRatio` and always use `smooth_pad = False` and a ratio of
    1/3 (A3): to reproduce them, the caller must pass these values, whatever the configured
    ones (stories 3.6 and 3.7). With the PAD of the `pad_profile` stage, capped at 5 unlike
    the R one, the results are also guaranteed to be the same as R if
    `5 × min(fma_can, fma_shrub) >= bd_max` (the CBD is capped at `bd_max` anyway).

    Args:
        pad (np.ndarray): Raw 1-D PAD profile of the pixel (m²/m³), one value per stratum,
            ordered by height; NaN for no data.
        h_bottom (np.ndarray): Height above ground of the bottom of each stratum (m), ascending,
            see `read_pad_heights`.
        dz (float): Thickness of the strata (m).
        fma_can (float): Fuel mass area of the canopy fuel (kg/m²) for this pixel.
        fma_shrub (float): Fuel mass area of the shrub fuel (kg/m²) for this pixel.
        smooth_pad (bool): Whether to smooth the PAD profile over 3 strata.
        ladder_fuel_bd_thresholds (Sequence[float]): CBD thresholds of the ladder fuel (kg/m³).
        canopy_height_threshold (float): Height (m) separating surface and canopy fuel (CHT).
        canopy_bd_thresh_frac (float): Fraction between CanBDmin and CanBDmax giving the
            crown base threshold.
        canopy_top_bd_thresholds (Sequence[float]): CBD thresholds of the canopy top (kg/m³).
        cbh_threshold_ratio (float): Fraction of CTH separating the upper and lower crown
            base searches.
        bd_max (float): Maximum CBD value (kg/m³).
        m_perrakis (float): M factor of the equivalent surface fuel load (Perrakis 2025).

    Returns:
        dict[str, float]: The metrics, keyed by their R band name, in the R order:
            `CLFL`, `FL_k1`, `FL_k2`, the 8 ladder metrics of each ladder threshold (ascending),
            `CTH_*` (one per canopy top threshold), `CDH`, `CBH`, `CBH2`, `FSG`, `CBHlow`,
            `CanBDmax`, `CanBDmin`, `CanBDlowmax`, `CanBDlowmin`, `CBDcan`, `CBD_cbh`,
            `CBD_cbh2`, `CBD_cbhlow`, `LFL`, `CFL`, `ProfType`
            (19 + 8 × n_distinct_ladder_thresholds + n_canopy_top_thresholds), all NaN when
            the profile has no data.
    """
    no_data = bool(np.isnan(pad).all())
    profile, _ = prepare_pad_profile(pad, smooth_pad)
    h_center = np.asarray(h_bottom, dtype=float) + dz / 2
    cht = float(canopy_height_threshold)

    canopy, cbd = compute_canopy_metrics(
        profile,
        h_center,
        dz,
        fma_can=fma_can,
        fma_shrub=fma_shrub,
        bd_max=bd_max,
        canopy_height_threshold=cht,
        canopy_bd_thresh_frac=canopy_bd_thresh_frac,
        canopy_top_bd_thresholds=canopy_top_bd_thresholds,
        cbh_threshold_ratio=cbh_threshold_ratio,
    )

    # 7. Fuel loads, on the CBD recomputed with CBH2
    clfl = float(np.sum(cbd[h_center > cht])) * dz
    fl_k1 = float(cbd[0]) * dz
    fl_k2 = float(cbd[1]) * dz if cbd.size > 1 else np.nan  # R: CBDval[2] is NA

    # 8. Ladder fuel metrics, for each ladder threshold
    ladders = compute_ladder_metrics(
        cbd,
        h_center,
        dz,
        cbh2=canopy["CBH2"],
        canopy_height_threshold=cht,
        ladder_fuel_bd_thresholds=ladder_fuel_bd_thresholds,
        m_perrakis=m_perrakis,
    )

    # 9. Default ladder metrics: those of the threshold closest to the median
    default = ladder_suffix(select_default_ladder_threshold(ladder_fuel_bd_thresholds))

    cth = {name: value for name, value in canopy.items() if name.startswith("CTH_")}

    metrics = {
        "CLFL": clfl,
        "FL_k1": fl_k1,
        "FL_k2": fl_k2,
        **ladders,
        **cth,
        "CDH": canopy["CDH"],
        "CBH": canopy["CBH"],
        "CBH2": canopy["CBH2"],
        "FSG": ladders[f"FSG_{default}"],
        "CBHlow": canopy["CBHlow"],
        "CanBDmax": canopy["CanBDmax"],
        "CanBDmin": canopy["CanBDmin"],
        "CanBDlowmax": canopy["CanBDlowmax"],
        "CanBDlowmin": canopy["CanBDlowmin"],
        "CBDcan": canopy["CBDcan"],
        # Always NaN in R: it looks for `zval == CBH`, a stratum centre equal to a bottom (A1)
        "CBD_cbh": np.nan,
        "CBD_cbh2": np.nan,
        "CBD_cbhlow": np.nan,
        "LFL": ladders[f"LadderLoad_{default}"],
        "CFL": ladders[f"CFL_{default}"],
        "ProfType": ladders[f"ProfType_{default}"],
    }

    # Metrics computed on the null profile first, so that the keys stay those of the contract
    return dict.fromkeys(metrics, np.nan) if no_data else metrics
