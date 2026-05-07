# `pad_metrics` Parameter Reference

Complete parameter list extracted from [`pad_metrics.R`](../../R/pad_metrics.R) and its Python equivalent.

---

## Group 1 — Vertical profile

| Parameter | Type | R default | Python default | Role |
|---|---|---|---|---|
| `z0` | float | `0` | `0.0` | Bottom height of the first stratum (m) |
| `dz` | float | `1` | `1.0` | Stratum thickness (m) |
| `nlayers` | int | `60` | `60` | Number of vertical strata. If `None`/`NULL`, derived from `max(Z)` |
| `ground_margin` | float | `0.1` | `0.1` | Margin above `z0` (m) excluded from the first stratum to avoid ground-reflection artefacts |

> These 4 parameters define the vertical column: from `z0 + ground_margin` up to `z0 + nlayers × dz`, split into strata of `dz` metres each.

---

## Group 2 — Radiative transfer model

| Parameter | Type | R default | Python default | Role |
|---|---|---|---|---|
| `G` | float | `0.5` | `0.5` | Ross-G leaf projection coefficient (0.5 = randomly oriented leaves) |
| `omega` | float | `0.77` | `0.77` | Clumping factor — `1` = homogeneous vegetation, `< 1` = clumped vegetation |
| `scanning_angle` | bool | `TRUE` | `True` | If `True`: cos θ computed from the aircraft trajectory. If `False`: cos θ = 1 (pulses assumed vertical) |

---

## Group 3 — Canopy cover estimation

| Parameter | Type | R default | Python default | Role |
|---|---|---|---|---|
| `cover_type` | str | `"all"` | `"all"` | `"all"` = all returns / `"first"` = first returns only |
| `height_cover` | float | `2` | `2.0` | Height threshold (m) above which canopy cover is estimated (`cover_h_pad`) |
| `use_cover` | bool | `TRUE` | `True` | If `True`: simple Beer-Lambert formula. If `False`: cover-normalised correction from Pimont et al. 2018 |

---

## Group 4 — Quality guards

| Parameter | Type | R default | Python default | Role |
|---|---|---|---|---|
| `limit_N_points` | int | `0` | `0` | Minimum number of points per pixel to compute PAD. Below this threshold → `NULL` / `None`. Set to `400` in operational use |
| `limit_flight_agl` | float | `800` | `800.0` | Minimum acceptable mean flight height above ground (m). Below this → likely trajectory error → `NULL` / `None` |
| `keep_N` | bool | `FALSE` | `False` | If `True`: include raw `Ni` and `N` vectors per stratum in the output |

---

## Group 5 — Temporal filtering

These parameters are present in **both R and Python**. They are applied independently per pixel so that the modal acquisition date is computed locally.

| Parameter | Type | R default | Python default | Role |
|---|---|---|---|---|
| `season_filter` | int vector | `1:12` | `list(range(1, 13))` | Calendar months to keep (e.g. `5:9` / `[5,6,7,8,9]` for summer) |
| `deviation_days` | float | `Inf` | `np.inf` | Maximum deviation in days around the local modal acquisition date. `Inf` = no filter |
| `gpstime_ref` | str | `"2011-09-14 01:46:40"` | `"2011-09-14 01:46:40"` | GPS time origin — LAS 1.4 standard (GPS epoch + 1×10⁹ s, UTC) |

---

## Group 6 — Reference grid *(Python / tile processing only)*

| Parameter | Type | Default | Role |
|---|---|---|---|
| `res` | float | — | Output raster pixel size in metres (e.g. `10.0`) |
| `start` | tuple[float, float] | `(0.0, 0.0)` | Reference origin for grid alignment — matches `start` in `lidR::pixel_metrics()` |

---

## Complete R function signature (as reference)

```r
pad_metrics(
  z0 = 0, dz = 1, nlayers = 60,
  ground_margin = 0.1,
  G = 0.5, omega = 0.77,
  scanning_angle = TRUE,
  cover_type = "all", height_cover = 2, use_cover = TRUE,
  limit_N_points = 0, limit_flight_agl = 800, keep_N = FALSE,
  season_filter = 1:12, deviation_days = Inf, gpstime_ref = "2011-09-14 01:46:40"
)
```