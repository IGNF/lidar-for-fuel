"""
Orchestrate PAD output rasters for one dalle: bin raw points into 10 m pixels
aligned to the CosiaFrance reference grid, compute 6 per-pixel products via
`pad_metrics_core`/`filter_gpstime`, and write 6 COG GeoTIFFs.

Per pixel group:
- `pad_metrics_core(dz=1, nlayers=60, keep_N=True)` feeds `pad_profile_1m`
  (`PAD_1_0`...`PAD_1_59`), `entering_rays` (`N_1_0`...`N_1_59`) and
  `intercept_ray` (`Ni_1_0`...`Ni_1_59`).
- `pad_metrics_core(dz=0.5, nlayers=4)` feeds `pad_sb_0.5m`
  (`PAD_0.5_0`...`PAD_0.5_1.5`).
- `Cover_2`/`Cover_4`/`Cover_6` from either call (identical regardless of
  dz/nlayers) feed `COVER`.
- An independent classification count on the `filter_gpstime`-filtered subset
  (same season/deviation_days/gpstime_ref as both `pad_metrics_core` calls)
  feeds `class_count`.

Pixels with zero raw points are skipped entirely (NoData on every band).
Pixels where `pad_metrics_core` returns `None` (quality guard) get NoData on
PAD/N/Ni/Cover bands only; `class_count` is computed independently and is
always populated when the pixel has raw points.
"""

import os
from typing import Iterable

import numpy as np
import rasterio

from lidar_for_fuel.pad_profile.filter_gpstime import filter_gpstime
from lidar_for_fuel.pad_profile.pad_metrics_core import pad_metrics_core
from lidar_for_fuel.pad_profile.pad_output_grid import (
    REFERENCE_CRS,
    bin_points_to_pixels,
    build_dalle_transform,
)

_DEFAULT_GPSTIME_REF = "2011-09-14 01:46:40"

CLASS_COUNT_VALUES = [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67]

_PAD_CAP = 5.0

_PAD_1M_BANDS = [f"PAD_1_{i}" for i in range(60)]
_N_1M_BANDS = [f"N_1_{i}" for i in range(60)]
_NI_1M_BANDS = [f"Ni_1_{i}" for i in range(60)]
_PAD_SB_05M_BANDS = ["PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"]
_COVER_BANDS = ["Cover_2", "Cover_4", "Cover_6"]
_CLASS_COUNT_BANDS = [f"Class_{c}" for c in CLASS_COUNT_VALUES] + ["Total"]

_FLOAT_NODATA = np.nan
_INT_NODATA = -1


def _write_cog(path, bands_array, band_names, transform, crs, nodata):
    """Write one COG GeoTIFF with named bands."""
    count, height, width = bands_array.shape
    with rasterio.open(
        path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=count,
        dtype=bands_array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(bands_array)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def write_pad_rasters(
    gpstime: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    h_abg: np.ndarray,
    z: np.ndarray,
    return_number: np.ndarray,
    classification: np.ndarray,
    x_sensor: np.ndarray,
    y_sensor: np.ndarray,
    z_sensor: np.ndarray,
    origin_x: float,
    origin_y: float,
    n_rows: int,
    n_cols: int,
    output_dir: str,
    dalle_name: str = "dalle",
    season_filter: Iterable[int] = range(1, 13),
    deviation_days: float = np.inf,
    gpstime_ref: str = _DEFAULT_GPSTIME_REF,
    **pad_metrics_kwargs,
) -> dict[str, str]:
    """Bin one dalle's points into pixels and write 6 PAD output COG rasters.

    Args:
        gpstime (np.ndarray): GPS time in seconds, one value per raw point.
        x (np.ndarray): Point easting (m, EPSG:2154).
        y (np.ndarray): Point northing (m, EPSG:2154).
        h_abg (np.ndarray): Normalized height above ground.
        z (np.ndarray): Raw, unnormalized elevation.
        return_number (np.ndarray): Return number within the pulse.
        classification (np.ndarray): LAS classification code.
        x_sensor (np.ndarray): Sensor easting at acquisition time.
        y_sensor (np.ndarray): Sensor northing.
        z_sensor (np.ndarray): Sensor altitude.
        origin_x (float): Dalle raster origin, top-left X (m, EPSG:2154).
        origin_y (float): Dalle raster origin, top-left Y (m, EPSG:2154).
        n_rows (int): Output raster height (pixels).
        n_cols (int): Output raster width (pixels).
        output_dir (str): Directory the 6 COG files are written into.
        dalle_name (str): Filename stem for the 6 output files. Default "dalle".
        season_filter (Iterable[int]): Calendar months (1-12) to keep, forwarded to
            both `pad_metrics_core` calls and to the `class_count` temporal filter.
        deviation_days (float): Max deviation in days around the local modal acquisition
            date, forwarded to both `pad_metrics_core` calls and to `class_count`.
        gpstime_ref (str): UTC reference datetime for gpstime=0, forwarded to both
            `pad_metrics_core` calls and to `class_count`.
        **pad_metrics_kwargs: Extra keyword arguments forwarded to both `pad_metrics_core`
            calls (e.g. `scanning_angle`, `use_cover`, `G`, `omega`, `limit_N_points`,
            `limit_flight_agl`). Must not include `dz`, `nlayers`, or `keep_N` (set
            internally per product).

    Returns:
        dict[str, str]: Mapping of product name (`pad_profile_1m`, `pad_sb_0.5m`,
            `entering_rays`, `intercept_ray`, `class_count`, `COVER`) to the written
            file path.

    Raises:
        ValueError: If the dalle origin is not aligned with the CosiaFrance reference
            grid, or `n_rows`/`n_cols` are not positive (propagated from
            `build_dalle_transform`); if the per-point arrays don't all have the same
            length; or if `pad_metrics_kwargs` includes a reserved key (`dz`, `nlayers`,
            `keep_N`). No files are written in any of these cases.
    """
    point_arrays = {
        "gpstime": gpstime,
        "x": x,
        "y": y,
        "h_abg": h_abg,
        "z": z,
        "return_number": return_number,
        "classification": classification,
        "x_sensor": x_sensor,
        "y_sensor": y_sensor,
        "z_sensor": z_sensor,
    }
    lengths = {name: len(arr) for name, arr in point_arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All point arrays must have the same length, got: {lengths}")

    reserved_kwargs = {"dz", "nlayers", "keep_N"} & pad_metrics_kwargs.keys()
    if reserved_kwargs:
        raise ValueError(f"pad_metrics_kwargs must not include {sorted(reserved_kwargs)} (set internally per product)")

    transform = build_dalle_transform(origin_x, origin_y, n_rows=n_rows, n_cols=n_cols)

    pad_1m = np.full((60, n_rows, n_cols), np.nan, dtype=np.float32)
    n_1m = np.full((60, n_rows, n_cols), _INT_NODATA, dtype=np.int32)
    ni_1m = np.full((60, n_rows, n_cols), _INT_NODATA, dtype=np.int32)
    pad_sb_05m = np.full((4, n_rows, n_cols), np.nan, dtype=np.float32)
    cover = np.full((3, n_rows, n_cols), np.nan, dtype=np.float32)
    class_count = np.full((len(_CLASS_COUNT_BANDS), n_rows, n_cols), _INT_NODATA, dtype=np.int32)

    rows, cols = bin_points_to_pixels(x, y, transform)
    in_bounds = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    flat_pixel = np.where(in_bounds, rows * n_cols + cols, -1)

    occupied_pixels = np.unique(flat_pixel[in_bounds])

    for flat_idx in occupied_pixels:
        mask = flat_pixel == flat_idx
        row = int(flat_idx // n_cols)
        col = int(flat_idx % n_cols)

        point_args = dict(
            gpstime=gpstime[mask],
            x=x[mask],
            y=y[mask],
            h_abg=h_abg[mask],
            z=z[mask],
            return_number=return_number[mask],
            classification=classification[mask],
            x_sensor=x_sensor[mask],
            y_sensor=y_sensor[mask],
            z_sensor=z_sensor[mask],
        )

        result_1m = pad_metrics_core(
            **point_args,
            dz=1.0,
            nlayers=60,
            keep_N=True,
            season_filter=season_filter,
            deviation_days=deviation_days,
            gpstime_ref=gpstime_ref,
            **pad_metrics_kwargs,
        )
        if result_1m is not None:
            pad_1m[:, row, col] = [min(result_1m[name], _PAD_CAP) for name in _PAD_1M_BANDS]
            n_1m[:, row, col] = [result_1m[name] for name in _N_1M_BANDS]
            ni_1m[:, row, col] = [result_1m[name] for name in _NI_1M_BANDS]
            # Cover_2/4/6 don't depend on dz/nlayers, and both calls share the same
            # temporal filter and quality-guard kwargs, so result_1m's Cover_* is
            # equivalent to result_sb's and the two calls always agree on guard outcome.
            cover[:, row, col] = [result_1m[name] for name in ("Cover_2", "Cover_4", "Cover_6")]

        result_sb = pad_metrics_core(
            **point_args,
            dz=0.5,
            nlayers=4,
            keep_N=False,
            season_filter=season_filter,
            deviation_days=deviation_days,
            gpstime_ref=gpstime_ref,
            **pad_metrics_kwargs,
        )
        if result_sb is not None:
            pad_sb_05m[:, row, col] = [min(result_sb[name], _PAD_CAP) for name in _PAD_SB_05M_BANDS]

        temporal_valid = filter_gpstime(
            gpstime[mask], months=season_filter, deviation_days=deviation_days, gpstime_ref=gpstime_ref
        )
        pixel_classification = classification[mask][temporal_valid]
        counts = [int(np.sum(pixel_classification == c)) for c in CLASS_COUNT_VALUES]
        counts.append(len(pixel_classification))
        class_count[:, row, col] = counts

    os.makedirs(output_dir, exist_ok=True)

    products = {
        "pad_profile_1m": (pad_1m, _PAD_1M_BANDS, _FLOAT_NODATA),
        "pad_sb_0.5m": (pad_sb_05m, _PAD_SB_05M_BANDS, _FLOAT_NODATA),
        "entering_rays": (n_1m, _N_1M_BANDS, _INT_NODATA),
        "intercept_ray": (ni_1m, _NI_1M_BANDS, _INT_NODATA),
        "class_count": (class_count, _CLASS_COUNT_BANDS, _INT_NODATA),
        "COVER": (cover, _COVER_BANDS, _FLOAT_NODATA),
    }

    paths: dict[str, str] = {}
    for product_name, (bands_array, band_names, nodata) in products.items():
        path = os.path.join(output_dir, f"{dalle_name}_{product_name}.tif")
        _write_cog(path, bands_array, band_names, transform, REFERENCE_CRS, nodata)
        paths[product_name] = path

    return paths
