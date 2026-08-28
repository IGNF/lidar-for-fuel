"""
Assemble a grid of per-pixel PAD metrics (calculate_pad_profile.pad_metrics_core output)
into georeferenced multi-band GeoTIFF rasters.
"""

import logging
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lidar_for_fuel.pad_profile.calculate_pad_profile import _format_num

logger = logging.getLogger(__name__)

_DAY_SECONDS = 86400
_FLOAT_NODATA = np.nan
_INT_NODATA = -9999


def _strata_band_names(prefix: str, dz: float, nlayers: int, z0: float) -> list[str]:
    """Band names for one stratified raster, e.g. PAD_1_0, PAD_1_1, ... PAD_1_{nlayers-1}."""
    min_layer = z0 + np.arange(nlayers) * dz
    return [f"{prefix}_{_format_num(dz)}_{_format_num(layer)}" for layer in min_layer]


def _band_array(pixel_results: np.ndarray, band_name: str, nodata: float) -> np.ndarray:
    """Extract one band as a 2D array from the grid of per-pixel result dicts."""
    values = [nodata if pixel is None else pixel.get(band_name, nodata) for pixel in pixel_results.ravel()]
    return np.array(values, dtype=np.float64).reshape(pixel_results.shape)


def _days_since_epoch_band(pixel_results: np.ndarray, band_name: str) -> np.ndarray:
    """Extract a Date_* band (Unix seconds) as whole days since 1970-01-01."""
    seconds = _band_array(pixel_results, band_name, nodata=np.nan)
    days = np.full(seconds.shape, _INT_NODATA, dtype=np.float64)
    valid = ~np.isnan(seconds)
    days[valid] = seconds[valid] // _DAY_SECONDS
    return days


def write_multiband_raster(
    output_path: str,
    bands: dict[str, np.ndarray],
    raster_origin: tuple[float, float],
    pixel_size: float,
    crs: str,
    dtype: str,
    nodata: float,
) -> None:
    """Write a single georeferenced multi-band GeoTIFF.

    Args:
        output_path (str): Destination .tif path.
        bands (dict[str, np.ndarray]): Band name -> 2D array (height, width), in
            write order. All arrays must share the same shape.
        raster_origin (tuple[float, float]): (x_min, y_max) of the raster's
            top-left pixel corner. Written on the tile's own native origin
            (CosiaFrance grid alignment is not applied here).
        pixel_size (float): Pixel size (m), used for both axes.
        crs (str): Spatial reference (e.g. "EPSG:2154").
        dtype (str): Output data type (e.g. "float32", "int32").
        nodata (float): Nodata value written to the raster metadata.

    Raises:
        ValueError: If `bands` is empty or its arrays don't all share one shape.
    """
    if not bands:
        raise ValueError("bands must contain at least one band")
    shapes = {array.shape for array in bands.values()}
    if len(shapes) > 1:
        raise ValueError(f"All bands must share the same shape, got {sorted(shapes)}")
    height, width = shapes.pop()

    transform = from_origin(raster_origin[0], raster_origin[1], pixel_size, pixel_size)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(bands),
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for i, (name, array) in enumerate(bands.items(), start=1):
            dst.write(array.astype(dtype), i)
            dst.set_band_description(i, name)

    logger.info("Wrote %d-band raster to %s", len(bands), output_path)


def export_pad_rasters(
    pixel_results: np.ndarray,
    raster_origin: tuple[float, float],
    pixel_size: float,
    crs: str,
    output_dir: str,
    tilename: str,
    z0: float,
    dz: float,
    nlayers: int,
    dz_low: float,
    nlayers_low: int,
    keep_classes: list,
) -> dict[str, str]:
    """Assemble a grid of per-pixel PAD metrics into the 7 output GeoTIFF rasters.

    Args:
        pixel_results (np.ndarray): Grid of per-pixel `pad_metrics_core` outputs,
            shape (height, width), dtype=object. Each cell is either the metrics
            dict for that pixel, or `None` where the quality guard rejected it
            (written as nodata on every band).
        raster_origin (tuple[float, float]): (x_min, y_max) of the raster's
            top-left pixel corner -- the tile's own native origin.
        pixel_size (float): Raster pixel size (m). 10 m per the PAD product spec.
        crs (str): Spatial reference (e.g. "EPSG:2154").
        output_dir (str): Directory the 7 GeoTIFFs are written into.
        tilename (str): Tile identifier used to prefix each output filename.
        z0 (float): Bottom height of the first stratum (m), as passed to `pad_metrics_core`.
        dz (float): Stratum thickness of the main PAD profile (m).
        nlayers (int): Number of strata of the main PAD profile. Must not be `None` --
            every pixel needs the same fixed band count for the raster to be well-formed.
        dz_low (float): Stratum thickness of the low-strata PAD band (m).
        nlayers_low (int): Number of strata of the low-strata PAD band. Must not be `None`.
        keep_classes (list): Classes tracked by `Class_*`, in the order the raster
            bands are written.

    Returns:
        dict[str, str]: Raster name -> output .tif path, for the 7 rasters written:
            pad_sb_0.5m, pad_profile_1m, class_count, entering_rays, intercept_ray,
            cover, dates_pad.

    Raises:
        ValueError: If `nlayers` or `nlayers_low` is `None`.
    """
    if nlayers is None or nlayers_low is None:
        raise ValueError("nlayers and nlayers_low must be fixed (not None) to export a raster")

    pad_low_bands = _strata_band_names("PAD", dz_low, nlayers_low, z0)
    pad_main_bands = _strata_band_names("PAD", dz, nlayers, z0)
    n_bands = _strata_band_names("N", dz, nlayers, z0)
    ni_bands = _strata_band_names("Ni", dz, nlayers, z0)
    class_bands = [f"Class_{code}" for code in keep_classes] + ["Total"]

    rasters = {
        "pad_sb_0.5m": (pad_low_bands, "float32", _FLOAT_NODATA),
        "pad_profile_1m": (pad_main_bands, "float32", _FLOAT_NODATA),
        "class_count": (class_bands, "int32", _INT_NODATA),
        "entering_rays": (n_bands, "int32", _INT_NODATA),
        "intercept_ray": (ni_bands, "int32", _INT_NODATA),
        "cover": (["Cover_2", "Cover_4", "Cover_6"], "float32", _FLOAT_NODATA),
        "dates_pad": (["Date_maj", "Date_min", "Date_max"], "int32", _INT_NODATA),
    }

    output_paths: dict[str, str] = {}
    for name, (band_names, dtype, nodata) in rasters.items():
        if name == "dates_pad":
            bands = {band: _days_since_epoch_band(pixel_results, band) for band in band_names}
        else:
            bands = {band: _band_array(pixel_results, band, nodata) for band in band_names}

        output_path = os.path.join(output_dir, f"{tilename}_{name}.tif")
        write_multiband_raster(
            output_path=output_path,
            bands=bands,
            raster_origin=raster_origin,
            pixel_size=pixel_size,
            crs=crs,
            dtype=dtype,
            nodata=nodata,
        )
        output_paths[name] = output_path

    return output_paths
