import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from lidar_for_fuel.pad_profile.calculate_pad_profile import _format_num

logger = logging.getLogger(__name__)


def _raster_transform(
    global_origin_x: float, global_origin_y: float, origin_pixel: tuple, resolution_factor: float
) -> Affine:
    """Affine transform anchored on the CosiaFrance grid corner `origin_pixel`.

    `origin_pixel` is the tile's own (ix, iy) grid index. The raster's top-left
    corner is therefore the *north-west* corner of pixel (ix=origin_pixel[0], iy=origin_pixel[1]).

    Args:
        global_origin_x (float): X of the CosiaFrance grid anchor.
        global_origin_y (float): Y of the CosiaFrance grid anchor (north edge).
        origin_pixel (tuple): CosiaFrance grid corner (ix, iy) the tile's window is anchored on.
        resolution_factor (float): Pixel size (m) of the CosiaFrance grid.

    Returns:
        Affine: Transform mapping (row, col) of the output array to real-world (x, y).
    """
    # Get the top-left corner in meters
    top_left_x = global_origin_x + origin_pixel[0] * resolution_factor
    top_left_y = global_origin_y + origin_pixel[1] * resolution_factor

    # Affine transform from image space (row, column) to the georeferenced coordinate space
    return Affine(resolution_factor, 0.0, top_left_x, 0.0, -resolution_factor, top_left_y)


def _select_stratum_columns(aggregated: pd.DataFrame, prefix: str) -> list[str]:
    """Select and order raster bands: filter by a given prefix (e.g. "PAD_1_"),
    then sort numerically by the suffix.

    Args:
        aggregated (pd.DataFrame): Per-pixel metrics, as returned by `compute_pixel_aggregates`.
        prefix (str): Column prefix to match, e.g. "PAD_1_" or "N_1_".

    Returns:
        list[str]: Matching column names, sorted ascending by their numeric suffix
            (e.g. PAD_1_0, PAD_1_1, ..., PAD_1_59 -- not alphabetically, which
            would put PAD_1_10 before PAD_1_2).
    """
    columns = [c for c in aggregated.columns if c.startswith(prefix)]

    return sorted(columns, key=lambda c: float(c[len(prefix) :]))


def _band_array(values: pd.Series, origin_pixel: tuple, nb_pixels: float, clip: tuple | None = None) -> np.ndarray:
    """Rasterize one pixel-indexed series into a (nb_pixels, nb_pixels) float32 grid.

    `values` is indexed like `compute_pixel_aggregates`'s output: a (pixel_y, pixel_x)
    MultiIndex in CosiaFrance grid coordinates. Cells with no data stay `NaN`.

    Args:
        values (pd.Series): One column of `aggregated` (or a derived series, e.g.
            `pl_factor`), indexed by (pixel_y, pixel_x).
        origin_pixel (tuple): CosiaFrance grid corner (ix, iy) the tile's window is anchored on.
        nb_pixels (float): Tile side length in pixels.
        clip (tuple | None): Optional (min, max) bounds the values are clipped to
            before being written (e.g. (0.0, 5.0) to cap PAD). Default None.

    Returns:
        np.ndarray: (nb_pixels, nb_pixels) float32 grid, `NaN` where no pixel matched.
    """
    size = int(nb_pixels)
    array = np.full((size, size), np.nan, dtype=np.float32)

    data = values.to_numpy(dtype=np.float64)

    # Clip = capping of PADs and cover.
    if clip is not None:
        data = np.clip(data, *clip)

    pixel_y = values.index.get_level_values("pixel_y").to_numpy()
    pixel_x = values.index.get_level_values("pixel_x").to_numpy()
    rows = (origin_pixel[1] - pixel_y).astype(int)
    cols = (pixel_x - origin_pixel[0]).astype(int)

    # A single line fills all the pixels at once.
    array[rows, cols] = data.astype(np.float32)

    return array


def _write_geotiff(path: Path, bands: dict[str, np.ndarray], transform: Affine, srid: str) -> None:
    """Write one multi-band GeoTIFF.

    Args:
        path (Path): Output file path. Parent directories are created if missing.
        bands (dict[str, np.ndarray]): Band name -> (height, width) float32 array,
            written in dict order (band 1 first).
        transform (Affine): Georeferencing transform (see `_raster_transform`).
        srid (str): Spatial reference of the output raster (e.g. "EPSG:2154").

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = next(iter(bands.values())).shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(bands),
        dtype="float32",
        crs=srid,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        for band_index, (name, array) in enumerate(bands.items(), start=1):
            dst.write(array, band_index)
            dst.set_band_description(band_index, name)


def export_raster(
    aggregated: pd.DataFrame,
    origin_pixel: tuple,
    nb_pixels: float,
    global_origin_x: float,
    global_origin_y: float,
    resolution_factor: float,
    dz: float,
    dz_low: float,
    srid: str,
    output_dir: str | Path,
    tile_stem: str,
) -> dict[str, Path]:
    """Assemble `compute_pixel_aggregates`'s per-pixel PAD metrics into 8 georeferenced,
    multi-band GeoTIFFs aligned on the CosiaFrance grid.

    Args:
        aggregated (pd.DataFrame): Per-pixel metrics, as returned by
            `compute_pixel_aggregates` with a PAD `aggregation` (see `build_pad_aggregation`).
            Indexed by (pixel_y, pixel_x) in CosiaFrance grid coordinates.
        origin_pixel (tuple): CosiaFrance grid corner (ix, iy) the tile's window is
            anchored on, as returned by `compute_pixel_aggregates`.
        nb_pixels (float): Tile side length in pixels, as returned by `compute_pixel_aggregates`.
        global_origin_x (float): X of the CosiaFrance grid anchor.
        global_origin_y (float): Y of the CosiaFrance grid anchor (north edge).
        resolution_factor (float): Pixel size (m) of the CosiaFrance grid.
        dz (float): Stratum thickness (m) of the main PAD profile, as passed to `pad_metrics_core`.
        dz_low (float): Stratum thickness (m) of the low-strata PAD band, as passed to `pad_metrics_core`.
        srid (str): Spatial reference of the output rasters (e.g. "EPSG:2154").
        output_dir (str | Path): Directory the 8 GeoTIFFs are written into.
        tile_stem (str): Basename (without extension) shared by the 8 output files,
            e.g. the input tile's own filename stem.

    Returns:
        dict[str, Path]: raster name -> path of the GeoTIFF written for it.
    """
    output_dir = Path(output_dir)
    transform = _raster_transform(global_origin_x, global_origin_y, origin_pixel, resolution_factor)

    dz_str = _format_num(dz)
    dz_low_str = _format_num(dz_low)

    class_columns = _select_stratum_columns(aggregated, "Class_") + ["Total"]

    pl_factor = 1.0 / aggregated["cos_theta"]
    pl_factor.name = "pl_factor"

    raster_columns = {
        "pad_sb_0.5m": [ <la liste des noms de colonnes correspondant>]
        },
        "pad_profile_1m": [ <la liste des noms de colonnes correspondant>],
        "class_count":[ <la liste des noms de colonnes correspondant>],
        "entering_rays": [ <la liste des noms de colonnes correspondant>],
        "intercept_ray":[ <la liste des noms de colonnes correspondant>],
        "pl_factor":[ <la liste des noms de colonnes correspondant>],
        "cover": [ <la liste des noms de colonnes correspondant>],
        "dates_pad": [ <la liste des noms de colonnes correspondant>],
    }
    raster_clip = {
        "pad_sb_0.5m":(0.0, 5.0)
        },
        "pad_profile_1m":(0.0, 5.0),
        "class_count":None,
        "entering_rays": None,
        "intercept_ray":None,
        "pl_factor":None,
        "cover": (0.0, 1.0),
        "dates_pad": None,
    }

    written: dict[str, Path] = {}
    for raster_name in raster_columns.keys():
        columns = raster_columns[raster_name]
        clip = raster_clip[raster_name]
        bands = {
            name: _band_array(aggregated[c], origin_pixel, nb_pixels, clip=clip) for col_name in columns)
        }
        path = output_dir / f"{tile_stem}_{raster_name}.tif"
        _write_geotiff(path, bands, transform, srid)
        written[raster_name] = path
        logger.info("Wrote %s (%d bands) to %s", raster_name, len(bands), path)

    return written
