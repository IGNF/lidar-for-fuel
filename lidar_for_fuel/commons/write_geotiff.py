"""
Write named float32 bands into one georeferenced, multi-band GeoTIFF (nodata NaN).
"""

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


def write_geotiff(path: Path, bands: dict[str, np.ndarray], transform: Affine, srid: str | CRS) -> None:
    """Write one multi-band GeoTIFF.

    Args:
        path (Path): Output file path. Parent directories are created if missing.
        bands (dict[str, np.ndarray]): Band name -> (height, width) float32 array,
            written in dict order (band 1 first).
        transform (Affine): Georeferencing transform of the output raster.
        srid (str | CRS): Spatial reference of the output raster (e.g. "EPSG:2154"), or the
            CRS of an input raster.

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
