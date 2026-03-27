"""
Download MNT LiDAR HD from IGN Géoplateforme for a given LiDAR tile.

Dataset: https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_MNT-LIDAR-HD
WMS flux: https://data.geopf.fr/wms-r/wms?service=wms&version=1.3.0&request=GetCapabilities

The tile bounding box is read directly from the LAS header using ign-pdaltools
"""
import logging
from pathlib import Path

import numpy as np
import requests
from pdaltools.las_info import get_bounds_from_header_info, las_info_metadata
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)


def is_dtm_nodata(content: bytes) -> bool:
    """Check if a DTM raster (in memory) contains only uniform values.

    Unlike an orthophoto where a failed download returns a white image (all 255),
    a failed DTM download typically returns a raster where all pixels share the
    same value (nodata fill or constant elevation). Any valid DTM over real terrain
    must contain varying elevation values.

    Args:
        content (bytes): Raw bytes of the downloaded GeoTIFF.

    Returns:
        bool: True if all pixels have the same value (DTM is uniform/nodata).
    """
    with MemoryFile(content) as memfile:
        with memfile.open() as dataset:
            band = dataset.read(1)
    return bool(np.all(band == band.flat[0]))


def download_dtm(
    tile_path: str,
    layer: str,
    output_dir: str,
    epsg: int = 2154,
    tile_width: int = 1000,
    resolution: float = 0.5,
    timeout: int = 60,
) -> str:
    """
    Download the MNT LiDAR HD from IGN Géoplateforme for the given LiDAR tile.

    The bounding box (minx, maxx, miny, maxy) is read directly from the LAS
    header via pdaltools.las_info, avoiding any dependency on the filename format.

    Args:
        tile_path (str): Path to the LiDAR tile (LAS/LAZ).
        layer (str): which kind of image is downloaded (IGNF_LIDAR-HD_MNT_ELEVATION)
        output_dir (str): Directory where the MNT GeoTIFF is saved.
        epsg (int): EPSG code of the coordinate reference system. Default: 2154
            (Lambert 93).
        tile_width (int): Tile size in metres. Default: 1000 (LiDAR HD tiles
            are 1 km × 1 km).
        resolution (float): Pixel size in metres. Default: 0.5 m (native
            resolution of the LiDAR HD MNT).
        timeout (int): Delay after which the request is canceled (in seconds) Default: 60.

    Returns:
        str: Absolute path to the downloaded MNT GeoTIFF.

    Raises:
        requests.HTTPError: If the WMS request fails.
    """
    metadata = las_info_metadata(tile_path)
    minx, maxx, miny, maxy = get_bounds_from_header_info(metadata)

    width_px = int(tile_width / resolution)
    height_px = int(tile_width / resolution)

    URL_GPP = "https://data.geopf.fr/wms-r/wms?"
    URL_FORMAT = "&EXCEPTIONS=text/xml&FORMAT=image/geotiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES="
    URL_EPSG = "&CRS=EPSG:" + str(epsg)
    URL_BBOX = f"&BBOX={minx},{miny},{maxx},{maxy}"
    URL_SIZE = f"&WIDTH={width_px}&HEIGHT={height_px}"
    url = URL_GPP + "LAYERS=" + layer + URL_FORMAT + URL_EPSG + URL_BBOX + URL_SIZE

    logger.info(
        "Downloading MNT for tile '%s' — bbox=[%s, %s, %s, %s]",
        Path(tile_path).name,
        minx,
        miny,
        maxx,
        maxy,
    )
    logger.debug("WMS URL: %s", url)

    response = requests.get(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()

    if is_dtm_nodata(response.content):
        raise ValueError(f"Downloaded DTM contains only uniform values (nodata): {layer}")

    output_path = Path(output_dir) / f"{Path(tile_path).stem}.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    logger.info("DTM saved: %s", output_path)
    return str(output_path)
