"""
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
Download DTM LiDAR HD from IGN Géoplateforme for a given LiDAR tile.
=======
Download MNT LiDAR HD from IGN Géoplateforme for a given LiDAR tile.
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
Download DTM LiDAR HD from IGN Géoplateforme for a given LiDAR tile.
>>>>>>> 5c437cc (replace MNT -> DTM):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
Download DTM LiDAR HD from IGN Géoplateforme for a given LiDAR tile.
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

Dataset: https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_MNT-LIDAR-HD
WMS flux: https://data.geopf.fr/wms-r/wms?service=wms&version=1.3.0&request=GetCapabilities

The tile bounding box is read directly from the LAS header using ign-pdaltools
"""
import logging
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
import os
=======
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
import os
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
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
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
    tilename: str,
    input_dir: str,
=======
    tile_path: str,
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
    tilename: str,
    input_dir: str,
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
    layer: str,
    output_dir: str,
    epsg: int = 2154,
    tile_width: int = 1000,
    resolution: float = 0.5,
    timeout: int = 60,
) -> str:
    """
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
    Download the DTM LiDAR HD from IGN Géoplateforme for the given LiDAR tile.
=======
    Download the MNT LiDAR HD from IGN Géoplateforme for the given LiDAR tile.
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
    Download the DTM LiDAR HD from IGN Géoplateforme for the given LiDAR tile.
>>>>>>> 5c437cc (replace MNT -> DTM):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
    Download the DTM LiDAR HD from IGN Géoplateforme for the given LiDAR tile.
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

    The bounding box (minx, maxx, miny, maxy) is read directly from the LAS
    header via pdaltools.las_info, avoiding any dependency on the filename format.

    Args:
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
        tilename (str): Filename of the LiDAR tile (LAS/LAZ), without directory.
        input_dir (str): Directory where the LiDAR tiles are stored.
        layer (str): which kind of image is downloaded (IGNF_LIDAR-HD_MNT_ELEVATION)
        output_dir (str): Directory where the DTM GeoTIFF is saved.
<<<<<<< HEAD
=======
        tile_path (str): Path to the LiDAR tile (LAS/LAZ).
        layer (str): which kind of image is downloaded (IGNF_LIDAR-HD_MNT_ELEVATION)
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
        output_dir (str): Directory where the MNT GeoTIFF is saved.
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
        output_dir (str): Directory where the DTM GeoTIFF is saved.
>>>>>>> 5c437cc (replace MNT -> DTM):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
        epsg (int): EPSG code of the coordinate reference system. Default: 2154
            (Lambert 93).
        tile_width (int): Tile size in metres. Default: 1000 (LiDAR HD tiles
            are 1 km × 1 km).
        resolution (float): Pixel size in metres. Default: 0.5 m (native
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
            resolution of the LiDAR HD DTM).
        timeout (int): Delay after which the request is canceled (in seconds) Default: 60.

    Returns:
        str: Absolute path to the downloaded DTM GeoTIFF.
<<<<<<< HEAD
=======
            resolution of the LiDAR HD MNT).
        timeout (int): Delay after which the request is canceled (in seconds) Default: 60.

    Returns:
        str: Absolute path to the downloaded MNT GeoTIFF.
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
            resolution of the LiDAR HD DTM).
        timeout (int): Delay after which the request is canceled (in seconds) Default: 60.

    Returns:
        str: Absolute path to the downloaded DTM GeoTIFF.
>>>>>>> 5c437cc (replace MNT -> DTM):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)

    Raises:
        requests.HTTPError: If the WMS request fails.
    """
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
    tile_path = os.path.join(input_dir, tilename)
=======
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
    tile_path = os.path.join(input_dir, tilename)
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
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
<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
        "Downloading DTM for tile '%s' — bbox=[%s, %s, %s, %s]",
        tilename,
=======
        "Downloading MNT for tile '%s' — bbox=[%s, %s, %s, %s]",
=======
        "Downloading DTM for tile '%s' — bbox=[%s, %s, %s, %s]",
>>>>>>> 5c437cc (replace MNT -> DTM):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
        Path(tile_path).name,
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
        "Downloading DTM for tile '%s' — bbox=[%s, %s, %s, %s]",
        tilename,
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
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

<<<<<<< HEAD
<<<<<<< HEAD:lidar_for_fuel/preprocessing/download_dtm_from_geoplateforme.py
    geotiff_filename = f"{Path(tilename).stem}.tif"
    output_path = Path(output_dir) / geotiff_filename
=======
    output_path = Path(output_dir) / f"{Path(tile_path).stem}.tif"
>>>>>>> cef44e0 (add function : download DTM from geoservice):lidar_for_fuel/pretreatment/download_dtm_from_geoplateforme.py
=======
    geotiff_filename = f"{Path(tilename).stem}.tif"
    output_path = Path(output_dir) / geotiff_filename
>>>>>>> 8f1bacb (refacto add_trajectory : name fileds, configs, main_preprocessing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)

    logger.info("DTM saved: %s", output_path)
    return str(output_path)
