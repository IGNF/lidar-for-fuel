"""
Download metadata LiDAR HD from IGN Géoplateforme for a given LiDAR tile.

Dataset: https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_MNT-LIDAR-HD
WMS flux: https://data.geopf.fr/wms-r/wms?service=wms&version=1.3.0&request=GetCapabilities

The tile bounding box is read directly from the LAS header using ign-pdaltools
"""
import logging
import os
from pathlib import Path
import json

import numpy as np
import requests
from pdaltools.las_info import get_bounds_from_header_info, las_info_metadata
from rasterio.io import MemoryFile

logger = logging.getLogger(__name__)

OVERLAP_TOLERANCE = 0.1


def _geometry_bounds(geometry: dict) -> tuple[float, float, float, float]:
    """Compute the bounding box of a GeoJSON geometry.

    Args:
        geometry (dict): GeoJSON geometry (any type built from nested coordinate lists).

    Returns:
        tuple[float, float, float, float]: Bounding box as (minx, maxx, miny, maxy).
    """
    xs: list[float] = []
    ys: list[float] = []

    def collect(coordinates) -> None:
        if isinstance(coordinates[0], (int, float)):
            xs.append(coordinates[0])
            ys.append(coordinates[1])
        else:
            for sub_coordinates in coordinates:
                collect(sub_coordinates)

    collect(geometry["coordinates"])
    return min(xs), max(xs), min(ys), max(ys)


def _overlaps(
    extent: tuple[float, float, float, float],
    other: tuple[float, float, float, float],
    tolerance: float = OVERLAP_TOLERANCE,
) -> bool:
    """Tell whether two extents overlap by more than a tolerance along both axes.

    Tiles that only touch the extent along an edge or a corner are not considered as
    overlapping, so that a tile-aligned extent matches only the tiles it really covers.

    Args:
        extent (tuple[float, float, float, float]): First extent as (minx, maxx, miny, maxy).
        other (tuple[float, float, float, float]): Second extent as (minx, maxx, miny, maxy).
        tolerance (float): Minimum overlap along each axis, in the unit of the extents.

    Returns:
        bool: True if both extents overlap by more than ``tolerance`` along both axes.
    """
    minx, maxx, miny, maxy = extent
    other_minx, other_maxx, other_miny, other_maxy = other
    return (
        min(maxx, other_maxx) - max(minx, other_minx) > tolerance
        and min(maxy, other_maxy) - max(miny, other_miny) > tolerance
    )


def _parse_feature(feature: dict) -> dict:
    """Flatten a WFS feature into a metadata dictionary.

    The layer exposes the metadata twice: as individual attributes (with dates suffixed by
    ``Z`` and the sensor as a raw PostgreSQL array) and as a JSON string in the ``metadata``
    attribute. The JSON values are the cleanest ones, so they take precedence over the
    attributes, which are kept for the download URLs they carry.

    Args:
        feature (dict): GeoJSON feature returned by the WFS service.

    Returns:
        dict: Metadata of the tile, plus its extent under the ``tile_extent`` key.
    """
    metadata = dict(feature["properties"])
    raw_metadata = metadata.pop("metadata", None)
    if raw_metadata:
        try:
            metadata.update({key.lower(): value for key, value in json.loads(raw_metadata).items()})
        except json.JSONDecodeError:
            logger.warning("Could not parse the 'metadata' attribute of a WFS feature: %s", raw_metadata)

    metadata["tile_extent"] = _geometry_bounds(feature["geometry"])
    return metadata



def get_mtd_from_stream(
    extent: tuple[float, float, float, float],
    epsg: int = 2154,
    resolution: float = 0.5,
    layer: str = "IGNF_LIDAR-HD_MNT_ELEVATION",
    timeout: int = 60,
) -> str:
    """
    Download the DTM LiDAR HD from IGN Géoplateforme for the given LiDAR tile.

    The bounding box (minx, maxx, miny, maxy) is read directly from the LAS
    header via pdaltools.las_info, avoiding any dependency on the filename format.

    Args:
        extent (tuple[float, float, float, float]): Bounding box coordinates (minx, maxx, miny, maxy).
        epsg (int): EPSG code of the coordinate reference system. Default: 2154
            (Lambert 93).
        tile_width (int): Tile size in metres. Default: 1000 (LiDAR HD tiles
            are 1 km × 1 km).
        resolution (float): Pixel size in metres. Default: 0.5 m (native
            resolution of the LiDAR HD DTM).
        timeout (int): Delay after which the request is canceled (in seconds) Default: 60.

    Returns:
        str: Absolute path to the downloaded DTM GeoTIFF.

    Raises:
        requests.HTTPError: If the WMS request fails.
    """
    minx, maxx, miny, maxy = extent

    URL_GPP = "https://data.geopf.fr/wms-r/wms?"
    URL_FORMAT = "&EXCEPTIONS=text/xml&FORMAT=image/geotiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetFeature&STYLES="
    URL_EPSG = "&CRS=EPSG:" + str(epsg)
    URL_BBOX = f"&BBOX={minx},{miny},{maxx},{maxy}"
    URL_SIZE = f"&WIDTH={(maxx - minx)/resolution}&HEIGHT={(maxy - miny)/resolution}"
    url = URL_GPP + "LAYERS=" + layer + URL_FORMAT + URL_EPSG + URL_BBOX + URL_SIZE

    logger.info(
        "Downloading DTM for tile '%s' — bbox=[%s, %s, %s, %s]",
        minx,
        miny,
        maxx,
        maxy,
    )
    logger.debug("WMS URL: %s", url)

    features = []
    response = requests.get(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    page = response.json()

    page_features = page.get("features", [])
    features.extend(page_features)

    metadata = [
        _parse_feature(feature)
        for feature in features
        if _overlaps(extent, _geometry_bounds(feature["geometry"]), 0.01) # overlap_tolerance
    ]

    return metadata
