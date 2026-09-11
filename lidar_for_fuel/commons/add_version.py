"""
Retrieve LiDAR HD metadata from the IGN Géoplateforme WFS service for a given extent.

Dataset: https://cartes.gouv.fr/rechercher-une-donnee/dataset/IGNF_LIDAR-HD
WFS flux: https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities

The ``IGNF_LIDAR-HD_METADONNEE:metadata`` layer holds one feature per 1 km x 1 km LiDAR HD
tile, describing how the tile was acquired and classified (mission code, sensor, acquisition
dates, classification process, reference systems, download URLs).

The extent is read directly from the LAS/LAZ headers, so no assumption is made on filenames.
"""
import json
import logging
from pathlib import Path

import laspy
import requests

logger = logging.getLogger(__name__)

WFS_URL = "https://data.geopf.fr/wfs/ows"
WFS_VERSION = "2.0.0"
LIDAR_HD_METADATA_TYPENAME = "IGNF_LIDAR-HD_METADONNEE:metadata"

# Requesting features by pages keeps the response size bounded on large extents.
PAGE_SIZE = 1000

# The server applies the BBOX filter on an approximate envelope and returns tiles that do not
# actually overlap the requested extent, so features are filtered again client-side. Its
# geometries are stored in EPSG:4326 and reprojected on the fly, which shifts the tile corners
# by a few millimetres: adjacent tiles would overlap the extent without this tolerance (in
# metres), which has to stay well below the size of the smallest queried extent.
OVERLAP_TOLERANCE = 0.1


def compute_extent(las_dir: Path) -> tuple[float, float, float, float]:
    """Compute the extent enclosing every LAS/LAZ file of a directory.

    Only the file headers are read, the point records are left untouched.

    Args:
        las_dir (Path): Directory containing the LiDAR tiles (``.las`` or ``.laz``).

    Returns:
        tuple[float, float, float, float]: Extent as (minx, maxx, miny, maxy), in the
            coordinate reference system of the tiles.

    Raises:
        FileNotFoundError: If the directory contains no LAS/LAZ file.
    """
    las_files = sorted(set(Path(las_dir).glob("*.las")) | set(Path(las_dir).glob("*.laz")))
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ file found in {las_dir}")

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for las_file in las_files:
        with laspy.open(las_file) as las:
            header = las.header
        minx = min(minx, header.mins[0])
        maxx = max(maxx, header.maxs[0])
        miny = min(miny, header.mins[1])
        maxy = max(maxy, header.maxs[1])

    logger.debug("Extent of %d tiles in %s: [%s, %s, %s, %s]", len(las_files), las_dir, minx, maxx, miny, maxy)
    return minx, maxx, miny, maxy


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
    typename: str = LIDAR_HD_METADATA_TYPENAME,
    epsg: int = 2154,
    timeout: int = 60,
    wfs_url: str = WFS_URL,
    overlap_tolerance: float = OVERLAP_TOLERANCE,
) -> list[dict]:
    """Get the LiDAR HD metadata of every tile overlapping an extent, from the WFS service.

    Args:
        extent (tuple[float, float, float, float]): Extent as (minx, maxx, miny, maxy).
        typename (str): Name of the queried WFS layer. Default: the LiDAR HD metadata layer.
        epsg (int): EPSG code of the extent, also used for the returned geometries.
            Default: 2154 (Lambert 93).
        timeout (int): Delay after which a request is canceled (in seconds). Default: 60.
        wfs_url (str): Base URL of the WFS service.
        overlap_tolerance (float): Minimum overlap, in metres, between the extent and a tile
            for that tile to be kept. Default: 0.1 (see ``OVERLAP_TOLERANCE``).

    Returns:
        list[dict]: One dictionary per LiDAR HD tile overlapping the extent.

    Raises:
        requests.HTTPError: If a WFS request fails.
        ValueError: If the extent is empty (its min bound is above its max bound).
    """
    minx, maxx, miny, maxy = extent
    if minx > maxx or miny > maxy:
        raise ValueError(f"Invalid extent (minx, maxx, miny, maxy): {extent}")

    parameters = {
        "SERVICE": "WFS",
        "VERSION": WFS_VERSION,
        "REQUEST": "GetFeature",
        "TYPENAMES": typename,
        "OUTPUTFORMAT": "application/json",
        "SRSNAME": f"EPSG:{epsg}",
        "BBOX": f"{minx},{miny},{maxx},{maxy},EPSG:{epsg}",
        "COUNT": PAGE_SIZE,
    }

    logger.info("Querying %s on extent [%s, %s, %s, %s]", typename, minx, maxx, miny, maxy)

    features = []
    start_index = 0
    while True:
        response = requests.get(wfs_url, params={**parameters, "STARTINDEX": start_index}, timeout=timeout)
        response.raise_for_status()
        page = response.json()

        page_features = page.get("features", [])
        features.extend(page_features)
        start_index += len(page_features)
        if not page_features or start_index >= page.get("numberMatched", 0):
            break

    metadata = [
        _parse_feature(feature)
        for feature in features
        if _overlaps(extent, _geometry_bounds(feature["geometry"]), overlap_tolerance)
    ]

    logger.info(
        "Found %d LiDAR HD tiles overlapping the extent (%d returned by the service)",
        len(metadata),
        len(features),
    )
    return metadata
