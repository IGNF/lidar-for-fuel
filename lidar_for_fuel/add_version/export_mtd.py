import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box


def _attribute_value(value):
    """Turn a metadata value into a scalar a GeoPackage attribute column can store.

    Lists, tuples and dictionaries (for instance the sensor list) are serialized
    as JSON strings. Scalars are left unchanged.

    Args:
        value: Metadata value.

    Returns:
        A scalar suitable for a GeoPackage attribute, or a JSON string.
    """
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def export_mtd(metadata: list[dict], output_path: Path, epsg: int = 2154) -> None:
    """Export the LiDAR HD tile metadata to a GeoPackage.

    Each tile is written as one feature in the ``metadata`` layer. The geometry
    is the rectangle of ``tile_extent``; the other entries become attributes.

    Args:
        metadata (list[dict]): One dictionary per LiDAR HD tile, as returned by
            ``get_mtd_from_stream`` (optionally enriched by ``add_version_to_mtd``).
        output_path (Path): Path of the GeoPackage file to write. Parent
            directories are created when missing. An existing file is replaced.
        epsg (int): EPSG code of the tile extents. Default: 2154 (Lambert 93).

    Raises:
        ValueError: If a tile dictionary has no ``tile_extent``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geometries = []
    records = []
    for tile in metadata:
        try:
            minx, maxx, miny, maxy = tile["tile_extent"]
        except KeyError as error:
            raise ValueError("Each metadata dictionary must contain a 'tile_extent'") from error
        geometries.append(box(minx, miny, maxx, maxy))
        records.append({key: _attribute_value(value) for key, value in tile.items() if key != "tile_extent"})

    geodataframe = gpd.GeoDataFrame(records, geometry=geometries, crs=f"EPSG:{epsg}")
    if output_path.exists():
        output_path.unlink()
    geodataframe.to_file(output_path, driver="GPKG", layer="metadata")