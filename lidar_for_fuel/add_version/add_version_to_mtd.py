from lidar_for_fuel._version import __version__


def add_version_to_mtd(metadata: list[dict]) -> list[dict]:
    """Add the lidar_for_fuel version to the metadata of each tile.

    Args:
        metadata (list[dict]): One dictionary per LiDAR HD tile, as returned by
            ``get_mtd_from_stream``.

    Returns:
        list[dict]: The same list, each tile dictionary now carrying
            ``version_lidarforfuel``.
    """
    for tile in metadata:
        tile["version_lidarforfuel"] = __version__
    return metadata