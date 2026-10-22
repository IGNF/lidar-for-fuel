from pathlib import Path
import logging
from pdaltools.las_info import get_bounds_from_header_info, las_info_metadata


logger = logging.getLogger(__name__)


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
        metadata = las_info_metadata(las_file)
        bounds = get_bounds_from_header_info(metadata)
        minx = min(minx, bounds[0])
        maxx = max(maxx, bounds[1])
        miny = min(miny, bounds[2])
        maxy = max(maxy, bounds[3])

    logger.debug("Extent of %d tiles in %s: [%s, %s, %s, %s]", len(las_files), las_dir, minx, maxx, miny, maxy)
    return minx, maxx, miny, maxy