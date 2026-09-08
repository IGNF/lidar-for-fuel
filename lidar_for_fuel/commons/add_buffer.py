"""
Add a buffer to a LiDAR tile by merging in points from neighboring tiles.
"""

import logging
import os
import tempfile
from contextlib import contextmanager

from pdaltools.las_add_buffer import create_las_with_buffer

logger = logging.getLogger(__name__)


@contextmanager
def create_buffered_las_file(
    input_dir: str,
    input_filename: str,
    buffer_width: float,
    spatial_ref: str,
    tile_width: int = 1000,
    tile_coord_scale: int = 1000,
):
    """Merge a tile with its neighbors and crop to tile bounds + buffer, in a temporary file.

    This avoids edge effects on PAD computation by feeding the pixels near the tile
    border with points from the neighboring tiles.

    Args:
        input_dir (str): Directory of pointclouds (where neighbor tiles are looked up).
        input_filename (str): Full path to the queried LiDAR tile.
        buffer_width (float): Width of the buffer to add around the tile (m).
        spatial_ref (str): Spatial reference to use to override the one from input las.
        tile_width (int): Tile width in meters. Default 1000.
        tile_coord_scale (int): Scale used in filenames to describe coordinates in meters.
            Default 1000.

    Yields:
        str: Path to the temporary buffered LAS file.
    """
    with tempfile.TemporaryDirectory(prefix="tmp_buffer_") as tmpdir:
        # Use .las (not .laz) to avoid a bug with LAZ compression, as done in ctview.
        buffered_filename = os.path.join(tmpdir, os.path.splitext(os.path.basename(input_filename))[0] + ".las")

        create_las_with_buffer(
            input_dir=input_dir,
            tile_filename=input_filename,
            output_filename=buffered_filename,
            buffer_width=buffer_width,
            spatial_ref=spatial_ref,
            tile_width=tile_width,
            tile_coord_scale=tile_coord_scale,
        )

        logger.info(
            "create_buffered_las_file: merged %s with buffer=%sm into %s",
            input_filename,
            buffer_width,
            buffered_filename,
        )

        try:
            yield buffered_filename
        except Exception:
            logger.error("create_buffered_las_file: failed while using %s", buffered_filename)
            raise
