import shutil
from pathlib import Path

import laspy

from lidar_for_fuel.commons.add_buffer import create_buffered_las_file

_INPUT_DIR = Path("data/buffer")
_CENTRAL_TILE = _INPUT_DIR / "Semis_2022_0691_6484_LA93_IGN69_pretraited.laz"
_BUFFER_WIDTH = 10


def test_create_buffered_las_file_extends_bounds_and_cleans_up_on_exit(tmp_path):
    """The buffered file is a usable, writable copy covering a larger extent than the
    original tile, and it is removed once the context manager exits."""
    original_header = laspy.open(_CENTRAL_TILE).header

    with create_buffered_las_file(
        input_dir=str(_INPUT_DIR),
        input_filename=str(_CENTRAL_TILE),
        buffer_width=_BUFFER_WIDTH,
        spatial_ref="EPSG:2154",
        tile_width=1000,
        tile_coord_scale=1000,
    ) as buffered_las_filename:
        path = Path(buffered_las_filename)
        assert path.exists()

        # The temporary file is a regular, usable file: it can be copied.
        copy_path = tmp_path / "copy.las"
        shutil.copy2(path, copy_path)
        assert copy_path.exists()

        buffered_header = laspy.open(copy_path).header
        # The buffer widens the extent on every side compared to the original tile.
        assert buffered_header.mins[0] < original_header.mins[0]
        assert buffered_header.mins[1] < original_header.mins[1]
        assert buffered_header.maxs[0] > original_header.maxs[0]
        assert buffered_header.maxs[1] > original_header.maxs[1]

    assert not path.exists()
    assert not path.parent.exists()
