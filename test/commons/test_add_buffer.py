from pathlib import Path

import laspy
import pytest

from lidar_for_fuel.commons.add_buffer import create_buffered_las_file

_INPUT_DIR = Path("data/buffer")
_CENTRAL_TILE = _INPUT_DIR / "Semis_2022_0691_6484_LA93_IGN69_pretraited.laz"
_BUFFER_WIDTH = 100


@pytest.fixture()
def central_tile():
    """Real 3x3 tile grid: Semis_2022_0691_6484 (central) + its 8 neighbors."""
    if not _CENTRAL_TILE.exists():
        pytest.skip(f"Real LAS {_CENTRAL_TILE} not found in workspace")
    return _CENTRAL_TILE


def test_create_buffered_las_file_yields_a_readable_las_with_points(central_tile):
    """The buffered file is a valid LAS readable by laspy, with points inside it."""
    with create_buffered_las_file(
        input_dir=str(_INPUT_DIR),
        input_filename=str(central_tile),
        buffer_width=_BUFFER_WIDTH,
        spatial_ref="EPSG:2154",
        tile_width=1000,
        tile_coord_scale=1000,
    ) as buffered_las_filename:
        assert Path(buffered_las_filename).exists()
        las = laspy.read(buffered_las_filename)
        assert len(las.points) > 0


def test_create_buffered_las_file_merges_points_from_neighboring_tiles(central_tile):
    """The buffered tile has more points than the central tile alone: neighbors were merged in."""
    central_only_point_count = laspy.open(central_tile).header.point_count

    with create_buffered_las_file(
        input_dir=str(_INPUT_DIR),
        input_filename=str(central_tile),
        buffer_width=_BUFFER_WIDTH,
        spatial_ref="EPSG:2154",
        tile_width=1000,
        tile_coord_scale=1000,
    ) as buffered_las_filename:
        las = laspy.read(buffered_las_filename)
        assert len(las.points) > central_only_point_count


def test_create_buffered_las_file_preserves_extra_dimensions(central_tile):
    """Extra dimensions common to the tile grid (dtm_marker, dsm_marker) survive buffering."""
    with create_buffered_las_file(
        input_dir=str(_INPUT_DIR),
        input_filename=str(central_tile),
        buffer_width=_BUFFER_WIDTH,
        spatial_ref="EPSG:2154",
        tile_width=1000,
        tile_coord_scale=1000,
    ) as buffered_las_filename:
        las = laspy.read(buffered_las_filename)
        extra_dims = set(las.point_format.extra_dimension_names)
        assert {"dtm_marker", "dsm_marker"}.issubset(extra_dims)


def test_create_buffered_las_file_removes_temp_file_on_exit(central_tile):
    """The temporary buffered file is cleaned up once the context manager exits."""
    with create_buffered_las_file(
        input_dir=str(_INPUT_DIR),
        input_filename=str(central_tile),
        buffer_width=_BUFFER_WIDTH,
        spatial_ref="EPSG:2154",
        tile_width=1000,
        tile_coord_scale=1000,
    ) as buffered_las_filename:
        path = Path(buffered_las_filename)
        assert path.exists()

    assert not path.exists()
    assert not path.parent.exists()
