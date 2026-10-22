import shutil
import pytest
from pathlib import Path
from lidar_for_fuel.add_version.compute_extent import compute_extent
from utils import DATA_DIR, TEST_TILE, TEST_TILE_EXTENT



def test_compute_extent_one_tile(tmp_path):
    shutil.copy(DATA_DIR / TEST_TILE, tmp_path / TEST_TILE)

    minx, maxx, miny, maxy = compute_extent(tmp_path)

    assert (minx, maxx, miny, maxy) == pytest.approx(TEST_TILE_EXTENT, abs=1.0)


def test_compute_extent_is_the_union_of_the_tiles(tmp_path):
    shutil.copy(DATA_DIR / TEST_TILE, tmp_path / TEST_TILE)
    shutil.copy(DATA_DIR / "test_semis_2022_0897_6577_LA93_IGN69_decimation.laz", tmp_path)

    minx, maxx, miny, maxy = compute_extent(tmp_path)

    assert minx == pytest.approx(751000.0, abs=1.0)
    assert maxx == pytest.approx(898000.0, abs=1.0)
    assert miny == pytest.approx(6576000.0, abs=1.0)
    assert maxy == pytest.approx(6690000.0, abs=1.0)


def test_compute_extent_without_any_tile(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LAS/LAZ file"):
        compute_extent(tmp_path)
