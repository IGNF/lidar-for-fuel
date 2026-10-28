import shutil

import geopandas as gpd
import pytest

from lidar_for_fuel._version import __version__
from lidar_for_fuel.main_add_version import add_version

from utils import DATA_DIR, TEST_TILE, TEST_TILE_EXTENT, FakeResponse, _tile_feature


def test_add_version_writes_the_version_in_a_geopackage(tmp_path, monkeypatch):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    shutil.copy(DATA_DIR / TEST_TILE, las_dir / TEST_TILE)
    output_path = tmp_path / "out" / "metadata.gpkg"

    print(output_path)
    # No network access: the WFS service returns the tile covered by TEST_TILE
    # plus a neighbour that only shares an edge and must be discarded.
    features = [_tile_feature(751000.0, 6690000.0), _tile_feature(752000.0, 6690000.0)]
    monkeypatch.setattr(
        "lidar_for_fuel.add_version.download_mtd_from_geoplateforme.requests.get",
        lambda *args, **kwargs: FakeResponse({"features": features, "numberMatched": len(features)}),
    )

    add_version(las_dir, output_path)

    geodataframe = gpd.read_file(output_path, layer="metadata")
    assert geodataframe.crs.to_epsg() == 2154
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6690"]
    assert list(geodataframe["version_lidarforfuel"]) == [__version__]
    minx, maxx, miny, maxy = TEST_TILE_EXTENT
    assert geodataframe.geometry.iloc[0].bounds == pytest.approx((minx, miny, maxx, maxy))


def test_add_version_without_any_tile(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LAS/LAZ file"):
        add_version(tmp_path, tmp_path / "metadata.gpkg")
