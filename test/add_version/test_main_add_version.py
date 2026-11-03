import shutil

import geopandas as gpd
import pytest
from omegaconf import OmegaConf
from utils import DATA_DIR, TEST_TILE, TEST_TILE_EXTENT, FakeResponse, _tile_feature

from lidar_for_fuel._version import __version__
from lidar_for_fuel.main_add_version import add_version, main

# Tile far from TEST_TILE, whose north-west corner is (897000, 6577000).
OTHER_TEST_TILE = "test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"


def _main_config(input_dir, output_dir, output_filename, input_filename=None):
    config = OmegaConf.load("configs/config.yaml")
    config.io.input_dir = str(input_dir)
    config.io.input_filename = input_filename
    config.add_version.output_dir = str(output_dir)
    config.add_version.output_filename = output_filename
    return config


def test_add_version_writes_the_version_in_a_geopackage(tmp_path, monkeypatch):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    shutil.copy(DATA_DIR / TEST_TILE, las_dir / TEST_TILE)
    output_path = tmp_path / "out" / "metadata.gpkg"

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


def test_add_version_on_one_tile(tmp_path, monkeypatch):
    las_filename = DATA_DIR / TEST_TILE

    print(tmp_path)

    add_version(las_filename, tmp_path / "metadata.gpkg")

    geodataframe = gpd.read_file(tmp_path / "metadata.gpkg", layer="metadata")
    assert geodataframe.crs.to_epsg() == 2154
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6690"]
    assert list(geodataframe["version_lidarforfuel"]) == [__version__]
    minx, maxx, miny, maxy = TEST_TILE_EXTENT
    assert geodataframe.geometry.iloc[0].bounds == pytest.approx((minx, miny, maxx, maxy))


def test_add_version_without_any_tile(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LAS/LAZ file"):
        add_version(tmp_path, tmp_path / "metadata.gpkg")


def test_main_writes_one_geopackage_for_all_the_tiles_of_a_directory(tmp_path, monkeypatch):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    for tile in (TEST_TILE, OTHER_TEST_TILE):
        shutil.copy(DATA_DIR / tile, las_dir / tile)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    features = [_tile_feature(751000.0, 6690000.0), _tile_feature(897000.0, 6577000.0)]
    monkeypatch.setattr(
        "lidar_for_fuel.add_version.download_mtd_from_geoplateforme.requests.get",
        lambda *args, **kwargs: FakeResponse({"features": features, "numberMatched": len(features)}),
    )

    main(_main_config(las_dir, output_dir, "metadata.gpkg"))

    assert [path.name for path in output_dir.iterdir()] == ["metadata.gpkg"]
    geodataframe = gpd.read_file(output_dir / "metadata.gpkg", layer="metadata")
    assert sorted(geodataframe["coordonnees_nw"]) == ["0751-6690", "0897-6577"]
    assert list(geodataframe["version_lidarforfuel"]) == [__version__, __version__]


def test_main_on_one_tile_writes_the_output_filename(tmp_path, monkeypatch):
    features = [_tile_feature(751000.0, 6690000.0)]
    monkeypatch.setattr(
        "lidar_for_fuel.add_version.download_mtd_from_geoplateforme.requests.get",
        lambda *args, **kwargs: FakeResponse({"features": features, "numberMatched": len(features)}),
    )

    main(_main_config(DATA_DIR, tmp_path, "metadata", input_filename=TEST_TILE))

    assert [path.name for path in tmp_path.iterdir()] == ["metadata.gpkg"]
    geodataframe = gpd.read_file(tmp_path / "metadata.gpkg", layer="metadata")
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6690"]
