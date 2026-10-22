from lidar_for_fuel.add_version.export_mtd import export_mtd
from lidar_for_fuel.add_version.add_version_to_mtd import add_version_to_mtd
from lidar_for_fuel.add_version.download_mtd_from_geoplateforme import _parse_feature
from lidar_for_fuel._version import __version__
import pytest
import json
import geopandas as gpd
from pathlib import Path
from utils import TEST_TILE_EXTENT, _tile_feature



def test_export_mtd_writes_one_feature_per_tile(tmp_path):
    metadata = add_version_to_mtd(
        [
            _parse_feature(_tile_feature(751000.0, 6690000.0)),
            _parse_feature(_tile_feature(751000.0, 6689000.0, code_mission="23LHDMH")),
        ]
    )
    output_path = tmp_path / "out" / "metadata.gpkg"

    export_mtd(metadata, output_path)

    geodataframe = gpd.read_file(output_path, layer="metadata")
    assert geodataframe.crs.to_epsg() == 2154
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6690", "0751-6689"]
    assert list(geodataframe["code_mission"]) == ["22LHDMH", "23LHDMH"]
    assert list(geodataframe["version_lidarforfuel"]) == [__version__, __version__]
    # Lists cannot be stored as GeoPackage attributes, so they are JSON strings.
    assert json.loads(geodataframe.iloc[0]["capteur"]) == ["Optech ALTM Galaxy T2000:5060485"]
    assert "tile_extent" not in geodataframe.columns
    minx, maxx, miny, maxy = TEST_TILE_EXTENT
    assert geodataframe.geometry.iloc[0].bounds == pytest.approx((minx, miny, maxx, maxy))
    # The input dictionaries are left unchanged.
    assert metadata[0]["capteur"] == ["Optech ALTM Galaxy T2000:5060485"]
    assert metadata[0]["tile_extent"] == TEST_TILE_EXTENT


def test_export_mtd_replaces_an_existing_file(tmp_path):
    output_path = tmp_path / "metadata.gpkg"
    export_mtd([_parse_feature(_tile_feature(751000.0, 6690000.0))], output_path)
    export_mtd([_parse_feature(_tile_feature(751000.0, 6689000.0))], output_path)

    geodataframe = gpd.read_file(output_path)
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6689"]


def test_export_mtd_without_a_tile_extent():
    with pytest.raises(ValueError, match="tile_extent"):
        export_mtd([{"code_mission": "22LHDMH"}], Path("unused.gpkg"))