"""
Tests for add_version.

The tests of the WFS query itself are offline (the HTTP layer is faked), except
test_get_mtd_from_stream_on_real_service which requires network access to data.geopf.fr
(IGN Géoplateforme).
"""
import json
import shutil
from pathlib import Path

import geopandas as gpd
import pytest

from lidar_for_fuel._version import __version__
from lidar_for_fuel.commons.add_version import (
    _geometry_bounds,
    _overlaps,
    _parse_feature,
    add_version,
    add_version_to_mtd,
    compute_extent,
    export_mtd,
    get_mtd_from_stream,
)

DATA_DIR = Path("data/pointcloud")
TEST_TILE = "test_semis_2024_0751_6690_LA93_IGN69.laz"
# Extent of TEST_TILE, i.e. the LiDAR HD tile whose north-west corner is (751000, 6690000).
TEST_TILE_EXTENT = (751000.0, 752000.0, 6689000.0, 6690000.0)


def _tile_feature(nw_x: float, nw_y: float, code_mission: str = "22LHDMH") -> dict:
    """Build a WFS feature mimicking a 1 km x 1 km LiDAR HD metadata tile."""
    metadata = {
        "capteur": ["Optech ALTM Galaxy T2000:5060485"],
        "code_mission": code_mission,
        "date_edition": "2025-10-15",
        "coordonnees_NW": f"{int(nw_x // 1000):04d}-{int(nw_y // 1000):04d}",
        "procede_classement": "IGN_AUTO_V5",
        "date_debut_acquisition": "2024-02-15",
    }
    return {
        "type": "Feature",
        "properties": {
            "capteur": '{"Optech ALTM Galaxy T2000:5060485"}',
            "code_mission": code_mission,
            "date_edition": "2025-10-15Z",
            "url_npl": "https://data.geopf.fr/telechargement/whatever.copc.laz",
            "metadata": json.dumps(metadata),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [nw_x, nw_y - 1000],
                    [nw_x + 1000, nw_y - 1000],
                    [nw_x + 1000, nw_y],
                    [nw_x, nw_y],
                    [nw_x, nw_y - 1000],
                ]
            ],
        },
    }


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_compute_extent_single_tile(tmp_path):
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


@pytest.mark.parametrize(
    "geometry, expected",
    [
        ({"type": "Point", "coordinates": [1.0, 2.0]}, (1.0, 1.0, 2.0, 2.0)),
        ({"type": "LineString", "coordinates": [[1.0, 5.0], [3.0, 2.0]]}, (1.0, 3.0, 2.0, 5.0)),
        (
            {"type": "MultiPolygon", "coordinates": [[[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 0.0]]]]},
            (0.0, 2.0, 0.0, 1.0),
        ),
    ],
)
def test_geometry_bounds(geometry, expected):
    assert _geometry_bounds(geometry) == expected


@pytest.mark.parametrize(
    "other, expected",
    [
        ((0.0, 1000.0, 0.0, 1000.0), True),  # identical
        ((500.0, 1500.0, 500.0, 1500.0), True),  # partial overlap
        ((250.0, 750.0, 250.0, 750.0), True),  # contained
        ((1000.0, 2000.0, 0.0, 1000.0), False),  # neighbour sharing an edge
        ((1000.0, 2000.0, 1000.0, 2000.0), False),  # neighbour sharing a corner
        ((1001.0, 2000.0, 0.0, 1000.0), False),  # disjoint
        ((999.995, 1999.995, 0.0, 1000.0), False),  # neighbour shifted by the reprojection noise
    ],
)
def test_overlaps(other, expected):
    assert _overlaps((0.0, 1000.0, 0.0, 1000.0), other) is expected


def test_parse_feature_prefers_the_json_metadata():
    metadata = _parse_feature(_tile_feature(751000.0, 6690000.0))

    # Taken from the JSON metadata, cleaner than the "2025-10-15Z" attribute.
    assert metadata["date_edition"] == "2025-10-15"
    assert metadata["capteur"] == ["Optech ALTM Galaxy T2000:5060485"]
    assert metadata["coordonnees_nw"] == "0751-6690"
    assert metadata["procede_classement"] == "IGN_AUTO_V5"
    # Only carried by the attributes.
    assert metadata["url_npl"].endswith(".copc.laz")
    # Redundant with the flattened keys.
    assert "metadata" not in metadata
    assert metadata["tile_extent"] == TEST_TILE_EXTENT


def test_parse_feature_with_an_unparsable_metadata_attribute(caplog):
    feature = _tile_feature(751000.0, 6690000.0)
    feature["properties"]["metadata"] = "not json"

    metadata = _parse_feature(feature)

    assert "Could not parse" in caplog.text
    # Falls back on the raw attributes.
    assert metadata["date_edition"] == "2025-10-15Z"


def test_get_mtd_from_stream_discards_the_tiles_outside_the_extent(monkeypatch):
    """The service filters on an approximate envelope and returns non-overlapping tiles."""
    features = [
        _tile_feature(750000.0, 6690000.0),  # west neighbour, only shares an edge
        _tile_feature(751000.0, 6690000.0),  # the only tile really covered
        _tile_feature(752000.0, 6690000.0),  # east neighbour, only shares an edge
    ]
    monkeypatch.setattr(
        "lidar_for_fuel.commons.add_version.requests.get",
        lambda *args, **kwargs: FakeResponse({"features": features, "numberMatched": len(features)}),
    )

    metadata = get_mtd_from_stream(TEST_TILE_EXTENT)

    assert [tile["coordonnees_nw"] for tile in metadata] == ["0751-6690"]


def test_get_mtd_from_stream_paginates(monkeypatch):
    pages = [
        {"features": [_tile_feature(751000.0, 6690000.0)], "numberMatched": 2},
        {"features": [_tile_feature(751000.0, 6689000.0)], "numberMatched": 2},
        {"features": [], "numberMatched": 2},
    ]
    start_indexes = []

    def fake_get(url, params, timeout):
        start_indexes.append(params["STARTINDEX"])
        return FakeResponse(pages[len(start_indexes) - 1])

    monkeypatch.setattr("lidar_for_fuel.commons.add_version.requests.get", fake_get)

    metadata = get_mtd_from_stream((751000.0, 752000.0, 6688000.0, 6690000.0))

    assert start_indexes == [0, 1]
    assert [tile["coordonnees_nw"] for tile in metadata] == ["0751-6690", "0751-6689"]


def test_get_mtd_from_stream_with_an_invalid_extent():
    with pytest.raises(ValueError, match="Invalid extent"):
        get_mtd_from_stream((752000.0, 751000.0, 6689000.0, 6690000.0))


def test_add_version_to_mtd_adds_the_package_version_to_each_tile():
    metadata = [
        _parse_feature(_tile_feature(751000.0, 6690000.0)),
        _parse_feature(_tile_feature(751000.0, 6689000.0)),
    ]

    result = add_version_to_mtd(metadata)

    assert result is metadata
    assert [tile["lidar_for_fuel_version"] for tile in metadata] == [__version__, __version__]
    assert metadata[0]["coordonnees_nw"] == "0751-6690"
    assert metadata[1]["coordonnees_nw"] == "0751-6689"


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
    assert list(geodataframe["lidar_for_fuel_version"]) == [__version__, __version__]
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


def test_add_version_exports_the_metadata_of_the_chantier(tmp_path, monkeypatch):
    shutil.copy(DATA_DIR / TEST_TILE, tmp_path / TEST_TILE)
    features = [_tile_feature(751000.0, 6690000.0)]
    monkeypatch.setattr(
        "lidar_for_fuel.commons.add_version.requests.get",
        lambda *args, **kwargs: FakeResponse({"features": features, "numberMatched": len(features)}),
    )
    output_path = tmp_path / "out" / "metadata.gpkg"

    add_version(tmp_path, output_path)

    geodataframe = gpd.read_file(output_path, layer="metadata")
    assert list(geodataframe["coordonnees_nw"]) == ["0751-6690"]
    assert list(geodataframe["lidar_for_fuel_version"]) == [__version__]
    assert geodataframe.crs.to_epsg() == 2154


def test_add_version_without_any_tile(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LAS/LAZ file"):
        add_version(tmp_path, tmp_path / "metadata.gpkg")


def test_get_mtd_from_stream_on_real_service():
    """Integration test: requires network access to data.geopf.fr (IGN Géoplateforme)."""
    metadata = get_mtd_from_stream(TEST_TILE_EXTENT)

    assert len(metadata) == 1
    tile = metadata[0]
    assert tile["coordonnees_nw"] == "0751-6690"
    assert tile["date_debut_acquisition"] == "2024-02-15"
    assert tile["systeme_altimetrique"] == "IGN69"
    assert tile["procede_classement"].startswith("IGN_AUTO")
