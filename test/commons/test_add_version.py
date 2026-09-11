"""
Tests for add_version.

The tests of the WFS query itself are offline (the HTTP layer is faked), except
test_get_mtd_from_stream_on_real_service which requires network access to data.geopf.fr
(IGN Géoplateforme).
"""
import json
import shutil
from pathlib import Path

import pytest

from lidar_for_fuel.commons.add_version import (
    _geometry_bounds,
    _overlaps,
    _parse_feature,
    compute_extent,
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


def test_get_mtd_from_stream_on_real_service():
    """Integration test: requires network access to data.geopf.fr (IGN Géoplateforme)."""
    metadata = get_mtd_from_stream(TEST_TILE_EXTENT)

    assert len(metadata) == 1
    tile = metadata[0]
    assert tile["coordonnees_nw"] == "0751-6690"
    assert tile["date_debut_acquisition"] == "2024-02-15"
    assert tile["systeme_altimetrique"] == "IGN69"
    assert tile["procede_classement"].startswith("IGN_AUTO")
