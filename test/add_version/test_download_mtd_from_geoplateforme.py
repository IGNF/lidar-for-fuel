from lidar_for_fuel.add_version.download_mtd_from_geoplateforme import get_mtd_from_stream

from utils import _tile_feature, TEST_TILE_EXTENT, FakeResponse

import pytest



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



