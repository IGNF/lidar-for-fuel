from lidar_for_fuel._version import __version__

from lidar_for_fuel.add_version.add_version_to_mtd import add_version_to_mtd
from lidar_for_fuel.add_version.download_mtd_from_geoplateforme import _parse_feature
from utils import _tile_feature


def test_add_version_to_mtd_adds_the_package_version_to_each_tile():
    metadata = [
        _parse_feature(_tile_feature(751000.0, 6690000.0)),
        _parse_feature(_tile_feature(751000.0, 6689000.0)),
    ]

    result = add_version_to_mtd(metadata)

    assert result is metadata
    assert [tile["version_lidarforfuel"] for tile in metadata] == [__version__, __version__]
    assert metadata[0]["coordonnees_nw"] == "0751-6690"
    assert metadata[1]["coordonnees_nw"] == "0751-6689"