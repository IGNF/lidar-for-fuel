import json
from pathlib import Path


DATA_DIR = Path("data/pointcloud")
TEST_TILE = "test_semis_2024_0751_6690_LA93_IGN69.laz"
# Extent of TEST_TILE, i.e. the LiDAR HD tile whose north-west corner is (751000, 6690000).
TEST_TILE_EXTENT = (751000.0, 752000.0, 6689000.0, 6690000.0)



class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload



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