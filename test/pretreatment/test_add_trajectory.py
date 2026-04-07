"""
Tests pour lidar_for_fuel.pretreatment.add_trajectory.

Stratégie
---------
Les tests sont organisés en trois niveaux :

1. **Unitaires sur les helpers internes**
   - ``_extract_axis_number`` : extraction du numéro d'axe depuis un nom de fichier.
   - ``_build_axis_index``    : construction de l'index axe → chemin.
   - ``_load_trajectory``     : chargement et tri d'un fichier trajectoire JSON.

2. **Tests de find_trajectories_for_tile**
   Tests fonctionnels avec données synthétiques (LAS minimal + JSON minimal).

3. **Tests de add_trajectory_to_points**
   - Interpolation linéaire vérifiée analytiquement sur des données synthétiques.
   - PointSourceId sans trajectoire → champs à NaN.
   - Test d'intégration sur les données réelles du projet.
"""

import json
import struct
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pytest
from shapely.geometry import Point

from lidar_for_fuel.pretreatment.add_trajectory import (
    _build_axis_index,
    _extract_axis_number,
    _load_trajectory,
    add_trajectory_to_points,
    find_trajectories_for_tile,
)

# ── Chemins vers les données réelles ──────────────────────────────────────────

_REAL_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69.laz")
_REAL_TRAJ_FOLDER = Path("data/trajectory")

# ── Constantes de référence extraites du fichier axe 12911 ───────────────────
# Feature 0 et 1 du JSON  (vérifiées manuellement)
_T0 = 392030462.7040609
_T1 = 392030462.7090609
_E0, _N0, _Z0 = 733590.749013, 6690173.6463, 1951.2585158654674
_E1, _N1, _Z1 = 733591.144455, 6690173.518372, 1951.2547131003066


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_trajectory_json(path: Path, timestamps: list[float], xs: list[float], ys: list[float], zs: list[float]) -> None:
    """Écrit un GeoJSON trajectoire minimal avec des Points 2D + properties z/timestamp."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"timestamp": t, "z": z},
        }
        for t, x, y, z in zip(timestamps, xs, ys, zs)
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _make_las_file(path: Path, psids: list[int], gps_times: list[float]) -> None:
    """Écrit un fichier LAS minimal avec les PointSourceId et GpsTime donnés."""
    header = laspy.LasHeader(point_format=6)
    las = laspy.LasData(header=header)
    n = len(psids)
    las.x = np.zeros(n, dtype=np.float64)
    las.y = np.zeros(n, dtype=np.float64)
    las.z = np.zeros(n, dtype=np.float64)
    las.point_source_id = np.array(psids, dtype=np.uint16)
    las.gps_time = np.array(gps_times, dtype=np.float64)
    las.write(str(path))


# ── 1. Tests unitaires sur les helpers ────────────────────────────────────────


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("20240215_105825_00_axe_12911_axe_12911.json", 12911),
        ("20240215_105825_00_axe_13011.json", 13011),
        ("no_axis_here.json", None),
        ("_axe_0.json", 0),
    ],
    ids=["double_axe", "single_axe", "no_axe", "axe_zero"],
)
def test_extract_axis_number(filename, expected):
    """Le dernier identifiant _axe_<id> est correctement extrait."""
    assert _extract_axis_number(filename) == expected


def test_build_axis_index(tmp_path):
    """L'index associe correctement chaque identifiant d'axe à son fichier."""
    (tmp_path / "20240215_axe_100_axe_100.json").touch()
    (tmp_path / "20240215_axe_200.json").touch()
    (tmp_path / "not_a_traj.txt").touch()

    index = _build_axis_index(tmp_path)

    assert set(index.keys()) == {100, 200}
    assert index[100].name == "20240215_axe_100_axe_100.json"
    assert index[200].name == "20240215_axe_200.json"


def test_load_trajectory_sorted(tmp_path):
    """La trajectoire est triée par timestamp croissant même si le fichier ne l'est pas."""
    traj_path = tmp_path / "axe_1.json"
    # Points dans l'ordre inverse
    _make_trajectory_json(traj_path, timestamps=[3.0, 1.0, 2.0], xs=[30.0, 10.0, 20.0], ys=[3.0, 1.0, 2.0], zs=[300.0, 100.0, 200.0])

    times, eastings, northings, elevations = _load_trajectory(traj_path)

    np.testing.assert_array_equal(times, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(eastings, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(northings, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(elevations, [100.0, 200.0, 300.0])


# ── 2. Tests de find_trajectories_for_tile ────────────────────────────────────


def test_find_trajectories_matches_psids(tmp_path):
    """Chaque PointSourceId présent dans la dalle est associé à sa trajectoire."""
    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()
    _make_trajectory_json(traj_folder / "flight_axe_100_axe_100.json", [1.0], [0.0], [0.0], [0.0])
    _make_trajectory_json(traj_folder / "flight_axe_200.json", [1.0], [0.0], [0.0], [0.0])

    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[100, 100, 200], gps_times=[1.0, 2.0, 3.0])

    result = find_trajectories_for_tile(str(las_path), str(traj_folder))

    assert set(result.keys()) == {100, 200}
    assert result[100].name == "flight_axe_100_axe_100.json"
    assert result[200].name == "flight_axe_200.json"


def test_find_trajectories_missing_psid_logs_warning(tmp_path, caplog):
    """Un PointSourceId sans trajectoire ne lève pas d'exception mais logue un avertissement."""
    import logging

    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()
    # Aucune trajectoire pour le PointSourceId 999
    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[999], gps_times=[1.0])

    with caplog.at_level(logging.WARNING):
        result = find_trajectories_for_tile(str(las_path), str(traj_folder))

    assert result == {}
    assert any("999" in msg for msg in caplog.messages)


def test_find_trajectories_missing_las_raises(tmp_path):
    """Un chemin LiDAR inexistant lève FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Dalle LiDAR introuvable"):
        find_trajectories_for_tile(str(tmp_path / "nope.laz"), str(tmp_path))


def test_find_trajectories_missing_folder_raises(tmp_path):
    """Un dossier trajectoire inexistant lève FileNotFoundError."""
    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[1], gps_times=[1.0])
    with pytest.raises(FileNotFoundError, match="Dossier trajectoire introuvable"):
        find_trajectories_for_tile(str(las_path), str(tmp_path / "nope"))


# ── 3. Tests de add_trajectory_to_points ─────────────────────────────────────


def test_add_trajectory_linear_interpolation(tmp_path):
    """L'interpolation est bien linéaire : au milieu de deux points trajectoire,
    les coordonnées interpolées sont la moyenne arithmétique."""
    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()

    # Trajectoire avec deux points distants de 1 seconde
    _make_trajectory_json(
        traj_folder / "flight_axe_100_axe_100.json",
        timestamps=[0.0, 1.0],
        xs=[0.0, 10.0],
        ys=[0.0, 20.0],
        zs=[100.0, 200.0],
    )

    # LAS avec un seul point au milieu temporel
    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[100], gps_times=[0.5])

    result = add_trajectory_to_points(str(las_path), str(traj_folder))

    assert "Easting" in result.dtype.names
    assert "Northing" in result.dtype.names
    assert "Elevation" in result.dtype.names
    assert "Time" in result.dtype.names

    np.testing.assert_allclose(result["Easting"][0], 5.0, atol=1e-6)
    np.testing.assert_allclose(result["Northing"][0], 10.0, atol=1e-6)
    np.testing.assert_allclose(result["Elevation"][0], 150.0, atol=1e-6)
    np.testing.assert_allclose(result["Time"][0], 0.5, atol=1e-9)


def test_add_trajectory_multiple_psids(tmp_path):
    """Chaque cluster PointSourceId reçoit les valeurs de sa propre trajectoire."""
    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()

    _make_trajectory_json(traj_folder / "axe_1.json", [0.0, 2.0], [100.0, 200.0], [10.0, 20.0], [1000.0, 2000.0])
    _make_trajectory_json(traj_folder / "axe_2.json", [0.0, 2.0], [500.0, 600.0], [50.0, 60.0], [5000.0, 6000.0])

    las_path = tmp_path / "tile.las"
    # PointSourceId 1 à t=1.0, PointSourceId 2 à t=1.0
    _make_las_file(las_path, psids=[1, 2], gps_times=[1.0, 1.0])

    result = add_trajectory_to_points(str(las_path), str(traj_folder))

    # Axe 1 à t=1 → milieu entre t=0 et t=2 → E=(100+200)/2=150
    np.testing.assert_allclose(result["Easting"][0], 150.0, atol=1e-6)
    # Axe 2 à t=1 → milieu entre t=0 et t=2 → E=(500+600)/2=550
    np.testing.assert_allclose(result["Easting"][1], 550.0, atol=1e-6)


def test_add_trajectory_missing_psid_gives_nan(tmp_path):
    """Un PointSourceId sans trajectoire associée produit des NaN dans les champs ajoutés."""
    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()
    # Pas de fichier pour PointSourceId 999

    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[999], gps_times=[1.0])

    result = add_trajectory_to_points(str(las_path), str(traj_folder))

    assert np.isnan(result["Easting"][0])
    assert np.isnan(result["Northing"][0])
    assert np.isnan(result["Elevation"][0])


def test_add_trajectory_preserves_original_fields(tmp_path):
    """Les champs LiDAR d'origine sont conservés intacts après l'enrichissement."""
    traj_folder = tmp_path / "traj"
    traj_folder.mkdir()
    _make_trajectory_json(traj_folder / "axe_1.json", [0.0, 1.0], [0.0, 10.0], [0.0, 10.0], [0.0, 100.0])

    las_path = tmp_path / "tile.las"
    _make_las_file(las_path, psids=[1], gps_times=[0.5])

    result = add_trajectory_to_points(str(las_path), str(traj_folder))

    # Champs LiDAR attendus (issus de point_format=6)
    for field in ("X", "Y", "Z", "GpsTime", "PointSourceId"):
        assert field in result.dtype.names, f"Champ '{field}' manquant après enrichissement."


@pytest.mark.skipif(not _REAL_LAS.exists() or not _REAL_TRAJ_FOLDER.exists(), reason="Données réelles absentes")
def test_add_trajectory_integration_real_data():
    """Test d'intégration sur la vraie dalle et les vraies trajectoires.

    Vérifie :
    - Tous les PointSourceId connus sont interpolés (pas de NaN).
    - Les valeurs interpolées sont dans des plages géographiques cohérentes
      (Lambert-93, zone France métropolitaine).
    - L'interpolation au milieu exact de deux points consécutifs du JSON
      axe 12911 retourne la moyenne attendue.
    """
    result = add_trajectory_to_points(str(_REAL_LAS), str(_REAL_TRAJ_FOLDER))

    assert "Easting" in result.dtype.names
    assert "Northing" in result.dtype.names
    assert "Elevation" in result.dtype.names
    assert "Time" in result.dtype.names

    # Aucun NaN attendu (les 6 PointSourceIds ont tous une trajectoire)
    assert not np.any(np.isnan(result["Easting"])), "Des NaN inattendus dans Easting."
    assert not np.any(np.isnan(result["Northing"])), "Des NaN inattendus dans Northing."
    assert not np.any(np.isnan(result["Elevation"])), "Des NaN inattendus dans Elevation."

    # Plages géographiques Lambert-93 raisonnables
    assert result["Easting"].min() > 100_000, "Easting hors plage Lambert-93."
    assert result["Easting"].max() < 1_300_000, "Easting hors plage Lambert-93."
    assert result["Northing"].min() > 6_000_000, "Northing hors plage Lambert-93."
    assert result["Northing"].max() < 7_200_000, "Northing hors plage Lambert-93."

    # Vérification analytique : interpolation au milieu de deux points consécutifs
    # de la trajectoire axe 12911 (valeurs vérifiées manuellement).
    t_mid = (_T0 + _T1) / 2.0
    mask_12911 = result["PointSourceId"] == 12911
    closest_idx = np.argmin(np.abs(result["GpsTime"][mask_12911] - t_mid))
    pt = result[mask_12911][closest_idx]

    np.testing.assert_allclose(pt["Easting"], (_E0 + _E1) / 2, rtol=1e-3)
    np.testing.assert_allclose(pt["Northing"], (_N0 + _N1) / 2, rtol=1e-3)
    np.testing.assert_allclose(pt["Elevation"], (_Z0 + _Z1) / 2, rtol=1e-3)
