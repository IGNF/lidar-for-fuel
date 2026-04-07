"""
Module pour associer les trajectoires capteur aux points LiDAR.

Contexte
--------
Lors d'une acquisition LiDAR aéroportée, l'avion enregistre sa position
(Easting, Northing, Elevation) à haute fréquence dans des fichiers de
trajectoire. Chaque point LiDAR est tiré d'un axe de vol identifié par
son champ ``PointSourceId``.

Ce module :

1. Lit la dalle LiDAR et recense les ``PointSourceId`` présents.
2. Associe chaque ``PointSourceId`` au fichier trajectoire correspondant,
   en cherchant le motif ``_axe_<id>`` dans le nom du fichier.
3. Pour chaque cluster de points partageant le même ``PointSourceId``,
   réalise une **interpolation linéaire** de la position capteur à l'instant
   ``GpsTime`` de chaque point LiDAR.
4. Retourne le tableau numpy enrichi des champs ``Easting``, ``Northing``,
   ``Elevation`` (position du capteur) et ``Time`` (timestamp trajectoire
   interpolé, identique à ``GpsTime``).

Format des fichiers trajectoire attendu
---------------------------------------
GeoJSON FeatureCollection de géométries **Point 2D** :
- ``geometry.coordinates`` : [X, Y] en projection métrique (ex. Lambert-93).
- ``properties.z``          : altitude du capteur (float).
- ``properties.timestamp``  : timestamp GPS (float, secondes GPS).

Exemple de nom de fichier : ``20240215_105825_00_axe_12911_axe_12911.json``
→ correspond au ``PointSourceId`` **12911**.
"""

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.lib.recfunctions as rfn
import pdal

logger = logging.getLogger(__name__)

_AXIS_PATTERN = re.compile(r"_axe_(\d+)")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_axis_number(filename: str) -> int | None:
    """Extrait le dernier identifiant d'axe d'un nom de fichier trajectoire.

    Exemple : ``'20240215_105825_00_axe_12911_axe_12911.json'`` → ``12911``.

    Args:
        filename: Nom du fichier (pas le chemin complet).

    Returns:
        Identifiant d'axe (int), ou ``None`` si le motif est absent.
    """
    matches = _AXIS_PATTERN.findall(filename)
    return int(matches[-1]) if matches else None


def _build_axis_index(trajectory_folder: Path) -> dict[int, Path]:
    """Construit un index ``{axe_id: chemin_fichier}`` pour le dossier trajectoire.

    Seuls les fichiers ``.json`` dont le nom contient ``_axe_<id>`` sont indexés.

    Args:
        trajectory_folder: Dossier contenant les fichiers trajectoire JSON.

    Returns:
        Dictionnaire associant chaque identifiant d'axe à son chemin de fichier.
    """
    index: dict[int, Path] = {}
    for path in trajectory_folder.glob("*.json"):
        axis = _extract_axis_number(path.name)
        if axis is not None:
            index[axis] = path
    return index


def _load_trajectory(trajectory_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge un fichier trajectoire JSON et retourne ses colonnes triées par timestamp.

    Les coordonnées X et Y viennent de la géométrie Point 2D.
    La coordonnée Z et le timestamp viennent des properties (``z`` et ``timestamp``).

    Args:
        trajectory_path: Chemin vers le fichier JSON trajectoire.

    Returns:
        Tuple ``(timestamps, eastings, northings, elevations)`` en arrays
        numpy float64, triés par timestamp croissant.

    Raises:
        ValueError: Si le fichier ne contient aucun point valide.
    """
    gdf = gpd.read_file(trajectory_path)

    # Extraction X, Y depuis la géométrie 2D
    eastings = gdf.geometry.x.to_numpy(dtype=np.float64)
    northings = gdf.geometry.y.to_numpy(dtype=np.float64)

    # Extraction Z et timestamp depuis les properties
    elevations = gdf["z"].to_numpy(dtype=np.float64)
    timestamps = gdf["timestamp"].to_numpy(dtype=np.float64)

    if len(timestamps) == 0:
        raise ValueError(f"Aucun point valide dans la trajectoire : {trajectory_path}")

    # Tri par timestamp croissant (requis par np.interp)
    sort_idx = np.argsort(timestamps)
    return timestamps[sort_idx], eastings[sort_idx], northings[sort_idx], elevations[sort_idx]


# ── API publique ───────────────────────────────────────────────────────────────


def find_trajectories_for_tile(
    input_file: str,
    trajectory_folder: str,
) -> dict[int, Path]:
    """Associe les trajectoires GeoJSON aux ``PointSourceId`` d'une dalle LiDAR.

    Lit la dalle pour lister les ``PointSourceId`` uniques, puis cherche dans
    *trajectory_folder* les fichiers dont le nom contient ``_axe_<id>`` avec
    un identifiant correspondant.

    Args:
        input_file: Chemin vers la dalle LiDAR (LAS/LAZ).
        trajectory_folder: Dossier contenant les fichiers trajectoire JSON.

    Returns:
        Dictionnaire ``{PointSourceId: chemin_fichier_trajectoire}``.

    Raises:
        FileNotFoundError: Si *input_file* ou *trajectory_folder* n'existent pas.
        ValueError: Si la dalle ne contient aucun point.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Dalle LiDAR introuvable : {input_file}")

    traj_folder = Path(trajectory_folder)
    if not traj_folder.exists():
        raise FileNotFoundError(f"Dossier trajectoire introuvable : {trajectory_folder}")

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=str(input_path), nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    if len(points) == 0:
        raise ValueError(f"La dalle ne contient aucun point : {input_file}")

    psids: list[int] = np.unique(points["PointSourceId"]).tolist()
    logger.info("%d PointSourceId(s) détecté(s) : %s", len(psids), psids)

    axis_index = _build_axis_index(traj_folder)

    result: dict[int, Path] = {}
    for psid in psids:
        if psid in axis_index:
            result[psid] = axis_index[psid]
            logger.info("PointSourceId %d → %s", psid, axis_index[psid].name)
        else:
            logger.warning("Aucune trajectoire trouvée pour PointSourceId %d.", psid)

    return result


def add_trajectory_to_points(
    input_file: str,
    trajectory_folder: str,
) -> np.ndarray:
    """Enrichit les points LiDAR avec la position interpolée du capteur.

    **Démarche (équivalent Python du code R fourni, avec interpolation linéaire)**

    Étape 1 — Lecture de la dalle
        La dalle LAS/LAZ est lue via PDAL. On récupère un tableau numpy structuré
        contenant tous les champs LiDAR (X, Y, Z, GpsTime, PointSourceId, …).

    Étape 2 — Association PointSourceId → trajectoire
        On liste les ``PointSourceId`` uniques de la dalle. Pour chacun, on
        identifie le fichier trajectoire correspondant grâce au motif
        ``_axe_<id>`` dans son nom (ex. ``_axe_12911``).

    Étape 3 — Chargement de la trajectoire
        Chaque fichier trajectoire est lu avec geopandas. On extrait :
        - X, Y depuis la géométrie Point 2D (projection Lambert-93).
        - Z depuis ``properties.z``.
        - Le timestamp depuis ``properties.timestamp``.
        Les points sont triés par timestamp croissant.

    Étape 4 — Interpolation linéaire par cluster de PointSourceId
        Le code R utilise ``RANN::nn2`` (plus proche voisin, k=1).
        Ici, on utilise ``numpy.interp`` qui réalise une **interpolation
        linéaire** entre les deux instants trajectoire encadrant chaque
        ``GpsTime`` LiDAR :

            alpha = (t_LiDAR - t_avant) / (t_après - t_avant)
            Easting  = E_avant  + alpha * (E_après  - E_avant)
            Northing = N_avant  + alpha * (N_après  - N_avant)
            Elevation= Elev_avant + alpha * (Elev_après - Elev_avant)

        Si ``GpsTime`` est hors de la plage trajectoire, ``numpy.interp``
        retourne la valeur de bord (extrapolation plate).

    Étape 5 — Ajout des champs au tableau
        Les quatre champs ``Easting``, ``Northing``, ``Elevation``, ``Time``
        sont ajoutés au tableau structuré via ``numpy.lib.recfunctions``.

    Args:
        input_file: Chemin vers la dalle LiDAR (LAS 1.4, LAS/LAZ).
        trajectory_folder: Dossier contenant les fichiers trajectoire JSON.

    Returns:
        Tableau numpy structuré identique à l'entrée, enrichi de quatre champs
        float64 supplémentaires :

        - ``Easting``   — Easting du capteur interpolé au GpsTime du point.
        - ``Northing``  — Northing du capteur interpolé.
        - ``Elevation`` — Altitude du capteur interpolée.
        - ``Time``      — Timestamp trajectoire (= GpsTime du point LiDAR).

        Les points sans trajectoire associée reçoivent ``NaN``.

    Raises:
        FileNotFoundError: Si *input_file* ou *trajectory_folder* n'existent pas.
        ValueError: Si la dalle ne contient aucun point.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Dalle LiDAR introuvable : {input_file}")

    traj_folder = Path(trajectory_folder)
    if not traj_folder.exists():
        raise FileNotFoundError(f"Dossier trajectoire introuvable : {trajectory_folder}")

    # ── Étape 1 : Lecture de la dalle ──────────────────────────────────────────
    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=str(input_path), nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    if len(points) == 0:
        raise ValueError(f"La dalle ne contient aucun point : {input_file}")

    logger.info("Dalle lue : %d points.", len(points))

    # ── Étape 2 : Association PointSourceId → trajectoire ─────────────────────
    axis_index = _build_axis_index(traj_folder)
    psids: list[int] = np.unique(points["PointSourceId"]).tolist()
    logger.info("PointSourceId(s) détecté(s) : %s", psids)

    # ── Étapes 3 & 4 : Interpolation par cluster ───────────────────────────────
    n = len(points)
    easting_out = np.full(n, np.nan, dtype=np.float64)
    northing_out = np.full(n, np.nan, dtype=np.float64)
    elevation_out = np.full(n, np.nan, dtype=np.float64)

    for psid in psids:
        if psid not in axis_index:
            logger.warning("Aucune trajectoire pour PointSourceId %d → champs à NaN.", psid)
            continue

        # Étape 3 : chargement et tri de la trajectoire
        traj_times, traj_e, traj_n, traj_z = _load_trajectory(axis_index[psid])
        logger.info("Trajectoire %s : %d points (%.2f → %.2f s).", axis_index[psid].name, len(traj_times), traj_times[0], traj_times[-1])

        # Étape 4 : interpolation linéaire (np.interp = piecewise-linear)
        mask = points["PointSourceId"] == psid
        gps_times = points["GpsTime"][mask]

        # np.interp(x, xp, fp) : interpolation linéaire de fp aux points x,
        # avec xp supposé croissant ; valeurs de bord pour l'extrapolation.
        easting_out[mask] = np.interp(gps_times, traj_times, traj_e)
        northing_out[mask] = np.interp(gps_times, traj_times, traj_n)
        elevation_out[mask] = np.interp(gps_times, traj_times, traj_z)

        logger.info("PointSourceId %d : %d points interpolés.", psid, int(mask.sum()))

    # ── Étape 5 : Ajout des champs au tableau structuré ───────────────────────
    # Time = GpsTime (identique au comportement du code R)
    points = rfn.append_fields(
        points,
        names=["Easting", "Northing", "Elevation", "Time"],
        data=[easting_out, northing_out, elevation_out, points["GpsTime"].copy()],
        dtypes=[np.float64, np.float64, np.float64, np.float64],
        usemask=False,
    )

    return points
