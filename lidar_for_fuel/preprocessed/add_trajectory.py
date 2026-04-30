
"""
Module to associate sensor trajectories with LiDAR points.

Context
-------
During an airborne LiDAR acquisition, the aircraft records its position
(Easting, Northing, Elevation) at high frequency in trajectory files.
Each LiDAR point is fired from a flight axis identified by its
``PointSourceId`` field.

This module:

1. Lists the unique ``PointSourceId`` values present in the point array.
2. Maps each ``PointSourceId`` to its corresponding trajectory file by
   searching for the pattern ``_axe_<id>`` in the filename.
3. For each cluster of points sharing the same ``PointSourceId``, performs
   **linear interpolation** of the sensor position at the ``GpsTime`` of
   each LiDAR point.
4. Returns the numpy array enriched with the fields ``Easting``, ``Northing``
   and ``Elevation`` (sensor position).

Expected trajectory file format
--------------------------------
GeoJSON FeatureCollection of **2D Point** geometries:
- ``geometry.coordinates`` : [X, Y] in a metric projection (e.g. Lambert-93).
- ``properties.z``          : sensor altitude (float).
- ``properties.timestamp``  : GPS timestamp (float, GPS seconds).

Example filename: ``20240215_105825_00_axe_12911_axe_12911.json``
→ matches ``PointSourceId`` **12911**.
"""

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.lib.recfunctions as rfn

logger = logging.getLogger(__name__)

_AXIS_PATTERN = re.compile(r"_axe_(\d+)")


def _extract_axis_number(filename: str) -> int | None:
    """Extract the last axis identifier from a trajectory filename.

    Example: ``'20240215_105825_00_axe_12911_axe_12911.json'`` → ``12911``.

    Args:
        filename (str): File name (not the full path).

    Returns:
        int: Axis identifier, or ``None`` if the pattern is absent.
    """
    matches = _AXIS_PATTERN.findall(filename)
    return int(matches[-1]) if matches else None


def _build_axis_index(trajectory_folder: Path) -> dict[int, Path]:
    """Build an ``{axis_id: file_path}`` index for the trajectory folder.

    Only ``.json`` files whose name contains ``_axe_<id>`` are indexed.

    Args:
        trajectory_folder (Path): Folder containing the JSON trajectory files.

    Returns:
        dict: Dictionary mapping each axis identifier to its file path.
    """
    index: dict[int, Path] = {}
    for path in trajectory_folder.glob("*.json"):
        axis = _extract_axis_number(path.name)
        if axis is not None:
            index[axis] = path
    return index


def _load_trajectory(trajectory_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a JSON trajectory file and return its columns sorted by timestamp.

    X and Y coordinates come from the 2D Point geometry.
    Z coordinate and timestamp come from the properties (``z`` and ``timestamp``).

    Args:
        trajectory_path (Path): Path to the JSON trajectory file.

    Returns:
        tuple: ``(timestamps, eastings, northings, elevations)`` as float64
        numpy arrays, sorted in ascending timestamp order.

    Raises:
        ValueError: If the file contains no valid points.
    """
    gdf = gpd.read_file(trajectory_path)

    # Extract X, Y from the 2D geometry
    eastings = gdf.geometry.x.to_numpy(dtype=np.float64)
    northings = gdf.geometry.y.to_numpy(dtype=np.float64)

    # Extract Z and timestamp from the properties
    elevations = gdf["z"].to_numpy(dtype=np.float64)
    timestamps = gdf["timestamp"].to_numpy(dtype=np.float64)

    if len(timestamps) == 0:
        raise ValueError(f"No valid points in trajectory file: {trajectory_path}")

    # Sort by ascending timestamp (required by np.interp)
    sort_idx = np.argsort(timestamps)
    return timestamps[sort_idx], eastings[sort_idx], northings[sort_idx], elevations[sort_idx]


def add_trajectory_to_points(
    points: np.ndarray,
    trajectory_folder: str,
) -> np.ndarray:
    """Enrich LiDAR points with the interpolated sensor position.

    Step 1 — PointSourceId → trajectory mapping
        List the unique ``PointSourceId`` values in the array. For each one,
        identify the corresponding trajectory file using the pattern
        ``_axe_<id>`` in its filename (e.g. ``_axe_12911``).

    Step 2 — Trajectory loading
        Each trajectory file is read with geopandas. Extracted fields:
        - X, Y from the 2D Point geometry (Lambert-93 projection).
        - Z from ``properties.z``.
        - Timestamp from ``properties.timestamp``.
        Points are sorted in ascending timestamp order.

    Step 3 — Linear interpolation per PointSourceId cluster
        ``numpy.interp`` performs **piecewise-linear interpolation** between
        the two trajectory instants bracketing each LiDAR ``GpsTime``:

            alpha    = (t_LiDAR - t_before) / (t_after - t_before)
            Easting  = E_before  + alpha * (E_after  - E_before)
            Northing = N_before  + alpha * (N_after  - N_before)
            Elevation= Z_before  + alpha * (Z_after  - Z_before)

        If ``GpsTime`` falls outside the trajectory range, ``numpy.interp``
        returns the boundary value (flat extrapolation).

    Step 4 — Append fields to the structured array
        The three fields ``Easting``, ``Northing``, ``Elevation`` are appended
        to the structured array via ``numpy.lib.recfunctions``.

    Args:
        points: Structured numpy array of LiDAR points (from ``las.points.array``
            or a PDAL pipeline array), containing at least the fields
            ``PointSourceId`` (or ``point_source_id``) and ``GpsTime``
            (or ``gps_time``).
        trajectory_folder: Folder containing the JSON trajectory files.

    Returns:
        np.ndarray: Structured numpy array identical to the input, enriched
        with three additional float64 fields:

        - ``Easting``   — Interpolated sensor easting at the point GpsTime.
        - ``Northing``  — Interpolated sensor northing.
        - ``Elevation`` — Interpolated sensor altitude.

        Points with no associated trajectory receive ``NaN``.

    Raises:
        FileNotFoundError: If *trajectory_folder* does not exist.
    """
    traj_folder = Path(trajectory_folder)
    if not traj_folder.exists():
        raise FileNotFoundError(f"Trajectory folder not found: {trajectory_folder}")

    # Detect naming convention (PDAL = PascalCase, laspy = snake_case)
    psid_field = "PointSourceId" if "PointSourceId" in points.dtype.names else "point_source_id"
    gpstime_field = "GpsTime" if "GpsTime" in points.dtype.names else "gps_time"

    # Step 1: PointSourceId → trajectory mapping
    axis_index = _build_axis_index(traj_folder)
    psids: list[int] = np.unique(points[psid_field]).tolist()
    logger.info("Detected PointSourceId(s): %s", psids)

    # Steps 2 & 3: interpolation per cluster
    n = len(points)
    easting_out = np.full(n, np.nan, dtype=np.float64)
    northing_out = np.full(n, np.nan, dtype=np.float64)
    elevation_out = np.full(n, np.nan, dtype=np.float64)

    for psid in psids:
        if psid not in axis_index:
            logger.warning("No trajectory found for PointSourceId %d → fields set to NaN.", psid)
            continue

        # Step 2: load and sort the trajectory
        traj_times, traj_e, traj_n, traj_z = _load_trajectory(axis_index[psid])
        logger.info("Trajectory %s: %d points (%.2f → %.2f s).", axis_index[psid].name, len(traj_times), traj_times[0], traj_times[-1])

        # Step 3: linear interpolation (np.interp = piecewise-linear)
        mask = points[psid_field] == psid
        gps_times = points[gpstime_field][mask]

        # np.interp(x, xp, fp): linear interpolation of fp at points x,
        # with xp assumed ascending; boundary values used for extrapolation.
        easting_out[mask] = np.interp(gps_times, traj_times, traj_e)
        northing_out[mask] = np.interp(gps_times, traj_times, traj_n)
        elevation_out[mask] = np.interp(gps_times, traj_times, traj_z)

        logger.info("PointSourceId %d: %d points interpolated.", psid, int(mask.sum()))

    # Step 4: append fields to the structured array
    points = rfn.append_fields(
        points,
        names=["Easting", "Northing", "Elevation"],
        data=[easting_out, northing_out, elevation_out],
        dtypes=[np.float64, np.float64, np.float64],
        usemask=False,
    )

    return points
