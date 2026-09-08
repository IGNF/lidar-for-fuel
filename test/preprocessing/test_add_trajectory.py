import json
from pathlib import Path

import laspy
import numpy as np
import pdal

from lidar_for_fuel.preprocessing.add_trajectory import add_trajectory_to_points

_REAL_TRAJ_FOLDER = Path("data/trajectory")

# Data for the comparison test against the R reference
_FILTERED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311.laz")
_PREPROCESSED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_preprocessed.laz"
)


def _make_points(psid_values: list[int], gps_times: list[float]) -> np.ndarray:
    dtype = np.dtype([("PointSourceId", np.uint16), ("GpsTime", np.float64)])
    arr = np.zeros(len(psid_values), dtype=dtype)
    arr["PointSourceId"] = psid_values
    arr["GpsTime"] = gps_times
    return arr


def _write_trajectory(
    folder: Path,
    axis_id: int,
    timestamps: list[float],
    eastings: list[float],
    northings: list[float],
    elevations: list[float],
) -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [e, n]},
            "properties": {"z": float(z), "timestamp": float(t)},
        }
        for t, e, n, z in zip(timestamps, eastings, northings, elevations)
    ]
    path = folder / f"traj_axe_{axis_id}_axe_{axis_id}.json"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def test_missing_trajectory_produces_nodata(tmp_path):
    """All points with an unmatched PointSourceId must get nodata (0) in all three fields."""
    points = _make_points(psid_values=[99, 99], gps_times=[1000.0, 2000.0])

    result = add_trajectory_to_points(points, str(tmp_path), nodata=0)

    assert "X_sensor" in result.dtype.names
    assert np.all(result["X_sensor"] == 0)
    assert np.all(result["Y_sensor"] == 0)
    assert np.all(result["Z_sensor"] == 0)


def test_mixed_psid_nodata_only_for_missing(tmp_path):
    """Points with a matching trajectory are interpolated; the rest get nodata (0)."""
    psids = [1, 1, 99]
    times = [100.0, 200.0, 500.0]
    points = _make_points(psids, times)

    _write_trajectory(
        tmp_path,
        axis_id=1,
        timestamps=[50.0, 150.0, 250.0],
        eastings=[700_000.0, 700_100.0, 700_200.0],
        northings=[6_500_000.0, 6_500_100.0, 6_500_200.0],
        elevations=[1500.0, 1510.0, 1520.0],
    )

    result = add_trajectory_to_points(points, str(tmp_path), nodata=0)

    mask_known = result["PointSourceId"] == 1
    mask_missing = result["PointSourceId"] == 99

    assert not np.any(result["X_sensor"][mask_known] == 0)
    assert not np.any(result["Y_sensor"][mask_known] == 0)
    assert not np.any(result["Z_sensor"][mask_known] == 0)

    assert np.all(result["X_sensor"][mask_missing] == 0)
    assert np.all(result["Y_sensor"][mask_missing] == 0)
    assert np.all(result["Z_sensor"][mask_missing] == 0)


def test_add_trajectory_against_r_reference():
    """Compare our linear interpolation against the R reference (nearest-neighbour).

    The pre-treated file was produced by R code using nearest-neighbour
    interpolation (RANN::nn2 k=1). Our implementation uses linear interpolation
    (numpy.interp). Results should be close but not identical.

    Checks:
    - Fields X_sensor, Y_sensor, Z_sensor are present in the result.
    - No NaN values (the PointSourceId in the tile has an associated trajectory).
    - X_sensor/Y_sensor/Z_sensor values are within coherent Lambert-93 ranges.
    - The deviation from the R reference is less than 5 m for position and altitude
      (conservative bound given the trajectory sampling frequency).
    """
    # ── Compute our linear interpolation ─────────────────────────────────────
    pipeline = pdal.Reader.las(filename=str(_FILTERED_LAS)).pipeline()
    pipeline.execute()
    points = pipeline.arrays[0]
    result = add_trajectory_to_points(points, str(_REAL_TRAJ_FOLDER))

    # ── Load the R reference ──────────────────────────────────────────────────
    las_ref = laspy.read(str(_PREPROCESSED_LAS))

    # ── 1. Field presence ─────────────────────────────────────────────────────
    assert "X_sensor" in result.dtype.names
    assert "Y_sensor" in result.dtype.names
    assert "Z_sensor" in result.dtype.names

    # ── 2. No NaN values ──────────────────────────────────────────────────────
    assert not np.any(np.isnan(result["X_sensor"])), "Unexpected NaN in X_sensor."
    assert not np.any(np.isnan(result["Y_sensor"])), "Unexpected NaN in Y_sensor."
    assert not np.any(np.isnan(result["Z_sensor"])), "Unexpected NaN in Z_sensor."

    # ── 3. Coherent Lambert-93 geographic ranges ──────────────────────────────
    assert result["X_sensor"].min() > 100_000, "X_sensor out of Lambert-93 range."
    assert result["X_sensor"].max() < 1_300_000, "X_sensor out of Lambert-93 range."
    assert result["Y_sensor"].min() > 6_000_000, "Y_sensor out of Lambert-93 range."
    assert result["Y_sensor"].max() < 7_200_000, "Y_sensor out of Lambert-93 range."

    # ── 4. Comparison against the R reference ────────────────────────────────
    ref_easting = np.asarray(las_ref["X_sensor"])
    ref_northing = np.asarray(las_ref["Y_sensor"])
    ref_elevation = np.asarray(las_ref["Z_sensor"])

    assert len(result) == len(
        ref_easting
    ), f"Point count mismatch: result={len(result)}, reference={len(ref_easting)}."

    # Tolerance of 5 m: maximum expected difference between linear interpolation
    # and nearest-neighbour for a trajectory sampled at ~200 Hz (5 ms steps)
    # with an aircraft flying at ~80 m/s (max displacement ~0.4 m per step).
    np.testing.assert_allclose(
        result["X_sensor"],
        ref_easting,
        atol=5.0,
        err_msg="X_sensor too far from R reference (>5 m).",
    )
    np.testing.assert_allclose(
        result["Y_sensor"],
        ref_northing,
        atol=5.0,
        err_msg="Y_sensor too far from R reference (>5 m).",
    )
    np.testing.assert_allclose(
        result["Z_sensor"],
        ref_elevation,
        atol=5.0,
        err_msg="Z_sensor too far from R reference (>5 m).",
    )
