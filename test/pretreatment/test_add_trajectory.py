from pathlib import Path

import laspy
import numpy as np
import pytest

from lidar_for_fuel.pretreatment.add_trajectory import (
    add_trajectory_to_points,
)

_REAL_TRAJ_FOLDER = Path("data/trajectory")

# Data for the comparison test against the R reference
_FILTERED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311.laz")
_PRETRAITED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz")


@pytest.mark.skipif(
    not _FILTERED_LAS.exists() or not _PRETRAITED_LAS.exists() or not _REAL_TRAJ_FOLDER.exists(),
    reason="Real data files absent (filtered tile or pre-treated tile missing)",
)
def test_add_trajectory_against_r_reference():
    """Compare our linear interpolation against the R reference (nearest-neighbour).

    The pre-treated file was produced by R code using nearest-neighbour
    interpolation (RANN::nn2 k=1). Our implementation uses linear interpolation
    (numpy.interp). Results should be close but not identical.

    Checks:
    - Fields Easting, Northing, Elevation are present in the result.
    - No NaN values (the PointSourceId in the tile has an associated trajectory).
    - Easting/Northing/Elevation values are within coherent Lambert-93 ranges.
    - The deviation from the R reference is less than 5 m for position and altitude
      (conservative bound given the trajectory sampling frequency).
    """
    # ── Compute our linear interpolation ─────────────────────────────────────
    points = laspy.read(str(_FILTERED_LAS)).points.array
    result = add_trajectory_to_points(points, str(_REAL_TRAJ_FOLDER))

    # ── Load the R reference ──────────────────────────────────────────────────
    las_ref = laspy.read(str(_PRETRAITED_LAS))

    # ── 1. Field presence ─────────────────────────────────────────────────────
    assert "Easting" in result.dtype.names
    assert "Northing" in result.dtype.names
    assert "Elevation" in result.dtype.names

    # ── 2. No NaN values ──────────────────────────────────────────────────────
    assert not np.any(np.isnan(result["Easting"])), "Unexpected NaN in Easting."
    assert not np.any(np.isnan(result["Northing"])), "Unexpected NaN in Northing."
    assert not np.any(np.isnan(result["Elevation"])), "Unexpected NaN in Elevation."

    # ── 3. Coherent Lambert-93 geographic ranges ──────────────────────────────
    assert result["Easting"].min() > 100_000, "Easting out of Lambert-93 range."
    assert result["Easting"].max() < 1_300_000, "Easting out of Lambert-93 range."
    assert result["Northing"].min() > 6_000_000, "Northing out of Lambert-93 range."
    assert result["Northing"].max() < 7_200_000, "Northing out of Lambert-93 range."

    # ── 4. Comparison against the R reference ────────────────────────────────
    ref_easting = np.asarray(las_ref["Easting"])
    ref_northing = np.asarray(las_ref["Northing"])
    ref_elevation = np.asarray(las_ref["Elevation"])

    assert len(result) == len(ref_easting), (
        f"Point count mismatch: result={len(result)}, reference={len(ref_easting)}."
    )

    # Tolerance of 5 m: maximum expected difference between linear interpolation
    # and nearest-neighbour for a trajectory sampled at ~200 Hz (5 ms steps)
    # with an aircraft flying at ~80 m/s (max displacement ~0.4 m per step).
    np.testing.assert_allclose(
        result["Easting"], ref_easting, atol=5.0,
        err_msg="Easting too far from R reference (>5 m).",
    )
    np.testing.assert_allclose(
        result["Northing"], ref_northing, atol=5.0,
        err_msg="Northing too far from R reference (>5 m).",
    )
    np.testing.assert_allclose(
        result["Elevation"], ref_elevation, atol=5.0,
        err_msg="Elevation too far from R reference (>5 m).",
    )
