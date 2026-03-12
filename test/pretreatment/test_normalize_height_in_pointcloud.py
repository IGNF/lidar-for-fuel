import json
import os
import shutil
from pathlib import Path

import laspy
import numpy as np
import pdal

from lidar_for_fuel.pretreatment.normalize_height_in_pointcloud import normalize_height

TMP_PATH = Path("./tmp/normalize_height")
SAMPLE_LAS = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
OUTPUT_LAS = TMP_PATH / "normalized.laz"
OUTPUT_DTM_MARKER = TMP_PATH / "normalized_dtm_marker.laz"
OUTPUT_FLAT = TMP_PATH / "normalized_flat.las"
OUTPUT_SLOPE = TMP_PATH / "normalized_slope.las"

# LAS point dtype with minimum required fields
_LAS_DTYPE = np.dtype(
    [
        ("X", np.float64),
        ("Y", np.float64),
        ("Z", np.float64),
        ("Intensity", np.uint16),
        ("ReturnNumber", np.uint8),
        ("NumberOfReturns", np.uint8),
        ("Classification", np.uint8),
    ]
)


def _make_synthetic_pipeline(rows: list) -> pdal.Pipeline:
    """Build an unexecuted PDAL Pipeline from a list of (x, y, z, classification) tuples."""
    pts = np.zeros(len(rows), dtype=_LAS_DTYPE)
    for i, (x, y, z, c) in enumerate(rows):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
        pts[i]["Classification"] = c
    return pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[pts])


def setup_module(module):
    """Clean and recreate tmp directory before tests."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def _make_pipeline():
    return pdal.Reader.las(filename=SAMPLE_LAS, override_srs="EPSG:2154", nosrs=True)


def test_normalize_height_in_pointcloud():
    """Test function produces valid normalized output with Z_ref in expected bounds."""
    normalize_height(
        _make_pipeline(),
        OUTPUT_LAS,
        "Classification",
        [1, 2, 3, 4, 5],
        height_filter=60.0,
        min_height=-3.0,
    )

    assert OUTPUT_LAS.exists(), "Output file created"

    las_out = laspy.read(OUTPUT_LAS)
    assert len(las_out.points) > 0, "Points preserved after processing"
    assert "Z_ref" in las_out.point_format.dimension_names, "Z_ref dimension added"
    assert np.all(las_out.Z_ref >= -3.0), "No points below min_height threshold"
    assert np.all(las_out.Z_ref <= 60.0), "No points above height_filter threshold"


def test_normalize_height_dtm_marker_preserves_classification():
    """When use_dtm_marker=True, Classification must not be altered."""
    normalize_height(
        _make_pipeline(),
        OUTPUT_DTM_MARKER,
        "Classification",
        [1, 2, 3, 4, 5],
        height_filter=60.0,
        min_height=-3.0,
        use_dtm_marker=True,
    )

    las_out = laspy.read(OUTPUT_DTM_MARKER)
    output_classes = np.array(las_out.classification)

    # The test data has only classes [2, 3, 4, 5] — Class 1 must not appear
    assert 1 not in output_classes, "Temporary reclassification leaked into output"
    assert 2 in output_classes, "Ground points (Class 2) must be present in output"
    assert np.all(las_out.Z_ref >= -3.0), "No points below min_height threshold"
    assert np.all(las_out.Z_ref <= 60.0), "No points above height_filter threshold"


def test_normalize_height_flat_ground():
    """On a flat ground at Z=100, Z_ref must equal Z - 100 for each non-ground point.

    Geometry (top view):
        Ground (Class 2) at Z=100 forms a grid covering [0,20]x[0,20].
        Non-ground (Class 3) are placed inside this area at known heights.

        Expected: Z_ref = Z - 100 (exactly, since the TIN reproduces a flat plane).
    """
    ground = [(0, 0, 100, 2), (20, 0, 100, 2), (0, 20, 100, 2), (20, 20, 100, 2), (10, 10, 100, 2)]
    # (x, y, z, class) — expected Z_ref in comment
    non_ground = [
        (5, 5, 103, 3),  # Z_ref = 3.0
        (10, 5, 105, 3),  # Z_ref = 5.0
        (8, 8, 102, 3),  # Z_ref = 2.0
    ]
    expected_z_ref = np.array([3.0, 5.0, 2.0])

    normalize_height(
        _make_synthetic_pipeline(ground + non_ground),
        OUTPUT_FLAT,
        "Classification",
        [2, 3],
        height_filter=60.0,
        min_height=-3.0,
    )

    las_out = laspy.read(OUTPUT_FLAT)
    non_ground_mask = np.array(las_out.classification) == 3
    z_ref_out = np.sort(np.array(las_out.Z_ref)[non_ground_mask])

    np.testing.assert_allclose(z_ref_out, np.sort(expected_z_ref), atol=1e-2)


def test_normalize_height_sloped_ground():
    """On a sloped ground where Z_ground = X, Z_ref must equal Z - X for each non-ground point.

    Geometry:
        Ground (Class 2): Z = X  (slope of 1 m/m along X axis)
        Non-ground (Class 3): Z = X + known_height  → expected Z_ref = known_height

        The TIN exactly reproduces a linear plane, so Z_ref should be constant
        regardless of the (X, Y) position of the non-ground point.
    """
    known_height = 5.0
    ground = [(0, 0, 0, 2), (20, 0, 20, 2), (0, 20, 0, 2), (20, 20, 20, 2), (10, 10, 10, 2)]
    non_ground = [
        (5, 5, 5 + known_height, 3),
        (10, 5, 10 + known_height, 3),
        (8, 8, 8 + known_height, 3),
    ]

    normalize_height(
        _make_synthetic_pipeline(ground + non_ground),
        OUTPUT_SLOPE,
        "Classification",
        [2, 3],
        height_filter=60.0,
        min_height=-3.0,
    )

    las_out = laspy.read(OUTPUT_SLOPE)
    non_ground_mask = np.array(las_out.classification) == 3
    z_ref_out = np.array(las_out.Z_ref)[non_ground_mask]

    np.testing.assert_allclose(z_ref_out, known_height, atol=1e-2)
