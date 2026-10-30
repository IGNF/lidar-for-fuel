import json

import laspy
import numpy as np
import pdal
import pytest

from lidar_for_fuel.preprocessing.filter_outliers import remove_outliers

_LAS_DTYPE = np.dtype(
    [
        ("X", np.float64),
        ("Y", np.float64),
        ("Z", np.float64),
        ("h_abg", np.float64),
        ("Intensity", np.uint16),
        ("ReturnNumber", np.uint8),
        ("NumberOfReturns", np.uint8),
        ("Classification", np.uint8),
    ]
)

# Isolated points far from the dense cluster — expected to be detected as outliers
_OUTLIER_XYZ = [(500.0, 500.0, 500.0), (1000.0, 1000.0, 1000.0)]


def _make_point_cloud_with_outliers() -> np.ndarray:
    """10x10 grid at ground altitude 100 m + 2 outliers; h_abg = Z - 100 m."""
    normal = [(float(x), float(y), 100.0) for x in range(10) for y in range(10)]
    all_pts = normal + _OUTLIER_XYZ
    pts = np.zeros(len(all_pts), dtype=_LAS_DTYPE)
    for i, (x, y, z) in enumerate(all_pts):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
        pts[i]["h_abg"] = z - 100.0
        pts[i]["Classification"] = 2
    return pts


def _make_pipeline(pts: np.ndarray) -> pdal.Pipeline:
    pipeline = pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[pts])
    pipeline.execute()
    return pipeline


def test_outliers_are_classified_as_18_and_preserved(tmp_path):
    """Keep points and heights, save/restore Z, and classify only outliers as 18."""
    pts = _make_point_cloud_with_outliers()
    assert np.all(pts["Z"] != pts["h_abg"])

    result = remove_outliers(_make_pipeline(pts), mean_k=5, multiplier=3.0)
    result |= pdal.Writer.las(
        filename=str(tmp_path / "filtered.las"),
        extra_dims="all",
        forward="all",
        minor_version="4",
    )
    result.execute()

    las = laspy.read(str(tmp_path / "filtered.las"))
    out_xyz = list(zip(las.x, las.y, las.z))
    expected_xyz = set(zip(pts["X"], pts["Y"], pts["Z"]))

    # Every input point must be preserved, regardless of output order.
    assert len(las.points) == len(pts)
    assert set(out_xyz) == expected_xyz

    # Check the Z -> Zref -> Z round trip and unchanged normalized heights.
    expected_by_xyz = {(p["X"], p["Y"], p["Z"]): p for p in pts}
    for xyz, zref, height in zip(out_xyz, las["Zref"], las["h_abg"]):
        point = expected_by_xyz[xyz]
        assert zref == point["Z"]
        assert height == point["h_abg"]

    # Only the isolated points must be reclassified; the dense cluster stays in class 2.
    outlier_xyz = set(_OUTLIER_XYZ)
    expected_classification = [18 if xyz in outlier_xyz else 2 for xyz in out_xyz]
    np.testing.assert_array_equal(las.classification, expected_classification)


@pytest.mark.parametrize("isolated_dimension", ["h_abg", "Z"])
def test_outlier_detection_uses_h_abg_and_restores_z(tmp_path, isolated_dimension):
    """Only height-normalized isolation matters; original coordinates survive export."""
    pts = _make_point_cloud_with_outliers()[:-2].copy()
    pts["Z"] = 200.0
    pts[isolated_dimension][-2:] = 500.0
    original = pts.copy()

    result = remove_outliers(_make_pipeline(pts), mean_k=5, multiplier=5.0)
    output = tmp_path / "normalized_outliers.las"
    result |= pdal.Writer.las(filename=str(output), extra_dims="all", minor_version="4")
    result.execute()

    las = laspy.read(output)
    assert len(las.points) == len(original)
    # Match points by XY rather than assuming that PDAL preserves their order.
    expected_by_xy = {(p["X"], p["Y"]): p for p in original}
    for x, y, z, height, classification in zip(las.x, las.y, las.z, las["h_abg"], las.classification):
        point = expected_by_xy.pop((x, y))
        assert z == point["Z"]
        assert height == point["h_abg"]
        assert classification == (18 if point["h_abg"] >= 500.0 else 2)
    assert not expected_by_xy
    np.testing.assert_array_equal(pts, original)


def test_outlier_detection_requires_h_abg():
    """Missing normalized heights must fail rather than silently using raw Z."""
    pts = _make_point_cloud_with_outliers()
    pts = pts[[name for name in pts.dtype.names if name != "h_abg"]].copy()

    with pytest.raises(RuntimeError, match="h_abg"):
        remove_outliers(_make_pipeline(pts)).execute()
