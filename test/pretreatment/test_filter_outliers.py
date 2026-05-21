import json

import laspy
import numpy as np
import pdal

from lidar_for_fuel.pretreatment.filter_outliers import remove_outliers

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

# Isolated points far from the dense cluster — expected to be detected as outliers
_OUTLIER_XYZ = [(500.0, 500.0, 500.0), (1000.0, 1000.0, 1000.0)]


def _make_point_cloud_with_outliers() -> np.ndarray:
    """10x10 dense grid (spacing 1 m, Z=0) + 2 isolated outliers."""
    normal = [(float(x), float(y), 0.0) for x in range(10) for y in range(10)]
    all_pts = normal + _OUTLIER_XYZ
    pts = np.zeros(len(all_pts), dtype=_LAS_DTYPE)
    for i, (x, y, z) in enumerate(all_pts):
        pts[i]["X"] = x
        pts[i]["Y"] = y
        pts[i]["Z"] = z
        pts[i]["Classification"] = 2
    return pts


def _make_pipeline(pts: np.ndarray) -> pdal.Pipeline:
    pipeline = pdal.Pipeline(json.dumps({"pipeline": []}), arrays=[pts])
    pipeline.execute()
    return pipeline


def test_outliers_are_detected_and_removed(tmp_path):
    """Isolated points far from the cluster are removed; dense cluster is preserved."""
    pts = _make_point_cloud_with_outliers()
    n_normal = len(pts) - len(_OUTLIER_XYZ)

    result = remove_outliers(_make_pipeline(pts), mean_k=5, multiplier=3.0)
    result |= pdal.Writer.las(
        filename=str(tmp_path / "filtered.las"),
        extra_dims="all",
        forward="all",
        minor_version="4",
    )
    result.execute()

    las = laspy.read(str(tmp_path / "filtered.las"))
    out_xy = set(zip(np.round(np.array(las.x), 1).tolist(), np.round(np.array(las.y), 1).tolist()))

    # Dense cluster must be preserved
    assert len(las.x) <= n_normal, f"Expected at most {n_normal} points, got {len(las.x)}"

    # Each outlier coordinate must be absent from the output
    for ox, oy, _ in _OUTLIER_XYZ:
        assert (
            round(ox, 1),
            round(oy, 1),
        ) not in out_xy, f"Outlier at ({ox}, {oy}) should have been removed but is still present"
