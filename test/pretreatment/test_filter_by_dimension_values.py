import json

import laspy
import numpy as np
import pdal
import pytest

from lidar_for_fuel.pretreatment.filter_points_by_dimension_values import (
    filter_by_dimension_values,
)

_TILE_PATH = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
_KEEP_CLASSES = [1, 2, 3, 4, 5, 9]


def _make_pipeline(tile_path: str) -> pdal.Pipeline:
    """Build an executed PDAL pipeline from a real LiDAR tile."""
    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=tile_path, override_srs="EPSG:2154", nosrs=True)
    pipeline.execute()
    return pipeline


@pytest.fixture()
def executed_pipeline() -> pdal.Pipeline:
    return _make_pipeline(_TILE_PATH)


def test_returns_pdal_pipeline(executed_pipeline):
    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    assert isinstance(result, pdal.Pipeline)


def test_pipeline_not_executed(executed_pipeline):
    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    with pytest.raises(RuntimeError):
        _ = result.arrays


def test_pipeline_contains_range_filter(executed_pipeline):
    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    stages = json.loads(result.pipeline)["pipeline"]
    types = [s.get("type") for s in stages if isinstance(s, dict)]
    assert "filters.range" in types


def test_range_filter_limits_correctness(executed_pipeline):
    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    stages = json.loads(result.pipeline)["pipeline"]
    range_stage = next(s for s in stages if isinstance(s, dict) and s.get("type") == "filters.range")
    for cls in _KEEP_CLASSES:
        assert f"Classification[{cls}:{cls}]" in range_stage["limits"]


def test_raises_on_empty_filter_values(executed_pipeline):
    with pytest.raises(ValueError, match="filter_values"):
        filter_by_dimension_values(executed_pipeline, "Classification", [])


def test_only_requested_classes_remain(executed_pipeline, tmp_path):
    """Execute the filtered pipeline and verify via laspy that only requested classes remain."""
    output_path = str(tmp_path / "filtered.las")

    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    result |= pdal.Writer.las(filename=output_path, extra_dims="all", forward="all", minor_version="4")
    result.execute()

    las = laspy.read(output_path)
    unique_classes = set(np.array(las.classification).tolist())
    assert unique_classes.issubset(
        set(_KEEP_CLASSES)
    ), f"Unexpected classification values found: {unique_classes - set(_KEEP_CLASSES)}"


def test_point_count_reduced(executed_pipeline, tmp_path):
    """Filtering should reduce or equal the original point count."""
    output_path = str(tmp_path / "filtered.las")
    original_count = len(executed_pipeline.arrays[0])

    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    result |= pdal.Writer.las(filename=output_path, extra_dims="all", forward="all", minor_version="4")
    result.execute()

    filtered_count = len(laspy.read(output_path).x)
    assert filtered_count <= original_count
