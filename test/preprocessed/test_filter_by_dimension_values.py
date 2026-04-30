import laspy
import numpy as np
import pdal
import pytest

from lidar_for_fuel.preprocessed.filter_points_by_dimension_values import (
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
    """filter_by_dimension_values returns a pdal.Pipeline object."""
    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    assert isinstance(result, pdal.Pipeline)


def test_raises_on_empty_filter_values(executed_pipeline):
    """filter_values=[] raises a ValueError."""
    with pytest.raises(ValueError, match="filter_values"):
        filter_by_dimension_values(executed_pipeline, "Classification", [])


def test_only_requested_classes_remain(executed_pipeline, tmp_path):
    """On a real LiDAR tile, only points with the requested classification values remain."""
    output_path = str(tmp_path / "filtered.las")

    result = filter_by_dimension_values(executed_pipeline, "Classification", _KEEP_CLASSES)
    result |= pdal.Writer.las(filename=output_path, extra_dims="all", forward="all", minor_version="4")
    result.execute()

    las = laspy.read(output_path)
    unique_classes = set(np.array(las.classification).tolist())
    assert unique_classes.issubset(
        set(_KEEP_CLASSES)
    ), f"Unexpected classification values found in output: {unique_classes - set(_KEEP_CLASSES)}"
