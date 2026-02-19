import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import laspy
import numpy as np

from lidar_for_fuel.pretreatment.normalize_height_in_pointcloud import normalize_height

TMP_PATH = Path("./tmp/normalize_height")
SAMPLE_LAS = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
OUTPUT_LAS = TMP_PATH / "normalized.laz"


def setup_module(module):
    """Clean and recreate tmp directory before tests."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def test_normalize_height_in_pointcloud():
    """Test function produces valid normalized LasData object."""
    las_in = laspy.read(SAMPLE_LAS)

    gps_times = las_in.gps_time
    gps_median = np.nanmedian(gps_times)

    # Offset LAS for Adjusted Standard GPS Time
    offset = 1000000000.0
    unix_median = gps_median + offset

    # Conversion UTC
    dt_median = datetime.fromtimestamp(unix_median, tz=UTC)
    print("GPS time median:", dt_median.strftime("%Y-%m-%d %H:%M:%S"))

    classes = las_in.classification
    print("Classes uniques :", np.unique(classes))

    normalize_height(
        SAMPLE_LAS,
        OUTPUT_LAS,
        "EPSG:2154",
        "Classification",
        [1, 2, 3, 4, 5],
        height_filter=60.0,
    )

    assert OUTPUT_LAS.exists(), "Output file created"

    las_out = laspy.read(OUTPUT_LAS)
    assert len(las_out.points) > 0, "Points preserved after processing"
    assert "Z_ref" in las_out.point_format.dimension_names, "Z_ref dimension added"
