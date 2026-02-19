import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pdal

from lidar_for_fuel.pretreatment.filter_deviation_day import filter_points_by_date

logging.basicConfig(level=logging.INFO)

TMP_PATH = Path("./tmp/test_deviation_day")
SAMPLE_LAS = "./data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz"
OUTPUT_LAS = TMP_PATH / "filtered.laz"


def setup_module(module):
    """Setup: clean and create tmp directory."""
    if TMP_PATH.is_dir():
        shutil.rmtree(TMP_PATH)
    os.makedirs(TMP_PATH)


def test_filter_deviation_day():
    # Initialisation du pipeline
    pipeline = pdal.Pipeline() | pdal.Reader.las(
        filename=SAMPLE_LAS, override_srs="epsg:2154", extra_dims="dtm_marker=uint8, dsm_marker=uint8", nosrs=True
    )
    pipeline.execute()
    arrays = pipeline.arrays

    num_points = len(arrays[0])
    print(f"Initial PDAL points: {num_points}")

    # Affichage des stats GpsTime
    if len(arrays) > 0:
        dims = arrays[0].dtype.names
        print(f"Available dimensions: {list(dims)}")

        if "GpsTime" in dims:
            GpsTime = arrays[0]["GpsTime"]
            print(f"GpsTime min: {np.min(GpsTime):.2f}")
            print(f"GpsTime max: {np.max(GpsTime):.2f}")
            print(f"GpsTime mean: {np.mean(GpsTime):.2f}")
            print(f"Number of unique GpsTime values: {np.unique(GpsTime).size}")
        else:
            print("No 'GpsTime' dimension found.")
    else:
        print("No arrays generated.")

    # Appel de la fonction de filtrage
    filtered_pipeline = filter_points_by_date(pipeline, 2)
    filtered_pipeline.execute()
    filtered_arrays = filtered_pipeline.arrays
    num_filtered = len(filtered_arrays[0])
    print(f"Nombre de points filtrés: {num_filtered}")

    assert num_filtered <= num_points
