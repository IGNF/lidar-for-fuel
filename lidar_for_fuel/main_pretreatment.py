#!/usr/bin/env python3
"""
Main script for LiDAR file validation in fPC pretreatment pipeline.
Validates single file or all files in directory.
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from lidar_for_fuel.pretreatment.filter_outliers import remove_outliers
from lidar_for_fuel.pretreatment.filter_points_by_date import filter_by_date
from lidar_for_fuel.pretreatment.filter_points_by_dimension_values import (
    filter_by_dimension_values,
)
from lidar_for_fuel.pretreatment.validate_lidar_file import check_lidar_file

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Normalize and add various attributes of the input LAS/LAZ file and save it as LAS file.

    It can run either on a single file, or on each file of a folder

    Args:
        config (DictConfig): hydra configuration (configs/configs_lidro.yaml by default)
        It contains the algorithm parameters and the input/output parameters
    """
    logging.basicConfig(level=logging.INFO)

    # Check input/output files and folders
    input_dir = config.io.input_dir
    if input_dir is None:
        raise ValueError("""config.io.input_dir is empty, please provide an input directory in the configuration""")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"""The input directory ({input_dir}) doesn't exist.""")

    output_dir = config.io.output_dir
    if output_dir is None:
        raise ValueError("""config.io.output_dir is empty, please provide an input directory in the configuration""")

    os.makedirs(output_dir, exist_ok=True)

    # If input filename is not provided, lidro runs on the whole input_dir directory
    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        """Lauch main.py on one tile

        Args:
            filename (str): filename to the LAS file
        """
        tilename = os.path.splitext(filename)[0]  # filename to the LAS file
        input_filename = os.path.join(input_dir, filename)  # path to the LAS file
        srid = config.io.spatial_reference
        logging.info(f"\nCheck data of 1 for tile : {tilename}")
        pipeline_check_lidar = check_lidar_file(input_filename, srid)

        logging.info(f"\nFilter deviation day of 1 for tile : {tilename}")
        deviation_days = config.pretreatment.filter_date.deviation_days
        gpstime_ref = config.pretreatment.filter_date.gpstime_ref
        pipeline_filter_date = filter_by_date(pipeline_check_lidar, deviation_days, gpstime_ref)

        logging.info(f"\nFilter dimension/values (classfication) of 1 for tile : {tilename}")
        dimension = config.pretreatment.filter.dimension
        values = config.pretreatment.filter.keep_values
        pipeline_filter_dimension = filter_by_dimension_values(pipeline_filter_date, dimension, values)

        logging.info(f"\nFilter outliers of 1 for tile : {tilename}")
        mean_k = config.pretreatment.filter_outlier.mean_k
        multiplier = config.pretreatment.filter_outlier.multiplier
        las = remove_outliers(pipeline_filter_dimension, mean_k, multiplier)

        return las

    if initial_las_filename:
        # Launch pretreatment by one tile:
        main_on_one_tile(initial_las_filename)

    else:
        # Lauch pretreatment tile by tile
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
