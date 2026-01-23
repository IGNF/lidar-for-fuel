#!/usr/bin/env python3
"""
Main script for LiDAR file validation in fPC pretreatment pipeline.
Validates single file or all files in directory.
"""
import os
import sys
import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


from lidarforfuel.fPCpretreatment.pointcloud.validate_lidar_file import check_lidar_file

@hydra.main(config_path="../configs/", config_name="configs_lidarforfuel.yaml", version_base="1.2")
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
        input_file = os.path.join(input_dir, filename)  # path to the LAS file
        logging.info(f"\nNormalize and add attributes  of 1 for tile : {tilename}")
        las = check_lidar_file(input_file)
        return las

    if initial_las_filename:
        # Lauch creating mask by one tile:
        las = main_on_one_tile(initial_las_filename)
        print(f"✅ SUCCESS: {len(las.points)} points loaded")
        print(f"   Version: {las.header.version}")
        print(f"   Point format: {las.header.point_format}")

    else:
        # Lauch creating Mask Hydro tile by tile
        for file in os.listdir(input_dir):
            las = main_on_one_tile(file)
            print(f"✅ SUCCESS: {len(las.points)} points loaded")
            print(f"   Version: {las.header.version}")
            print(f"   Point format: {las.header.point_format}")


if __name__ == "__main__":
    main()

