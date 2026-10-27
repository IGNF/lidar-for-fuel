#!/usr/bin/env python3
"""
Main script for computing PAD (Plant Area Density) metrics by tiles.
Runs on a single pre-processed LAS/LAZ tile or all tiles in a directory.
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from lidar_for_fuel.add_version.compute_extent import compute_extent
from lidar_for_fuel.add_version.download_mtd_from_geoplateforme import get_mtd_from_stream
from lidar_for_fuel.add_version.add_version_to_mtd import add_version_to_mtd
from lidar_for_fuel.add_version.export_mtd import export_mtd



logger = logging.getLogger(__name__)


def add_version(las_dir: Path, output_path: Path, epsg: int = 2154) -> None:
    """Add the lidar_for_fuel version to the metadata of a LAS/LAZ file.
    Args:
        las_dir (Path): Directory containing the LiDAR tiles (``.las`` or ``.laz``).
        output_path (Path): Path of the GeoPackage file to write. Parent directories are
            created when missing. An existing file is replaced.
        epsg (int): EPSG code of the tiles and of the written geometries.
        Default: 2154 (Lambert 93).
    """

    logger.info("Adding the lidar_for_fuel version to the metadata of %s", las_dir)

    extent = compute_extent(las_dir)
    metadata = get_mtd_from_stream(extent, epsg=epsg)
    export_mtd(add_version_to_mtd(metadata), output_path, epsg=epsg)


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """
    Args:
        config (DictConfig): hydra configuration (configs/config.yaml by default).
    """
    logging.basicConfig(level=logging.INFO)

    input_dir = config.io.input_dir
    if input_dir is None:
        raise ValueError("config.io.input_dir is empty, please provide an input directory in the configuration")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory ({input_dir}) doesn't exist.")

    output_dir = config.io.output_dir
    output_filename = config.io.output_filename

    add_version(input_dir, os.path.join(output_dir, output_filename))



if __name__ == "__main__":
    main()
