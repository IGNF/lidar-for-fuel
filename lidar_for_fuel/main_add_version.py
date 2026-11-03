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

from lidar_for_fuel.add_version.add_version_to_mtd import add_version_to_mtd
from lidar_for_fuel.add_version.compute_extent import compute_extent
from lidar_for_fuel.add_version.download_mtd_from_geoplateforme import (
    get_mtd_from_stream,
)
from lidar_for_fuel.add_version.export_mtd import export_mtd

logger = logging.getLogger(__name__)


def add_version(input_filename: str, output_path: Path, epsg: int = 2154) -> None:
    """Add the lidar_for_fuel version to the metadata of a LAS/LAZ file or of a directory of tiles.
    Args:
        input_filename (str): Path to the input LAS/LAZ file, or to a directory whose tiles
            are all written to the same GeoPackage.
        output_path (Path): Path of the GeoPackage file to write. Parent directories are
            created when missing. An existing file is replaced.
        epsg (int): EPSG code of the tiles and of the written geometries.
        Default: 2154 (Lambert 93).
    """
    logger.info("Adding the lidar_for_fuel version to the metadata of %s", input_filename)

    if os.path.isdir(input_filename):
        las_files = sorted(set(Path(input_filename).glob("*.las")) | set(Path(input_filename).glob("*.laz")))
        if not las_files:
            raise FileNotFoundError(f"No LAS/LAZ file found in {input_filename}")
    else:
        las_files = [input_filename]

    extent = compute_extent(las_files)
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

    output_dir = config.add_version.output_dir
    if output_dir is None:
        raise ValueError(
            "config._add_version.output_dir is empty, please provide an output directory in the configuration"
        )
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"The output directory ({output_dir}) doesn't exist.")

    output_filename = config.add_version.output_filename
    if output_filename is None:
        raise ValueError(
            "config._add_version.output_filename is empty, please provide an output filename in the configuration"
        )

    if not output_filename.endswith(".gpkg"):
        output_filename = output_filename + ".gpkg"

    # Without an input filename, every tile of the input directory goes into a single GeoPackage.
    initial_las_filename = config.io.input_filename
    input_path = os.path.join(input_dir, initial_las_filename) if initial_las_filename else input_dir

    add_version(
        input_filename=input_path,
        output_path=os.path.join(output_dir, output_filename),
        epsg=int(str(config.io.spatial_reference).split(":")[-1]),
    )


if __name__ == "__main__":
    main()
