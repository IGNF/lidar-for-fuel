#!/usr/bin/env python3
"""
Main script for LiDAR file validation in PAD PROFIL.
Validates single file or all files in directory.
"""

import logging
import os

import numpy as np
import hydra
import pdal
from omegaconf import DictConfig

from lidar_for_fuel.pad_profil.compute_Nz_U import Nz_U
from lidar_for_fuel.pad_profil.validate_lidar_preprocessing_file import check_lidar_file

logger = logging.getLogger(__name__)


def pad_profil_one_tile(
    input_filename: str,
    output_path: str,
    srid: str = "EPSG:2154",
    scanning_angle: bool = True,
    limit_flight_agl: float = 800,
) -> None:
    """Run the full pad profil pipeline on one tile.

    Args:
        input_filename: Path to the input LAS/LAZ file.
        output_path: Path for the output LAZ file.
        srid: Spatial reference of the input file. Default: EPSG:2154.
        scanning_angle: If False, returns 1.0 (vertical pulses assumed, no correction).
        limit_flight_agl: Minimum acceptable mean sensor height above ground (m).
            Below this threshold the trajectory is considered aberrant and
            None is returned with a warning.

    """
    try:
        check_lidar_file(input_filename)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Validation failed for %s: %s — tile skipped.", input_filename, e)
        return

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=input_filename, override_srs=srid, nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    X = points["X"].astype(np.float64)
    Y = points["Y"].astype(np.float64)
    h_abg = points["h_abg"].astype(np.float64)
    X_sensor = points["X_sensor"].astype(np.float64)
    Y_sensor = points["Y_sensor"].astype(np.float64)
    Z_sensor = points["Z_sensor"].astype(np.float64)

    nz_u = Nz_U(X, Y, h_abg, X_sensor, Y_sensor, Z_sensor, scanning_angle, limit_flight_agl)
    if nz_u is None:
        logger.warning("Nz_U could not be computed for %s — tile skipped.", input_filename)
        return


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """ Compute PAD metrics from the input LAS/LAZ file and save it as severals RASTER file.

    It can run either on a single file, or on each file of a folder.

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
    if output_dir is None:
        raise ValueError("config.io.output_dir is empty, please provide an input directory in the configuration")
    os.makedirs(output_dir, exist_ok=True)

    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        logging.info(f"\nProcessing tile : {os.path.splitext(filename)[0]}")
        pad_profil_one_tile(
            input_filename=os.path.join(input_dir, filename),
            output_path=os.path.join(output_dir, os.path.splitext(filename)[0] + ".laz"),
            srid=config.io.spatial_reference,
            nodata_value=config.dtm.nodata_value,
            scanning_angle=config.pad_profil.Nz_U.scanning_angle,
            limit_flight_agl=config.pad_profil.Nz_U.limit_flight_agl,
        )

    if initial_las_filename:
        main_on_one_tile(initial_las_filename)
    else:
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
