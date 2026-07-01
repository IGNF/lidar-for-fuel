#!/usr/bin/env python3
"""
Main script for computing PAD (Plant Area Density) metrics by tiles.
Runs on a single pre-processed LAS/LAZ tile or all tiles in a directory.
"""

import logging
import os

import hydra
import numpy as np
import pdal
from omegaconf import DictConfig

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core
from lidar_for_fuel.pad_profile.validate_lidar_preprocessing_file import (
    check_lidar_file,
)

logger = logging.getLogger(__name__)


def pad_profile_one_tile(
    input_filename: str,
    srid: str,
    scanning_angle: bool,
    limit_N_points: int,
    limit_flight_agl: float,
    deviation_days: float,
    z0: float,
    dz: float,
    nlayers: int | None,
    ground_margin: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute PAD metrics for one tile.

    Args:
        input_filename (str): Path to the input LAS/LAZ file.
        srid (str): Spatial reference of the input file. Default: EPSG:2154.
        scanning_angle (bool): If True, estimate cos(theta) from the trajectory.
                               If False, returns 1.0 (vertical pulses assumed, no correction).
                               Default True.
        limit_N_points (int): Minimum number of point in the pixel/plot for computing profiles & metrics.
                            Default 400 points.
        limit_flight_agl (float):  Limit flight height above ground in m.
                                   If the distance between the flight height and the ground and (Elevation - Zref)
                                   is lower than `limit_flight_agl`, NULL is returned.
                                   Default 800 meters.
        deviation_days (float): Max deviation in days around the local modal acquisition date.
                                `inf` = no filter.
                                Default `inf`.
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness (m). Default 1.
        nlayers (int | None): Number of strata above z0. If None, derived from
            `max(h_abg)`. Default 60.
        ground_margin (float): Margin above `z0` excluded from the first stratum
            (m). Default 0.1.

    Returns:
        tuple[float, np.ndarray, np.ndarray, np.ndarray] | None: `(cos_theta, ni, n,
        min_layer)`, or None if a quality guard fails. See `pad_metrics_core`.
    """
    # Validate pointclouds after preprocessing
    check_lidar_file(input_filename)

    # Extract pointclouds with attributes
    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=input_filename, override_srs=srid, nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    x = points["X"].astype(np.float64)
    y = points["Y"].astype(np.float64)
    h_abg = points["h_abg"].astype(np.float64)
    z = points["Z"].astype(np.float64)
    gpstime = points["GpsTime"].astype(np.float64)
    return_number = points["ReturnNumber"].astype(np.float64)
    classification = points["Classification"].astype(np.float64)
    x_sensor = points["X_sensor"].astype(np.float64)
    y_sensor = points["Y_sensor"].astype(np.float64)
    z_sensor = points["Z_sensor"].astype(np.float64)

    # Calcule PAD PROFILE by TILE
    results = pad_metrics_core(
        gpstime=gpstime,
        x=x,
        y=y,
        h_abg=h_abg,
        z=z,
        return_number=return_number,
        classification=classification,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        scanning_angle=scanning_angle,
        limit_N_points=limit_N_points,
        limit_flight_agl=limit_flight_agl,
        deviation_days=deviation_days,
        z0=z0,
        dz=dz,
        nlayers=nlayers,
        ground_margin=ground_margin,
    )

    logger.info("Computed PAD metrics by tiles in %s", input_filename)
    return results


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Compute PAD metrics from the input LAS/LAZ file and save it as several RASTER files.

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

    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        logging.info(f"\nProcessing tile : {os.path.splitext(filename)[0]}")

        pad_profile_one_tile(
            input_filename=os.path.join(input_dir, filename),
            srid=config.io.spatial_reference,
            deviation_days=config.commons.filter_date.deviation_days,
            scanning_angle=config.pad_profile.cos_theta.scanning_angle,
            limit_N_points=config.pad_profile.cos_theta.limit_N_points,
            limit_flight_agl=config.pad_profile.cos_theta.limit_flight_agl,
            z0=config.pad_profile.compute_ni_n.z0,
            dz=config.pad_profile.compute_ni_n.dz,
            nlayers=config.pad_profile.compute_ni_n.nlayers,
            ground_margin=config.pad_profile.compute_ni_n.ground_margin,
        )

    if initial_las_filename:
        main_on_one_tile(initial_las_filename)
    else:
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
