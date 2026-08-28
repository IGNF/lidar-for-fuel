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
from pdaltools.las_info import get_tile_origin_using_header_info

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core
from lidar_for_fuel.pad_profile.export_raster import export_pad_rasters
from lidar_for_fuel.pad_profile.validate_lidar_preprocessing_file import (
    check_lidar_file,
)

logger = logging.getLogger(__name__)


def pad_profile_one_tile(
    input_filename: str,
    srid: str,
    keep_classes: list,
    scanning_angle: bool,
    limit_N_points: int,
    limit_flight_agl: float,
    deviation_days: int,
    z0: float,
    dz: float,
    nlayers: int | None,
    dz_low: float,
    nlayers_low: int | None,
    ground_margin: float,
    cover_type: str,
    height_cover: float,
    use_cover: bool,
    G: float,
    omega: float,
    keep_values: list,
) -> dict[str, float] | None:
    """Compute PAD metrics for one tile.

    Args:
        input_filename (str): Path to the input LAS/LAZ file.
        srid (str): Spatial reference of the input file. Default: EPSG:2154.
        keep_classes(list): Classes to keep for counting points. Default: [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67].
        scanning_angle (bool): If True, estimate cos(theta) from the trajectory.
                               If False, returns 1.0 (vertical pulses assumed, no correction).
                               Default True.
        limit_N_points (int): Minimum number of point in the pixel/plot for computing profiles & metrics.
                            Default 400 points.
        limit_flight_agl (float):  Limit flight height above ground in m.
                                   If the distance between the flight height and the ground and (Elevation - Zref)
                                   is lower than `limit_flight_agl`, NULL is returned.
                                   Default 800 meters.
        deviation_days (int): Max deviation in days around the local modal acquisition date.
                                Default 0 (only the modal calendar day is retained).
        z0 (float): Bottom height of the first stratum (m). Default 0.
        dz (float): Stratum thickness of the main PAD profile (m). Default 1.
        nlayers (int | None): Number of strata above z0 for the main PAD
            profile. If None, derived from `max(h_abg)`. Default 60.
        dz_low (float): Stratum thickness of the low-strata PAD band (m).
            Default 0.5.
        nlayers_low (int | None): Number of strata above z0 for the
            low-strata PAD band. If None, derived from `max(h_abg)`. Default 4.
        ground_margin (float): Margin above `z0` excluded from the first stratum
            (m). Default 0.1.
        cover_type (str): Either "first" (cover estimated from first returns
            only) or "all" (cover estimated from all returns).
        height_cover (float): Height threshold (m) used for `cover_h_pad`.
        use_cover (bool): If False, `cover_h_pad` is `NaN`; the 2/4/6 m
            cover fractions are always computed.
        G (float): Leaf projection ratio. Default 0.5.
        omega (float): Clumping factor. Default 1.
        keep_values (list): Classes to keep for counting Ni. Default: [2, 3, 4, 5, 9].

    Returns:
        dict[str, float] | None: `None` if a quality guard fails, otherwise a dict
        matching R's named-list output. See `pad_metrics_core`.
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
        dz_low=dz_low,
        nlayers_low=nlayers_low,
        ground_margin=ground_margin,
        cover_type=cover_type,
        height_cover=height_cover,
        use_cover=use_cover,
        G=G,
        omega=omega,
        keep_values=keep_values,
        keep_classes=keep_classes,
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

    output_dir = config.io.output_dir
    if output_dir is None:
        raise ValueError("config.io.output_dir is empty, please provide an output directory in the configuration")
    os.makedirs(output_dir, exist_ok=True)

    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        logging.info(f"\nProcessing tile : {os.path.splitext(filename)[0]}")

        input_filename = os.path.join(input_dir, filename)
        results = pad_profile_one_tile(
            input_filename=input_filename,
            srid=config.io.spatial_reference,
            keep_classes=config.commons.class_count.keep_classes,
            deviation_days=config.commons.filter_date.deviation_days,
            scanning_angle=config.pad_profile.cos_theta.scanning_angle,
            limit_N_points=config.pad_profile.cos_theta.limit_N_points,
            limit_flight_agl=config.pad_profile.cos_theta.limit_flight_agl,
            z0=config.pad_profile.compute_ni_n.z0,
            dz=config.pad_profile.compute_ni_n.dz,
            nlayers=config.pad_profile.compute_ni_n.nlayers,
            dz_low=config.pad_profile.compute_ni_n.dz_low,
            nlayers_low=config.pad_profile.compute_ni_n.nlayers_low,
            ground_margin=config.pad_profile.compute_ni_n.ground_margin,
            cover_type=config.pad_profile.compute_cover.cover_type,
            height_cover=config.pad_profile.compute_cover.height_cover,
            use_cover=config.pad_profile.compute_cover.use_cover,
            G=config.pad_profile.compute_pad.G,
            omega=config.pad_profile.compute_pad.omega,
            keep_values=config.pad_profile.compute_pad.keep_values,
        )

        if results is None:
            logger.warning("No raster exported for %s: quality guard rejected the tile.", input_filename)
            return

        # One tile == one pixel for now (no sub-tile gridding yet): the raster's
        # pixel size is the tile width itself, anchored on the tile's own native origin.
        tile_origin = get_tile_origin_using_header_info(input_filename, tile_width=config.tile_geometry.tile_width)
        pixel_results = np.array([[results]], dtype=object)

        export_pad_rasters(
            pixel_results=pixel_results,
            raster_origin=tile_origin,
            pixel_size=config.tile_geometry.tile_width,
            crs=config.io.spatial_reference,
            output_dir=output_dir,
            tilename=os.path.splitext(filename)[0],
            z0=config.pad_profile.compute_ni_n.z0,
            dz=config.pad_profile.compute_ni_n.dz,
            nlayers=config.pad_profile.compute_ni_n.nlayers,
            dz_low=config.pad_profile.compute_ni_n.dz_low,
            nlayers_low=config.pad_profile.compute_ni_n.nlayers_low,
            keep_classes=config.commons.class_count.keep_classes,
        )

    if initial_las_filename:
        main_on_one_tile(initial_las_filename)
    else:
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
