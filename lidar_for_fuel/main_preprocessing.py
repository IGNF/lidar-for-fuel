#!/usr/bin/env python3
"""
Main script for LiDAR file validation in fPC preprocessed pipeline.
Validates single file or all files in directory.
"""

import logging
import os
import tempfile

import hydra
import pdal
from omegaconf import DictConfig

from lidar_for_fuel.commons.filter_points_by_dimension_values import (
    filter_by_dimension_values,
)
from lidar_for_fuel.preprocessing.add_trajectory import add_trajectory_to_points
from lidar_for_fuel.preprocessing.download_dtm_from_geoplateforme import download_dtm
from lidar_for_fuel.preprocessing.filter_outliers import remove_outliers
from lidar_for_fuel.preprocessing.normalize_height_by_dtm import (
    add_h_abg,
    filter_z_by_height,
)
from lidar_for_fuel.preprocessing.validate_lidar_file import check_lidar_file

from pdaltools.check_las import check_pdal_can_open_file_with_retry_decorator

logger = logging.getLogger(__name__)


@check_pdal_can_open_file_with_retry_decorator(delay=10, filepath="output_path")
def preprocess_one_tile(
    input_filename: str,
    trajectory_dir: str,
    output_path: str,
    srid: str = "EPSG:2154",
    filter_dimension: str = "Classification",
    filter_values: list = [1, 2, 3, 4, 5, 9, 17, 64, 67],
    nodata_value: float = -9999.0,
    min_height_filter: float = -3.0,
    height_filter: float = 80.0,
    trajectory_nodata: int = 0,
    mean_k: int = 5,
    multiplier: float = 10.0,
    dtm_layer: str = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93",
    dtm_epsg: int = 2154,
    tile_width: int = 1000,
    dtm_resolution: float = 0.5,
    dtm_timeout: int = 60,
    dtm_path: str = None,
) -> None:
    """Run the full preprocessing pipeline on one tile.

    If ``dtm_path`` is provided the DTM download is skipped and that file is
    used directly. This allows tests to inject a local DTM without a network
    call while keeping the download step as the default production behaviour.

    Args:
        input_filename: Path to the input LAS/LAZ file.
        trajectory_dir: Folder containing JSON trajectory files.
        output_path: Path for the output LAZ file.
        srid: Spatial reference of the input file. Default: EPSG:2154.
        filter_dimension: LiDAR dimension used for class filtering. Default: Classification.
        filter_values: Classification values to keep (config.preprocessing.filter.keep_values).
        nodata_value: Value assigned to h_abg for NoData DTM pixels. Default: -9999.
        min_height_filter: Minimum height above ground to keep (m). Default: -3.
        height_filter: Maximum height above ground to keep (m). Default: 80.
        trajectory_nodata: Value for sensor fields when no trajectory is found. Default: 0.
        mean_k: Number of nearest neighbours for outlier detection. Default: 5.
        multiplier: Standard deviation multiplier for outlier detection. Default: 10.
        dtm_layer: IGN Géoplateforme layer identifier for DTM download.
        dtm_epsg: EPSG code for DTM download. Default: 2154.
        tile_width: Tile width in metres. Default: 1000.
        dtm_resolution: DTM pixel size in metres. Default: 0.5.
        dtm_timeout: Timeout in seconds for the DTM download. Default: 60.
        dtm_path: Path to an existing DTM GeoTIFF. If provided, skips the download.
    """
    pipeline = check_lidar_file(input_filename, srid)
    pipeline = filter_by_dimension_values(pipeline, filter_dimension, filter_values)
    pipeline.execute()
    points = pipeline.arrays[0]

    filename = os.path.basename(input_filename)
    input_dir = os.path.dirname(input_filename)

    if dtm_path is None:
        with tempfile.TemporaryDirectory() as tmp_dtm_dir:
            dtm_path = download_dtm(
                filename, input_dir, dtm_layer, tmp_dtm_dir, dtm_epsg, tile_width, dtm_resolution, dtm_timeout
            )
            points = add_h_abg(points, dtm_path, nodata_value=nodata_value)
            points = filter_z_by_height(points, min_height_filter, height_filter)
    else:
        points = add_h_abg(points, dtm_path, nodata_value=nodata_value)
        points = filter_z_by_height(points, min_height_filter, height_filter)

    points = add_trajectory_to_points(points, trajectory_dir, trajectory_nodata)

    pipeline_out = pdal.Pipeline(arrays=[points])
    pipeline_out = remove_outliers(pipeline_out, mean_k, multiplier)
    (
        pipeline_out
        | pdal.Writer.las(
            filename=output_path,
            minor_version=4,
            forward="all",
            extra_dims="all",
        )
    ).execute()


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Normalize and add various attributes of the input LAS/LAZ file and save it as LAS file.

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

    trajectory_dir = config.io.input_trajectory_dir
    if trajectory_dir is None:
        raise ValueError(
            "config.io.input_trajectory_dir is empty, please provide an input directory in the configuration"
        )
    if not os.path.isdir(trajectory_dir):
        raise FileNotFoundError(f"The input directory ({trajectory_dir}) doesn't exist.")

    output_dir = config.io.output_dir
    if output_dir is None:
        raise ValueError("config.io.output_dir is empty, please provide an input directory in the configuration")
    os.makedirs(output_dir, exist_ok=True)

    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        logging.info(f"\nProcessing tile : {os.path.splitext(filename)[0]}")
        preprocess_one_tile(
            input_filename=os.path.join(input_dir, filename),
            trajectory_dir=trajectory_dir,
            output_path=os.path.join(output_dir, os.path.splitext(filename)[0] + ".laz"),
            srid=config.io.spatial_reference,
            filter_dimension=config.preprocessing.filter.dimension,
            filter_values=list(config.preprocessing.filter.keep_values),
            nodata_value=config.dtm.nodata_value,
            min_height_filter=config.preprocessing.normalize.min_height_filter,
            height_filter=config.preprocessing.normalize.height_filter,
            trajectory_nodata=config.preprocessing.trajectory.nodata,
            mean_k=config.preprocessing.filter_outlier.mean_k,
            multiplier=config.preprocessing.filter_outlier.multiplier,
            dtm_layer=config.dtm.download.dtm_layer,
            dtm_epsg=config.dtm.download.epsg,
            tile_width=config.tile_geometry.tile_width,
            dtm_resolution=config.dtm.download.resolution,
            dtm_timeout=config.dtm.download.timeout,
        )

    if initial_las_filename:
        main_on_one_tile(initial_las_filename)
    else:
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
