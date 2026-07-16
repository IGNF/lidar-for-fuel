#!/usr/bin/env python3
"""
Main script for computing PAD (Plant Area Density) metrics by tiles.
Runs on a single pre-processed LAS/LAZ tile or all tiles in a directory.
"""

import logging
import os

import hydra
import numpy as np
import pandas as pd
import pdal
import rasterio
from omegaconf import DictConfig
from rasterio.transform import from_origin

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core
from lidar_for_fuel.pad_profile.validate_lidar_preprocessing_file import (
    check_lidar_file,
)

logger = logging.getLogger(__name__)


def transform_points_coordinates(
    points_df: pd.DataFrame,
    origin_x: float,
    origin_y: float,
    resolution_factor: float,
) -> pd.DataFrame:
    """Apply vectorized coordinate transforms using external origin and resolution data."""
    transformed_df = points_df.copy()

    if "X" in transformed_df.columns:
        # Normalize X coordinates: (X - origin) / pixel_size → pixel space
        transformed_df["X"] = (transformed_df["X"].astype(np.float64) - origin_x) / resolution_factor
    if "Y" in transformed_df.columns:
        # Normalize Y coordinates: (Y - origin) / pixel_size → pixel space
        transformed_df["Y"] = (transformed_df["Y"].astype(np.float64) - origin_y) / resolution_factor

    return transformed_df


def create_raster_from_points(
    points_df: pd.DataFrame,
    origin_x: float,
    origin_y: float,
    resolution_factor: float,
    output_path: str,
    value_column: str = "h_abg",
    aggregation: str = "max",
) -> np.ndarray:
    """Create a GeoTIFF raster from transformed point cloud using pixel indices and aggregated values.

    Args:
        points_df (pd.DataFrame): DataFrame with transformed X, Y coordinates and value column.
        origin_x (float): X origin for GeoTIFF transform.
        origin_y (float): Y origin for GeoTIFF transform.
        resolution_factor (float): Pixel size in map units.
        output_path (str): Path to save the GeoTIFF file.
        value_column (str): Column name to aggregate (default: "h_abg" for height above ground).
        aggregation (str): Aggregation method ("max", "mean", "count", etc.).

    Returns:
        np.ndarray: The raster as a 2D numpy array.
    """
    df = points_df.copy()

    # round values to give int indices
    df["pixel_x"] = np.floor(df["X"]).astype(int)
    df["pixel_y"] = np.floor(df["Y"]).astype(int)

    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in DataFrame.")

    # Aggregate points within each pixel
    if aggregation == "max":
        raster_data = df.groupby(["pixel_y", "pixel_x"])[value_column].max()
    elif aggregation == "mean":
        raster_data = df.groupby(["pixel_y", "pixel_x"])[value_column].mean()
    elif aggregation == "count":
        raster_data = df.groupby(["pixel_y", "pixel_x"]).size()
    else:
        raise ValueError(f"Aggregation method '{aggregation}' not supported.")

    # Create empty raster grid
    n_rows = df["pixel_y"].max() + 1
    n_cols = df["pixel_x"].max() + 1
    raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # Vectorized assignment: extract indices and values, then place in raster
    if len(raster_data) > 0:
        rows, cols = zip(*raster_data.index)
        raster[rows, cols] = raster_data.values

    # Geotransform: top-left corner + pixel size for each dimension (grid Cosia France)
    transform = from_origin(origin_x, origin_y, resolution_factor, resolution_factor)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype=raster.dtype,
        crs="EPSG:2154",
        transform=transform,
    ) as dst:
        dst.write(raster, 1)

    logger.info("Raster saved to %s", output_path)
    return raster


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

    points_df = pd.DataFrame(points)
    x = points_df["X"].to_numpy(dtype=np.float64)
    y = points_df["Y"].to_numpy(dtype=np.float64)
    h_abg = points_df["h_abg"].to_numpy(dtype=np.float64)
    z = points_df["Z"].to_numpy(dtype=np.float64)
    gpstime = points_df["GpsTime"].to_numpy(dtype=np.float64)
    return_number = points_df["ReturnNumber"].to_numpy(dtype=np.float64)
    classification = points_df["Classification"].to_numpy(dtype=np.float64)
    x_sensor = points_df["X_sensor"].to_numpy(dtype=np.float64)
    y_sensor = points_df["Y_sensor"].to_numpy(dtype=np.float64)
    z_sensor = points_df["Z_sensor"].to_numpy(dtype=np.float64)

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

    initial_las_filename = config.io.input_filename

    def main_on_one_tile(filename):
        logging.info(f"\nProcessing tile : {os.path.splitext(filename)[0]}")

        pad_profile_one_tile(
            input_filename=os.path.join(input_dir, filename),
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

    if initial_las_filename:
        main_on_one_tile(initial_las_filename)
    else:
        for file in os.listdir(input_dir):
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
