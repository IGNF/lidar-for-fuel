import logging
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)


def points_to_dataframe(points: np.ndarray) -> pd.DataFrame:
    """Convert a PDAL structured numpy array to a DataFrame."""
    return pd.DataFrame(points)


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
        # origin_y is the north edge (max Y)
        transformed_df["Y"] = (origin_y - transformed_df["Y"].astype(np.float64)) / resolution_factor

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
