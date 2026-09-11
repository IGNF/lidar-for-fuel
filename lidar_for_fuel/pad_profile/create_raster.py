import logging
from typing import Callable

import numpy as np
import pandas as pd

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


def compute_pixel_aggregates(
    points_df: pd.DataFrame,
    global_origin_x: float,
    global_origin_y: float,
    tile_origin_x: float,
    tile_origin_y: float,
    tile_size: float,
    resolution_factor: float,
    aggregation: Callable,
) -> tuple[pd.DataFrame, tuple, float]:
    """Assign a point cloud to the CosiaFrance pixel grid and run an aggregation per pixel.

    Args:
        points_df (pd.DataFrame): DataFrame with raw X, Y coordinates (and the columns
            needed by `aggregation`), covering at least the tile + its buffer.
        global_origin_x (float): X of the CosiaFrance grid anchor (e.g. 98029.75).
        global_origin_y (float): Y of the CosiaFrance grid anchor (north edge, e.g. 6045536.75).
        tile_origin_x (float): X of the tile's own (raw, unaligned) top-left corner.
        tile_origin_y (float): Y of the tile's own (raw, unaligned) top-left corner.
        tile_size (float): Tile side length in map units (e.g. 1000 for a 1 km dalle).
        resolution_factor (float): Pixel size in map units (e.g. 10 for the CosiaFrance grid).
        aggregation (callable): Function run on each pixel's points (e.g. the PAD profile calc).

    Returns:
        tuple: (aggregated, origin_pixel, nb_pixels). `aggregated` is a DataFrame indexed by
            (pixel_y, pixel_x), one column per output band, holding only the pixels where
            `aggregation` returned a result (pixels where it returned None are dropped ->
            NoData once rasterized). `origin_pixel` is the CosiaFrance grid corner (ix, iy)
            this window is anchored on and `nb_pixels` is the tile's side length in pixels;
            together with `global_origin_x`/`global_origin_y` and `resolution_factor`, they
            hold everything needed to build the output raster's affine transform.
    """
    df = points_df.copy()

    def get_pixel_index(x, y, origin_x, origin_y):
        pixel_x = (x - origin_x) // resolution_factor
        pixel_y = (y - origin_y) // resolution_factor + 1  # pixel origin above the upmost point
        return pixel_x, pixel_y

    # --- Step 1: extract the CosiaFrance window -------------------
    # Extract the CosiaFrance grid covering the tile from an origin point and a pixel size.
    # origin_pixel = nearest (floor) CosiaFrance grid corner to the tile's own raw corner
    # (tile_origin_x, tile_origin_y).
    origin_pixel = get_pixel_index(tile_origin_x, tile_origin_y, global_origin_x, global_origin_y)

    # / ! \ nb_pixels is computed from tile_size alone (never from the buffer)
    nb_pixels = tile_size // resolution_factor

    # --- Step 2: assign each point of the cloud to a pixel (ix, iy) --------------------
    # / ! \ Indices are computed relative to the grid's global origin, not the tile's own origin
    df["pixel_x"], df["pixel_y"] = get_pixel_index(df["X"], df["Y"], global_origin_x, global_origin_y)

    # Keep only points whose pixel falls within the tile's window (excluding the
    # buffer): half-open interval so no pixel is ever counted twice between tiles.
    df = df[
        (df["pixel_x"] >= origin_pixel[0])
        & (df["pixel_x"] < origin_pixel[0] + nb_pixels)
        & (df["pixel_y"] <= origin_pixel[1])
        & (df["pixel_y"] > origin_pixel[1] - nb_pixels)
    ]
    if df.empty:
        raise ValueError(
            f"No point of the tile (tile_origin=({tile_origin_x}, {tile_origin_y})) falls "
            "within the extracted CosiaFrance window: check the buffer loaded around the "
            "tile and the coordinates passed to compute_pixel_aggregates."
        )

    # --- Step 3: run the calculation (aggregation) on each pixel's points ---------
    aggregated = df.groupby(["pixel_y", "pixel_x"]).apply(aggregation)
    if isinstance(aggregated, pd.Series):
        aggregated = aggregated.dropna()  # pixels where aggregation returned None -> NoData
        if not aggregated.empty and isinstance(aggregated.iloc[0], dict):
            aggregated = pd.DataFrame(aggregated.tolist(), index=aggregated.index)
        else:
            # `aggregation` returned a scalar (not a dict): a single output band.
            aggregated = aggregated.to_frame(name=aggregated.name or "value")
    if aggregated.empty:
        # Every pixel's `aggregation` returned None (quality guard)
        raise ValueError(
            f"`aggregation` returned no result for the tile (tile_origin=({tile_origin_x}, "
            f"{tile_origin_y})): every pixel failed"
        )

    return aggregated, origin_pixel, nb_pixels
