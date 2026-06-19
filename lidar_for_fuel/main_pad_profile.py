#!/usr/bin/env python3
"""
Main script to run write_pad_rasters on one already-preprocessed LAS/LAZ dalle
(or every dalle of a folder), the way main_preprocessing.py runs the preprocessing
pipeline.

Input is the *already-preprocessed* output of main_preprocessing.py: LAS/LAZ with
fields X, Y, Z, GpsTime, ReturnNumber, Classification plus extra dims h_abg,
X_sensor, Y_sensor, Z_sensor.
"""

import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pdal
from omegaconf import DictConfig
from pdaltools.las_info import get_buffered_bounds_from_filename

from lidar_for_fuel.pad_profile.pad_output_grid import PIXEL_SIZE
from lidar_for_fuel.pad_profile.write_pad_rasters import write_pad_rasters

logger = logging.getLogger(__name__)

_LAS_EXTENSIONS = (".las", ".laz")
_REQUIRED_FIELDS = (
    "GpsTime",
    "X",
    "Y",
    "h_abg",
    "Z",
    "ReturnNumber",
    "Classification",
    "X_sensor",
    "Y_sensor",
    "Z_sensor",
)


def process_one_dalle(
    input_filename: str,
    output_dir: str,
    tile_width: int = 1000,
    tile_coord_scale: int = 1000,
    deviation_days: float = np.inf,
    gpstime_ref: str = "2011-09-14 01:46:40",
    **pad_metrics_kwargs,
) -> dict[str, str]:
    """Run write_pad_rasters on one already-preprocessed LAS/LAZ dalle.

    Reads the dalle via PDAL, maps its fields to write_pad_rasters' kwargs, derives
    the dalle's native origin/shape from its filename via pdaltools.las_info, and
    writes the 6 PAD output COG rasters.

    Args:
        input_filename: Path to the preprocessed LAS/LAZ file. Its basename must
            follow the IGN tiling convention expected by
            `pdaltools.las_info.parse_filename` (`{prefix1}_{prefix2}_{coordx}_{coordy}_{suffix}`).
        output_dir: Directory the 6 COG files are written into.
        tile_width: Tile width in metres, used to derive the dalle bounds from its
            filename. Default 1000.
        tile_coord_scale: Scale used in the filename to describe coordinates
            (usually kilometres). Default 1000.
        deviation_days: Max deviation in days around the local modal acquisition
            date, forwarded to `write_pad_rasters`. Default `inf` (no filter).
        gpstime_ref: UTC reference datetime for gpstime=0, forwarded to
            `write_pad_rasters`.
        **pad_metrics_kwargs: Extra keyword arguments forwarded to
            `write_pad_rasters` (and in turn to `pad_metrics_core`), e.g. `G`,
            `omega`, `scanning_angle`, `use_cover`, `cover_type`, `height_cover`,
            `limit_N_points`, `limit_flight_agl`.

    Returns:
        dict[str, str]: Mapping of product name to the written file path (the same
            mapping returned by `write_pad_rasters`).

    Raises:
        ValueError: If the dalle's filename doesn't match the expected IGN tiling
            convention (propagated from `pdaltools.las_info.parse_filename`); if
            `tile_width` is not an exact multiple of `PIXEL_SIZE`; if the input file
            is missing one of the expected preprocessed fields/extra-dims; or if
            `write_pad_rasters`/`pad_metrics_core` reject the inputs.
    """
    if tile_width % PIXEL_SIZE != 0:
        raise ValueError(f"tile_width ({tile_width}) must be an exact multiple of PIXEL_SIZE ({PIXEL_SIZE})")

    logger.info("Processing tile : %s", input_filename)

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=input_filename)
    pipeline.execute()
    points = pipeline.arrays[0]

    missing_fields = [field for field in _REQUIRED_FIELDS if field not in points.dtype.names]
    if missing_fields:
        raise ValueError(
            f"{input_filename} is missing expected preprocessed field(s) {missing_fields}; "
            "it must be the output of main_preprocessing.py."
        )

    # get_buffered_bounds_from_filename returns xs=[minX, maxX], ys=[minY, maxY];
    # the dalle's top-left origin is (minX, maxY).
    xs, ys = get_buffered_bounds_from_filename(
        input_filename, buffer_width=0, tile_width=tile_width, tile_coord_scale=tile_coord_scale
    )
    origin_x = xs[0]
    origin_y = ys[1]
    n_rows = n_cols = int(tile_width / PIXEL_SIZE)

    dalle_name = Path(input_filename).stem

    paths = write_pad_rasters(
        gpstime=points["GpsTime"],
        x=points["X"],
        y=points["Y"],
        h_abg=points["h_abg"],
        z=points["Z"],
        return_number=points["ReturnNumber"],
        classification=points["Classification"],
        x_sensor=points["X_sensor"],
        y_sensor=points["Y_sensor"],
        z_sensor=points["Z_sensor"],
        origin_x=origin_x,
        origin_y=origin_y,
        n_rows=n_rows,
        n_cols=n_cols,
        output_dir=output_dir,
        dalle_name=dalle_name,
        deviation_days=deviation_days,
        gpstime_ref=gpstime_ref,
        **pad_metrics_kwargs,
    )

    logger.info("Wrote PAD rasters for %s: %s", dalle_name, list(paths.values()))

    return paths


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Run write_pad_rasters on one preprocessed LAS/LAZ dalle, or each file of a folder.

    Args:
        config (DictConfig): hydra configuration (configs/config.yaml by default).
    """
    logging.basicConfig(level=logging.INFO)

    input_dir = config.pad_profile.input_dir
    if input_dir is None:
        raise ValueError(
            "config.pad_profile.input_dir is empty, please provide an input directory in the configuration"
        )
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory ({input_dir}) doesn't exist.")

    output_dir = config.pad_profile.output_dir
    if output_dir is None:
        raise ValueError(
            "config.pad_profile.output_dir is empty, please provide an output directory in the configuration"
        )
    os.makedirs(output_dir, exist_ok=True)

    pad_metrics_kwargs = {
        key: config.pad_profile[key]
        for key in (
            "G",
            "omega",
            "scanning_angle",
            "use_cover",
            "cover_type",
            "height_cover",
            "limit_N_points",
            "limit_flight_agl",
        )
        if config.pad_profile[key] is not None
    }

    def main_on_one_dalle(filename):
        process_one_dalle(
            input_filename=os.path.join(input_dir, filename),
            output_dir=output_dir,
            tile_width=config.tile_geometry.tile_width,
            tile_coord_scale=config.tile_geometry.tile_coord_scale,
            deviation_days=config.commons.filter_date.deviation_days,
            gpstime_ref=config.commons.filter_date.gpstime_ref,
            **pad_metrics_kwargs,
        )

    initial_las_filename = config.pad_profile.input_filename

    if initial_las_filename:
        main_on_one_dalle(initial_las_filename)
    else:
        las_files = [file for file in os.listdir(input_dir) if file.lower().endswith(_LAS_EXTENSIONS)]
        if not las_files:
            logger.warning("No %s file found in %s", _LAS_EXTENSIONS, input_dir)
        for file in las_files:
            main_on_one_dalle(file)


if __name__ == "__main__":
    main()
