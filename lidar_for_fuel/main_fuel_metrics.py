#!/usr/bin/env python3
"""
Main script for computing the fuel metrics by tiles (step 3 of the LidarForFuel chain).
Runs on a single `{stem}_pad_profile_1m.tif` PAD raster or on all of those of a directory.
"""

import logging
import os

import hydra
from affine import Affine
from omegaconf import DictConfig, OmegaConf

from lidar_for_fuel.fuel_metrics.export_fuel_metrics import export_fuel_metrics
from lidar_for_fuel.fuel_metrics.prepare_cbd_profile import read_fma
from lidar_for_fuel.fuel_metrics.validate_fuel_metrics_params import (
    validate_fuel_metrics_params,
)

logger = logging.getLogger(__name__)

# Name of the PAD raster written by step 2, among its 8 rasters written in the same directory
_PAD_RASTER_SUFFIX = "_pad_profile_1m.tif"


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Compute the fuel metrics of PAD rasters and save each as a `{stem}_fuel_metrics.tif` GeoTIFF.

    It can run either on a single file (`io.input_filename`), or on each `*_pad_profile_1m.tif`
    file of a folder. The `fuel_metrics` parameters are checked once, before the first tile.

    Args:
        config (DictConfig): hydra configuration (configs/config.yaml by default).

    Raises:
        ValueError: If `io.input_dir` or `io.output_dir` is empty, or if a `fuel_metrics`
            parameter is invalid (the message names its key).
        FileNotFoundError: If `io.input_dir` does not exist.
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

    params = OmegaConf.to_container(config.fuel_metrics, resolve=True)
    validate_fuel_metrics_params(**params)
    for key in ("fma_can", "fma_shrub"):
        # A value and a map path are checked here, before the first tile; the map values tile by tile
        if isinstance(params[key], str):
            if not os.path.isfile(params[key]):
                raise ValueError(f"{key}: the FMA map {params[key]} is not an existing file.")
        else:
            try:
                read_fma(params[key], Affine.identity(), (1, 1))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key}: {error}") from error

    os.makedirs(output_dir, exist_ok=True)

    def main_on_one_tile(filename):
        logger.info("Processing tile : %s", os.path.splitext(filename)[0])
        export_fuel_metrics(os.path.join(input_dir, filename), output_dir, **params)

    if config.io.input_filename:
        main_on_one_tile(config.io.input_filename)
    else:
        files = sorted(file for file in os.listdir(input_dir) if file.endswith(_PAD_RASTER_SUFFIX))
        if not files:
            logger.warning(
                "No *%s file in the input directory (%s): nothing to process", _PAD_RASTER_SUFFIX, input_dir
            )
        for file in files:
            main_on_one_tile(file)


if __name__ == "__main__":
    main()
