#!/usr/bin/env python3
"""
Main script for computing the fuel metrics by tiles (step 3 of the LidarForFuel chain).
Runs on a single `{stem}_pad_profile_1m.tif` PAD raster or on all of those of a directory.
"""

import logging
import os

import hydra
from omegaconf import DictConfig

from lidar_for_fuel.fuel_metrics.export_fuel_metrics import export_fuel_metrics

logger = logging.getLogger(__name__)

# Name of the PAD raster written by step 2, among its 8 rasters written in the same directory
_PAD_RASTER_SUFFIX = "_pad_profile_1m.tif"


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig):
    """Compute the fuel metrics of PAD rasters and save each as a `{stem}_fuel_metrics.tif` GeoTIFF.

    It can run either on a single file (`io.input_filename`), or on each `*_pad_profile_1m.tif`
    file of a folder.

    Args:
        config (DictConfig): hydra configuration (configs/config.yaml by default).

    Raises:
        ValueError: If `io.input_dir` or `io.output_dir` is empty.
        FileNotFoundError: If `io.input_dir` or a FMA map (`fuel_metrics.fma_can` or
            `fuel_metrics.fma_shrub` given as a path) does not exist.
    """
    logging.basicConfig(level=logging.INFO)

    input_dir = config.io.input_dir
    if input_dir is None:
        raise ValueError("config.io.input_dir is empty, please provide an input directory in the configuration")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"The input directory ({input_dir}) doesn't exist.")
    for key in ("fma_can", "fma_shrub"):
        # A FMA given as a path is the FMA map, read tile by tile: check it exists before the first tile
        fma = config.fuel_metrics[key]
        if isinstance(fma, str) and not os.path.isfile(fma):
            raise FileNotFoundError(f"The FMA map of config.fuel_metrics.{key} ({fma}) doesn't exist.")

    output_dir = config.io.output_dir
    if output_dir is None:
        raise ValueError("config.io.output_dir is empty, please provide an output directory in the configuration")

    os.makedirs(output_dir, exist_ok=True)

    def main_on_one_tile(filename):
        logger.info("Processing tile : %s", os.path.splitext(filename)[0])
        export_fuel_metrics(
            pad_path=os.path.join(input_dir, filename),
            output_dir=output_dir,
            fma_can=config.fuel_metrics.fma_can,
            fma_shrub=config.fuel_metrics.fma_shrub,
            smooth_pad=config.fuel_metrics.smooth_pad,
            ladder_fuel_bd_thresholds=config.fuel_metrics.ladder_fuel_bd_thresholds,
            canopy_height_threshold=config.fuel_metrics.canopy_height_threshold,
            canopy_bd_thresh_frac=config.fuel_metrics.canopy_bd_thresh_frac,
            canopy_top_bd_thresholds=config.fuel_metrics.canopy_top_bd_thresholds,
            cbh_threshold_ratio=config.fuel_metrics.cbh_threshold_ratio,
            bd_max=config.fuel_metrics.bd_max,
            m_perrakis=config.fuel_metrics.m_perrakis,
            pad_max=config.fuel_metrics.pad_max,
        )

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
