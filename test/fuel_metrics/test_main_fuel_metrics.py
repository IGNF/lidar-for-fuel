import inspect
import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from hydra import compose, initialize

from lidar_for_fuel.fuel_metrics.export_fuel_metrics import export_fuel_metrics
from lidar_for_fuel.main_fuel_metrics import main
from lidar_for_fuel.pad_profile.export_raster import _write_geotiff

# R (lidarforfuel) PAD raster of a 500 m tile (story 3.1): 186 bands, only its PAD_1_* bands
# make a valid step 3 input
_REF_PAD_RASTER_1M = Path(__file__).resolve().parents[2] / "data/ref_inrae/pad_profile/512500_6854997_pad_1m.tif"

_TRANSFORM = Affine(10.0, 0.0, 512489.75, 0.0, -10.0, 6854996.75)
_CRS = "EPSG:2154"


def _config(*overrides):
    """configs/config.yaml with `overrides`, as `python -m lidar_for_fuel.main_fuel_metrics` builds it."""
    with initialize(config_path="../../configs", version_base="1.2"):
        return compose(config_name="config", overrides=list(overrides))


def _write_reference_pad(path):
    """Write the PAD_1_* bands of the R reference PAD raster at `path` (a step 2 like input)."""
    with rasterio.open(_REF_PAD_RASTER_1M) as src:
        bands = [i for i, name in enumerate(src.descriptions, start=1) if name.startswith("PAD_1_")]
        names = [src.descriptions[i - 1] for i in bands]
        pad = src.read(bands)
        transform, crs = src.transform, src.crs
    _write_geotiff(path, dict(zip(names, pad)), transform, crs)
    return path


def _write_small_pad(path):
    """Write a 2 x 3 PAD raster of 8 strata of 1 m."""
    profile = np.array([0.05, 0.05, 0.03, 0.03, 0.0, 0.2, 0.3, 0.1], dtype=np.float32)
    pad = np.repeat(profile[:, None, None], 6, axis=1).reshape(-1, 2, 3)
    _write_geotiff(path, {f"PAD_1_{h}": band for h, band in enumerate(pad)}, _TRANSFORM, _CRS)
    return path


# --- Folder or file (AC-3.7.2, AC-3.7.5) ------------------------------------------------------


def test_main_fuel_metrics_processes_the_pad_rasters_of_a_folder(tmp_path, caplog):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    for stem in ["512500_6854997", "513000_6854997"]:
        _write_reference_pad(input_dir / f"{stem}_pad_profile_1m.tif")
    # Another raster written by step 2 in the same folder: not a step 3 input
    _write_geotiff(input_dir / "512500_6854997_cover.tif", {"Cover_2": np.zeros((2, 3))}, _TRANSFORM, _CRS)

    with caplog.at_level(logging.INFO):
        main(_config(f"io.input_dir={input_dir}", f"io.output_dir={output_dir}"))

    written = sorted(path.name for path in output_dir.iterdir())
    assert written == ["512500_6854997_fuel_metrics.tif", "513000_6854997_fuel_metrics.tif"]
    for name in written:
        with rasterio.open(output_dir / name) as dst:
            assert dst.count == 45  # default config (AC-3.7.5)
            assert not np.isnan(dst.read()).all()
    processed = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Processing tile")]
    assert processed == [
        "Processing tile : 512500_6854997_pad_profile_1m",
        "Processing tile : 513000_6854997_pad_profile_1m",
    ]


def test_main_fuel_metrics_processes_only_the_input_filename(tmp_path):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    _write_small_pad(input_dir / "a_pad_profile_1m.tif")
    _write_small_pad(input_dir / "b_pad_profile_1m.tif")

    main(
        _config(
            f"io.input_dir={input_dir}",
            "io.input_filename=b_pad_profile_1m.tif",
            f"io.output_dir={output_dir}",
        )
    )

    assert [path.name for path in output_dir.iterdir()] == ["b_fuel_metrics.tif"]


def test_main_fuel_metrics_folder_without_pad_raster_logs_a_warning(tmp_path, caplog):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    _write_geotiff(input_dir / "a_cover.tif", {"Cover_2": np.zeros((2, 3))}, _TRANSFORM, _CRS)

    with caplog.at_level(logging.INFO):
        main(_config(f"io.input_dir={input_dir}", f"io.output_dir={output_dir}"))

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [f"No *_pad_profile_1m.tif file in the input directory ({input_dir}): nothing to process"]
    assert not any(output_dir.iterdir())


def test_main_fuel_metrics_passes_the_parameters_on_to_export_fuel_metrics(tmp_path, caplog):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    path = _write_small_pad(input_dir / "a_pad_profile_1m.tif")
    with rasterio.open(path, "r+") as dst:
        pad = dst.read()
        pad[6, 0, 0] = 3.0  # above 2
        dst.write(pad)

    with caplog.at_level(logging.INFO, logger="lidar_for_fuel.fuel_metrics.export_fuel_metrics"):
        main(_config(f"io.input_dir={input_dir}", f"io.output_dir={output_dir}", "fuel_metrics.pad_max=2"))

    summaries = [r.getMessage() for r in caplog.records if r.name == "lidar_for_fuel.fuel_metrics.export_fuel_metrics"]
    assert "PAD values brought back into [0, 2] on read: 1 above 2, 0 +inf, 0 negative" in summaries[0]


# --- io errors: same messages as main_pad_profile (AC-3.7.2) ----------------------------------


def test_main_fuel_metrics_input_dir_null_raises(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"^config.io.input_dir is empty, please provide an input directory in the configuration$",
    ):
        main(_config(f"io.output_dir={tmp_path}"))


def test_main_fuel_metrics_input_dir_missing_raises(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match=rf"^The input directory \({missing}\) doesn't exist.$"):
        main(_config(f"io.input_dir={missing}", f"io.output_dir={tmp_path}"))


def test_main_fuel_metrics_output_dir_null_raises(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"^config.io.output_dir is empty, please provide an output directory in the configuration$",
    ):
        main(_config(f"io.input_dir={tmp_path}"))


# --- Parameters checked at start, before any tile ---------------------------------------------


@pytest.mark.parametrize(
    "override, key",
    [
        ("fuel_metrics.bd_max=0", "bd_max"),
        ("fuel_metrics.pad_max=null", "pad_max"),
        ("fuel_metrics.canopy_top_bd_thresholds=[0.01,0.01004]", "canopy_top_bd_thresholds"),
        ("fuel_metrics.ladder_fuel_bd_thresholds=[]", "ladder_fuel_bd_thresholds"),
        ("fuel_metrics.fma_can=0.6", "fma_can"),
        ("fuel_metrics.fma_shrub=nan", "fma_shrub"),
        ("fuel_metrics.fma_can=null", "fma_can"),
        ("fuel_metrics.fma_shrub=missing.tif", "fma_shrub"),
    ],
)
def test_main_fuel_metrics_invalid_parameter_raises_before_any_tile(tmp_path, caplog, override, key):
    input_dir, output_dir = tmp_path / "in", tmp_path / "out"
    input_dir.mkdir()
    _write_small_pad(input_dir / "a_pad_profile_1m.tif")

    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match=key):
        main(_config(f"io.input_dir={input_dir}", f"io.output_dir={output_dir}", override))

    assert not any(r.getMessage().startswith("Processing tile") for r in caplog.records)
    assert not output_dir.exists() or not any(output_dir.iterdir())


# --- Config contract (review of story 3.4) ----------------------------------------------------


def test_config_fuel_metrics_keys_are_the_parameters_of_export_fuel_metrics():
    parameters = set(inspect.signature(export_fuel_metrics).parameters) - {"pad_path", "output_dir"}

    assert set(_config().fuel_metrics) == parameters
