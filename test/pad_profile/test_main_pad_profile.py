import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio

from lidar_for_fuel.main_pad_profile import pad_profile_one_tile

TMP_PATH = Path("./tmp/cmain_pad_profile")

# CosiaFrance grid anchor and pixel size (configs/config.yaml: pad_profile.create_raster).
_GLOBAL_ORIGIN_X = 98029.75
_GLOBAL_ORIGIN_Y = 6045536.75
_RESOLUTION_FACTOR = 10.0

# File produced by the preprocessing pipeline: has all 4 required extra dims.
PREPROCESSED_LAS = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_preprocessed.laz")

# 3x3 grid of real neighboring tiles, used to exercise real buffer merging (as opposed to
# a single isolated tile copied into an empty directory, which has no neighbors to merge in).
BUFFER_DIR = Path("data/buffer")
CENTRAL_TILE = BUFFER_DIR / "Semis_2022_0691_6484_LA93_IGN69_preprocessed.laz"

# The 8 rasters `export_raster` writes, one GeoTIFF each (see export_raster.raster_columns).
_RASTER_NAMES = (
    "pad_sb_0.5m",
    "pad_profile_1m",
    "class_count",
    "entering_rays",
    "intercept_ray",
    "pl_factor",
    "cover",
    "dates_pad",
)

_PAD_PARAMS = dict(
    keep_classes=[1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67],
    limit_N_points=1,
    limit_flight_agl=0.0,
    deviation_days=36_500,  # ~100 years: wide enough to keep every point, including neighbors'
    scanning_angle=True,
    z0=0.0,
    dz=1.0,
    nlayers=60,
    dz_low=0.5,
    nlayers_low=4,
    ground_margin=0.1,
    cover_type="all",
    height_cover=2.0,
    use_cover=True,
    G=0.5,
    omega=0.77,
    keep_values=[2, 3, 4, 5, 9],
)


def _populated_cells(raster_path: Path, band: int = 1) -> set[tuple[int, int]]:
    """(row, col) of every cell of `band` that carries a value (i.e. is not NoData).

    Args:
        raster_path (Path): GeoTIFF written by `export_raster`.
        band (int): 1-indexed band to read. Default 1.

    Returns:
        set[tuple[int, int]]: Grid positions holding a non-NaN value.
    """
    with rasterio.open(raster_path) as src:
        array = src.read(band)

    rows, cols = np.nonzero(~np.isnan(array))

    return set(zip(rows.tolist(), cols.tolist()))


def test_pad_profile_one_tile_real_las_writes_coherent_rasters(tmp_path):
    """Run pad_profile_one_tile on the real pre-treated LAS and assert the 8 GeoTIFFs are
    written, sized on the tile's CosiaFrance window, with coherent values.

    `pad_profile_one_tile` returns nothing: the rasters on disk are its whole output, so
    every assertion reads them back. Band layout and grid alignment are covered per-raster
    in test_export_raster.py -- here only the end-to-end wiring is checked.

    The test is skipped if the LAS file is not present in the workspace.
    """
    real_las = PREPROCESSED_LAS

    if not real_las.exists():
        pytest.skip(f"Real LAS {real_las} not found in workspace")

    # pdaltools' buffer tile-naming parser expects <prefix1>_<prefix2>_<coordX>_<coordY>_<suffix>
    # (e.g. Semis_2024_0751_6690_...). Strip the fixture's leading "test_" segment so the tile
    # coordinates are parsed from the right fields.
    TMP_PATH.mkdir(parents=True, exist_ok=True)
    tile = TMP_PATH / real_las.name.removeprefix("test_")
    shutil.copy(real_las, tile)

    # Lower quality guards so the pixels pass pad_metrics_core and reach the rasters.
    pad_profile_one_tile(
        input_filename=str(tile),
        input_dir=str(tile.parent),
        buffer_width=100,
        tile_width=1000,
        tile_coord_scale=1000,
        srid="EPSG:2154",
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        output_dir=str(tmp_path),
        **_PAD_PARAMS,
    )

    written = {name: tmp_path / f"{tile.stem}_{name}.tif" for name in _RASTER_NAMES}
    for name, path in written.items():
        assert path.is_file(), f"missing raster {name}"

    with rasterio.open(written["pad_profile_1m"]) as src:
        # nb_pixels = 1000 m tile / 10 m pixels
        assert src.width == 100 and src.height == 100
        assert src.crs.to_string() == "EPSG:2154"
        assert src.count == _PAD_PARAMS["nlayers"]
        pad_profile = src.read()

    with rasterio.open(written["pad_sb_0.5m"]) as src:
        assert src.count == _PAD_PARAMS["nlayers_low"]

    # At least one pixel passed the quality guards, otherwise the run wrote 8 empty grids
    # and every assertion below would hold vacuously.
    assert not np.isnan(pad_profile).all()
    # export_raster clips PAD to [0, 5]; NaN cells are pixels that produced no value.
    assert np.nanmin(pad_profile) >= 0.0
    assert np.nanmax(pad_profile) <= 5.0

    with rasterio.open(written["cover"]) as src:
        assert src.count == 3  # Cover_2, Cover_4, Cover_6
        cover = src.read()

    assert np.nanmin(cover) >= 0.0
    assert np.nanmax(cover) <= 1.0


def test_pad_profile_one_tile_buffered_rasters_cover_more_pixels_than_unbuffered(tmp_path):
    """`compute_pixel_aggregates` windows its output back down to the tile's own
    CosiaFrance footprint, so the buffer no longer inflates the overall point count the
    way the old whole-tile call did (see git history) -- it only fills in the pixels
    that straddle the tile's own boundary (the misalignment between the LiDAR HD tiling
    and the CosiaFrance grid, see test_create_raster.py). Compare buffered vs unbuffered
    (buffer_width=0) runs of the same tile: the buffer must strictly add populated cells
    to the output raster, all of them among the cells the unbuffered run left NoData.

    Skipped if the buffer tile set is not present in the workspace.
    """
    if not CENTRAL_TILE.exists():
        pytest.skip(f"Buffer tile set not found in workspace: {CENTRAL_TILE}")

    unbuffered_dir = tmp_path / "unbuffered"
    buffered_dir = tmp_path / "buffered"

    common_kwargs = dict(
        input_filename=str(CENTRAL_TILE),
        input_dir=str(BUFFER_DIR),
        tile_width=1000,
        tile_coord_scale=1000,
        srid="EPSG:2154",
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        **_PAD_PARAMS,
    )

    pad_profile_one_tile(buffer_width=0, output_dir=str(unbuffered_dir), **common_kwargs)
    pad_profile_one_tile(buffer_width=50, output_dir=str(buffered_dir), **common_kwargs)

    raster_name = f"{CENTRAL_TILE.stem}_pad_profile_1m.tif"
    unbuffered_raster = unbuffered_dir / raster_name
    buffered_raster = buffered_dir / raster_name

    # Both runs window on the tile's own footprint, so the two grids must be superposable
    # -- otherwise comparing (row, col) sets below would be meaningless.
    with rasterio.open(unbuffered_raster) as unbuffered_src, rasterio.open(buffered_raster) as buffered_src:
        assert unbuffered_src.transform == buffered_src.transform
        assert (unbuffered_src.width, unbuffered_src.height) == (buffered_src.width, buffered_src.height)

    unbuffered_cells = _populated_cells(unbuffered_raster)
    buffered_cells = _populated_cells(buffered_raster)

    assert buffered_cells
    assert len(buffered_cells) > len(unbuffered_cells)
    assert unbuffered_cells <= buffered_cells
