from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pytest
import rasterio

from lidar_for_fuel.pad_profile.create_raster import (
    build_pad_aggregation,
    compute_pixel_aggregates,
)
from lidar_for_fuel.pad_profile.export_raster import export_raster

_GLOBAL_ORIGIN_X = 98029.75
_GLOBAL_ORIGIN_Y = 6045536.75
_RESOLUTION_FACTOR = 10.0
_TILE_SIZE = 20.0  # 2x2 pixels at resolution_factor=10.0
_SRID = "EPSG:2154"

# Real 1 km tile + its 30 m buffer (same fixture as test_create_raster.py's
# real_buffered_tile test): raw tile corner (691000, 6484000), CosiaFrance origin
# (59297, 43847) -- see that test's docstring for how these are derived.
_REAL_LAS = (
    Path(__file__).resolve().parents[2] / "data/pointcloud/Semis_2022_0691_6484_LA93_IGN69_preprocessed_30m.las"
)

_PAD_PARAMS = dict(
    scanning_angle=False,
    limit_N_points=1,
    limit_flight_agl=0.0,
    deviation_days=36_500,
    z0=0.0,
    dz=1.0,
    nlayers=2,
    dz_low=0.5,
    nlayers_low=1,
    ground_margin=0.0,
    cover_type="all",
    height_cover=2.0,
    use_cover=False,
    G=0.5,
    omega=1.0,
    keep_values=[2, 3, 4, 5, 9],
    keep_classes=[2, 3, 4, 5, 9],
)


def _pixel_points_df(x: float, y: float, n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GpsTime": np.full(n, 100.0),
            "X": np.full(n, x),
            "Y": np.full(n, y),
            "h_abg": np.linspace(0.5, 3.0, n),
            "Z": np.full(n, 200.0),
            "ReturnNumber": np.full(n, 1.0),
            "Classification": np.full(n, 4.0),
            "X_sensor": np.zeros(n),
            "Y_sensor": np.zeros(n),
            "Z_sensor": np.zeros(n),
        }
    )


@pytest.fixture(scope="module")
def aggregated_single_pixel():
    """One populated pixel (south-west corner) in a 2x2-pixel tile aligned on the grid origin."""
    points_df = _pixel_points_df(_GLOBAL_ORIGIN_X + 1.0, _GLOBAL_ORIGIN_Y - 1.0)
    aggregated, origin_pixel, nb_pixels = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=_GLOBAL_ORIGIN_X,
        tile_origin_y=_GLOBAL_ORIGIN_Y,
        tile_size=_TILE_SIZE,
        resolution_factor=_RESOLUTION_FACTOR,
        aggregation=build_pad_aggregation(**_PAD_PARAMS),
    )
    return aggregated, origin_pixel, nb_pixels


@pytest.fixture(scope="module")
def exported_single_pixel(tmp_path_factory, aggregated_single_pixel):
    """The 8 GeoTIFFs `aggregated_single_pixel` exports to, written once for the module.

    Every test below that inspects the *unmodified* export reads these same files, so the
    call is hoisted here rather than repeated per test (the parametrized band-count test
    alone used to re-write all 8 rasters 8 times over). Tests that need a different input
    -- value clipping, `pl_factor` passthrough -- call `export_raster` themselves.

    Returns:
        tuple[dict[str, Path], Path]: raster name -> written path, and their directory.
    """
    aggregated, origin_pixel, nb_pixels = aggregated_single_pixel
    output_dir = tmp_path_factory.mktemp("export_raster_single_pixel")

    written = export_raster(
        aggregated,
        origin_pixel=origin_pixel,
        nb_pixels=nb_pixels,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        dz=_PAD_PARAMS["dz"],
        dz_low=_PAD_PARAMS["dz_low"],
        srid=_SRID,
        output_dir=output_dir,
        tile_stem="test_tile",
    )

    return written, output_dir


def test_export_raster_writes_eight_georeferenced_geotiffs(exported_single_pixel):
    """The 8 expected files are written under `tile_stem`, and they are georeferenced:
    float32 with NaN nodata, on `_SRID`, anchored on the CosiaFrance grid corner."""
    written, output_dir = exported_single_pixel

    assert set(written) == {
        "pad_sb_0.5m",
        "pad_profile_1m",
        "class_count",
        "entering_rays",
        "intercept_ray",
        "pl_factor",
        "cover",
        "dates_pad",
    }
    for name, path in written.items():
        assert path == output_dir / f"test_tile_{name}.tif"
        assert path.is_file()

    with rasterio.open(written["pad_profile_1m"]) as src:
        assert src.dtypes[0] == "float32"
        assert np.isnan(src.nodata)
        assert src.crs.to_string() == _SRID
        assert src.width == 2 and src.height == 2
        # origin_pixel=(0, 1): top-left corner is (global_origin_x, global_origin_y + 1 * res)
        assert src.transform.c == pytest.approx(_GLOBAL_ORIGIN_X)
        assert src.transform.f == pytest.approx(_GLOBAL_ORIGIN_Y + _RESOLUTION_FACTOR)
        assert src.transform.a == pytest.approx(_RESOLUTION_FACTOR)
        assert src.transform.e == pytest.approx(-_RESOLUTION_FACTOR)


@pytest.mark.parametrize(
    "raster_name,expected_band_count",
    [
        ("pad_sb_0.5m", 1),  # nlayers_low=1
        ("pad_profile_1m", 2),  # nlayers=2
        ("class_count", 6),  # 5 keep_classes + Total
        ("entering_rays", 2),
        ("intercept_ray", 2),
        ("pl_factor", 1),
        ("cover", 3),
        ("dates_pad", 3),
    ],
)
def test_export_raster_band_count_matches_spec(exported_single_pixel, raster_name, expected_band_count):
    written, _ = exported_single_pixel

    with rasterio.open(written[raster_name]) as src:
        assert src.count == expected_band_count


def test_export_raster_places_pixel_value_at_correct_row_col_and_leaves_rest_nodata(
    exported_single_pixel, aggregated_single_pixel
):
    """The single populated pixel (south-west of the tile) lands at row=1, col=0; every
    other cell of the 2x2 grid stays NaN (NoData)."""
    aggregated, _, _ = aggregated_single_pixel
    expected_value = aggregated["PAD_1_0"].iloc[0]
    written, _ = exported_single_pixel

    with rasterio.open(written["pad_profile_1m"]) as src:
        band = src.read(1)  # PAD_1_0 is the first stratum band
        assert band.shape == (2, 2)
        assert band[1, 0] == pytest.approx(expected_value)
        untouched = np.delete(band.flatten(), 2)  # flat index of (1,0) in a 2x2 array
        assert np.isnan(untouched).all()


def test_export_raster_caps_pad_values_at_five(tmp_path, aggregated_single_pixel):
    aggregated, origin_pixel, nb_pixels = aggregated_single_pixel
    aggregated = aggregated.copy()
    aggregated["PAD_1_0"] = 42.0  # simulate an out-of-range PAD value

    written = export_raster(
        aggregated,
        origin_pixel=origin_pixel,
        nb_pixels=nb_pixels,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        dz=_PAD_PARAMS["dz"],
        dz_low=_PAD_PARAMS["dz_low"],
        srid=_SRID,
        output_dir=tmp_path,
        tile_stem="test_tile",
    )

    with rasterio.open(written["pad_profile_1m"]) as src:
        band = src.read(1)
        assert np.nanmax(band) == pytest.approx(5.0)


def test_export_raster_pl_factor_passes_through_unchanged(tmp_path, aggregated_single_pixel):
    """export_raster only reorganizes/exports existing columns: `pl_factor` (1 /
    cos_theta) is already computed upstream by `pad_metrics_core`, not re-derived
    here from `cos_theta`."""
    aggregated, origin_pixel, nb_pixels = aggregated_single_pixel
    aggregated = aggregated.copy()
    aggregated["pl_factor"] = 2.0

    written = export_raster(
        aggregated,
        origin_pixel=origin_pixel,
        nb_pixels=nb_pixels,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=_RESOLUTION_FACTOR,
        dz=_PAD_PARAMS["dz"],
        dz_low=_PAD_PARAMS["dz_low"],
        srid=_SRID,
        output_dir=tmp_path,
        tile_stem="test_tile",
    )

    with rasterio.open(written["pl_factor"]) as src:
        band = src.read(1)
        assert band[1, 0] == pytest.approx(2.0)


def test_export_raster_band_descriptions_match_column_names(exported_single_pixel):
    written, _ = exported_single_pixel

    with rasterio.open(written["pad_profile_1m"]) as src:
        assert src.descriptions == ("PAD_1_0", "PAD_1_1")
    with rasterio.open(written["dates_pad"]) as src:
        assert src.descriptions == ("Date_maj", "Date_min", "Date_max")


def test_export_raster_real_las_tile_writes_inspectable_geotiffs():
    """Run the full PAD pipeline + export_raster on the real 0691/6484 tile.

    Unlike the other tests here, this writes to a persistent `./tmp` folder (not
    pytest's auto-cleaned `tmp_path`) so the resulting GeoTIFFs can be opened in QGIS
    afterward to visually check alignment and values -- e.g. `pad_profile_1m` against
    the tile's known vegetation cover.
    """
    if not _REAL_LAS.is_file():
        pytest.skip(f"Real LAS fixture not found in workspace: {_REAL_LAS}")

    cloud = laspy.read(_REAL_LAS)
    points_df = pd.DataFrame(
        {
            "GpsTime": np.asarray(cloud.gps_time),
            "X": np.asarray(cloud.x),
            "Y": np.asarray(cloud.y),
            "h_abg": np.asarray(cloud.h_abg),
            "Z": np.asarray(cloud.z),
            "ReturnNumber": np.asarray(cloud.return_number),
            "Classification": np.asarray(cloud.classification),
            "X_sensor": np.asarray(cloud.X_sensor),
            "Y_sensor": np.asarray(cloud.Y_sensor),
            "Z_sensor": np.asarray(cloud.Z_sensor),
        }
    )

    # Same lenient guards as test_main_pad_profile.py's real-LAS test: only relax the
    # quality gates, keep every other PAD parameter at its production default.
    real_pad_params = dict(
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

    aggregated, origin_pixel, nb_pixels = compute_pixel_aggregates(
        points_df,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        tile_origin_x=691000.0,
        tile_origin_y=6484000.0,
        tile_size=1000.0,
        resolution_factor=10.0,
        aggregation=build_pad_aggregation(**real_pad_params),
    )
    assert origin_pixel == (59297, 43847)  # see test_create_raster.py's real-tile test
    assert nb_pixels == 100

    output_dir = Path("./tmp/test_export_raster_real_tile")
    written = export_raster(
        aggregated,
        origin_pixel=origin_pixel,
        nb_pixels=nb_pixels,
        global_origin_x=_GLOBAL_ORIGIN_X,
        global_origin_y=_GLOBAL_ORIGIN_Y,
        resolution_factor=10.0,
        dz=real_pad_params["dz"],
        dz_low=real_pad_params["dz_low"],
        srid=_SRID,
        output_dir=output_dir,
        tile_stem="Semis_2022_0691_6484_LA93_IGN69",
    )
    print(f"\nGeoTIFFs written to {output_dir.resolve()} -- open pad_profile_1m.tif in QGIS to inspect")

    assert set(written) == {
        "pad_sb_0.5m",
        "pad_profile_1m",
        "class_count",
        "entering_rays",
        "intercept_ray",
        "pl_factor",
        "cover",
        "dates_pad",
    }

    with rasterio.open(written["pad_profile_1m"]) as src:
        assert src.count == 60
        assert src.crs.to_string() == _SRID
        ground_stratum = src.read(1)  # PAD_1_0: should show real vegetation signal on this tile
        assert np.isfinite(ground_stratum).any(), "Expected at least some populated pixels on this real tile"
        assert np.nanmax(ground_stratum) <= 5.0  # PAD is capped at 5, per AC

    with rasterio.open(written["class_count"]) as src:
        total = src.read(src.count)  # "Total" is always the last band
        populated = np.nansum(total)
        # Buffer points outside the tile's own CosiaFrance window are dropped by
        # compute_pixel_aggregates, so the raster's total is <= the input point count.
        assert 0 < populated <= len(points_df)
