import numpy as np
import pytest
import rasterio

from lidar_for_fuel.pad_profile.export_raster import (
    export_pad_rasters,
    write_multiband_raster,
)

_ORIGIN = (691000.0, 6484000.0)
_PIXEL_SIZE = 10.0
_CRS = "EPSG:2154"


# ── write_multiband_raster ───────────────────────────────────────────────────


def test_write_multiband_raster_round_trips_values_and_band_names(tmp_path):
    bands = {
        "A": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "B": np.array([[5.0, 6.0], [7.0, 8.0]]),
    }
    output_path = str(tmp_path / "out.tif")

    write_multiband_raster(
        output_path=output_path,
        bands=bands,
        raster_origin=_ORIGIN,
        pixel_size=_PIXEL_SIZE,
        crs=_CRS,
        dtype="float32",
        nodata=np.nan,
    )

    with rasterio.open(output_path) as src:
        assert src.count == 2
        assert src.descriptions == ("A", "B")
        np.testing.assert_allclose(src.read(1), bands["A"])
        np.testing.assert_allclose(src.read(2), bands["B"])


def test_write_multiband_raster_sets_correct_georeferencing(tmp_path):
    bands = {"A": np.zeros((3, 4))}
    output_path = str(tmp_path / "out.tif")

    write_multiband_raster(
        output_path=output_path,
        bands=bands,
        raster_origin=_ORIGIN,
        pixel_size=_PIXEL_SIZE,
        crs=_CRS,
        dtype="float32",
        nodata=np.nan,
    )

    with rasterio.open(output_path) as src:
        assert src.crs.to_string() == _CRS
        assert src.transform.c == _ORIGIN[0]
        assert src.transform.f == _ORIGIN[1]
        assert src.transform.a == _PIXEL_SIZE
        assert src.transform.e == -_PIXEL_SIZE
        assert src.width == 4
        assert src.height == 3


def test_write_multiband_raster_raises_on_empty_bands(tmp_path):
    with pytest.raises(ValueError, match="at least one band"):
        write_multiband_raster(
            output_path=str(tmp_path / "out.tif"),
            bands={},
            raster_origin=_ORIGIN,
            pixel_size=_PIXEL_SIZE,
            crs=_CRS,
            dtype="float32",
            nodata=np.nan,
        )


def test_write_multiband_raster_raises_on_mismatched_shapes(tmp_path):
    bands = {"A": np.zeros((2, 2)), "B": np.zeros((3, 3))}
    with pytest.raises(ValueError, match="same shape"):
        write_multiband_raster(
            output_path=str(tmp_path / "out.tif"),
            bands=bands,
            raster_origin=_ORIGIN,
            pixel_size=_PIXEL_SIZE,
            crs=_CRS,
            dtype="float32",
            nodata=np.nan,
        )


# ── export_pad_rasters ───────────────────────────────────────────────────────

_KEEP_CLASSES = [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67]
_DZ, _NLAYERS = 1.0, 3
_DZ_LOW, _NLAYERS_LOW = 0.5, 2
_Z0 = 0.0


def _make_pixel_dict(seed: float) -> dict:
    """A minimal pad_metrics_core-shaped dict for a 3-layer main profile / 2-layer low band."""
    d = {}
    for i, layer in enumerate([0.0, 1.0, 2.0]):
        d[f"PAD_{1}_{int(layer)}"] = seed + i
        d[f"N_1_{int(layer)}"] = 100 + i
        d[f"Ni_1_{int(layer)}"] = 10 + i
    for i, layer in enumerate([0.0, 0.5]):
        key_layer = "0" if layer == 0.0 else "0.5"
        d[f"PAD_0.5_{key_layer}"] = seed + i
    for code in _KEEP_CLASSES:
        d[f"Class_{code}"] = code
    d["Total"] = sum(_KEEP_CLASSES)
    d["Cover_2"] = 0.2
    d["Cover_4"] = 0.4
    d["Cover_6"] = 0.6
    d["Date_maj"] = 1_700_000_000 // 86400 * 86400  # already day-aligned, like modal_time_unix
    d["Date_min"] = d["Date_maj"] - 86400
    d["Date_max"] = d["Date_maj"] + 86400
    return d


@pytest.fixture()
def pixel_grid():
    grid = np.empty((2, 2), dtype=object)
    grid[0, 0] = _make_pixel_dict(1.0)
    grid[0, 1] = _make_pixel_dict(2.0)
    grid[1, 0] = None  # rejected pixel -> nodata everywhere
    grid[1, 1] = _make_pixel_dict(4.0)
    return grid


def _export(pixel_grid, tmp_path):
    return export_pad_rasters(
        pixel_results=pixel_grid,
        raster_origin=_ORIGIN,
        pixel_size=_PIXEL_SIZE,
        crs=_CRS,
        output_dir=str(tmp_path),
        tilename="Semis_2022_0691_6484",
        z0=_Z0,
        dz=_DZ,
        nlayers=_NLAYERS,
        dz_low=_DZ_LOW,
        nlayers_low=_NLAYERS_LOW,
        keep_classes=_KEEP_CLASSES,
    )


def test_export_pad_rasters_writes_the_7_expected_rasters(pixel_grid, tmp_path):
    output_paths = _export(pixel_grid, tmp_path)

    assert set(output_paths) == {
        "pad_sb_0.5m",
        "pad_profile_1m",
        "class_count",
        "entering_rays",
        "intercept_ray",
        "cover",
        "dates_pad",
    }
    for path in output_paths.values():
        with rasterio.open(path) as src:
            assert src.width == 2
            assert src.height == 2


def test_export_pad_rasters_band_names_and_counts_match_spec(pixel_grid, tmp_path):
    output_paths = _export(pixel_grid, tmp_path)

    with rasterio.open(output_paths["pad_sb_0.5m"]) as src:
        assert src.count == 2
        assert src.descriptions == ("PAD_0.5_0", "PAD_0.5_0.5")

    with rasterio.open(output_paths["pad_profile_1m"]) as src:
        assert src.count == 3
        assert src.descriptions == ("PAD_1_0", "PAD_1_1", "PAD_1_2")

    with rasterio.open(output_paths["class_count"]) as src:
        assert src.count == len(_KEEP_CLASSES) + 1
        assert src.descriptions[-1] == "Total"

    with rasterio.open(output_paths["entering_rays"]) as src:
        assert src.descriptions == ("N_1_0", "N_1_1", "N_1_2")

    with rasterio.open(output_paths["intercept_ray"]) as src:
        assert src.descriptions == ("Ni_1_0", "Ni_1_1", "Ni_1_2")

    with rasterio.open(output_paths["cover"]) as src:
        assert src.descriptions == ("Cover_2", "Cover_4", "Cover_6")

    with rasterio.open(output_paths["dates_pad"]) as src:
        assert src.descriptions == ("Date_maj", "Date_min", "Date_max")


def test_export_pad_rasters_nodata_pixel_is_written_as_nodata(pixel_grid, tmp_path):
    output_paths = _export(pixel_grid, tmp_path)

    with rasterio.open(output_paths["pad_profile_1m"]) as src:
        band = src.read(1)
        assert np.isnan(band[1, 0])  # rejected pixel
        assert not np.isnan(band[0, 0])

    with rasterio.open(output_paths["class_count"]) as src:
        band = src.read(1)
        assert band[1, 0] == -9999


def test_export_pad_rasters_converts_dates_to_days_since_epoch(pixel_grid, tmp_path):
    output_paths = _export(pixel_grid, tmp_path)

    with rasterio.open(output_paths["dates_pad"]) as src:
        band = src.read(1)  # Date_maj
        expected_days = _make_pixel_dict(1.0)["Date_maj"] // 86400
        assert band[0, 0] == expected_days
        assert band[1, 0] == -9999  # rejected pixel -> nodata, not a bogus day count


def test_export_pad_rasters_raises_when_nlayers_is_none(pixel_grid, tmp_path):
    with pytest.raises(ValueError, match="nlayers"):
        export_pad_rasters(
            pixel_results=pixel_grid,
            raster_origin=_ORIGIN,
            pixel_size=_PIXEL_SIZE,
            crs=_CRS,
            output_dir=str(tmp_path),
            tilename="tile",
            z0=_Z0,
            dz=_DZ,
            nlayers=None,
            dz_low=_DZ_LOW,
            nlayers_low=_NLAYERS_LOW,
            keep_classes=_KEEP_CLASSES,
        )
