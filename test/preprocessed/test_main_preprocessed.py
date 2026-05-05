from pathlib import Path

import laspy
import numpy as np
import pdal
import rasterio
from rasterio.transform import from_bounds

from lidar_for_fuel.preprocessed.add_trajectory import add_trajectory_to_points
from lidar_for_fuel.preprocessed.filter_outliers import remove_outliers
from lidar_for_fuel.preprocessed.filter_points_by_date import filter_by_date
from lidar_for_fuel.preprocessed.filter_points_by_dimension_values import filter_by_dimension_values
from lidar_for_fuel.preprocessed.normalize_height_by_dtm import add_Zref, filter_z_by_height
from lidar_for_fuel.preprocessed.validate_lidar_file import check_lidar_file

# Real data – must be present in the working directory
_TILE_WITH_TRAJ = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311.laz")
_TILE_NO_TRAJ = Path("data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz")
_TRAJ_FOLDER = Path("data/trajectory")
_DTM_TILE1 = Path("data/DTM/Semis_2024_0751_6690_LA93_IGN69_10M.tif")

# Pipeline parameters matching config.yaml defaults
_SRID = "EPSG:2154"
_DEVIATION_DAYS = 14
_GPSTIME_REF = "2011-09-14 01:46:40"
_FILTER_DIM = "Classification"
_FILTER_VALS = [1, 2, 3, 4, 5, 9, 17, 64, 65, 67]
_NODATA = -9999.0
_MIN_HEIGHT = -3.0
_MAX_HEIGHT = 80.0
_MEAN_K = 5
_MULTIPLIER = 10.0


def _make_flat_dtm(path: Path, z_value: float, bounds: tuple) -> str:
    """Write a 200×200 flat GeoTIFF (EPSG:2154) at constant ground elevation z_value."""
    west, south, east, north = bounds
    data = np.full((200, 200), z_value, dtype=np.float32)
    transform = from_bounds(west, south, east, north, 200, 200)
    with rasterio.open(
        path, "w", driver="GTiff", height=200, width=200,
        count=1, dtype="float32", crs="EPSG:2154",
        transform=transform, nodata=_NODATA,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _run_full_pipeline(
    input_laz: Path,
    trajectory_folder: Path,
    dtm_path: str,
    output_laz: Path,
) -> None:
    """Run the full preprocessing pipeline, mirroring main_on_one_tile in main_preprocessed.py.

    The only difference is that the DTM path is injected directly instead of being downloaded
    from the IGN Géoplateforme, which avoids network dependency in tests.
    """
    pipeline = check_lidar_file(str(input_laz), _SRID)
    pipeline = filter_by_date(pipeline, _DEVIATION_DAYS, _GPSTIME_REF)
    pipeline = filter_by_dimension_values(pipeline, _FILTER_DIM, _FILTER_VALS)
    pipeline.execute()
    points = pipeline.arrays[0]

    points = add_Zref(points, dtm_path, nodata_value=_NODATA)
    points = filter_z_by_height(points, _MIN_HEIGHT, _MAX_HEIGHT)
    points = add_trajectory_to_points(points, str(trajectory_folder))

    pipeline_out = pdal.Pipeline(arrays=[points])
    pipeline_out = remove_outliers(pipeline_out, _MEAN_K, _MULTIPLIER)
    (pipeline_out | pdal.Writer.las(
        filename=str(output_laz),
        minor_version=4,
        forward="all",
        extra_dims="all",
    )).execute()


def test_pipeline_with_trajectory(tmp_path):
    """Full pipeline on a tile whose PointSourceId (13111) has a matching trajectory file.

    Tile bounds (from header): X=[751000, 752000], Y=[6689110, 6690000], Z=[178.68, 262.09].
    Real DTM (data/DTM/Semis_2024_0751_6690_LA93_IGN69_10M.tif): 10 m resolution, EPSG:2154,
    Z_sol ∈ [178.51, 235.83] → Z_ref = Z − Z_sol; some high canopy points (Z_ref > 80 m) filtered.

    Checks:
    - Extra dims Easting, Northing, Elevation, Z_ref are present in the output.
    - At least some points survive the full pipeline.
    - No NaN in Easting/Northing/Elevation: the trajectory covers all points.
    - Z_ref is not nodata: the real DTM covers the entire tile.
    - Easting/Northing fall within coherent Lambert-93 ranges.
    """
    output = tmp_path / "out.laz"
    _run_full_pipeline(_TILE_WITH_TRAJ, _TRAJ_FOLDER, str(_DTM_TILE1), output)

    las = laspy.read(str(output))
    extra = {d.name for d in las.point_format.extra_dimensions}

    # ── 1. All four extra fields present ─────────────────────────────────────
    assert "Easting" in extra
    assert "Northing" in extra
    assert "Elevation" in extra
    assert "Z_ref" in extra

    # ── 2. Pipeline produced output points ───────────────────────────────────
    assert las.header.point_count > 0

    # ── 3. No NaN in sensor position: PointSourceId 13111 has a trajectory ───
    assert not np.any(np.isnan(las["Easting"])), "Unexpected NaN in Easting."
    assert not np.any(np.isnan(las["Northing"])), "Unexpected NaN in Northing."
    assert not np.any(np.isnan(las["Elevation"])), "Unexpected NaN in Elevation."

    # ── 4. Z_ref is valid: real DTM fully covers the tile ────────────────────
    assert not np.any(np.asarray(las["Z_ref"]) == _NODATA), "Unexpected nodata in Z_ref."

    # ── 5. Easting/Northing in Lambert-93 bounds ─────────────────────────────
    assert np.all(las["Easting"] > 100_000)
    assert np.all(las["Easting"] < 1_300_000)
    assert np.all(las["Northing"] > 6_000_000)
    assert np.all(las["Northing"] < 7_200_000)


def test_pipeline_without_trajectory_produces_nan(tmp_path):
    """Full pipeline on a tile whose PointSourceIds (820, 821, 822, 900, 4004) have no trajectory.

    Tile bounds (from header): X=[897000, 898000], Y=[6576000, 6577000], Z=[374.49, 449.76].
    Synthetic DTM at 374 m → Z_ref = Z − 374 ∈ [0, 75]; all points pass filter_z_by_height.

    Checks:
    - Extra dims Easting, Northing, Elevation, Z_ref are present in the output.
    - At least some points survive the full pipeline.
    - All Easting/Northing/Elevation are 0 (nodata): no trajectory matches any PointSourceId.
    - Z_ref is not nodata: the synthetic DTM covers the entire tile.
    """
    dtm = _make_flat_dtm(
        tmp_path / "dtm.tif",
        z_value=374.0,
        bounds=(896500, 6575500, 898500, 6577500),
    )
    output = tmp_path / "out.laz"
    _run_full_pipeline(_TILE_NO_TRAJ, _TRAJ_FOLDER, dtm, output)

    las = laspy.read(str(output))
    extra = {d.name for d in las.point_format.extra_dimensions}

    # ── 1. All four extra fields present ─────────────────────────────────────
    assert "Easting" in extra
    assert "Northing" in extra
    assert "Elevation" in extra
    assert "Z_ref" in extra

    # ── 2. Pipeline produced output points ───────────────────────────────────
    assert las.header.point_count > 0

    # ── 3. All sensor fields are 0 (nodata): no trajectory matches any PointSourceId
    for field in ("Easting", "Northing", "Elevation"):
        values = np.asarray(las[field])
        assert np.all(values == 0), (
            f"Expected nodata (0) in {field} for all points (no matching trajectory), "
            f"but got: {values[values != 0][:5]}"
        )

    # ── 4. Z_ref is valid: synthetic DTM fully covers the tile ───────────────
    assert not np.any(np.asarray(las["Z_ref"]) == _NODATA), "Unexpected nodata in Z_ref."
