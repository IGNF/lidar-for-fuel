from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from lidar_for_fuel.main_preprocessing import preprocess_one_tile

# Real data – must be present in the working directory
_TILE_WITH_TRAJ = Path("data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311.laz")
_TILE_NO_TRAJ = Path("data/pointcloud/test_semis_2022_0897_6577_LA93_IGN69_decimation.laz")
_TRAJ_FOLDER = Path("data/trajectory")
_DTM_TILE1 = Path("data/DTM/Semis_2024_0751_6690_LA93_IGN69_10M.tif")

_NODATA = -9999.0


def _make_flat_dtm(path: Path, z_value: float, bounds: tuple) -> str:
    """Write a 200×200 flat GeoTIFF (EPSG:2154) at constant ground elevation z_value."""
    west, south, east, north = bounds
    data = np.full((200, 200), z_value, dtype=np.float32)
    transform = from_bounds(west, south, east, north, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype="float32",
        crs="EPSG:2154",
        transform=transform,
        nodata=_NODATA,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def test_pipeline_with_trajectory(tmp_path):
    """Full pipeline on a tile whose PointSourceId (13111) has a matching trajectory file.

    Tile bounds (from header): X=[751000, 752000], Y=[6689110, 6690000], Z=[178.68, 262.09].
    Real DTM (data/DTM/Semis_2024_0751_6690_LA93_IGN69_10M.tif): 10 m resolution, EPSG:2154,
    Z_sol ∈ [178.51, 235.83] → h_abg = Z − Z_sol; some high canopy points (h_abg > 80 m) filtered.

    Checks:
    - Extra dims X_sensor, Y_sensor, Z_sensor, h_abg are present in the output.
    - At least some points survive the full pipeline.
    - No NaN / no nodata in X_sensor/Y_sensor/Z_sensor: the trajectory covers all points.
    - h_abg is not nodata: the real DTM covers the entire tile.
    - X_sensor/Y_sensor fall within coherent Lambert-93 ranges.
    """
    output = tmp_path / "out.laz"
    preprocess_one_tile(
        input_filename=str(_TILE_WITH_TRAJ),
        trajectory_dir=str(_TRAJ_FOLDER),
        output_path=str(output),
        dtm_path=str(_DTM_TILE1),
    )

    las = laspy.read(str(output))
    extra = {d.name for d in las.point_format.extra_dimensions}

    # ── 1. All four extra fields present ─────────────────────────────────────
    assert "X_sensor" in extra
    assert "Y_sensor" in extra
    assert "Z_sensor" in extra
    assert "h_abg" in extra

    # ── 2. Pipeline produced output points ───────────────────────────────────
    assert las.header.point_count > 0

    # ── 3. No NaN / no nodata in sensor position: PointSourceId 13111 has a trajectory ───
    assert not np.any(np.isnan(las["X_sensor"])), "Unexpected NaN in X_sensor."
    assert not np.any(np.isnan(las["Y_sensor"])), "Unexpected NaN in Y_sensor."
    assert not np.any(np.isnan(las["Z_sensor"])), "Unexpected NaN in Z_sensor."
    assert not np.any(np.asarray(las["X_sensor"]) == _NODATA), "Unexpected nodata (0) in X_sensor."
    assert not np.any(np.asarray(las["Y_sensor"]) == _NODATA), "Unexpected nodata (0) in Y_sensor."
    assert not np.any(np.asarray(las["Z_sensor"]) == _NODATA), "Unexpected nodata (0) in Z_sensor."

    # ── 4. h_abg is valid: real DTM fully covers the tile ────────────────────
    assert not np.any(np.asarray(las["h_abg"]) == _NODATA), "Unexpected nodata in h_abg."

    # ── 5. X_sensor/Y_sensor in Lambert-93 bounds ─────────────────────────────
    assert np.all(las["X_sensor"] > 100_000)
    assert np.all(las["X_sensor"] < 1_300_000)
    assert np.all(las["Y_sensor"] > 6_000_000)
    assert np.all(las["Y_sensor"] < 7_200_000)


def test_pipeline_without_trajectory_produces_no_data(tmp_path):
    """Full pipeline on a tile whose PointSourceIds (820, 821, 822, 900, 4004) have no trajectory.

    Tile bounds (from header): X=[897000, 898000], Y=[6576000, 6577000], Z=[374.49, 449.76].
    Synthetic DTM at 374 m → h_abg = Z − 374 ∈ [0, 75]; all points pass filter_z_by_height.

    Checks:
    - Extra dims X_sensor, Y_sensor, Z_sensor, h_abg are present in the output.
    - At least some points survive the full pipeline.
    - All X_sensor/Y_sensor/Z_sensor are 0 (nodata): no trajectory matches any PointSourceId.
    - h_abg is not nodata: the synthetic DTM covers the entire tile.
    """
    dtm = _make_flat_dtm(
        tmp_path / "dtm.tif",
        z_value=374.0,
        bounds=(896500, 6575500, 898500, 6577500),
    )
    output = tmp_path / "out.laz"
    preprocess_one_tile(
        input_filename=str(_TILE_NO_TRAJ),
        trajectory_dir=str(_TRAJ_FOLDER),
        output_path=str(output),
        dtm_path=dtm,
    )

    las = laspy.read(str(output))
    extra = {d.name for d in las.point_format.extra_dimensions}

    # ── 1. All four extra fields present ─────────────────────────────────────
    assert "X_sensor" in extra
    assert "Y_sensor" in extra
    assert "Z_sensor" in extra
    assert "h_abg" in extra

    # ── 2. Pipeline produced output points ───────────────────────────────────
    assert las.header.point_count > 0

    # ── 3. All sensor fields are 0 (nodata): no trajectory matches any PointSourceId
    for field in ("X_sensor", "Y_sensor", "Z_sensor"):
        values = np.asarray(las[field])
        assert np.all(values == 0), (
            f"Expected nodata (0) in {field} for all points (no matching trajectory), "
            f"but got: {values[values != 0][:5]}"
        )

    # ── 4. h_abg is valid: synthetic DTM fully covers the tile ───────────────
    assert not np.any(np.asarray(las["h_abg"]) == _NODATA), "Unexpected nodata in h_abg."
