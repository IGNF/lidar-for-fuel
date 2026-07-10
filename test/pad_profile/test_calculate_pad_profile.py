from pathlib import Path

import numpy as np
import pdal
import pytest

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core


_TRACKED_CLASSES = [1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67]

_REAL_PRETRAITED_LAS = Path(
    "data/pointcloud/test_semis_2024_0751_6690_LA93_IGN69_filter_trajectory_1311_pretraited.laz"
)
_SECONDS_PER_DAY = 86_400

# pad_metrics_core has no defaults of its own for these scalar params (the
# defaults documented in compute_ni_n/main_pad_profile live one level up).
_DEFAULT_PARAMS = dict(
    limit_flight_agl=0.0,
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
    keep_classes=[1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67],
)

_MAIN_PAD_KEYS = [f"PAD_1_{i}" for i in range(10)]
_MAIN_NI_KEYS = [f"Ni_1_{i}" for i in range(10)]
_MAIN_N_KEYS = [f"N_1_{i}" for i in range(10)]
_LOW_PAD_KEYS = ["PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"]
_CLASS_KEYS = [f"Class_{code}" for code in _TRACKED_CLASSES] + ["Total"]


def _points(n: int, gpstime: np.ndarray) -> dict:
    zeros = np.zeros(n, dtype=np.float64)
    return dict(
        gpstime=gpstime,
        x=zeros.copy(),
        y=zeros.copy(),
        h_abg=zeros.copy(),
        z=zeros.copy(),
        return_number=zeros.copy(),
        classification=zeros.copy(),
        x_sensor=zeros.copy(),
        y_sensor=zeros.copy(),
        z_sensor=np.full(n, 1000.0, dtype=np.float64),
    )


def test_pad_metrics_core_returns_cos_theta_between_0_and_1():
    gpstime = np.zeros(3, dtype=np.float64)
    points = _points(3, gpstime)

    # Use scanning_angle=False to get deterministic cos_theta == 1.0
    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    assert result is None or (isinstance(result, dict) and 0.0 <= float(result["cos_theta"]) <= 1.0)


def test_pad_metrics_core_output_format():
    """Verify the output format: `None` when there are too few points, otherwise a
    dict with `Date_maj`/`Date_min`/`Date_max`, `Cover_h_pad`, `Cover_2`, `Cover_4`,
    `Cover_6`, `cos_theta`, `Class_*`/`Total`, one `PAD_{dz}_{min_layer}` key per
    stratum for both the main profile (60 by default) and the low-strata band (4 by
    default), and `Ni_*`/`N_*` for the main profile."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    # Too few points -> None
    result_none = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=n + 1,
        deviation_days=0,
    )
    assert result_none is None

    # Enough points -> dict
    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )
    assert isinstance(result, dict)
    assert isinstance(result["cos_theta"], float)
    expected_keys = {
        "Date_maj",
        "Date_min",
        "Date_max",
        "Cover_h_pad",
        "Cover_2",
        "Cover_4",
        "Cover_6",
        "cos_theta",
        *_CLASS_KEYS,
        *_MAIN_PAD_KEYS,
        *_MAIN_NI_KEYS,
        *_MAIN_N_KEYS,
        *_LOW_PAD_KEYS,
    }
    assert set(result) == expected_keys
    assert len(result) == len(expected_keys)

    # classification=0.0 is never in keep_values -> no veg/ground points -> 0 cover everywhere
    assert result["Cover_h_pad"] == result["Cover_2"] == result["Cover_4"] == result["Cover_6"] == 0.0
    # classification=0.0 is not a tracked class either -> every Class_* is 0, but
    # Total still counts all points regardless of classification.
    assert all(result[f"Class_{code}"] == 0 for code in _TRACKED_CLASSES)
    assert result["Total"] == n
    # No veg/ground points at all -> Z_veg_gnd is empty -> every stratum (both
    # resolutions) counts as "empty" (min_empty = -inf) -> PAD forced to 0
    # everywhere, regardless of the nonzero Gf/NRD produced by the Laplace
    # correction on empty strata.
    assert all(result[key] == 0.0 for key in _MAIN_PAD_KEYS)
    assert all(result[key] == 0.0 for key in _LOW_PAD_KEYS)
    # No vegetation/ground hits anywhere, but every stratum still counts the
    # full point total via the cumulative sum.
    assert all(result[key] == 0 for key in _MAIN_NI_KEYS)
    assert all(result[key] == n for key in _MAIN_N_KEYS)


def test_pad_metrics_core_output_key_order_matches_channel_spec():
    """Locks in the exact output-list channel order from the PAD output spec:
    PAD (low-strata band, then main profile), Class_*/Total, N_*, Ni_*, Cover_*,
    cos_theta, Date_*. Downstream raster-band assembly relies on this order, so
    it's a contract, not an implementation detail."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    expected_order = [
        *_LOW_PAD_KEYS,
        *_MAIN_PAD_KEYS,
        *_CLASS_KEYS,
        *_MAIN_N_KEYS,
        *_MAIN_NI_KEYS,
        "Cover_h_pad",
        "Cover_2",
        "Cover_4",
        "Cover_6",
        "cos_theta",
        "Date_maj",
        "Date_min",
        "Date_max",
    ]
    assert list(result) == expected_order


def test_pad_metrics_core_class_counts_include_non_veg_ground_classes():
    """`Class_*`/`Total` are computed on all points after the temporal filter,
    before the vegetation/ground (`keep_values`) subsetting -- so a class like 1,
    which is tracked but excluded from PAD computation by `keep_values`, is still
    counted."""
    n = 10
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    points["classification"][:5] = 1.0  # tracked (keep_classes), not in keep_values
    points["classification"][5:] = 6.0  # tracked (keep_classes), not in keep_values

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    assert result["Class_1"] == 5
    assert result["Class_6"] == n - 5
    assert result["Total"] == n


def test_pad_metrics_core_date_min_max_match_deviation_days_window():
    """`Date_min`/`Date_max` are the theoretical bounds of the ±deviation_days
    window around `Date_maj` (the modal acquisition day), not the actual min/max
    GPS time of the retained points."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    deviation_days = 3

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=deviation_days,
    )

    assert result["Date_min"] == pytest.approx(result["Date_maj"] - deviation_days * _SECONDS_PER_DAY)
    assert result["Date_max"] == pytest.approx(result["Date_maj"] + deviation_days * _SECONDS_PER_DAY)


def test_pad_metrics_core_pad_nonzero_for_stratum_with_vegetation():
    """Contrasts with `test_pad_metrics_core_output_format`'s all-veg-free case:
    once a stratum actually contains vegetation/ground hits, its PAD is nonzero,
    while strata above the highest vegetation point are still forced to 0."""
    n = 20
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    # 5 vegetation returns (class 2, in keep_values) landing in stratum min_layer=5
    # (h_abg=5.5 falls in (5, 6] with dz=1).
    points["h_abg"][10:15] = 5.5
    points["classification"][10:15] = 2.0

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    # The 5 vegetation hits landed in stratum min_layer=5, as expected.
    assert result["Ni_1_5"] == 5
    assert result["PAD_1_5"] > 0.0
    # max(h_abg[veg_gnd]) = 5.5 -> min_empty = ceil(5.5/1)*1 = 6.0 -> the
    # stratum holding the vegetation itself (min_layer=5) survives, but every
    # stratum above it is still forced to 0.
    assert all(result[f"PAD_1_{i}"] == 0.0 for i in range(6, 60))


def test_pad_metrics_core_format_num_matches_r_paste_for_non_integer_dz():
    """Regression test for the deferred-work gap: `_format_num` (R's `paste()`
    number-stringification) was previously only exercised for `dz` in {1, 0.5, 10}
    in a sibling reference port. Lock in the exact key names for a non-integer
    main-profile `dz` distinct from the default low-strata `dz_low=0.5`, so the
    two PAD sets don't collide."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    params = dict(_DEFAULT_PARAMS, dz=0.25, nlayers=4)

    result = pad_metrics_core(
        **points,
        **params,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    assert {"PAD_0.25_0", "PAD_0.25_0.25", "PAD_0.25_0.5", "PAD_0.25_0.75"} <= set(result)


def test_pad_metrics_core_scanning_angle_false_returns_one():
    """With `scanning_angle=False` the function should return cos_theta == 1.0."""
    gpstime = np.zeros(5, dtype=np.float64)
    points = _points(5, gpstime)

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    assert result["cos_theta"] == 1.0


def test_pad_metrics_core_scanning_angle_does_not_affect_other_outputs():
    """`scanning_angle` only feeds `cos_theta` (and, transitively through it, every
    `PAD_*` key, which is cos_theta-scaled by construction); it must have zero effect
    on any other key (Date_*/Cover_*/Class_*), since none of the helpers feeding them
    take cos_theta as input."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    result_true = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=True,
        limit_N_points=1,
        deviation_days=0,
    )
    result_false = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=0,
    )

    assert result_true is not None and result_false is not None
    assert set(result_true) == set(result_false)
    # cos_theta and every PAD_* key are expected to differ (PAD is cos_theta-scaled
    # by construction) -- everything else must be identical regardless of scanning_angle.
    for key in result_true:
        if key == "cos_theta" or key.startswith("PAD_"):
            continue
        np.testing.assert_array_equal(result_true[key], result_false[key])


def test_pad_metrics_core_real_las_returns_coherent_output_values():
    """Run `pad_metrics_core` directly (no file-level validation) on the real
    pre-treated LAS used by `test_main_pad_profile.py`, to exercise the core PAD
    computation itself against a real point cloud rather than synthetic points.

    Skipped if the LAS file is not present in the workspace.
    """
    real_las = _REAL_PRETRAITED_LAS
    if not real_las.exists():
        pytest.skip(f"Real LAS {real_las} not found in workspace")

    pipeline = pdal.Pipeline() | pdal.Reader.las(filename=str(real_las), override_srs="EPSG:2154", nosrs=True)
    pipeline.execute()
    points = pipeline.arrays[0]

    result = pad_metrics_core(
        gpstime=points["GpsTime"].astype(np.float64),
        x=points["X"].astype(np.float64),
        y=points["Y"].astype(np.float64),
        h_abg=points["h_abg"].astype(np.float64),
        z=points["Z"].astype(np.float64),
        return_number=points["ReturnNumber"].astype(np.float64),
        classification=points["Classification"].astype(np.float64),
        x_sensor=points["X_sensor"].astype(np.float64),
        y_sensor=points["Y_sensor"].astype(np.float64),
        z_sensor=points["Z_sensor"].astype(np.float64),
        scanning_angle=True,
        limit_N_points=1,
        limit_flight_agl=0.0,
        deviation_days=14,
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
        keep_classes=[1, 2, 3, 4, 5, 6, 9, 17, 18, 64, 66, 67],
    )

    assert isinstance(result, dict)
    assert 0.0 <= result["cos_theta"] <= 1.0
    pad_keys = [key for key in result if key.startswith("PAD_")]
    assert len(pad_keys) == 60 + 4
    for cover_key in ("Cover_2", "Cover_4", "Cover_6"):
        assert 0.0 <= result[cover_key] <= 1.0
    assert result["Total"] > 0
    assert result["Date_min"] == pytest.approx(result["Date_maj"] - 14 * _SECONDS_PER_DAY)
    assert result["Date_max"] == pytest.approx(result["Date_maj"] + 14 * _SECONDS_PER_DAY)
