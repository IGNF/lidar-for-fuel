import numpy as np

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core

# pad_metrics_core has no defaults of its own for these scalar params (the
# defaults documented in compute_ni_n/main_pad_profile live one level up).
_DEFAULT_PARAMS = dict(
    limit_flight_agl=0.0,
    z0=0.0,
    dz=1.0,
    nlayers=60,
    ground_margin=0.1,
    cover_type="all",
    height_cover=2.0,
    use_cover=True,
    G=0.5,
    omega=0.77,
    keep_N=False,
)


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
        deviation_days=np.inf,
    )

    assert result is None or (isinstance(result, dict) and 0.0 <= float(result["cos_theta"]) <= 1.0)


def test_pad_metrics_core_output_format():
    """Verify the output format: `None` when there are too few points, otherwise a
    dict matching R's named-list output -- `date`, `Cover_h_pad`, `Cover_2`, `Cover_4`,
    `Cover_6`, `cos_theta`, plus one `PAD_{dz}_{min_layer}` key per stratum (60 by
    default), and no `Ni_*`/`N_*` keys since `keep_N=False`."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    # Too few points -> None
    result_none = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=n + 1,
        deviation_days=np.inf,
    )
    assert result_none is None

    # Enough points -> dict
    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )
    assert isinstance(result, dict)
    assert len(result) == 6 + 60
    assert isinstance(result["cos_theta"], float)
    pad_keys = [f"PAD_1_{i}" for i in range(60)]
    assert set(result) == {"date", "Cover_h_pad", "Cover_2", "Cover_4", "Cover_6", "cos_theta", *pad_keys}
    # classification=0.0 is never in _KEEP_VALUES -> no veg/ground points -> 0 cover everywhere
    assert result["Cover_h_pad"] == result["Cover_2"] == result["Cover_4"] == result["Cover_6"] == 0.0
    # No veg/ground points at all -> Z_veg_gnd is empty -> every stratum counts as
    # "empty" (min_empty = -inf) -> PAD forced to 0 everywhere, regardless of the
    # nonzero Gf/NRD produced by the Laplace correction on empty strata.
    assert all(result[key] == 0.0 for key in pad_keys)


def test_pad_metrics_core_keep_n_adds_ni_n_keys():
    """`keep_N=True` adds `Ni_{dz}_{min_layer}`/`N_{dz}_{min_layer}` per stratum,
    on top of the keys already covered by `test_pad_metrics_core_output_format`."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    params = dict(_DEFAULT_PARAMS, keep_N=True)

    result = pad_metrics_core(
        **points,
        **params,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    assert len(result) == 6 + 60 * 3
    # No vegetation/ground hits anywhere (see test_pad_metrics_core_output_format),
    # but every stratum still counts the full point total via the cumulative sum.
    assert all(result[f"Ni_1_{i}"] == 0 for i in range(60))
    assert all(result[f"N_1_{i}"] == n for i in range(60))


def test_pad_metrics_core_format_num_matches_r_paste_for_non_integer_dz():
    """Regression test for the deferred-work gap: `_format_num` (R's `paste()`
    number-stringification) was previously only exercised for `dz` in {1, 0.5, 10}
    in a sibling reference port. Lock in the exact key names for `dz=0.5`."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)
    params = dict(_DEFAULT_PARAMS, dz=0.5, nlayers=4)

    result = pad_metrics_core(
        **points,
        **params,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    assert {"PAD_0.5_0", "PAD_0.5_0.5", "PAD_0.5_1", "PAD_0.5_1.5"} <= set(result)


def test_pad_metrics_core_scanning_angle_false_returns_one():
    """With `scanning_angle=False` the function should return cos_theta == 1.0."""
    gpstime = np.zeros(5, dtype=np.float64)
    points = _points(5, gpstime)

    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    assert result["cos_theta"] == 1.0


def test_pad_metrics_core_scanning_angle_does_not_affect_other_outputs():
    """`scanning_angle` only feeds `cos_theta` (and, transitively through it, every
    `PAD_*` key, which is cos_theta-scaled by construction); it must have zero effect
    on any other key (date/Cover_*), since none of the helpers feeding them take
    cos_theta as input."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    result_true = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=True,
        limit_N_points=1,
        deviation_days=np.inf,
    )
    result_false = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    assert result_true is not None and result_false is not None
    assert set(result_true) == set(result_false)
    # cos_theta and every PAD_* key are expected to differ (PAD is cos_theta-scaled
    # by construction) -- everything else must be identical regardless of scanning_angle.
    for key in result_true:
        if key == "cos_theta" or key.startswith("PAD_"):
            continue
        np.testing.assert_array_equal(result_true[key], result_false[key])
