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

    assert result is None or (isinstance(result, tuple) and 0.0 <= float(result[0]) <= 1.0)


def test_pad_metrics_core_output_format():
    """Verify the output format: `None` when there are too few points, otherwise a
    `(cos_theta, ni, n, min_layer)` tuple shaped for the default 60 strata."""
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

    # Enough points -> (cos_theta, ni, n, min_layer)
    result = pad_metrics_core(
        **points,
        **_DEFAULT_PARAMS,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )
    cos_theta, ni, n_per_stratum, min_layer = result
    assert isinstance(cos_theta, float)
    assert len(ni) == len(n_per_stratum) == len(min_layer) == 60
    assert np.issubdtype(ni.dtype, np.integer)
    assert np.issubdtype(n_per_stratum.dtype, np.integer)
    assert min_layer.dtype == np.float64
    # All points sit at h_abg=0.0, within the dropped below-ground stratum: no
    # vegetation hit is recorded anywhere, but every stratum still counts the
    # full point total via the cumulative sum (see compute_ni_n).
    np.testing.assert_array_equal(ni, np.zeros(60, dtype=ni.dtype))
    np.testing.assert_array_equal(n_per_stratum, np.full(60, n, dtype=n_per_stratum.dtype))


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

    cos_theta, _, _, _ = result
    assert cos_theta == 1.0
