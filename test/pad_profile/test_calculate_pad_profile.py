import numpy as np

from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core


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
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    assert result is None or (isinstance(result, tuple) and 0.0 <= float(result[0]) <= 1.0)


def test_pad_metrics_core_returns_none_when_too_few_points():
    """If the number of points is below `limit_N_points`, the function returns None."""
    gpstime = np.zeros(2, dtype=np.float64)
    points = _points(2, gpstime)

    result = pad_metrics_core(
        **points,
        scanning_angle=False,
        limit_N_points=5,
        deviation_days=np.inf,
    )

    assert result is None


def test_pad_metrics_core_scanning_angle_false_returns_one():
    """With `scanning_angle=False` the function should return cos_theta == 1.0."""
    gpstime = np.zeros(5, dtype=np.float64)
    points = _points(5, gpstime)

    result = pad_metrics_core(
        **points,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    cos_theta, _, _, _ = result
    assert cos_theta == 1.0


def test_pad_metrics_core_returns_ni_n_min_layer_with_default_strata_shape():
    """`Ni`/`N`/`min_layer` are wired in and shaped for the default 60 strata."""
    n = 5
    gpstime = np.zeros(n, dtype=np.float64)
    points = _points(n, gpstime)

    result = pad_metrics_core(
        **points,
        scanning_angle=False,
        limit_N_points=1,
        deviation_days=np.inf,
    )

    _, ni, n_per_stratum, min_layer = result
    assert len(ni) == len(n_per_stratum) == len(min_layer) == 60
    assert np.issubdtype(ni.dtype, np.integer)
    assert np.issubdtype(n_per_stratum.dtype, np.integer)
    # All points sit at h_abg=0.0, within the dropped below-ground stratum: no
    # vegetation hit is recorded anywhere, but every stratum still counts the
    # full point total via the cumulative sum (see compute_ni_n).
    np.testing.assert_array_equal(ni, np.zeros(60, dtype=ni.dtype))
    np.testing.assert_array_equal(n_per_stratum, np.full(60, n, dtype=n_per_stratum.dtype))
