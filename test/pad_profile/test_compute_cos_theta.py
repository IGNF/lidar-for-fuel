import numpy as np

from lidar_for_fuel.pad_profile.compute_cos_theta import compute_cos_theta


def test_scanning_angle_false_returns_one_and_no_mean_agl():
    x = y = z = x_sensor = y_sensor = z_sensor = np.zeros(3)
    veg_gnd = np.array([True, False, True])

    cos_theta, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=False,
    )

    assert cos_theta == 1.0
    assert mean_flight_agl is None


def test_scanning_angle_false_with_no_veg_gnd_points_still_returns_one():
    """Empty veg_gnd must not produce NaN -- this is the R bug we deliberately don't reproduce."""
    x = y = z = x_sensor = y_sensor = z_sensor = np.zeros(3)
    veg_gnd = np.zeros(3, dtype=bool)

    cos_theta, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=False,
    )

    assert cos_theta == 1.0
    assert mean_flight_agl is None


def test_scanning_angle_true_sensor_directly_above_point():
    """Sensor exactly above the point: no horizontal offset, cos_theta == 1."""
    x = y = np.zeros(2)
    x_sensor = y_sensor = np.zeros(2)
    z = np.array([0.0, 10.0])
    z_sensor = np.array([1000.0, 1000.0])
    veg_gnd = np.array([True, True])

    cos_theta, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=True,
    )

    assert np.isclose(cos_theta, 1.0)
    assert np.isclose(mean_flight_agl, np.mean([1000.0, 990.0]))


def test_scanning_angle_true_with_horizontal_offset():
    """Sensor offset horizontally: cos_theta computed by hand from flight_agl/norm_u."""
    x = np.array([0.0])
    y = np.array([0.0])
    z = np.array([0.0])
    x_sensor = np.array([300.0])
    y_sensor = np.array([0.0])
    z_sensor = np.array([400.0])
    veg_gnd = np.array([True])

    flight_agl = 400.0 - 0.0
    norm_u = np.sqrt(300.0**2 + 0.0**2 + flight_agl**2)
    expected_cos_theta = abs(flight_agl / norm_u)

    cos_theta, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=True,
    )

    assert cos_theta == expected_cos_theta
    assert mean_flight_agl == 400.0


def test_scanning_angle_true_ignores_nan_in_mean_flight_agl():
    x = y = np.zeros(3)
    x_sensor = y_sensor = np.zeros(3)
    z = np.array([0.0, 0.0, np.nan])
    z_sensor = np.array([100.0, 200.0, 300.0])
    veg_gnd = np.array([True, True, False])

    _, mean_flight_agl = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=True,
    )

    assert mean_flight_agl == 150.0  # nanmean([100.0, 200.0])


def test_scanning_angle_true_only_averages_over_veg_gnd():
    x = y = np.zeros(3)
    x_sensor = y_sensor = np.zeros(3)
    z = np.zeros(3)
    z_sensor = np.array([100.0, 200.0, 300.0])
    veg_gnd = np.array([True, False, False])

    cos_theta, _ = compute_cos_theta(
        x=x,
        y=y,
        z=z,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        veg_gnd=veg_gnd,
        scanning_angle=True,
    )

    # Vertical sensor above each point -> cos_theta == 1 regardless of which point is kept.
    assert cos_theta == 1.0
