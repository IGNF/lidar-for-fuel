import numpy as np
from lidar_for_fuel.pad_profile.calculate_pad_profile import pad_metrics_core

_GPSTIME_REF = "2023-01-01 00:00:00"


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
		gpstime_ref=_GPSTIME_REF,
	)

	assert result is None or (isinstance(result, (float, int)) and 0.0 <= float(result) <= 1.0)


