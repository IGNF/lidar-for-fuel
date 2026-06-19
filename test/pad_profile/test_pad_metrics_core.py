import math

import numpy as np
import pytest

from lidar_for_fuel.pad_profile.pad_metrics_core import pad_metrics_core

_REF = "2011-09-14 01:46:40"


def _call(
    h_abg,
    classification=None,
    return_number=None,
    gpstime=None,
    z=None,
    x=None,
    y=None,
    x_sensor=None,
    y_sensor=None,
    z_sensor=None,
    **kwargs,
):
    """Build a minimal points set, filling sensible defaults, and call pad_metrics_core."""
    h_abg = np.asarray(h_abg, dtype=np.float64)
    n = len(h_abg)
    classification = np.full(n, 2) if classification is None else np.asarray(classification)
    return_number = np.ones(n, dtype=np.int64) if return_number is None else np.asarray(return_number)
    gpstime = np.zeros(n) if gpstime is None else np.asarray(gpstime, dtype=np.float64)
    z = np.zeros(n) if z is None else np.asarray(z, dtype=np.float64)
    x = np.zeros(n) if x is None else np.asarray(x, dtype=np.float64)
    y = np.zeros(n) if y is None else np.asarray(y, dtype=np.float64)
    x_sensor = np.zeros(n) if x_sensor is None else np.asarray(x_sensor, dtype=np.float64)
    y_sensor = np.zeros(n) if y_sensor is None else np.asarray(y_sensor, dtype=np.float64)
    z_sensor = np.zeros(n) if z_sensor is None else np.asarray(z_sensor, dtype=np.float64)

    return pad_metrics_core(
        gpstime=gpstime,
        x=x,
        y=y,
        h_abg=h_abg,
        z=z,
        return_number=return_number,
        classification=classification,
        x_sensor=x_sensor,
        y_sensor=y_sensor,
        z_sensor=z_sensor,
        **kwargs,
    )


def test_too_few_points_returns_none():
    result = _call([1.0, 2.0], limit_N_points=5)
    assert result is None


def test_aberrant_trajectory_returns_none():
    n = 5
    result = _call(
        [1.0] * n,
        z=[100.0] * n,
        z_sensor=[200.0] * n,  # flight_agl = 100 < limit_flight_agl
        scanning_angle=True,
        limit_flight_agl=800.0,
    )
    assert result is None


def test_scanning_angle_false_gives_cos_theta_one():
    result = _call([1.0, 2.0, 3.0], scanning_angle=False, use_cover=False, nlayers=1, dz=10.0)
    assert result is not None
    assert result["cos_theta"] == 1.0


def test_scanning_angle_true_computes_cos_theta_from_trajectory():
    result = _call(
        [1.0],
        x=[300.0],
        y=[400.0],
        z=[0.0],
        x_sensor=[0.0],
        y_sensor=[0.0],
        z_sensor=[1000.0],
        scanning_angle=True,
        limit_flight_agl=800.0,
        use_cover=False,
        nlayers=1,
        dz=10.0,
    )
    assert result is not None
    # flight_agl=1000, norm_U=sqrt(300^2+400^2+1000^2)=1118.034 -> Nz_U=1000/1118.034
    assert result["cos_theta"] == pytest.approx(0.8944271910, rel=1e-9)


def test_nrdc_correction_when_nrd_is_zero_with_ground_margin_and_empty_strata():
    # 3 points within the ground margin band (<=0.1) -> layer0 has Ni=0, N=3 -> NRD=0 -> corrected.
    result = _call(
        [0.05, 0.05, 0.05],
        scanning_angle=False,
        use_cover=False,
        G=1.0,
        omega=1.0,
        dz=1.0,
        nlayers=3,
        ground_margin=0.1,
        limit_N_points=0,
    )
    assert result is not None
    # NRD = (0+1)/(3+2) = 0.2 -> Gf=0.8 -> PAD = -log(0.8) * dz/(dz-ground_margin)
    expected = -math.log(0.8) * (1.0 / 0.9)
    assert result["PAD_1_0"] == pytest.approx(expected, rel=1e-9)
    assert result["PAD_1_1"] == 0.0
    assert result["PAD_1_2"] == 0.0


def test_nrdc_correction_when_nrd_is_one_with_ground_margin_and_empty_strata():
    # 4 points exactly in stratum 0, nothing below -> Ni == N -> NRD=1 -> corrected.
    result = _call(
        [0.5, 0.5, 0.5, 0.5],
        scanning_angle=False,
        use_cover=False,
        G=1.0,
        omega=1.0,
        dz=1.0,
        nlayers=3,
        ground_margin=0.1,
        limit_N_points=0,
    )
    assert result is not None
    # NRD = (4+1)/(4+2) = 5/6 -> Gf=1/6 -> PAD = log(6) * dz/(dz-ground_margin)
    expected = -math.log(1.0 / 6.0) * (1.0 / 0.9)
    assert result["PAD_1_0"] == pytest.approx(expected, rel=1e-9)
    assert result["PAD_1_1"] == 0.0
    assert result["PAD_1_2"] == 0.0


def test_cover_h_pad_zero_falls_back_to_no_cover_formula():
    # Same point set as the NRD=0 case, but with use_cover=True and nothing above height_cover.
    result = _call(
        [0.05, 0.05, 0.05],
        scanning_angle=False,
        use_cover=True,
        height_cover=2.0,
        G=1.0,
        omega=1.0,
        dz=1.0,
        nlayers=3,
        ground_margin=0.1,
        limit_N_points=0,
    )
    assert result is not None
    assert result["Cover_h_pad"] == 0.0
    expected = -math.log(0.8) * (1.0 / 0.9)
    assert result["PAD_1_0"] == pytest.approx(expected, rel=1e-9)


def test_cover_fallback_when_nrd_equals_cover_h_pad():
    # height_cover=1, no ground_margin. Layer1 (min_layer=1) has NRD == Cover_h_pad exactly.
    result = _call(
        [0.5, 0.5, 0.5, 1.5, 1.5],
        scanning_angle=False,
        use_cover=True,
        height_cover=1.0,
        G=1.0,
        omega=1.0,
        dz=1.0,
        nlayers=2,
        ground_margin=0.0,
        limit_N_points=0,
    )
    assert result is not None
    assert result["Cover_h_pad"] == pytest.approx(0.4, rel=1e-9)
    # Fallback formula: PAD = -log(Gf) (NRDc-corrected at layer0: NRD=(3+1)/(3+2)=0.8 -> Gf=0.2)
    assert result["PAD_1_0"] == pytest.approx(-math.log(0.2), rel=1e-9)
    # layer1: NRD=2/5=0.4 (no correction needed) -> Gf=0.6
    assert result["PAD_1_1"] == pytest.approx(-math.log(0.6), rel=1e-9)


def test_use_cover_general_case():
    result = _call(
        [0.5, 0.5, 1.5, 2.5, 2.5, 2.5],
        gpstime=[1000.0] * 6,
        scanning_angle=False,
        use_cover=True,
        height_cover=1.0,
        G=1.0,
        omega=1.0,
        dz=1.0,
        nlayers=3,
        ground_margin=0.0,
        limit_N_points=0,
    )
    assert result is not None
    assert result["date"] == pytest.approx(1000.0)
    assert result["Cover_h_pad"] == pytest.approx(4.0 / 6.0, rel=1e-9)
    assert result["PAD_1_0"] == pytest.approx(1.386294361, rel=1e-7)
    assert result["PAD_1_1"] == pytest.approx(0.462098120, rel=1e-7)
    assert result["PAD_1_2"] == pytest.approx(0.924196240, rel=1e-7)


def test_cover_type_first_vs_all():
    h_abg = [1.0, 3.0, 5.0, 7.0]
    return_number = [1, 2, 1, 2]

    result_all = _call(h_abg, return_number=return_number, use_cover=False, nlayers=1, dz=10.0, scanning_angle=False)
    assert result_all["Cover_2"] == pytest.approx(0.75)
    assert result_all["Cover_4"] == pytest.approx(0.5)
    assert result_all["Cover_6"] == pytest.approx(0.25)
    assert math.isnan(result_all["Cover_h_pad"])

    result_first = _call(
        h_abg,
        return_number=return_number,
        cover_type="first",
        use_cover=False,
        nlayers=1,
        dz=10.0,
        scanning_angle=False,
    )
    assert result_first["Cover_2"] == pytest.approx(0.5)
    assert result_first["Cover_4"] == pytest.approx(0.5)
    assert result_first["Cover_6"] == pytest.approx(0.0)


def test_keep_n_includes_raw_counts():
    result = _call(
        [0.5, 0.5, 1.5],
        scanning_angle=False,
        use_cover=False,
        nlayers=2,
        dz=1.0,
        ground_margin=0.0,
        keep_N=True,
    )
    assert result is not None
    assert result["Ni_1_0"] == 2
    assert result["N_1_0"] == 2
    assert result["Ni_1_1"] == 1
    assert result["N_1_1"] == 3


def test_invalid_cover_type_raises():
    with pytest.raises(ValueError):
        _call([1.0, 2.0], cover_type="bogus", scanning_angle=False)


def test_flight_agl_mean_ignores_nan_like_r_na_rm_true():
    # 4 valid points (flight_agl=900, well above the limit) + 1 point with NaN z_sensor.
    # R uses mean(flight_agl, na.rm=TRUE), so the NaN point must not pollute mean_agl.
    n = 5
    result = _call(
        [1.0] * n,
        z=[0.0] * n,
        z_sensor=[900.0, 900.0, 900.0, 900.0, np.nan],
        scanning_angle=True,
        limit_flight_agl=800.0,
        use_cover=False,
        nlayers=1,
        dz=10.0,
    )
    assert result is not None


def test_empty_after_temporal_filter_does_not_crash():
    # All points fall outside the season filter -> empty arrays reach the rest of the
    # function (limit_N_points defaults to 0, so the length guard does not catch this).
    result = _call(
        [5.0, 10.0],
        gpstime=[200.0 * 86400.0, 201.0 * 86400.0],
        season_filter=[9],
        scanning_angle=False,
        use_cover=True,
        height_cover=2.0,
        nlayers=2,
        dz=1.0,
        gpstime_ref=_REF,
        limit_N_points=0,
    )
    assert result is not None
    assert result["PAD_1_0"] == 0.0
    assert result["PAD_1_1"] == 0.0
    assert math.isnan(result["Cover_h_pad"])


def test_temporal_filter_excludes_out_of_season_points():
    # 2 points in September (kept), 1 point ~200 days later in April (excluded).
    gpstime = [0.0, 0.0, 200.0 * 86400.0]
    result = _call(
        [0.5, 0.5, 99.0],
        gpstime=gpstime,
        season_filter=[9],
        scanning_angle=False,
        use_cover=False,
        nlayers=1,
        dz=10.0,
        gpstime_ref=_REF,
        limit_N_points=0,
    )
    assert result is not None
    # Only the 2 September points remain -> date is their mean (0.0), not influenced by the April point.
    assert result["date"] == pytest.approx(0.0)
