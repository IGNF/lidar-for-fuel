import numpy as np
import pytest

from lidar_for_fuel.pad_profile.build_vertical_strata import build_vertical_strata


def test_default_params_produce_60_layers():
    breaks, min_layer = build_vertical_strata()
    # breaks: [-inf, 0.1, 1, 2, ..., 60] -> 62 edges
    assert len(breaks) == 62
    assert breaks[0] == -np.inf
    assert breaks[1] == pytest.approx(0.1)
    assert breaks[-1] == pytest.approx(60.0)
    # min_layer (pre-shift): [-inf, 0, 1, ..., 59] -> 61 entries
    assert len(min_layer) == 61
    assert min_layer[0] == -np.inf
    assert min_layer[1] == pytest.approx(0.0)
    assert min_layer[-1] == pytest.approx(59.0)


def test_custom_nlayers_and_dz():
    breaks, min_layer = build_vertical_strata(z0=0.0, dz=0.5, nlayers=4, ground_margin=0.1)
    # seq: 0, 0.5, 1, 1.5, 2 -> 5 values -> breaks length 6
    assert len(breaks) == 6
    np.testing.assert_allclose(breaks[1:], [0.1, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(min_layer[1:], [0.0, 0.5, 1.0, 1.5])


def test_ground_margin_only_shifts_the_zero_break():
    breaks, _ = build_vertical_strata(z0=0.0, dz=1.0, nlayers=3, ground_margin=0.25)
    np.testing.assert_allclose(breaks[1:], [0.25, 1.0, 2.0, 3.0])


def test_zero_ground_margin_leaves_breaks_untouched():
    breaks, _ = build_vertical_strata(z0=0.0, dz=1.0, nlayers=3, ground_margin=0.0)
    np.testing.assert_allclose(breaks[1:], [0.0, 1.0, 2.0, 3.0])


def test_nlayers_none_derives_zmax_from_data_ceiling():
    z_values = np.array([0.2, 1.1, 4.3])
    breaks, min_layer = build_vertical_strata(z0=0.0, dz=1.0, nlayers=None, ground_margin=0.1, z_values=z_values)
    # ceil(4.3 / 1) * 1 = 5.0 -> seq 0..5 -> 6 values -> breaks length 7
    assert len(breaks) == 7
    assert breaks[-1] == pytest.approx(5.0)
    assert len(min_layer) == 6


def test_nlayers_none_without_z_values_raises():
    with pytest.raises(ValueError):
        build_vertical_strata(nlayers=None, z_values=None)


def test_nlayers_none_with_empty_z_values_raises():
    with pytest.raises(ValueError):
        build_vertical_strata(nlayers=None, z_values=np.array([]))
