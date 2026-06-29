import numpy as np
import pytest

from lidar_for_fuel.pad_profile.build_vertical_strata import build_vertical_strata

# ── structural properties (default params) ──────────────────────────────────────


def test_breaks_starts_with_neginf():
    breaks, _ = build_vertical_strata()
    assert breaks[0] == -np.inf


def test_min_layer_first_element_is_neginf():
    _, min_layer = build_vertical_strata()
    assert min_layer[0] == -np.inf


def test_inner_breaks_spacing():
    breaks, _ = build_vertical_strata(z0=0.0, dz=1.0, nlayers=60, ground_margin=0.1)
    inner = breaks[2:]  # skip -Inf and the shifted 0.1 break
    assert np.allclose(np.diff(inner), 1.0)


def test_returns_independent_arrays():
    breaks, min_layer = build_vertical_strata(z0=0.0, dz=1.0, ground_margin=0.1)
    # min_layer must not be a view of breaks (ground margin must not affect it)
    assert not np.shares_memory(breaks, min_layer)


def test_output_dtypes():
    breaks, min_layer = build_vertical_strata()
    assert breaks.dtype == np.float64
    assert min_layer.dtype == np.float64


# ── breaks shape and top edge, for various z0/dz/nlayers ────────────────────────


@pytest.mark.parametrize(
    "z0,dz,nlayers,expected_n_breaks,expected_n_min_layer,expected_last_break",
    [
        (0.0, 1.0, 60, 62, 61, 60.0),  # default params
        (0.0, 0.5, 120, 122, 121, 60.0),  # half-metre strata
        (0.0, 2.0, 30, 32, 31, 60.0),  # 2 m strata
        (2.0, 1.0, 10, 12, 11, 12.0),  # nonzero z0
    ],
    ids=["default", "dz_0.5m", "dz_2m", "nonzero_z0"],
)
def test_breaks_shape_and_top(z0, dz, nlayers, expected_n_breaks, expected_n_min_layer, expected_last_break):
    breaks, min_layer = build_vertical_strata(z0=z0, dz=dz, nlayers=nlayers)
    assert len(breaks) == expected_n_breaks
    assert len(min_layer) == expected_n_min_layer
    assert breaks[-1] == expected_last_break


# ── ground_margin shift ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "z0,ground_margin,expected_break_1",
    [
        (0.0, 0.1, 0.1),  # default margin shifts the height-0 break
        (0.0, 0.0, 0.0),  # zero margin is a no-op
        (2.0, 0.5, 2.0),  # nonzero z0: no break equals 0, nothing shifts
    ],
    ids=["default_margin", "zero_margin", "nonzero_z0_no_shift"],
)
def test_ground_margin_shift(z0, ground_margin, expected_break_1):
    breaks, min_layer = build_vertical_strata(z0=z0, dz=1.0, nlayers=10, ground_margin=ground_margin)
    assert breaks[1] == pytest.approx(expected_break_1)
    # min_layer is captured before the shift: always reports the unshifted z0
    assert min_layer[1] == z0


# ── nlayers=None, derived from h_abg ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "h_abg,dz,expected_last_break",
    [
        (np.array([0.0, 5.3, 12.7, 8.1]), 1.0, 13.0),  # rounds up to next multiple
        (np.array([0.0, 11.3]), 0.5, 11.5),  # rounds up with dz=0.5
        (np.array([0.0, 10.0]), 1.0, 10.0),  # exact multiple, no rounding needed
    ],
    ids=["rounds_up", "dz_half", "exact_multiple"],
)
def test_nlayers_none_derives_top_from_h_abg(h_abg, dz, expected_last_break):
    breaks, _ = build_vertical_strata(z0=0.0, dz=dz, nlayers=None, h_abg=h_abg)
    assert breaks[-1] == pytest.approx(expected_last_break)


@pytest.mark.parametrize("h_abg", [None, np.array([])], ids=["none", "empty_array"])
def test_nlayers_none_requires_nonempty_h_abg(h_abg):
    with pytest.raises(ValueError, match="h_abg"):
        build_vertical_strata(nlayers=None, h_abg=h_abg)
