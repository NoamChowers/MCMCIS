import pytest
import numpy as np

from perm_pval.core.proposals import propose_localized_swaps, resolve_n_swap_pairs


def test_resolve_n_swap_pairs_accepts_fractional_proposal_size():
    assert resolve_n_swap_pairs(10, 8, 0.25) == 2


def test_resolve_n_swap_pairs_accepts_integer_proposal_size():
    assert resolve_n_swap_pairs(10, 8, 3) == 3


@pytest.mark.parametrize("proposal_size", [0, 9])
def test_resolve_n_swap_pairs_rejects_integer_outside_group_bounds(proposal_size):
    with pytest.raises(ValueError, match="1 <= proposal_size <= min group size"):
        resolve_n_swap_pairs(10, 8, proposal_size)


@pytest.mark.parametrize("proposal_size", [0.0, -0.1, 1.1])
def test_resolve_n_swap_pairs_rejects_fraction_outside_unit_interval(proposal_size):
    with pytest.raises(ValueError, match="0 < proposal_fraction <= 1"):
        resolve_n_swap_pairs(10, 8, proposal_size)


def test_resolve_n_swap_pairs_rejects_bool():
    with pytest.raises(TypeError, match="proposal_size"):
        resolve_n_swap_pairs(10, 8, True)


def test_propose_localized_swaps_uses_exact_block_swap():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    rng = np.random.default_rng(123)

    y_prop = propose_localized_swaps(y, rng, n_swap_pairs=3)

    assert int(np.sum(y_prop)) == int(np.sum(y))
    assert int(np.sum((y == 1) & (y_prop == 0))) == 3
    assert int(np.sum((y == 0) & (y_prop == 1))) == 3
    assert int(np.sum(y != y_prop)) == 6
    assert np.array_equal(y, np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8))


def test_propose_localized_swaps_rejects_too_many_pairs():
    y = np.array([1, 1, 0], dtype=np.int8)
    rng = np.random.default_rng(123)

    with pytest.raises(ValueError, match="smaller group size"):
        propose_localized_swaps(y, rng, n_swap_pairs=2)
