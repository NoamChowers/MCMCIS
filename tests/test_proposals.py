import pytest

from perm_pval.core.proposals import resolve_n_swap_pairs


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
