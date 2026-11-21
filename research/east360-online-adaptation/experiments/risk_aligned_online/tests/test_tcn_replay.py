import pytest

from experiments.risk_aligned_online.tcn_online import validate_order


def test_locked_order_must_be_complete_unique_and_preserved():
    order = [{'shot': 9, 'proxy': 1.}, {'shot': 2, 'proxy': 2.}]
    assert validate_order(order, [2, 9]) == order
    with pytest.raises(ValueError):
        validate_order(order[:1], [2, 9])
    with pytest.raises(ValueError):
        validate_order([order[0], order[0]], [2, 9])
    with pytest.raises(ValueError):
        validate_order([{'shot': 9, 'proxy': float('nan')}, order[1]], [2, 9])
