import pytest

from experiments.risk_aligned_online.adaptivity_capacity_20261102.verify_neural import (
    replay_ids,
)


def row(shot, flag, family="gates", witness=False):
    return dict(shot=shot, disruptive=flag, family=family, score=dict(witness=witness))


def test_empty_replay_rejected():
    with pytest.raises(ValueError):
        replay_ids([])


def test_first_last_and_first_witness_for_each_family():
    rows = [
        row(10, True),
        row(10, True, "ffn_lora"),
        row(11, True, witness=True),
        row(11, True, "ffn_lora"),
        row(12, True),
        row(12, True, "ffn_lora", True),
        row(13, True, witness=True),
        row(14, True),
        row(20, False),
        row(21, False),
        row(22, False, "ffn_lora", True),
        row(23, False),
    ]
    assert replay_ids(rows) == [10, 11, 12, 14, 20, 22, 23]


def test_one_case_preflight():
    assert replay_ids([row(10, True), row(10, True, "ffn_lora")]) == [10]
