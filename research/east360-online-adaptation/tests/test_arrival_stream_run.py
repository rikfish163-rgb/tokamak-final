import pytest

from experiments.risk_aligned_online.arrival_stream_20261030.run import OutcomeGate


def test_gate_requires_own_startup_seal_and_all_suffix_seals():
    reads = []
    gate = OutcomeGate(list(range(1, 361)), lambda d, s: reads.append(s) or (False, 1.))
    with pytest.raises(AssertionError):
        gate.read_outcome('EAST', 1)
    for sid in range(1, 21):
        gate.seal(sid, 'a'*64)
        assert gate.read_outcome('EAST', sid) == (False, 1.)
    gate.seal(21, 'b'*64)
    with pytest.raises(AssertionError):
        gate.read_outcome('EAST', 21)
    for sid in range(22, 361):
        gate.seal(sid, 'c'*64)
    assert gate.read_outcome('EAST', 21) == (False, 1.)
    assert reads == list(range(1, 22))
    assert all(r['sealed_count'] == 360 for r in gate.accesses if r['shot'] > 20)


def test_gate_rejects_wrong_device_order_duplicate_and_short_seal():
    gate = OutcomeGate(list(range(1, 361)), lambda d, s: (True, 2.))
    with pytest.raises(ValueError):
        gate.seal(2, 'a'*64)
    with pytest.raises(ValueError):
        gate.seal(1, 'bad')
    gate.seal(1, 'a'*64)
    with pytest.raises(ValueError):
        gate.seal(1, 'a'*64)
    with pytest.raises(AssertionError):
        gate.read_outcome('J-TEXT', 1)
    with pytest.raises(AssertionError):
        gate.read_outcome('EAST', 999)


def test_gate_needs_full_distinct_stream():
    with pytest.raises(ValueError):
        OutcomeGate(list(range(20)), None)
    with pytest.raises(ValueError):
        OutcomeGate([1]*360, None)
