import copy

import numpy as np
import pytest
import torch
from torch import nn

from experiments.risk_aligned_online.context_vector_20261030 import stream


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Module()
        self.base.register_buffer('center', torch.zeros(23))

    def forward(self, x, mask, states, slow, mode, forecast=False):
        return dict(logit=x[:, -1, 0]+states[:, 0])


def signal(n=210):
    values = np.zeros((n, 23), np.float32)
    values[:, 0] = np.arange(n)/200
    return dict(values=values, available=np.ones_like(values, bool))


@pytest.fixture
def fake_update(monkeypatch):
    calls = []
    def update(net, prefix, origin, slow, policies):
        assert set(prefix) == {'values', 'available'}
        assert len(prefix['values']) == origin+26
        calls.append(origin)
        states = {}
        for v in stream.VARIANTS:
            a = torch.zeros(9, 8192)
            if prefix['available'][-25:].any():
                a[1:6, 0] = float(prefix['values'][-1, 0])
            states[v] = a
        return states, dict(state_norm={key: float(states[v][i].norm()) for v in stream.VARIANTS
                            for i, arm in enumerate(stream.ARMS) if i in range(1,6)
                            for key in (v+'__'+arm,)}, cos_real={}, max_bound=2.)
    monkeypatch.setattr(stream, 'new_states', update)
    return calls


def test_causal_maturity_replacement_and_reset(fake_update):
    x = signal()
    net = ToyNet()
    curves, receipts = stream.predict(net, x, None, {})
    assert fake_update == [127, 152, 177]
    assert [(r['maturity'], r['first_affected'], r['apply_count']) for r in receipts] == [(152,153,25),(177,178,25),(202,203,7)]
    assert set(curves) == set(stream.NAMES)
    np.testing.assert_array_equal(curves['cold'], curves['identical'])
    real = curves['raw_fixed__real']
    assert real[152] == curves['cold'][152]
    assert real[153] > curves['cold'][153]
    same, _ = stream.predict(net, x, None, {})
    for key in curves:
        np.testing.assert_array_equal(curves[key], same[key])


def test_future_signal_does_not_rewrite_predictions(fake_update):
    original = signal()
    changed = copy.deepcopy(original)
    changed['values'][165:] += 100
    a, _ = stream.predict(ToyNet(), original, None, {})
    b, _ = stream.predict(ToyNet(), changed, None, {})
    short, _ = stream.predict(ToyNet(), {k:v[:165] for k,v in original.items()}, None, {})
    for key in a:
        np.testing.assert_array_equal(a[key][:165], b[key][:165])
        np.testing.assert_array_equal(a[key][:165], short[key])


def test_no_feedback_explicitly_resets_temporary_state(fake_update):
    x = signal(190)
    x['available'][153:178] = False
    curves, _ = stream.predict(ToyNet(), x, None, {})
    assert curves['raw_fixed__real'][177] > curves['cold'][177]
    np.testing.assert_array_equal(curves['raw_fixed__real'][178:], curves['cold'][178:])


@pytest.mark.parametrize('size,counts', [(1,[]),(128,[]),(152,[]),(153,[0]),(154,[1])])
def test_short_or_partial_feedback(size, counts, fake_update):
    curves, records = stream.predict(ToyNet(), signal(size), None, {})
    assert [r['apply_count'] for r in records] == counts
    assert all(len(v) == size for v in curves.values())


def test_label_input_rejected(fake_update):
    with pytest.raises(ValueError):
        stream.predict(ToyNet(), dict(signal(), outcome=1), None, {})
