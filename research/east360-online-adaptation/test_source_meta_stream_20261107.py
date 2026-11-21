from copy import deepcopy

import numpy as np
import torch

from experiments.risk_aligned_online.fast_meta import temporary_step
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.simple_data import window
from run_source_meta_stream_20261107 import native_curves, schedule


def case(n=220):
    torch.manual_seed(17)
    model = Simple23(Config(dim=8, heads=2, layers=1, fast_site='tokens',
                            fast_rule='normalized'), np.zeros(23), np.ones(23))
    model.eval().requires_grad_(False)
    signal = dict(values=np.random.default_rng(17).normal(size=(n, 23)).astype('float32'),
                  available=np.ones((n, 23), bool))
    return model, signal


def test_complete_horizon_precedes_the_first_adapted_prediction():
    assert not len(schedule(153)[0])
    origins, queries = schedule(154)
    np.testing.assert_array_equal(origins, [127])
    np.testing.assert_array_equal(queries, [153])
    assert np.all(origins + 25 < queries)


def test_native_stream_matches_the_source_training_inner_step():
    model, signal = case()
    frozen = deepcopy(model.state_dict())
    curves, info = native_curves(model, signal)
    x, a = window(signal, 127, 128)
    qx, qa = window(signal, 153, 128)
    episode = {k: torch.as_tensor(v[None]) for k, v in dict(
        x=x, a=a, truth=signal['values'][128:153], mask=signal['available'][128:153]).items()}
    fast, _ = temporary_step(model, episode, create_graph=False)
    with torch.no_grad():
        expected = model(torch.as_tensor(qx[None]), torch.as_tensor(qa[None]), fast)['logit'].sigmoid()
    np.testing.assert_allclose(curves['intra'][153], expected.item(), rtol=1e-5, atol=1e-7)
    assert info['updates'] > 0
    assert info['state_norm_max'] > 0
    assert info['wrong_norm_error'] < 1e-6
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


def test_truncation_and_changed_future_do_not_change_sealed_prefix():
    model, signal = case()
    full, _ = native_curves(model, signal)
    stop = 190
    prefix, _ = native_curves(model, {k: v[:stop].copy() for k, v in signal.items()})
    changed = {k: v.copy() for k, v in signal.items()}
    changed['values'][stop:] += 10000
    other, _ = native_curves(model, changed)
    for name in full:
        np.testing.assert_allclose(prefix[name], full[name][:stop], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(other[name][:stop], full[name][:stop], rtol=0, atol=0)


def test_preparation_shots_remain_static_for_every_control():
    model, signal = case()
    curves, info = native_curves(model, signal, warm=True)
    for curve in curves.values():
        np.testing.assert_array_equal(curve, curves['static'])
    assert info['updates'] == 0
