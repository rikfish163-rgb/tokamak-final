"""The source unroll and deployed direction update must be the same algorithm."""

from copy import deepcopy

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from test_risk_meta_step_20261107 import teacher_case
from test_risk_meta_sequence_20261108 import episode_sequence
from risk_feature_intra_20261107 import update
from risk_meta_sequence_20261108 import teacher_sequence
from risk_direction_intra_20261108 import DirectionAdamW, direction_curves


def test_direction_sequence_matches_real_accumulated_optimizer_and_preserves_templates():
    model, _ = teacher_case()
    original = deepcopy(model.state_dict())
    native = deepcopy(model)
    native.decoder.requires_grad_(False)
    episodes, qx, qa = episode_sequence()
    opt = DirectionAdamW(native.weight, lr=1e-4)
    with sdpa_kernel(SDPBackend.MATH):
        result = teacher_sequence(model, episodes, qx, qa, torch.ones(1),
                                  credit='last', step_rule='momentum_direction')
        for expected, episode in zip(result['steps'], episodes, strict=True):
            update(native, opt, episode)
            torch.testing.assert_close(expected['weight'], native.weight, atol=3e-7, rtol=0)
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(expected[key], opt.state[native.weight][key], atol=3e-7, rtol=1e-5)
        torch.testing.assert_close(result['logit'], native.risk(qx, qa)['logit'], atol=4e-6, rtol=0)
        grads = torch.autograd.grad(result['risk_loss'], tuple(model.decoder.parameters()))
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_direction_meta_keeps_inner_updates_label_free_and_credits_forward_identical():
    model, _ = teacher_case()
    episodes, qx, qa = episode_sequence()
    with sdpa_kernel(SDPBackend.MATH):
        results = [teacher_sequence(model, episodes, qx, qa, torch.tensor([float(y)]),
                                   credit=credit, step_rule='momentum_direction')
                   for y, credit in ((0, 'full'), (1, 'last'), (1, 'none'))]
        for other in results[1:]:
            torch.testing.assert_close(results[0]['logit'], other['logit'], atol=0, rtol=0)
            for a, b in zip(results[0]['steps'], other['steps'], strict=True):
                torch.testing.assert_close(a['weight'], b['weight'], atol=0, rtol=0)
        none = torch.autograd.grad(results[-1]['risk_loss'], tuple(model.decoder.parameters()), allow_unused=True)
        assert all(g is None or g.count_nonzero() == 0 for g in none)


def test_direction_clock_prefix_reset_and_warm_are_causal():
    from test_joint_stream_20261107 import signal
    import numpy as np

    model, _ = teacher_case()
    s = signal(193)
    with sdpa_kernel(SDPBackend.MATH):
        curves, info = direction_curves(model.risk, model.decoder, s)
        assert info['releases'] == [153, 185]
        assert info['updates']['real'] == 2
        np.testing.assert_allclose(curves['real'][:153], curves['static'][:153], atol=2e-6, rtol=0)
        assert abs(curves['real'][153:]-curves['static'][153:]).max() > 0
        prefix, _ = direction_curves(model.risk, model.decoder, {k:v[:154] for k,v in s.items()})
        np.testing.assert_allclose(prefix['real'], curves['real'][:154], atol=2e-6, rtol=0)
        again, _ = direction_curves(model.risk, model.decoder, s)
        np.testing.assert_array_equal(again['real'], curves['real'])
        warm, warm_info = direction_curves(model.risk, model.decoder, s, warm=True)
        np.testing.assert_array_equal(warm['real'], warm['static'])
        assert warm_info['updates']['real'] == 0
