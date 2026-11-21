"""Signal-only training, risk projection, and the actual feedback clock."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.simple23 import Config, Simple23
from joint_trajectory_20261107 import JointTrajectory
from run_joint_trajectory_20261107 import (
    ARMS, eligible_origins, energy, features, joint_curves, sample_signals,
    trajectory_risks, update, virtual_risks,
)


def base_model():
    torch.manual_seed(71)
    return Simple23(Config(dim=8, heads=4, layers=1, fast_site='tokens'),
                    torch.zeros(23), torch.ones(23)).eval().requires_grad_(False)


def signal(n=164):
    values = np.random.default_rng(72).normal(0., .15, (n, 23)).astype(np.float32)
    values[:, [17, 22]] = abs(values[:, [17, 22]])
    available = np.ones((n, 23), bool)
    available[:, 2] = False
    return dict(values=values, available=available)


def test_uniform_source_sampler_ignores_risk_labels_and_respects_complete_future():
    s = signal(190)
    s.update(y=np.zeros(190), valid=np.arange(190) < 180)
    first = eligible_origins([s])
    s['y'][:] = 1
    second = eligible_origins([s])
    np.testing.assert_array_equal(first[0][1], second[0][1])
    assert first[0][1][0] == 127
    assert first[0][1][-1] == 154
    episode = sample_signals([s], first, np.random.default_rng(9), 4, 'cpu')
    assert episode['x'].shape == (4, 128, 23)
    assert episode['truth'].shape == (4, 25, 23)
    assert not {'y', 'down', 'is_disrupt'} & set(episode)
    for i, origin in enumerate(episode['origin'].tolist()):
        torch.testing.assert_close(episode['x'][i], torch.from_numpy(s['values'][origin-127:origin+1]))
        torch.testing.assert_close(episode['truth'][i], torch.from_numpy(s['values'][origin+1:origin+26]))


def test_feature_capture_keeps_actual_source_risk_and_mask():
    base, s = base_model(), signal(128)
    x, a = torch.from_numpy(s['values'])[None], torch.from_numpy(s['available'])[None]
    tokens, valid, logits = features(base, x, a)
    assert tokens.shape == (1, 17, 8)
    assert valid.shape == (1, 17) and valid[:, -1].all()
    torch.testing.assert_close(logits, base(x, a)['logit'])
    assert not tokens.requires_grad
    assert len(base.forecast_head._forward_pre_hooks) == 0


def test_source_feature_capture_preserves_requested_internal_gradient_path():
    base, s = base_model(), signal(128)
    weight = base.encoder.layers[0].linear2.weight
    weight.requires_grad_(True)
    x, a = torch.from_numpy(s['values'])[None], torch.from_numpy(s['available'])[None]
    tokens, _, _ = features(base, x, a)
    assert tokens.requires_grad
    tokens[:, -1, 0].sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(weight.grad) > 0


def test_training_and_feedback_use_the_predictors_actual_objective():
    from joint_trajectory_20261107 import joint_energy_score

    class DoubleScore(JointTrajectory):
        def loss(self, encoded, probabilities, target, available):
            return 2*joint_energy_score(encoded, probabilities, target, available)

    base, s = base_model(), signal(190)
    e = sample_signals([s], eligible_origins([s]), np.random.default_rng(5), 2, 'cpu')
    e['tokens'], e['token_valid'], _ = features(base, e['x'], e['a'])
    model = DoubleScore(base.center, base.scale, dim=8, scenarios=4)
    output = model(e['tokens'], e['token_valid'], e['x'][:, -1], e['a'][:, -1])
    mask = e['mask'] & output['available']
    target = torch.asinh((torch.where(mask, e['truth'], model.center)-model.center)/model.scale)
    expected = 2*joint_energy_score(output['encoded'], output['probabilities'], target, mask).mean()
    torch.testing.assert_close(energy(model, e), expected)


def test_joint_risk_uses_whole_scenarios_not_risk_of_average_signal():
    class CurvedRisk(torch.nn.Module):
        def forward(self, x, a):
            return {'logit': x[:, -1, 0].square() - 1.}

    x = torch.zeros(1, 128, 23)
    a = torch.ones_like(x, dtype=torch.bool)
    raw = torch.zeros(1, 2, 25, 23)
    raw[:, 0, :, 0], raw[:, 1, :, 0] = -2., 2.
    output = dict(raw=raw, available=torch.ones(1, 25, 23, dtype=torch.bool),
                  probabilities=torch.tensor([[.5, .5]]))
    result = trajectory_risks(CurvedRisk(), x, a, output)
    torch.testing.assert_close(result['joint'], torch.sigmoid(torch.tensor([3.])))
    torch.testing.assert_close(result['mean'], torch.sigmoid(torch.tensor([-1.])))


def test_nonfinite_generated_signal_is_not_silently_reclassified_as_missing():
    base = base_model()
    x = torch.zeros(1, 128, 23)
    observed = torch.ones_like(x, dtype=torch.bool)
    raw = torch.zeros(1, 1, 25, 23)
    raw[0, 0, 3, 4] = float('inf')
    with pytest.raises(FloatingPointError, match='not a missing diagnostic'):
        virtual_risks(base, x, observed, raw, torch.ones(1, 25, 23, dtype=torch.bool))


def test_real_update_changes_temporal_and_generation_weights_but_probability_control_does_not():
    base, s = base_model(), signal(190)
    episode = sample_signals([s], eligible_origins([s]), np.random.default_rng(4), 2, 'cpu')
    episode['tokens'], episode['token_valid'], _ = features(base, episode['x'], episode['a'])
    torch.manual_seed(10)
    full = JointTrajectory(base.center, base.scale, dim=8, scenarios=4)
    probability = deepcopy(full)
    original = {k: v.detach().clone() for k, v in full.named_parameters()}
    probability.requires_grad_(False).probability.requires_grad_(True)
    for model in (full, probability):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-4, weight_decay=0.)
        info = update(model, opt, episode, original)
        assert np.isfinite(info['energy'])
    assert not torch.equal(full.temporal.self_attn.in_proj_weight, original['temporal.self_attn.in_proj_weight'])
    assert not torch.equal(full.trajectory.weight, original['trajectory.weight'])
    for name, parameter in probability.named_parameters():
        if not name.startswith('probability.'):
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


def test_actual_intra_replay_seals_feedback_and_resets_temporary_parameters():
    base, s = base_model(), signal()
    predictor = JointTrajectory(base.center, base.scale, dim=8, scenarios=4).eval()
    before = {k: v.detach().clone() for k, v in predictor.state_dict().items()}
    traces, info = joint_curves(base, predictor, s)
    assert set(traces) == set(ARMS)
    assert info['release_indices'] == [153]
    assert info['updates']['intra'] == 1
    for prefix in (152, 154):
        truncated, _ = joint_curves(base, predictor, {k: v[:prefix] for k, v in s.items()})
        for name in traces:
            np.testing.assert_allclose(traces[name][:prefix], truncated[name], atol=3e-6, rtol=0)
    np.testing.assert_allclose(traces['joint_intra'][:153], traces['joint_frozen'][:153], atol=3e-6, rtol=0)
    assert max(abs(traces['joint_intra'][153:]-traces['joint_frozen'][153:])) > 0
    assert info['parameter_change']['intra']['temporal'] > 0
    assert info['parameter_change']['intra']['trajectory'] > 0
    for key, value in predictor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_cold_start_shots_never_use_predictor_or_feedback():
    base, s = base_model(), signal()
    predictor = JointTrajectory(base.center, base.scale, dim=8, scenarios=4).eval()
    traces, info = joint_curves(base, predictor, s, warm=True)
    assert not info['release_indices']
    assert not any(info['updates'].values())
    for value in traces.values():
        np.testing.assert_array_equal(value, traces['static'])


def test_selected_feedback_branches_keep_both_readouts_and_reproduce_full_replay():
    import run_joint_trajectory_20261107 as runner
    from joint_motion_20261107 import OrderedTrajectory

    base, s = base_model(), signal()
    predictor = OrderedTrajectory(base.center, base.scale, dim=8, scenarios=4, motion_weight=.472)
    full, _ = joint_curves(base, predictor, s)
    context = runner.prediction_context(base, s)
    selected, info = runner.adaptive_curves(base, predictor, s, context,
        arms=('probability_only', 'wrong_time'))
    assert set(selected) == {'joint_probability_only', 'mean_probability_only',
                             'joint_wrong_time', 'mean_wrong_time'}
    assert info['updates'] == {'probability_only': 1, 'wrong_time': 1}
    for name in ('joint_probability_only', 'joint_wrong_time'):
        np.testing.assert_allclose(selected[name], full[name], atol=3e-6, rtol=0)
    for name in selected:
        np.testing.assert_allclose(selected[name][:127], full['static'][:127], atol=0, rtol=0)
