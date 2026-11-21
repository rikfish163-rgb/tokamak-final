"""CPU regressions for the separate finite-step forecast-head prior."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import fast_meta, io23, simple_data
from experiments.risk_aligned_online.fast_direction import episode
from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004 import prepare_prior as common
from experiments.risk_aligned_online.redesign_20261004 import prepare_finite_prior as finite


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def setup():
    torch.manual_seed(53)
    cfg = Config(window=128, patch=16, dim=64, heads=4, layers=1,
                 horizon=25, fast_site='tokens', fast_rule='normalized')
    net = Simple23(cfg, np.linspace(-.2, .2, 23), np.linspace(.5, 1.5, 23)).eval()
    with torch.no_grad():
        net.fast_origin.copy_(torch.linspace(-.3, .3, 128))
    net.requires_grad_(False)
    shot = dict(shot=1, device='J-TEXT',
                values=np.random.default_rng(53).normal(size=(190, 23)).astype('float32'),
                available=np.ones((190, 23), bool), valid=np.ones(190, bool),
                y=np.r_[np.zeros(150), np.ones(40)].astype('float32'))
    return net, episode(shot, 128, cfg, 'cpu'), shot


@pytest.mark.parametrize('forecast_weight', [0., .1])
def test_finite_objective_matches_undetached_manual_meta_gradient(setup, forecast_weight):
    net, e, _ = setup
    reference = deepcopy(net)
    for p in reference.forecast_head.parameters():
        p.requires_grad_(True)
    updated, signal, _, _ = fast_meta.temporary_step_components(reference, e, create_graph=True)
    expected_risk = F.binary_cross_entropy_with_logits(reference(e['qx'], e['qa'], updated)['logit'], e['y'])
    expected_objective = expected_risk + forecast_weight*signal
    expected_objective.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.forecast_head.parameters(), 1., error_if_nonfinite=True)
    expected_grad = [p.grad.detach().clone() for p in reference.forecast_head.parameters()]
    before = deepcopy(net.state_dict())
    optimizer = torch.optim.SGD(net.forecast_head.parameters(), lr=.01)
    log = finite.train_step(net, optimizer, e, forecast_weight=forecast_weight)
    assert norm > 0 and log['head_grad_norm'] > 0
    assert log['objective'] == pytest.approx(expected_objective.item(), abs=1e-7)
    assert log['risk_after'] == pytest.approx(expected_risk.item(), abs=1e-7)
    for p, expected in zip(net.forecast_head.parameters(), expected_grad):
        torch.testing.assert_close(p.grad, expected, rtol=1e-6, atol=1e-8)
    assert not torch.equal(net.forecast_head.weight, before['forecast_head.weight'])
    assert all(p.grad is None for name, p in net.named_parameters() if not name.startswith('forecast_head.'))


def test_static_classifier_scales_origin_logits_and_flags_are_unchanged(setup):
    net, e, _ = setup
    frozen = common.classifier_state(net)
    cfg = net.cfg.to_dict()
    flags = [p.requires_grad for p in net.parameters()]
    with torch.no_grad():
        risk_before = net(e['qx'], e['qa'])['logit'].clone()
    log = finite.train_step(net, torch.optim.AdamW(net.forecast_head.parameters(), lr=.001, weight_decay=0.), e)
    common.assert_classifier_unchanged(net, frozen)
    assert net.cfg.to_dict() == cfg and not net.training
    assert [p.requires_grad for p in net.parameters()] == flags
    assert log['fast_step'] == .05 and log['fast_radius'] == 1.
    with torch.no_grad():
        assert torch.equal(net(e['qx'], e['qa'])['logit'], risk_before)


def test_risk_audit_dispatch_matches_and_after_keeps_outer_graph(setup, monkeypatch):
    net, e, _ = setup
    forward, query_paths = net.forward, []
    def dispatch_trap(values, available, fast=None):
        result = forward(values, available, fast)
        if values is e['qx']:
            path = (torch.is_grad_enabled(), fast.requires_grad, fast.is_leaf)
            query_paths.append(path)
            if path[:2] != (True, True):
                result = dict(result, logit=result['logit']+.002)
        return result
    monkeypatch.setattr(net, 'forward', dispatch_trap)
    with torch.no_grad():
        log = finite.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01), e, forecast_weight=0.)
        assert not torch.is_grad_enabled()
    assert query_paths == [(True, True, True), (True, True, False)]
    assert log['head_grad_norm'] > 0


def test_exact_source_step_batch_replication_and_zero_delta_coordinates(setup):
    from experiments.risk_aligned_online.redesign_20261004.model import ContextModel
    net, e, _ = setup
    updated, _, fast, gradient = common.temporary_step_components(net, e, create_graph=True)
    expected = normalized_step(fast, fast.detach(), gradient, step=.05, radius=1.)
    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
    assert fast.shape == (1, 128) and torch.equal(fast, net.initial_fast(1))
    assert (updated-fast).norm().item() <= .050001
    doubled = {k: torch.cat([v, v]) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
    twice, _, _, _ = common.temporary_step_components(net, doubled, create_graph=True)
    torch.testing.assert_close(updated, twice[:1], rtol=2e-4, atol=1e-6)
    context = ContextModel(net).eval()
    delta, _, zero, context_gradient = fast_meta.temporary_step_components(context, e, create_graph=True)
    assert not zero.any()
    torch.testing.assert_close(gradient, context_gradient, rtol=0, atol=0)
    torch.testing.assert_close(updated, net.initial_fast(1)+delta, rtol=1e-6, atol=5e-8)


def test_zero_inner_gradient_has_finite_double_backward(setup):
    net, e, _ = setup
    with torch.no_grad():
        net.forecast_head.weight.zero_()
    log = finite.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.0001), e, forecast_weight=0.)
    assert log['signal_grad_norm'] == 0.
    assert log['fast_delta_norm'] == 0.
    assert np.isfinite(log['head_grad_norm']) and np.isfinite(log['objective'])
    assert all(torch.isfinite(p.grad).all() for p in net.forecast_head.parameters())


def test_risk_labels_change_only_outer_gradient_not_inner_step(setup):
    net, e, _ = setup
    other = deepcopy(net)
    first, _, _, _ = common.temporary_step_components(net, e, create_graph=False)
    second, _, _, _ = common.temporary_step_components(net, dict(e, y=1-e['y']), create_graph=False)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    a = finite.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01), e, forecast_weight=0.)
    b = finite.train_step(other, torch.optim.SGD(other.forecast_head.parameters(), lr=.01),
                          dict(e, y=1-e['y']), forecast_weight=0.)
    assert a['head_grad_norm'] > 0 and b['head_grad_norm'] > 0
    assert not torch.equal(net.forecast_head.weight, other.forecast_head.weight)


def test_optimizer_cannot_update_classifier_and_flags_restore_on_failure(setup):
    net, e, _ = setup
    frozen = common.classifier_state(net)
    with pytest.raises(ValueError, match='forecast_head'):
        finite.train_step(net, torch.optim.AdamW(net.parameters(), lr=.001), e)
    common.assert_classifier_unchanged(net, frozen)
    with pytest.raises(FloatingPointError, match='nonfinite'):
        finite.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01),
                          dict(e, y=torch.full_like(e['y'], float('nan'))))
    assert not any(p.requires_grad for p in net.parameters())
    common.assert_classifier_unchanged(net, frozen)


class SourceOnlyData:
    def __init__(self):
        self.events = []

    def source_roles(self):
        return dict(train=[1, 2], val=[3], test=[4])

    def advanced_times(self):
        class TrainTimes(dict):
            def __getitem__(self, key):
                assert key in (1, 2), 'holdout AdvancedTime access'
                return super().__getitem__(key)
        return TrainTimes({1: 5., 2: 40., 3: 100., 4: 100.})

    def read_shot(self, device, sid):
        assert device == 'J-TEXT' and sid in (1, 2)
        self.events.append(('signals', sid))
        return dict(shot=sid, device=device,
                    values=np.random.default_rng(sid).normal(size=(200, 23)).astype('float32'),
                    available=np.ones((200, 23), bool), times=np.arange(200)/1000.)

    def attach_outcome(self, shot, device, advanced):
        assert set(advanced) == {1, 2}
        self.events.append(('label', shot['shot']))
        return simple_data.attach_outcome(shot, device, advanced)


def test_cpu_main_uses_train_only_and_saves_compatible_truthful_prior(setup, tmp_path, monkeypatch):
    net, _, _ = setup
    frozen = common.classifier_state(net)
    api, loader = SourceOnlyData(), common.load_source_train
    def source_outcome(device, sid):
        assert device == 'J-TEXT' and sid in (1, 2)
        return True, .199
    monkeypatch.setattr(io23, 'read_outcome', source_outcome)
    monkeypatch.setattr(common, 'load_parent', lambda path: (net, {}))
    monkeypatch.setattr(common, 'load_source_train', lambda: loader(data_api=api, expected_train_count=2))
    output = tmp_path/'finite-prior'
    report = finite.main(['--output', str(output), '--steps', '2', '--batch-size', '2'])
    assert api.events == [('signals', 1), ('label', 1), ('signals', 2), ('label', 2)]
    saved = torch.load(output/'prior.pt', map_location='cpu', weights_only=True)
    assert set(saved) == {'forecast_head', 'protocol'}
    assert set(saved['forecast_head']) == {'weight', 'bias'}
    protocol = saved['protocol']
    assert protocol['objective'] == 'finite_step_BCE + forecast_weight * direct_H25_SmoothL1'
    assert protocol['forecast_weight'] == .1 and protocol['alignment_weight'] == 0.
    assert protocol['risk_after_graph_retained'] is True
    assert protocol['source_only'] and protocol['static_classifier_bitwise_unchanged']
    assert protocol['parent_sha256'] == common.PARENT_SHA256
    assert protocol['train_ids'] == [1, 2] and protocol['episodes_seen'] == 4
    assert protocol['source_label'] == 'max(25ms,AdvancedTime)'
    assert protocol['source_qualification_lead_ms'] == 10.
    assert protocol['source_model_config'] == net.cfg.to_dict()
    assert protocol['fast_step'] == .05 and protocol['fast_radius'] == 1.
    for name, path in finite.code_paths().items():
        assert protocol['code_sha256'][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['prior_sha256'] == hashlib.sha256((output/'prior.pt').read_bytes()).hexdigest()
    assert json.loads((output/'training.json').read_text()) == report
    assert len(report['history']) == 2
    common.assert_classifier_unchanged(net, frozen)
    with pytest.raises(FileExistsError):
        finite.main(['--output', str(output)])


def test_provenance_builder_does_not_inherit_cosine_objective_or_hide_mutation(setup):
    net, _, _ = setup
    frozen = common.classifier_state(net)
    metadata = dict(train_ids=[1], parent_sha256=common.PARENT_SHA256,
        code_sha256={'prepare_finite_prior.py': 'a'*64}, steps=128, batch_size=8,
        seed=131, lr=1e-4, forecast_weight=0., episode_sha256='b'*64)
    artifact = finite.build_prior(net, frozen, **metadata)
    assert 'cosine' not in artifact['protocol']['objective']
    assert artifact['protocol']['alignment_weight'] == 0.
    assert artifact['protocol']['forecast_weight'] == 0.
    with torch.no_grad():
        net.scale[0] += 1.
    with pytest.raises(AssertionError, match='scale'):
        finite.build_prior(net, frozen, **metadata)


def test_parent_sha_rejected_before_deserialization_or_data_access(tmp_path, monkeypatch):
    checkpoint = tmp_path/'wrong-parent.pt'
    checkpoint.write_bytes(b'not the trusted source')
    def forbidden(*args, **kwargs):
        raise AssertionError('wrong parent was loaded or labels were accessed')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(common, 'load_source_train', forbidden)
    with pytest.raises(ValueError, match='SHA256'):
        finite.main(['--checkpoint', str(checkpoint), '--output', str(tmp_path/'output')])


def test_training_code_change_fails_before_output(setup, tmp_path, monkeypatch):
    net, _, shot = setup
    changed, real_sha, real_train = [False], common.sha256, finite.train_step
    monkeypatch.setattr(common, 'load_parent', lambda path: (net, {}))
    monkeypatch.setattr(common, 'load_source_train', lambda: ([shot], [1]))
    def tracked_sha(path):
        if changed[0] and Path(path).name == 'prepare_finite_prior.py':
            return '0'*64
        return real_sha(path)
    def mark_changed(*args, **kwargs):
        result = real_train(*args, **kwargs)
        changed[0] = True
        return result
    monkeypatch.setattr(common, 'sha256', tracked_sha)
    monkeypatch.setattr(finite, 'train_step', mark_changed)
    output = tmp_path/'must-not-be-created'
    with pytest.raises(AssertionError, match='code changed'):
        finite.main(['--output', str(output), '--steps', '1', '--batch-size', '2'])
    assert not output.exists()


def test_cli_defaults_maintenance_optional_and_invalid_budget():
    parser = finite.make_parser()
    args = parser.parse_args(['--output', '/tmp/not-executed-finite-prior'])
    assert args.steps == 128 and args.batch_size == 8 and args.device == 'cpu'
    assert args.forecast_weight == .1 and args.lr == 1e-4 and args.seed == 131
    common.validate_args(parser.parse_args(['--output', '/tmp/not-executed', '--forecast-weight', '0']))
    for extra in (['--steps', '0'], ['--batch-size', '0'], ['--lr', 'nan'], ['--forecast-weight', '-1']):
        with pytest.raises(ValueError):
            common.validate_args(parser.parse_args(['--output', '/tmp/not-executed', *extra]))


def test_frozen_cosine_trainer_hash_unchanged():
    assert common.sha256(common.__file__) == '469c568e685ebf6d9451eb5ccdeddcca98eb8195897934814d7eb8b13952a49f'


def test_locked_real_parent_on_synthetic_episode_cpu_only(setup):
    _, e, _ = setup
    net, parent = common.load_parent(common.DEFAULT_PARENT)
    assert net.center.device.type == 'cpu'
    assert net.cfg.to_dict() == parent['model_config']
    frozen = common.classifier_state(net)
    with torch.no_grad():
        original = net(e['qx'], e['qa'])['logit'].clone()
    # A synthetic opposite-class label ensures the gradient probe is not
    # hidden by a confidently solved example. No source/target labels are read.
    synthetic = dict(e, y=(original < 0).to(e['y'].dtype))
    log = finite.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01),
                            synthetic, forecast_weight=0.)
    assert log['head_grad_norm'] > 0 and np.isfinite(log['head_grad_norm'])
    common.assert_classifier_unchanged(net, frozen)
    with torch.no_grad():
        assert torch.equal(original, net(e['qx'], e['qa'])['logit'])
