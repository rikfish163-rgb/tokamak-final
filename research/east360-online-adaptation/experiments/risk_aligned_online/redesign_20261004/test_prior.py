"""CPU-only regression tests for the optional, source-TRAIN-only prior."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import fast_meta, io23, simple_data
from experiments.risk_aligned_online.fast_direction import episode
from experiments.risk_aligned_online.fast_update import normalized_step
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004 import prepare_prior as prior


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def setup():
    torch.manual_seed(41)
    cfg = Config(window=128, patch=16, dim=64, heads=4, layers=1,
                 horizon=25, fast_site='tokens', fast_rule='normalized')
    net = Simple23(cfg, np.linspace(-.2, .2, 23), np.linspace(.5, 1.5, 23)).eval()
    with torch.no_grad():
        net.fast_origin.copy_(torch.linspace(-.3, .3, 128))
    shot = dict(shot=1, device='J-TEXT',
                values=np.random.default_rng(41).normal(size=(190, 23)).astype('float32'),
                available=np.ones((190, 23), bool), valid=np.ones(190, bool),
                y=np.r_[np.zeros(150), np.ones(40)].astype('float32'))
    e = episode(shot, 128, cfg, 'cpu')
    return net, e, shot


def test_alignment_detaches_risk_direction_and_has_double_backward():
    signal = torch.tensor([[1., 0.]], dtype=torch.float64, requires_grad=True)
    risk = torch.tensor([[0., 1.]], dtype=torch.float64, requires_grad=True)
    loss = prior.alignment_loss(signal, risk)
    first, = torch.autograd.grad(loss, signal, create_graph=True)
    second, = torch.autograd.grad(first.square().sum(), signal)
    assert first[0, 1] < 0
    assert torch.isfinite(second).all()
    assert risk.grad is None
    assert torch.autograd.gradcheck(lambda s: prior.alignment_loss(s, risk), (signal,))
    assert torch.autograd.gradgradcheck(lambda s: prior.alignment_loss(s, risk), (signal,))


@pytest.mark.parametrize('signal,risk', [([0., 0.], [1., 0.]),
                                        ([1., 0.], [0., 0.]),
                                        ([0., 0.], [0., 0.])])
def test_alignment_zero_norm_is_finite_to_second_order(signal, risk):
    signal = torch.tensor([signal], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([risk], dtype=torch.float64)
    loss = prior.alignment_loss(signal, target)
    first, = torch.autograd.grad(loss, signal, create_graph=True)
    second, = torch.autograd.grad(first.sum(), signal)
    assert torch.isfinite(loss) and torch.isfinite(first).all() and torch.isfinite(second).all()
    if not target.any():
        assert loss.item() == 0 and not first.any()


def test_actual_step_matches_source_and_zero_delta_coordinates(setup):
    net, e, _ = setup
    updated, loss, fast, gradient = prior.temporary_step_components(net, e, create_graph=True)
    original, original_loss, _, _ = fast_meta.temporary_step_components(net, e, create_graph=True)
    expected = normalized_step(fast, fast.detach(), gradient, step=.05, radius=1.)
    torch.testing.assert_close(updated, original, rtol=0, atol=0)
    torch.testing.assert_close(loss, original_loss, rtol=0, atol=0)
    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
    assert fast.shape == (1, 128)
    assert torch.equal(fast, net.initial_fast(1))
    assert (updated-fast).norm().item() <= .050001
    duplicated = {k: torch.cat([v, v]) if isinstance(v, torch.Tensor) else v for k, v in e.items()}
    twice, _, _, _ = prior.temporary_step_components(net, duplicated, create_graph=True)
    torch.testing.assert_close(updated, twice[:1], rtol=2e-4, atol=1e-6)


def test_inner_step_does_not_access_risk_labels(setup):
    net, e, _ = setup
    class NoLabel(dict):
        def __getitem__(self, key):
            if key == 'y':
                raise AssertionError('inner step read risk label')
            return super().__getitem__(key)
    first, _, _, _ = prior.temporary_step_components(net, NoLabel(e), create_graph=False)
    second, _, _, _ = prior.temporary_step_components(net, dict(e, y=1-e['y']), create_graph=False)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_context_zero_delta_has_same_gradients_and_actual_step(setup):
    from experiments.risk_aligned_online.redesign_20261004.model import ContextModel
    net, e, _ = setup
    wrapper = ContextModel(net).eval()
    absolute, _, _, source_gradient = prior.temporary_step_components(net, e, create_graph=True)
    delta, _, initial, context_gradient = fast_meta.temporary_step_components(wrapper, e, create_graph=True)
    assert not initial.any()
    torch.testing.assert_close(source_gradient, context_gradient, rtol=0, atol=0)
    torch.testing.assert_close(absolute, net.initial_fast(1)+delta, rtol=1e-6, atol=5e-8)
    source = net(e['qx'], e['qa'], absolute)
    adapted = wrapper(e['qx'], e['qa'], delta)
    for key in source:
        torch.testing.assert_close(source[key], adapted[key], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('forecast_weight', [0., .1])
def test_training_changes_only_forecast_head_and_preserves_static_logits(setup, forecast_weight):
    net, e, _ = setup
    for p in net.parameters():
        p.requires_grad_(False)
    frozen = prior.classifier_state(net)
    before = deepcopy(net.state_dict())
    config = net.cfg.to_dict()
    with torch.no_grad():
        logits = net(e['qx'], e['qa'])['logit'].clone()
    optimizer = torch.optim.AdamW(net.forecast_head.parameters(), lr=1e-3, weight_decay=0.)
    log = prior.train_step(net, optimizer, e, forecast_weight=forecast_weight)
    assert log['head_grad_norm'] > 0
    assert log['fast_step'] == .05 and log['fast_radius'] == 1.
    assert all(np.isfinite(v) for v in log.values())
    changed = [k for k, v in net.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith('forecast_head.') for k in changed)
    prior.assert_classifier_unchanged(net, frozen)
    assert net.cfg.to_dict() == config
    assert all(not p.requires_grad for p in net.parameters())
    with torch.no_grad():
        assert torch.equal(logits, net(e['qx'], e['qa'])['logit'])


def test_optimizer_cannot_include_classifier(setup):
    net, e, _ = setup
    before = deepcopy(net.state_dict())
    with pytest.raises(ValueError, match='forecast_head'):
        prior.train_step(net, torch.optim.AdamW(net.parameters(), lr=.001), e)
    assert all(torch.equal(v, before[k]) for k, v in net.state_dict().items())


def test_risk_step_audit_uses_matching_forward_dispatch(setup, monkeypatch):
    net, e, _ = setup
    original = net.forward
    query_paths = []
    def numerical_dispatch_trap(values, available, fast=None):
        output = original(values, available, fast)
        if values is e['qx']:
            path = (torch.is_grad_enabled(), fast.requires_grad)
            query_paths.append(path)
            if path != (True, True):
                output = dict(output, logit=output['logit'] + .002)
        return output
    monkeypatch.setattr(net, 'forward', numerical_dispatch_trap)
    optimizer = torch.optim.SGD(net.forecast_head.parameters(), lr=.0001)
    prior.train_step(net, optimizer, e)
    assert query_paths == [(True, True), (True, True)]


def test_classifier_audit_detects_even_signed_zero_buffer_change(setup):
    net, _, _ = setup
    with torch.no_grad():
        net.center[0] = 0.
    state = prior.classifier_state(net)
    with torch.no_grad():
        net.center[0] = -0.
    with pytest.raises(AssertionError, match='center'):
        prior.assert_classifier_unchanged(net, state)


class SourceOnlyData:
    def __init__(self):
        self.events = []

    def source_roles(self):
        return dict(train=[1, 2], val=[3], test=[4])

    def advanced_times(self):
        class TrainTimes(dict):
            def __getitem__(self, sid):
                assert sid in (1, 2), 'holdout AdvancedTime access'
                return super().__getitem__(sid)
        return TrainTimes({1: 5., 2: 40., 3: 100., 4: 100.})

    def read_shot(self, device, sid):
        assert device == 'J-TEXT' and sid in (1, 2)
        self.events.append(('signals', sid))
        return dict(shot=sid, device=device, values=np.zeros((200, 23), np.float32),
                    available=np.ones((200, 23), bool), times=np.arange(200)/1000.)

    def attach_outcome(self, shot, device, advanced):
        assert set(advanced) == {1, 2}
        self.events.append(('label', shot['shot']))
        return simple_data.attach_outcome(shot, device, advanced)


def test_loader_uses_only_source_train_and_original_max25_labels(monkeypatch):
    api = SourceOnlyData()
    def source_outcome(device, sid):
        assert device == 'J-TEXT' and sid in (1, 2)
        return True, .199
    monkeypatch.setattr(io23, 'read_outcome', source_outcome)
    shots, ids = prior.load_source_train(data_api=api, expected_train_count=2)
    assert ids == [1, 2]
    assert [s['horizon'] for s in shots] == [25., 40.]
    assert [s['y'].sum() for s in shots] == [26., 41.]
    assert api.events == [('signals', 1), ('label', 1), ('signals', 2), ('label', 2)]


def test_bad_roster_rejected_before_reading_signals_or_labels():
    api = SourceOnlyData()
    api.source_roles = lambda: dict(train=[1, 3], val=[3], test=[4])
    with pytest.raises(ValueError, match='overlap'):
        prior.load_source_train(data_api=api, expected_train_count=2)
    assert not api.events


@pytest.mark.parametrize('device,sid', [('EAST', 1), ('J-TEXT', 4)])
def test_episode_pool_rejects_target_or_holdout_before_label_access(setup, device, sid):
    net, _, shot = setup
    class LabelTrap(dict):
        def __getitem__(self, key):
            if key == 'y':
                raise AssertionError('unauthorized label read')
            return super().__getitem__(key)
    forbidden = LabelTrap(dict(shot, device=device, shot=sid))
    with pytest.raises(ValueError, match='TRAIN'):
        prior.source_episode_pools([forbidden], net.cfg, [1])


def test_artifact_has_only_two_forecast_tensors_and_locked_provenance(setup, tmp_path):
    net, _, _ = setup
    frozen = prior.classifier_state(net)
    code_hash = hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest()
    artifact = prior.build_prior(net, frozen, train_ids=[1, 2],
        parent_sha256=prior.PARENT_SHA256, code_sha256={'prepare_prior.py': code_hash},
        steps=128, batch_size=8, seed=131, lr=1e-4, forecast_weight=.1,
        episode_sha256='a'*64)
    assert set(artifact) == {'forecast_head', 'protocol'}
    assert set(artifact['forecast_head']) == {'weight', 'bias'}
    protocol = artifact['protocol']
    assert protocol['parent_sha256'] == prior.PARENT_SHA256
    assert protocol['source_only'] and protocol['static_classifier_bitwise_unchanged']
    assert protocol['train_ids'] == [1, 2]
    assert protocol['code_sha256']['prepare_prior.py'] == code_hash
    assert protocol['source_model_config'] == net.cfg.to_dict()
    assert protocol['fast_step'] == .05 and protocol['fast_radius'] == 1.
    assert protocol['source_qualification_lead_ms'] == 10.
    assert protocol['source_label'] == 'max(25ms,AdvancedTime)'
    path = tmp_path/'prior.pt'
    torch.save(artifact, path)
    assert set(torch.load(path, weights_only=True)) == {'forecast_head', 'protocol'}
    other = deepcopy(net)
    other.forecast_head.load_state_dict(artifact['forecast_head'], strict=True)
    prior.assert_classifier_unchanged(other, frozen)


def test_parent_hash_is_checked_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path/'source.pt'
    path.write_bytes(b'not a trusted checkpoint')
    def forbidden(*args, **kwargs):
        raise AssertionError('checkpoint deserialized before SHA verification')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='SHA256'):
        prior.load_parent(path)


def test_locked_parent_configuration_and_scales_load_unchanged():
    net, parent = prior.load_parent(prior.DEFAULT_PARENT)
    assert net.cfg.to_dict() == parent['model_config']
    assert net.initial_fast(1).shape == (1, 128)
    assert not any(p.requires_grad for p in net.parameters())
    assert parent['threshold'] == .7258946895599365
    for key, value in net.state_dict().items():
        assert prior._tensor_bytes(value) == prior._tensor_bytes(parent['model'][key])


def test_cpu_cli_orchestration_saves_prior_hash_and_manifest(setup, tmp_path, monkeypatch):
    net, _, shot = setup
    before = prior.classifier_state(net)
    monkeypatch.setattr(prior, 'load_parent', lambda path: (net, {}))
    monkeypatch.setattr(prior, 'load_source_train', lambda: ([shot], [1]))
    output = tmp_path/'offline-prior'
    report = prior.main(['--output', str(output), '--steps', '2', '--batch-size', '2'])
    artifact = torch.load(output/'prior.pt', map_location='cpu', weights_only=True)
    assert report['prior_sha256'] == hashlib.sha256((output/'prior.pt').read_bytes()).hexdigest()
    assert artifact['protocol']['episodes_seen'] == 4
    assert artifact['protocol']['train_ids'] == [1]
    assert len(artifact['protocol']['episode_sha256']) == 64
    assert len(report['history']) == 2
    prior.assert_classifier_unchanged(net, before)
    with pytest.raises(FileExistsError):
        prior.main(['--output', str(output), '--steps', '1'])


def test_cli_defaults_and_invalid_budget():
    parser = prior.make_parser()
    args = parser.parse_args(['--output', '/tmp/not-executed-prior'])
    assert args.steps == 128 and args.batch_size == 8
    assert args.device == 'cpu'
    for steps in (64, 128):
        prior.validate_args(parser.parse_args(['--output', '/tmp/not-executed-prior', '--steps', str(steps)]))
    for extra in (['--steps', '0'], ['--batch-size', '0'], ['--lr', 'nan'],
                  ['--forecast-weight', '-1']):
        with pytest.raises(ValueError):
            prior.validate_args(parser.parse_args(['--output', '/tmp/not-executed-prior', *extra]))
