"""CPU-only regressions for the independent gradient-balanced risk prior."""
from copy import deepcopy
import hashlib
import io
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
from experiments.risk_aligned_online.redesign_20261004 import risk_gradient_prior as prior


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


def manual_objective(net, e):
    net.eval()
    net.requires_grad_(False)
    head = tuple(net.forecast_head.parameters())
    for p in head:
        p.requires_grad_(True)
    updated, signal, initial, inner = common.temporary_step_components(net, e, create_graph=True)
    torch.testing.assert_close(updated, normalized_step(initial, initial.detach(), inner,
                               step=.05, radius=1.), rtol=0, atol=0)
    risk = F.binary_cross_entropy_with_logits(net(e['qx'], e['qa'], updated)['logit'], e['y'])
    gr = torch.autograd.grad(risk, head, retain_graph=True)
    gs = torch.autograd.grad(signal, head, retain_graph=True)
    rv = torch.cat([g.detach().reshape(-1).double() for g in gr])
    sv = torch.cat([g.detach().reshape(-1).double() for g in gs])
    rn, sn = float(rv.norm()), float(sv.norm())
    ratio = .1*sn/(rn+1e-12)
    alpha = float(np.clip(ratio, 1., 10000.))
    return alpha*risk+.1*signal, dict(risk=risk, signal=signal, rn=rn, sn=sn,
        ratio=ratio, alpha=alpha, cosine=float((rv*sv).sum()/rv.norm()/sv.norm()))


def test_diagnostic_matches_actual_second_order_head_gradients_without_any_mutation(setup):
    net, e, _ = setup
    net.train()
    net.encoder.eval()  # Read-only includes mixed submodule modes and existing .grad.
    for i, p in enumerate(net.parameters()):
        p.grad = torch.full_like(p, i+1.)
    state, flags = deepcopy(net.state_dict()), [p.requires_grad for p in net.parameters()]
    modes = [m.training for m in net.modules()]
    gradients = [p.grad.clone() for p in net.parameters()]
    _, expected = manual_objective(deepcopy(net), e)
    with torch.no_grad():
        log = prior.diagnostic_step(net, e)
        assert not torch.is_grad_enabled()
    assert log['risk_head_grad_norm'] == pytest.approx(expected['rn'], rel=1e-6)
    assert log['signal_head_grad_norm'] == pytest.approx(expected['sn'], rel=1e-6)
    assert log['maintenance_head_grad_norm'] == pytest.approx(.1*expected['sn'])
    assert log['maintenance_to_risk_head_ratio'] == pytest.approx(expected['ratio'])
    assert log['head_gradient_cosine'] == pytest.approx(expected['cosine'], abs=1e-7)
    assert log['head_gradient_conflict'] == (expected['cosine'] < 0.)
    assert log['alpha'] == expected['alpha'] and log['gradient_status'] == 'ready'
    assert not log['optimizer_step_applied']
    assert [p.requires_grad for p in net.parameters()] == flags
    assert [m.training for m in net.modules()] == modes
    for name, tensor in net.state_dict().items():
        assert common._tensor_bytes(tensor) == common._tensor_bytes(state[name])
    for p, g in zip(net.parameters(), gradients):
        assert torch.equal(p.grad, g)


def test_balanced_training_matches_detached_alpha_manual_double_backward(setup):
    net, e, _ = setup
    reference = deepcopy(net)
    objective, expected = manual_objective(reference, e)
    objective.backward()  # Must still work AFTER both audit autograd.grad calls.
    norm = torch.nn.utils.clip_grad_norm_(reference.forecast_head.parameters(), 1., error_if_nonfinite=True)
    expected_grads = [p.grad.clone() for p in reference.forecast_head.parameters()]
    frozen = common.classifier_state(net)
    before = deepcopy(net.state_dict())
    with torch.no_grad():
        original = net(e['qx'], e['qa'])['logit'].clone()
    log = prior.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01), e)
    assert log['objective'] == pytest.approx(float(objective.detach()), abs=1e-7)
    assert log['risk_after'] == pytest.approx(float(expected['risk'].detach()), abs=1e-7)
    assert log['head_grad_norm'] == pytest.approx(float(norm), rel=1e-6)
    assert log['alpha'] == expected['alpha'] and log['optimizer_step_applied']
    for p, g in zip(net.forecast_head.parameters(), expected_grads):
        torch.testing.assert_close(p.grad, g, rtol=1e-6, atol=1e-8)
    assert not torch.equal(net.forecast_head.weight, before['forecast_head.weight'])
    common.assert_classifier_unchanged(net, frozen)
    assert all(p.grad is None for name, p in net.named_parameters() if not name.startswith('forecast_head.'))
    assert not any(p.requires_grad for p in net.parameters())
    with torch.no_grad():
        assert torch.equal(original, net(e['qx'], e['qa'])['logit'])


@pytest.mark.parametrize('risk,signal,alpha,conflict', [
    ([2., 0.], [1., 0.], 1., False),
    ([1e-7, 0.], [-1., 0.], 10000., True),
    ([.01, 0.], [0., 1.], .1/(.01+1e-12), False),
])
def test_balance_bounds_cosine_and_detachment(risk, signal, alpha, conflict):
    r, s = torch.tensor(risk, requires_grad=True), torch.tensor(signal, requires_grad=True)
    log = prior.gradient_balance((r,), (s,))
    assert log['alpha'] == pytest.approx(alpha)
    assert isinstance(log['alpha'], float) and not r.grad and not s.grad
    assert log['head_gradient_conflict'] == conflict
    assert -1. <= log['head_gradient_cosine'] <= 1.
    json.dumps(log, allow_nan=False)


@pytest.mark.parametrize('which', ['risk', 'signal'])
def test_exact_zero_head_gradient_is_a_finite_rejection_not_a_fallback(which):
    r, s = torch.ones(3), torch.ones(3)
    (r if which == 'risk' else s).zero_()
    log = prior.gradient_balance((r,), (s,))
    assert log['gradient_status'] == 'rejected'
    assert log['rejection_reason'] == f'zero_{which}_head_gradient'
    assert log['head_gradient_cosine'] is None and log['head_gradient_conflict'] is None
    assert 1. <= log['alpha'] <= 10000.
    json.dumps(log, allow_nan=False)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_or_missing_head_gradients_rejected(bad):
    with pytest.raises(FloatingPointError, match='nonfinite'):
        prior.gradient_balance((torch.tensor([bad]),), (torch.ones(1),))
    with pytest.raises(FloatingPointError, match='missing'):
        prior.gradient_balance((None,), (torch.ones(1),))


def test_zero_inner_gradient_fails_closed_and_restores_everything(setup):
    net, e, _ = setup
    with torch.no_grad():
        net.forecast_head.weight.zero_()
    saved, modes = deepcopy(net.state_dict()), [m.training for m in net.modules()]
    log = prior.diagnostic_step(net, e)
    assert log['signal_fast_grad_norm'] == 0. and log['fast_delta_norm'] == 0.
    assert log['gradient_status'] == 'rejected' and log['rejection_reason'] == 'zero_inner_signal_gradient'
    json.dumps(log, allow_nan=False)
    optimizer = torch.optim.AdamW(net.forecast_head.parameters(), lr=.0001, weight_decay=0.)
    with pytest.raises(FloatingPointError, match='zero_inner_signal_gradient'):
        prior.train_step(net, optimizer, e)
    assert not optimizer.state
    assert not any(p.requires_grad for p in net.parameters())
    assert [m.training for m in net.modules()] == modes
    for name, tensor in net.state_dict().items():
        assert torch.equal(tensor, saved[name])


def test_before_after_dispatch_per_episode_audits_and_actual_fast_update(setup, monkeypatch):
    net, one, _ = setup
    e = {k: torch.cat([v, v]) if isinstance(v, torch.Tensor) else v for k, v in one.items()}
    e['y'][1] = 1-e['y'][0]
    forward, paths = net.forward, []
    def record(values, available, fast=None):
        if values is e['qx']:
            paths.append((torch.is_grad_enabled(), fast.requires_grad, fast.is_leaf))
        return forward(values, available, fast)
    monkeypatch.setattr(net, 'forward', record)
    log = prior.diagnostic_step(net, e)
    assert paths == [(True, True, True), (True, True, False)]
    assert log['future_labels'] == e['y'].tolist()
    assert len(log['risk_before_per_episode']) == len(log['risk_after_per_episode']) == 2
    np.testing.assert_allclose(log['risk_delta_per_episode'],
        np.asarray(log['risk_after_per_episode'])-np.asarray(log['risk_before_per_episode']), atol=1e-8)
    assert log['risk_before'] == pytest.approx(np.mean(log['risk_before_per_episode']))
    assert log['risk_after'] == pytest.approx(np.mean(log['risk_after_per_episode']))
    assert 0. < log['fast_delta_norm'] <= .050001
    assert log['fast_step'] == .05 and log['fast_radius'] == 1.
    updated, _, _, _ = common.temporary_step_components(net, e, create_graph=True)
    singleton, _, _, _ = common.temporary_step_components(net, one, create_graph=True)
    torch.testing.assert_close(updated[:1], singleton, rtol=2e-4, atol=1e-6)


def test_detaching_actual_step_cannot_silently_become_maintenance_only(setup, monkeypatch):
    net, e, _ = setup
    real = common.temporary_step_components
    def broken(*args, **kwargs):
        updated, signal, initial, gradient = real(*args, **kwargs)
        return updated.detach().requires_grad_(True), signal, initial, gradient
    monkeypatch.setattr(common, 'temporary_step_components', broken)
    with pytest.raises(FloatingPointError, match='missing'):
        prior.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01), e)


def test_invalid_optimizer_labels_and_maintenance_never_update(setup):
    net, e, _ = setup
    state = deepcopy(net.state_dict())
    with pytest.raises(ValueError, match='forecast_head'):
        prior.train_step(net, torch.optim.AdamW(net.parameters(), lr=.001), e)
    with pytest.raises(FloatingPointError, match='nonfinite'):
        prior.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.01),
                         dict(e, y=torch.full_like(e['y'], float('nan'))))
    with pytest.raises(ValueError, match='0.1'):
        prior.diagnostic_step(net, e, forecast_weight=0.)
    with pytest.raises(ValueError, match='label'):
        prior.diagnostic_step(net, dict(e, y=torch.full_like(e['y'], 2.)))
    with pytest.raises(ValueError, match='observed'):
        prior.diagnostic_step(net, dict(e, mask=torch.zeros_like(e['mask'])))
    assert not any(p.requires_grad for p in net.parameters())
    for name, value in net.state_dict().items():
        assert torch.equal(value, state[name])


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


@pytest.mark.parametrize('train', [False, True])
def test_cpu_main_source_only_digest_provenance_and_compatible_memory_artifact(setup, monkeypatch, train):
    net, _, _ = setup
    state, frozen = deepcopy(net.state_dict()), common.classifier_state(net)
    api, loader = SourceOnlyData(), common.load_source_train
    monkeypatch.setattr(io23, 'read_outcome', lambda device, sid:
        (True, .199) if device == 'J-TEXT' and sid in (1, 2) else pytest.fail('non-TRAIN label access'))
    monkeypatch.setattr(common, 'load_parent', lambda path: (net, {}))
    monkeypatch.setattr(common, 'load_source_train', lambda: loader(data_api=api, expected_train_count=2))
    # No test artifacts are written; even CLI serialization is intercepted.
    saved, writes, directories = {}, {}, []
    def save_in_memory(artifact, path):
        stream = io.BytesIO()
        real_save(artifact, stream)
        stream.seek(0)
        saved[Path(path).name] = torch.load(stream, map_location='cpu', weights_only=True)
    real_save = torch.save
    monkeypatch.setattr(torch, 'save', save_in_memory)
    real_sha = common.sha256
    monkeypatch.setattr(common, 'sha256', lambda path:
        'c'*64 if Path(path).name == 'riskprior.pt' else real_sha(path))
    monkeypatch.setattr(Path, 'mkdir', lambda path, **kw: directories.append((str(path), kw)))
    monkeypatch.setattr(Path, 'write_text', lambda path, content, **kw: writes.update({path.name: content}))
    output = '/tmp/not-created-risk-gradient-prior'
    report = prior.main(['--output', output, '--steps', '2', '--batch-size', '2', *(['--train'] if train else [])])
    assert api.events == [('signals', 1), ('label', 1), ('signals', 2), ('label', 2)]
    common.assert_classifier_unchanged(net, frozen)
    shots, ids = loader(data_api=SourceOnlyData(), expected_train_count=2)
    pools = common.source_episode_pools(shots, net.cfg, ids)
    rng, digest = np.random.default_rng(131), hashlib.sha256()
    for _ in range(2):
        e = fast_meta.sample(shots, pools, net.cfg, rng, 2, 'cpu')
        for key in sorted(e):
            digest.update(key.encode())
            digest.update(common._tensor_bytes(e[key]))
    protocol = report['protocol']
    assert protocol['episode_sha256'] == digest.hexdigest()
    assert protocol['source_only'] and protocol['static_classifier_bitwise_unchanged']
    assert protocol['parent_sha256'] == common.PARENT_SHA256 and protocol['train_ids'] == [1, 2]
    assert protocol['target_labels_accessed'] == protocol['holdout_outcomes_accessed'] == 0
    assert not protocol['support110_used'] and not protocol['jtext_online_branch']
    assert protocol['source_label'] == 'max(25ms,AdvancedTime)' and protocol['episodes_seen'] == 4
    assert protocol['alignment_weight'] == 0. and protocol['forecast_weight'] == .1
    assert protocol['alpha_detached'] and not protocol['alpha_meta_learned']
    assert protocol['objective'] == 'detached_alpha * actual_post_fast_step_future_BCE + 0.1 * direct_H25_SmoothL1'
    assert 'cosine' not in protocol['objective']
    for name, path in prior.code_paths().items():
        assert protocol['code_sha256'][name] == real_sha(path)
    assert len(report['history']) == 2
    if train:
        artifact = saved['riskprior.pt']
        assert set(artifact) == {'forecast_head', 'protocol'}
        assert set(artifact['forecast_head']) == {'weight', 'bias'}
        assert artifact['protocol'] == protocol and protocol['optimizer_steps'] == 2
        replacement = deepcopy(net)
        replacement.forecast_head.load_state_dict(artifact['forecast_head'], strict=True)
        common.assert_classifier_unchanged(replacement, frozen)
        assert json.loads(writes['training.json']) == report and len(directories) == 1
        assert report['prior_sha256'] == 'c'*64
    else:
        assert not saved and not writes and not directories
        assert protocol['optimizer_steps'] == 0
        for name, value in net.state_dict().items():
            assert torch.equal(value, state[name])


def test_builder_checks_classifier_bytes_and_replaces_cosine_provenance(setup):
    net, _, _ = setup
    frozen = common.classifier_state(net)
    metadata = dict(train_ids=[1], parent_sha256=common.PARENT_SHA256,
        code_sha256={'risk_gradient_prior.py': 'a'*64}, steps=128, batch_size=8,
        seed=131, lr=1e-4, forecast_weight=.1, episode_sha256='b'*64)
    artifact = prior.build_prior(net, frozen, **metadata)
    assert artifact['protocol']['alpha_bounds'] == [1., 10000.]
    assert artifact['protocol']['risk_after_graph_retained']
    assert artifact['protocol']['risk_weight'] == 'detached_per_batch_alpha'
    with torch.no_grad():
        net.center[0] += 1.
    with pytest.raises(AssertionError, match='center'):
        prior.build_prior(net, frozen, **metadata)


def test_cli_defaults_opt_in_trainer_and_fixed_bounded_contract():
    parser = prior.make_parser()
    args = parser.parse_args(['--output', '/tmp/not-created'])
    assert (args.steps, args.batch_size, args.seed, args.device) == (128, 8, 131, 'cpu')
    assert args.lr == 1e-4 and args.forecast_weight == .1 and not args.train and args.prior is None
    prior.validate_args(args)
    for extra in (['--steps', '0'], ['--steps', '129'], ['--batch-size', '9'],
                  ['--lr', 'nan'], ['--forecast-weight', '0'], ['--seed', '132'],
                  ['--train', '--prior', '/tmp/not-loaded.pt']):
        with pytest.raises(ValueError):
            prior.validate_args(parser.parse_args(['--output', '/tmp/not-created', *extra]))


def fixed_artifact(net):
    return dict(forecast_head=deepcopy(net.forecast_head.state_dict()), protocol=dict(
        parent_sha256=common.PARENT_SHA256, source_only=True,
        static_classifier_bitwise_unchanged=True, source_model_config=net.cfg.to_dict(),
        source_label='max(25ms,AdvancedTime)', train_ids=[1], target_labels_accessed=0,
        holdout_outcomes_accessed=0, support110_used=False))


def test_fixed_saved_head_loader_safe_source_only_and_read_only(setup, monkeypatch):
    net, _, _ = setup
    artifact = fixed_artifact(net)
    artifact['forecast_head']['bias'].add_(.01)
    frozen, loads = common.classifier_state(net), []
    monkeypatch.setattr(common, 'sha256', lambda path: 'd'*64)
    def load(path, **kwargs):
        loads.append(kwargs)
        return artifact
    monkeypatch.setattr(torch, 'load', load)
    identity = prior.load_fixed_prior(net, Path('/tmp/not-read-fixed.pt'))
    assert identity['sha256'] == 'd'*64 and identity['protocol'] == artifact['protocol']
    assert loads == [dict(map_location='cpu', weights_only=True)]
    assert torch.equal(net.forecast_head.bias, artifact['forecast_head']['bias'])
    common.assert_classifier_unchanged(net, frozen)


@pytest.mark.parametrize('bad', ['parent', 'target', 'holdout', 'source', 'classifier', 'shape', 'nan', 'dtype'])
def test_fixed_saved_head_rejects_unsafe_provenance_and_tensors_before_loading(setup, monkeypatch, bad):
    net, _, _ = setup
    saved = deepcopy(net.state_dict())
    artifact = fixed_artifact(net)
    if bad == 'parent':
        artifact['protocol']['parent_sha256'] = '0'*64
    elif bad == 'target':
        artifact['protocol']['target_labels_accessed'] = 1
    elif bad == 'holdout':
        artifact['protocol']['holdout_outcomes_accessed'] = 1
    elif bad == 'source':
        artifact['protocol']['source_only'] = False
    elif bad == 'classifier':
        artifact['protocol']['static_classifier_bitwise_unchanged'] = False
    elif bad == 'shape':
        artifact['forecast_head']['weight'] = torch.zeros(1)
    elif bad == 'nan':
        artifact['forecast_head']['bias'][0] = float('nan')
    else:
        artifact['forecast_head']['weight'] = artifact['forecast_head']['weight'].double()
    monkeypatch.setattr(common, 'sha256', lambda path: 'd'*64)
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: artifact)
    with pytest.raises(ValueError, match='prior'):
        prior.load_fixed_prior(net, '/tmp/not-read-fixed.pt')
    for name, value in net.state_dict().items():
        assert torch.equal(value, saved[name])


def test_fixed_diagnostic_main_preserves_selected_head_and_all_audits_in_stdout(setup, monkeypatch, capsys):
    net, _, shot = setup
    frozen = common.classifier_state(net)
    artifact = fixed_artifact(net)
    artifact['forecast_head']['bias'].add_(.01)
    real_sha = common.sha256
    monkeypatch.setattr(common, 'sha256', lambda path:
        'd'*64 if Path(path).name == 'fixed.pt' else real_sha(path))
    monkeypatch.setattr(common, 'load_parent', lambda path: (net, {}))
    monkeypatch.setattr(common, 'load_source_train', lambda: ([shot], [1]))
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: artifact)
    def forbidden(*args, **kwargs):
        pytest.fail('read-only fixed diagnostic attempted a write')
    monkeypatch.setattr(Path, 'mkdir', forbidden)
    monkeypatch.setattr(Path, 'write_text', forbidden)
    monkeypatch.setattr(torch, 'save', forbidden)
    report = prior.main(['--output', '/tmp/not-created', '--prior', '/tmp/fixed.pt',
                         '--steps', '2', '--batch-size', '2'])
    assert report['protocol']['fixed_head_prior_sha256'] == 'd'*64
    assert report['protocol']['optimizer_steps'] == 0
    assert report['protocol']['audit_scope'] == 'fixed saved head; not changing training-time audits'
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['history'] == report['history']
    assert torch.equal(net.forecast_head.bias, artifact['forecast_head']['bias'])
    common.assert_classifier_unchanged(net, frozen)


def test_riskprior_artifact_loads_through_unchanged_run_load_parent(setup, monkeypatch):
    from experiments.risk_aligned_online.redesign_20261004 import run
    net, _, _ = setup
    frozen = common.classifier_state(net)
    saved = dict(model_config=net.cfg.to_dict(), source_center=net.center.numpy(),
                 source_scale=net.scale.numpy(), model=deepcopy(net.state_dict()), forecast_trained=True)
    artifact = fixed_artifact(net)
    artifact['forecast_head']['bias'].add_(.01)
    gate = dict(qualified=True, checkpoint_sha256=common.PARENT_SHA256)
    monkeypatch.setattr(run, 'qualify_source', lambda path: gate)
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs:
        artifact if str(path).endswith('riskprior.pt') else saved)
    wrapper, _, _, protocol = run.load_parent('/tmp/not-read-source.pt', 'cpu',
        context_mode='ordered', prior='/tmp/not-read-riskprior.pt')
    assert protocol == artifact['protocol']
    assert torch.equal(wrapper.base.forecast_head.bias, artifact['forecast_head']['bias'])
    common.assert_classifier_unchanged(wrapper.base, frozen)


def test_wrong_source_hash_rejected_before_deserialization_or_data(monkeypatch):
    monkeypatch.setattr(common, 'sha256', lambda path: '0'*64)
    def forbidden(*args, **kwargs):
        pytest.fail('untrusted parent loaded or source data accessed')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(common, 'load_source_train', forbidden)
    with pytest.raises(ValueError, match='SHA256'):
        prior.main(['--output', '/tmp/not-created'])


def test_frozen_existing_prior_and_source_hashes():
    assert common.sha256(common.__file__) == '469c568e685ebf6d9451eb5ccdeddcca98eb8195897934814d7eb8b13952a49f'
    assert common.sha256(Path(common.__file__).with_name('prepare_finite_prior.py')) == 'af6569d3543031c8d0b368e062d16370268ed6e18ed4e1411136f41eb8ad6c6d'
    assert common.sha256(common.DEFAULT_PARENT) == common.PARENT_SHA256


def test_locked_actual_simple23_cpu_second_backward_with_synthetic_episode(setup):
    _, e, _ = setup
    net, parent = common.load_parent(common.DEFAULT_PARENT)
    assert net.center.device.type == 'cpu' and net.cfg.to_dict() == parent['model_config']
    frozen = common.classifier_state(net)
    with torch.no_grad():
        logits = net(e['qx'], e['qa'])['logit'].clone()
    synthetic = dict(e, y=(logits < 0).to(e['y'].dtype))
    audit = prior.diagnostic_step(net, synthetic)
    assert audit['risk_head_grad_norm'] > 0. and audit['gradient_status'] == 'ready'
    log = prior.train_step(net, torch.optim.SGD(net.forecast_head.parameters(), lr=.001), synthetic)
    assert log['head_grad_norm'] > 0. and np.isfinite(log['objective'])
    common.assert_classifier_unchanged(net, frozen)
    with torch.no_grad():
        assert torch.equal(logits, net(e['qx'], e['qa'])['logit'])
