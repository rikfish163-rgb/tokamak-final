"""Small CPU-only checks for the diagnostic runner's frozen contracts."""

import json
import shutil
import uuid

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030 import run_full as runner


def test_loads_all_four_real_completed_policies_on_cpu():
    previous = runner.read(runner.PREVIOUS / 'protocol.json')
    policies, hashes = runner.load_policies(runner.POLICY, previous, device='cpu')
    assert tuple(policies) == runner.VARIANTS
    assert set(hashes) >= {'complete.json', 'protocol.json'}
    for variant, payload in policies.items():
        assert set(payload) == {'controller', 'mean_context', 'scale', 'radius'}
        assert not payload['controller'].training
        assert all(not p.requires_grad for p in payload['controller'].parameters())
        assert payload['mean_context'].shape == (358,)
        assert all(t.device.type == 'cpu' for t in payload['scale'].values())
        assert hashes[f'{variant}/foldall.pt'] == runner.read(
            runner.POLICY / variant / 'complete.json')['policy_sha256']


def test_policy_loader_rejects_tampering_and_failed_gate_relabel(tmp_path):
    source = runner.POLICY
    for name in ('complete.json', 'protocol.json'):
        shutil.copy2(source / name, tmp_path / name)
    for variant in runner.VARIANTS:
        (tmp_path / variant).mkdir()
        for name in ('foldall.pt', 'complete.json'):
            shutil.copy2(source / variant / name, tmp_path / variant / name)
    previous = runner.read(runner.PREVIOUS / 'protocol.json')
    checkpoint = tmp_path / 'raw_fixed' / 'foldall.pt'
    with checkpoint.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='policy'):
        runner.load_policies(tmp_path, previous, device='cpu')
    shutil.copy2(source / 'raw_fixed' / 'foldall.pt', checkpoint)
    completion = tmp_path / 'complete.json'
    body = json.loads(completion.read_text())
    body['variants']['raw_fixed']['full_stream_gate'] = True
    completion.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='gate|completion'):
        runner.load_policies(tmp_path, previous, device='cpu')


def test_gate_requires_durable_complete_barrier_even_for_cold(tmp_path):
    gate = runner.GlobalOutcomeGate(list(range(1, 361)), lambda _d, s: (False, s))
    for shot in range(1, 361):
        gate.seal(shot, 'a' * 64)
        if shot in (1, 21, 360):
            with pytest.raises(AssertionError):
                gate.read_outcome('EAST', shot)
    barrier = tmp_path / 'all_predictions_sealed.json'
    barrier.write_text(json.dumps({'count': 360, 'seals': gate.seals}))
    gate.release(barrier)
    assert gate.read_outcome('EAST', 21) == (False, 21)
    assert gate.accesses == [{'shot': 21, 'sealed_count': 360}]


def receipt(n=154):
    return dict(origin=127, maturity=152, first_affected=153,
                apply_count=min(25, n - 153),
                state_norm={name: 0. for name in runner.NAMES},
                cos_real={name: None for name in runner.NAMES}, max_bound=0.,
                feedback_supported=False)


def test_receipts_require_exact_maturity_and_undefined_zero_cosines():
    policies = {v: {'radius': .1} for v in runner.VARIANTS}
    runner.validate_receipts([receipt()], 154, policies)
    wrong = receipt()
    wrong['first_affected'] = 152
    with pytest.raises(ValueError, match='timing|receipt'):
        runner.validate_receipts([wrong], 154, policies)
    wrong = receipt()
    wrong['state_norm']['raw_fixed__real'] = .11
    with pytest.raises(ValueError, match='bound|receipt'):
        runner.validate_receipts([wrong], 154, policies)


def test_exact_cold_archive_is_mandatory():
    old = np.array([.1, .5], dtype=np.float32)
    assert runner.check_cold_archive(old.copy(), old, .5) == (0., 0)
    changed = old.copy()
    changed[0] = np.nextafter(changed[0], np.float32(1.))
    with pytest.raises(ValueError, match='exactly archived'):
        runner.check_cold_archive(changed, old, .5)


def test_real_archived_startup_shot_has_matching_per_shot_and_global_seals():
    previous = runner.read(runner.PREVIOUS / 'protocol.json')
    barrier = runner.read(runner.PREVIOUS / 'all_predictions_sealed.json')
    times, cold, seal_hash = runner.load_archived_shot(runner.PREVIOUS, 0, previous['order'][0], barrier)
    assert len(times) == len(cold) > 0
    assert len(seal_hash) == 64


def test_full_run_refuses_cpu_without_creating_output(monkeypatch):
    output = runner.io23.ROOT / 'probe_outputs' / ('cpu_gate_' + uuid.uuid4().hex)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA required'):
        runner.run(output)
    assert not output.exists()


def test_reporting_uses_strict_lead_and_keeps_all_variants(monkeypatch):
    ids = list(range(1, 361))
    monkeypatch.setattr(runner.fixed_test_protocol, 'load_test_contract', lambda _device: {
        'shot_ids': ids, 'expected_disrupt': 180, 'expected_safe': 180, 'lead_ms': 30.})
    times = np.array([6.9, 9.97])
    rows = {}
    for arm in runner.NAMES:
        rows[arm] = []
        for sid in ids:
            disruptive = sid <= 180
            scores = np.array([0., .8 if disruptive else 0.])
            if arm == 'raw_fixed__real' and sid in (1, 181):
                scores[0] = .8
            rows[arm].append(dict(shot=sid, times=times, scores=scores,
                                  is_disrupt=disruptive, down=10. if disruptive else None))
    results, predictions, paired, controls, caps = runner.report_predictions(rows, .5)
    assert set(results) == set(runner.NAMES)
    assert set(results['cold']) == {'full360', 'cold20', 'suffix340', 'last90'}
    assert results['cold']['full360']['tp'] == 0  # exactly 30 ms is not timely
    assert results['raw_fixed__real']['full360']['tp'] == 1
    assert results['raw_fixed__real']['full360']['early_over3000'] == 1
    assert predictions['raw_fixed__real'][0]['first_alarm_time_seconds'] == 6.9
    assert paired['full360']['raw_fixed__real'] == {'rescued': [1], 'harmed': [181]}
    assert controls['raw_fixed']['reverse']['full360'] == {'rescued': [1], 'harmed': [181]}
    assert caps['raw_fixed__real']['full360']['0.1'] is True


@pytest.mark.parametrize('blocked', ['read_outcome', 'labels_at', 'manifest', 'gate'])
def test_prediction_cannot_access_labels_before_global_seal(tmp_path, monkeypatch, blocked):
    ids = list(range(1, 361))
    gate = runner.GlobalOutcomeGate(ids, lambda *_: (False, None))
    (tmp_path / 'shots').mkdir()
    times = np.array([0., 1.])
    cold = np.array([.2, .3], np.float32)
    monkeypatch.setattr(runner, 'load_archived_shot', lambda *_: (times, cold, 'b' * 64))
    monkeypatch.setattr(runner.io23, 'path_for', lambda *_: tmp_path / 'raw')
    (tmp_path / 'raw').write_bytes(b'raw')
    monkeypatch.setattr(runner.simple_data, 'read_shot', lambda *_a, **_k: {
        'times': times, 'values': np.zeros((2, 23), np.float32),
        'available': np.ones((2, 23), bool)})
    monkeypatch.setattr(runner, 'apply_input_contract', lambda x: x)
    monkeypatch.setattr(runner, 'transform', lambda x, _m: x)

    def leak(*_args):
        if blocked == 'read_outcome':
            runner.io23.read_outcome('EAST', 21)
        elif blocked == 'labels_at':
            runner.io23.labels_at(times, False, None, 125.)
        elif blocked == 'manifest':
            runner.fixed_test_protocol.load_test_contract('EAST')
        else:
            gate.read_outcome('EAST', 1)
        raise AssertionError('leak was not blocked')

    monkeypatch.setattr(runner, 'predict', leak)
    raw_hash = runner.sha(tmp_path / 'raw')
    with pytest.raises(AssertionError, match='outcome|label|manifest|barrier'):
        runner.seal_predictions(tmp_path, ids, gate, tmp_path, {},
                                {str(s): raw_hash for s in ids}, {}, None, None,
                                {v: {'radius': .1} for v in runner.VARIANTS}, .5)
    assert len(gate.seals) == 20
    assert not gate.accesses
    assert not (tmp_path / 'all_predictions_sealed.json').exists()


def test_toy_full_stream_seals_22_curves_before_any_outcome(tmp_path, monkeypatch):
    ids = list(range(1, 361))
    gate = runner.GlobalOutcomeGate(ids, lambda _d, s: (s % 2 == 0, None))
    (tmp_path / 'shots').mkdir()
    times = np.array([0., 1.])
    cold = np.array([.2, .3], np.float32)
    monkeypatch.setattr(runner, 'load_archived_shot', lambda *_: (times, cold, 'b' * 64))
    monkeypatch.setattr(runner.io23, 'path_for', lambda *_: tmp_path / 'raw')
    (tmp_path / 'raw').write_bytes(b'raw')
    monkeypatch.setattr(runner.simple_data, 'read_shot', lambda *_a, **_k: {
        'times': times, 'values': np.zeros((2, 23), np.float32),
        'available': np.ones((2, 23), bool)})
    monkeypatch.setattr(runner, 'apply_input_contract', lambda x: x)
    monkeypatch.setattr(runner, 'transform', lambda x, _m: x)
    monkeypatch.setattr(runner, 'predict', lambda *_: ({n: cold.copy() for n in runner.NAMES}, []))
    raw_hash = runner.sha(tmp_path / 'raw')
    receipts = runner.seal_predictions(
        tmp_path, ids, gate, tmp_path, {}, {str(s): raw_hash for s in ids}, {},
        None, None, {v: {'radius': .1} for v in runner.VARIANTS}, .5)
    assert len(receipts) == len(gate.seals) == 360
    assert not gate.accesses
    barrier = runner.read(tmp_path / 'all_predictions_sealed.json')
    assert barrier['count'] == 360 and barrier['seals'] == gate.seals
    for index in (1, 20, 21, 360):
        with np.load(tmp_path / 'shots' / f'{index:04d}_{index}.npz', allow_pickle=False) as data:
            assert set(data.files) == {'times', *runner.NAMES}
            for arm in runner.NAMES:
                np.testing.assert_array_equal(data[arm], cold)
        assert receipts[index - 1]['startup_archived'] is (index <= 20)
    assert gate.read_outcome('EAST', 21) == (False, None)


def test_toy_runner_writes_diagnostic_completion_only_after_full_seal(tmp_path, monkeypatch):
    """Exercise the full orchestration with tiny files and no real GPU or data."""
    root = tmp_path
    probes, archive, reference = root / 'probe_outputs', root / 'archive', root / 'reference'
    probes.mkdir()
    (archive / 'shots').mkdir(parents=True)
    (reference / 'models').mkdir(parents=True)
    plan, parent, raw = root / 'plan.md', root / 'parent.pt', root / 'raw'
    plan.write_text('synthetic plan')
    parent.write_bytes(b'synthetic source')
    raw.write_bytes(b'synthetic data')
    (archive / 'complete.json').write_text('{}')
    ids = list(range(1, 361))
    times, cold = np.array([0., 1.]), np.array([.2, .3], np.float32)
    barrier = {'count': 360, 'seals': []}
    for index, sid in enumerate(ids, 1):
        stem = f'{index:04d}_{sid}'
        path = archive / 'shots' / (stem + '.npz')
        np.savez(path, times=times, cold=cold)
        digest = runner.sha(path)
        (archive / 'shots' / (stem + '_seal.json')).write_text(
            json.dumps({'shot': sid, 'index': index, 'sha256': digest}))
        barrier['seals'].append({'shot': sid, 'sha256': digest})
    (archive / 'all_predictions_sealed.json').write_text(json.dumps(barrier))
    (reference / 'order.json').write_text(json.dumps([{'shot': sid} for sid in ids]))
    (reference / 'input_map.json').write_text(json.dumps({'calibration_ids': ids[:20]}))
    (reference / 'input_hashes.json').write_text(json.dumps({str(s): runner.sha(raw) for s in ids}))
    torch.save({'slow': torch.zeros(1)}, reference / 'models' / 'cold.pt')
    ref_hashes = {name: runner.sha(reference / name) for name in
                  ('order.json', 'input_map.json', 'input_hashes.json', 'models/cold.pt')}
    (archive / 'protocol.json').write_text(json.dumps({
        'order': ids, 'source_sha256': runner.sha(parent), 'threshold': .5,
        'reference': 'reference', 'reference_hashes': ref_hashes}))
    monkeypatch.setattr(runner.io23, 'ROOT', root)
    monkeypatch.setattr(runner, 'PREVIOUS', archive)
    monkeypatch.setattr(runner, 'PREVIOUS_SHA', runner.sha(archive / 'complete.json'))
    monkeypatch.setattr(runner, 'PLAN', plan)
    monkeypatch.setattr(runner, 'DEFAULT_PARENT', parent)
    monkeypatch.setattr(runner, 'PARENT_SHA256', runner.sha(parent))
    monkeypatch.setattr(runner, 'code_files', lambda: {plan})
    monkeypatch.setattr(runner.io23, 'path_for', lambda *_: raw)
    monkeypatch.setattr(runner.io23, 'read_outcome', lambda _d, s: (s <= 180, 10. if s <= 180 else None))
    monkeypatch.setattr(runner.fixed_test_protocol, 'load_test_contract', lambda _device: {
        'shot_ids': ids, 'expected_disrupt': 180, 'expected_safe': 180, 'lead_ms': 30.})
    monkeypatch.setattr(runner.simple_data, 'read_shot', lambda *_a, **_k: {
        'times': times, 'values': np.zeros((2, 23), np.float32),
        'available': np.ones((2, 23), bool)})
    monkeypatch.setattr(runner, 'apply_input_contract', lambda x: x)
    monkeypatch.setattr(runner, 'transform', lambda x, _m: x)
    monkeypatch.setattr(runner, 'predict', lambda *_: ({n: cold.copy() for n in runner.NAMES}, []))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU toy')
    original_load = torch.load
    monkeypatch.setattr(torch, 'load', lambda path, **kwargs: original_load(
        path, **{**kwargs, 'map_location': 'cpu'}))

    class ToyNet(torch.nn.Linear):
        def cuda(self):
            return self

    monkeypatch.setattr(runner, 'load_parent', lambda _p: (ToyNet(1, 1), {'threshold': .5}))
    monkeypatch.setattr(runner, 'MatrixAttention', lambda base: base)
    monkeypatch.setattr(runner, 'load_policies', lambda *_a, **_k: (
        {v: {'controller': torch.nn.Linear(1, 1).eval(), 'mean_context': torch.zeros(358),
             'scale': {}, 'radius': .1} for v in runner.VARIANTS}, {}))
    output = probes / 'toy'
    complete = runner.run(output)
    assert complete['complete'] is True
    assert complete['scope'] == 'DIAGNOSTIC_FULL360_AFTER_FAILED_COMPONENT_GATE'
    assert not complete['promoted'] and not complete['scientific_goal_achieved']
    assert set(complete['results']) == set(runner.NAMES)
    assert all(item['sealed_count'] == 360 for item in runner.read(output / 'accesses.json'))
    assert runner.read(output / 'all_predictions_sealed.json')['count'] == 360
