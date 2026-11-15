"""In-memory CPU tests; never open live run artifacts or real shot data."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online.context_vector_20261030 import replay_full as replay
from experiments.risk_aligned_online.context_vector_20261030 import stream


@pytest.fixture
def order():
    return list(range(1, 21)) + [70323] + list(range(1000, 1339))


@pytest.fixture
def comparisons():
    full = {name: {'rescued': [], 'harmed': []} for name in replay.NAMES}
    full['raw_fixed__real']['rescued'] = [1001, 1004]
    full['scaled_calibrated__risk_axis']['harmed'] = [1000, 1001]
    return {'full360': full, 'suffix340': copy.deepcopy(full)}


def test_default_selects_union_across_all_22_arms_in_run_order(order, comparisons):
    assert replay.select_shots(order, comparisons) == [70323, 1000, 1001, 1004]


def test_explicit_subset_does_not_consult_outcome_report(order):
    assert replay.select_shots(order, None, [1002, 70323]) == [70323, 1002]


@pytest.mark.parametrize('shots', [[], [1], [1000, 1000], [999999], [True]])
def test_explicit_shots_reject_empty_cold20_duplicates_unknown_and_bool(order, shots):
    with pytest.raises(ValueError, match='suffix|distinct'):
        replay.select_shots(order, None, shots)


def test_first_suffix_identity_and_paired_report_must_be_consistent(order, comparisons):
    altered = order.copy()
    altered[20], altered[21] = altered[21], altered[20]
    with pytest.raises(ValueError, match='70323'):
        replay.select_shots(altered, comparisons)
    comparisons['full360']['raw_fixed__real']['rescued'].append(1)
    with pytest.raises(ValueError, match='suffix|paired'):
        replay.select_shots(order, comparisons)


def test_selection_rejects_missing_arm_and_divergent_sections(order, comparisons):
    del comparisons['full360']['identical']
    with pytest.raises(ValueError, match='arm|paired'):
        replay.select_shots(order, comparisons)
    comparisons['full360']['identical'] = {'rescued': [], 'harmed': []}
    comparisons['suffix340']['raw_fixed__real']['rescued'] = []
    with pytest.raises(ValueError, match='section|paired'):
        replay.select_shots(order, comparisons)


def arrays():
    times = np.arange(154, dtype=np.float64) / 1000
    cold = np.linspace(.1, .4, len(times), dtype=np.float32)
    curves = {name: cold.copy() for name in replay.NAMES}
    receipt = dict(origin=127, maturity=152, first_affected=153, apply_count=1,
                   state_norm={name: 0. for name in replay.NAMES},
                   cos_real={name: None for name in replay.NAMES}, max_bound=0.,
                   feedback_supported=False)
    return times, curves, [receipt]


def test_compare_all_curves_timestamps_and_receipts_exactly():
    times, curves, updates = arrays()
    saved = {'times': times.copy(), **{k: v.copy() for k, v in curves.items()}}
    result = replay.compare_exact(times, curves, updates, saved, copy.deepcopy(updates))
    assert result['times_max_abs_error'] == 0
    assert set(result['curves_max_abs_error']) == set(replay.NAMES)
    assert max(result['curves_max_abs_error'].values()) == 0
    assert result['receipts_exact'] is True


@pytest.mark.parametrize('kind', ['time', 'curve', 'dtype', 'missing', 'extra', 'receipt', 'nan'])
def test_comparison_detects_any_mismatch(kind):
    times, curves, updates = arrays()
    saved = {'times': times.copy(), **{k: v.copy() for k, v in curves.items()}}
    if kind == 'time':
        times[0] = np.nextafter(times[0], 1.)
    elif kind == 'curve':
        curves['scaled_calibrated__risk_axis'][153] = np.nextafter(np.float32(.4), np.float32(1.))
    elif kind == 'dtype':
        curves['cold'] = curves['cold'].astype(np.float64)
    elif kind == 'missing':
        del curves['raw_fixed__real']
    elif kind == 'extra':
        curves['unknown'] = curves['cold']
    elif kind == 'receipt':
        updates[0]['apply_count'] = 0
    else:
        curves['cold'][0] = np.nan
    sealed = arrays()[2]
    with pytest.raises((ValueError, AssertionError), match='time|curve|receipt|finite|arm'):
        replay.compare_exact(times, curves, updates, saved, sealed)


def test_exact_receipt_comparison_distinguishes_bool_and_integer():
    times, curves, sealed = arrays()
    updates = copy.deepcopy(sealed)
    updates[0]['feedback_supported'] = 0
    with pytest.raises(ValueError, match='receipt'):
        replay.compare_exact(times, curves, updates, {'times': times, **curves}, sealed)


@pytest.mark.parametrize('output', ['/fake/elsewhere/r.json', '/fake/run/sub/r.json',
                                   '/fake/run/r.txt', '/fake/run/../run2/r.json'])
def test_output_must_be_new_json_directly_inside_run(output, monkeypatch):
    monkeypatch.setattr(replay.os.path, 'lexists', lambda _p: False)
    with pytest.raises(ValueError, match='output|JSON'):
        replay.check_output_path(Path('/fake/run'), Path(output))


def test_existing_file_or_symlink_is_never_overwritten(monkeypatch):
    monkeypatch.setattr(replay.os.path, 'lexists', lambda _p: True)
    with pytest.raises(FileExistsError):
        replay.check_output_path(Path('/fake/run'), Path('/fake/run/result.json'))


def test_cli_uses_exclusive_create_for_report(monkeypatch):
    captured = {}

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, data):
            captured['payload'] = data

        def flush(self):
            captured['flushed'] = True

        def fileno(self):
            return 12

    def open_file(path, mode, **_kwargs):
        captured['path'], captured['mode'] = path, mode
        return FakeHandle()

    monkeypatch.setattr('sys.argv', ['replay_full', '--run', '/fake/run', '--output',
                                     '/fake/run/result.json', '--shots', '70323'])
    monkeypatch.setattr(replay.os.path, 'lexists', lambda _path: False)
    monkeypatch.setattr(replay.os, 'fsync', lambda fd: captured.update(fd=fd))
    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(replay, 'replay', lambda run, shots: {'shots': shots, 'replay_passed': True})
    replay.main()
    assert captured['path'] == Path('/fake/run/result.json') and captured['mode'] == 'x'
    assert captured['flushed'] and captured['fd'] == 12
    assert json.loads(captured['payload']) == {'shots': [70323], 'replay_passed': True}


def test_cli_exclusive_open_fails_if_output_appears_after_precheck(monkeypatch):
    monkeypatch.setattr('sys.argv', ['replay_full', '--run', '/fake/run', '--output', '/fake/run/result.json'])
    monkeypatch.setattr(replay.os.path, 'lexists', lambda _path: False)
    monkeypatch.setattr(replay, 'replay', lambda *_: {'shots': [70323]})

    def concurrent_file(_path, mode, **_kwargs):
        assert mode == 'x'
        raise FileExistsError('another writer created output')

    monkeypatch.setattr(Path, 'open', concurrent_file)
    with pytest.raises(FileExistsError, match='another writer'):
        replay.main()


@pytest.mark.parametrize('error', [FileNotFoundError('complete'), ValueError('global360 barrier')])
def test_missing_completion_or_barrier_blocks_archives_and_prediction(monkeypatch, error):
    monkeypatch.setattr(replay.audit, 'require_completed_barrier', lambda *_: (_ for _ in ()).throw(error))
    monkeypatch.setattr(replay.audit, 'verify_provenance', lambda *_: pytest.fail('premature archive read'))
    monkeypatch.setattr(replay.runner, 'predict', lambda *_: pytest.fail('premature prediction'))
    with pytest.raises(type(error), match='complete|barrier'):
        replay.replay(Path('/fake/run'), [70323])


def test_real_barrier_validator_blocks_a_partial_360_before_archive_reads(monkeypatch, order):
    artifacts = {name: 'other-hash' for name in ('protocol.json', 'code_snapshot.zip',
                 'all_predictions_sealed.json', 'per_shot.json', 'receipts.json', 'accesses.json')}
    artifacts.update({'protocol.json': 'protocol-hash', 'all_predictions_sealed.json': 'barrier-hash'})
    documents = {
        'complete.json': {'complete': True, 'artifact_sha256': artifacts},
        'protocol.json': {'order': order},
        'all_predictions_sealed.json': {'count': 359,
            'seals': [{'shot': sid, 'sha256': 'a' * 64} for sid in order],
            'receipt_sha256': ['b' * 64] * 360},
    }
    monkeypatch.setattr(Path, 'is_file', lambda self: self.name == 'complete.json')
    monkeypatch.setattr(replay.audit, 'read_json', lambda path: documents[Path(path).name])
    monkeypatch.setattr(replay.audit, 'sha', lambda path: artifacts[Path(path).name])
    monkeypatch.setattr(replay.audit, 'verify_provenance', lambda *_: pytest.fail('premature archive read'))
    with pytest.raises(ValueError, match='global360'):
        replay.replay(Path('/fake/run'), [70323])


def test_real_seal_validator_checks_all_360_current_and_previous_shots(monkeypatch, order):
    directory, previous = Path('/fake/run'), Path('/fake/old')
    positions = {sid: index for index, sid in enumerate(order)}
    barrier = {'seals': [{'shot': sid, 'sha256': 'new'} for sid in order],
               'receipt_sha256': ['new-seal'] * 360}
    old_barrier = {'seals': [{'shot': sid, 'sha256': 'old'} for sid in order]}
    receipts = [dict(shot=sid, index=i + 1, sha256='new', archive_seal_sha256='old-seal',
                     archive_max_abs_error=0., archive_crossing_disagreements=0,
                     startup_archived=i < 20, outcomes_used_in_prediction=False, seal_sha256='new-seal')
                for i, sid in enumerate(order)]
    hashed = []

    def digest(path):
        path = Path(path)
        hashed.append(path)
        prefix = 'old' if path.is_relative_to(previous) else 'new'
        return prefix + '-seal' if path.name.endswith('_seal.json') else prefix

    def read(path):
        path = Path(path)
        if path.name == 'receipts.json':
            return receipts
        if path.name == 'accesses.json':
            return [{'shot': sid, 'sealed_count': 360} for sid in order]
        sid = int(path.name.split('_')[1])
        index = positions[sid]
        if path.is_relative_to(previous):
            return {'shot': sid, 'index': index + 1, 'sha256': 'old'}
        return {**{k: v for k, v in receipts[index].items() if k != 'seal_sha256'}, 'updates': []}

    monkeypatch.setattr(replay.audit, 'PREVIOUS', previous)
    monkeypatch.setattr(replay.audit, 'sha', digest)
    monkeypatch.setattr(replay.audit, 'read_json', read)
    with replay.no_outcomes():
        replay.audit.verify_all_seals(directory, {'order': order}, barrier, old_barrier)
    assert len(hashed) == 360 * 4
    barrier['receipt_sha256'][150] = 'tampered'
    with pytest.raises(ValueError, match='seal'):
        replay.audit.verify_all_seals(directory, {'order': order}, barrier, old_barrier)


@pytest.mark.parametrize('module,name', [('io23', 'read_outcome'), ('io23', 'labels_at'),
                                       ('io23', 'cohort_ids'), ('simple_data', 'attach_outcome'),
                                       ('fixed_test_protocol', 'load_test_contract')])
def test_outcome_and_class_manifest_apis_are_blocked(module, name):
    with replay.no_outcomes():
        with pytest.raises(AssertionError, match='outcome|label|manifest'):
            getattr(getattr(replay.runner, module), name)()


@pytest.mark.parametrize('changed', ['source', 'slow', 'controller', 'mean', 'scale', 'radius',
                                     'training', 'requires_grad'])
def test_frozen_snapshot_detects_weights_slow_policy_and_scale_mutations(changed):
    net = torch.nn.Linear(2, 1).eval().requires_grad_(False)
    slow = torch.zeros(2)
    policies = {v: dict(controller=torch.nn.Linear(1, 1).eval().requires_grad_(False),
                        mean_context=torch.zeros(2), scale={'center': torch.zeros(2)}, radius=.1)
                for v in stream.VARIANTS}
    snapshot = replay.snapshot_frozen(net, slow, policies)
    replay.assert_frozen(net, slow, policies, snapshot)
    with torch.no_grad():
        policy = policies['raw_fixed']
        if changed == 'source':
            net.weight[0, 0] += 1
        elif changed == 'slow':
            slow[0] = 1
        elif changed == 'controller':
            policy['controller'].weight[0, 0] += 1
        elif changed == 'mean':
            policy['mean_context'][0] = 1
        elif changed == 'scale':
            policy['scale']['center'][0] = 1
        elif changed == 'radius':
            policy['radius'] = .2
        elif changed == 'training':
            policy['controller'].train()
        else:
            policy['controller'].weight.requires_grad_(True)
    with pytest.raises(AssertionError, match='frozen|policy'):
        replay.assert_frozen(net, slow, policies, snapshot)


@pytest.mark.parametrize('case', ['matching', 'timeline', 'outcome', 'archive_changed'])
def test_one_shot_uses_current_runner_and_only_signal_input(monkeypatch, case):
    times, curves, updates = arrays()
    current = {'times': times.copy(), **{k: v.copy() for k, v in curves.items()}}
    old = {'times': times.copy(), 'cold': curves['cold'].copy()}
    seal = {'updates': copy.deepcopy(updates), 'shot': 70323, 'index': 21,
            'sha256': 'new', 'archive_seal_sha256': 'old-seal',
            'outcomes_used_in_prediction': False, 'startup_archived': False}
    barrier = {'seals': [{}] * 20 + [{'shot': 70323, 'sha256': 'new'}],
               'receipt_sha256': [''] * 20 + ['seal']}
    old_barrier = {'seals': [{}] * 20 + [{'shot': 70323, 'sha256': 'old'}]}
    monkeypatch.setattr(replay.audit, 'verified_npz', lambda path, digest: current if digest == 'new' else old)
    monkeypatch.setattr(replay.audit, 'read_json', lambda _p: seal)

    def digest(path):
        if str(path).endswith('_seal.json'):
            return 'old-seal' if str(path).startswith('/fake/old/') else 'seal'
        if str(path).endswith('.npz'):
            return ('old' if str(path).startswith('/fake/old/')
                    else 'changed' if case == 'archive_changed' else 'new')
        return 'raw'

    monkeypatch.setattr(replay.audit, 'sha', digest)
    monkeypatch.setattr(replay.runner.io23, 'path_for', lambda *_: Path('/fake/raw'))
    raw = {'times': times.copy(), 'values': np.zeros((154, 23), np.float32),
           'available': np.ones((154, 23), bool), 'other_nonlabel_field': 1}
    if case == 'timeline':
        raw['times'][0] = -1
    monkeypatch.setattr(replay.runner.simple_data, 'read_shot', lambda *_a, **_k: raw)
    monkeypatch.setattr(replay.runner, 'apply_input_contract', lambda x: x)
    monkeypatch.setattr(replay.runner, 'transform', lambda signal, _mapping: signal)

    calls = []

    def predict(_net, signal, _slow, _policies):
        calls.append(True)
        assert set(signal) == {'values', 'available'}
        if case == 'outcome':
            replay.runner.io23.read_outcome('EAST', 70323)
        return curves, updates

    monkeypatch.setattr(replay.runner, 'predict', predict)
    policies = {v: {'radius': .1} for v in stream.VARIANTS}
    with replay.no_outcomes():
        if case != 'matching':
            with pytest.raises((ValueError, AssertionError), match='time|outcome|changed'):
                replay.replay_shot(Path('/fake/run'), Path('/fake/old'), 20, 70323,
                                   barrier, old_barrier, {'70323': 'raw'}, {}, None, None, policies)
            assert bool(calls) is (case != 'timeline')
            return
        result = replay.replay_shot(Path('/fake/run'), Path('/fake/old'), 20, 70323,
                                    barrier, old_barrier, {'70323': 'raw'}, {}, None, None, policies)
    assert result['shot'] == 70323
    assert result['times_max_abs_error'] == 0
    assert max(result['curves_max_abs_error'].values()) == 0


def test_cpu_toy_replays_actual_padded_stream_and_receipts(monkeypatch):
    """Exercise the real 22-curve stream using only synthetic in-memory signals."""
    class ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Module()
            self.base.register_buffer('center', torch.zeros(23))

        def forward(self, x, mask, states, slow, mode, forecast=False):
            return {'logit': x[:, -1, 0] + states[:, 0]}

    def zero_update(net, prefix, origin, slow, policies):
        assert origin == 127 and len(prefix['values']) == 153
        return ({v: torch.zeros(9, 8192) for v in stream.VARIANTS},
                dict(state_norm={name: 0. for name in replay.NAMES},
                     cos_real={name: None for name in replay.NAMES},
                     max_bound=0., feedback_supported=False))

    monkeypatch.setattr(stream, 'new_states', zero_update)
    times = np.arange(154, dtype=np.float64) / 1000
    values = np.zeros((154, 23), np.float32)
    values[:, 0] = np.linspace(-1., 1., 154)
    signal = {'values': values, 'available': np.ones_like(values, dtype=bool)}
    policies = {v: {'radius': .1} for v in stream.VARIANTS}
    net = ToyNet().eval()
    with replay.no_outcomes():
        curves, updates = replay.runner.predict(net, signal, None, policies)
        saved = {'times': times, **curves}
        old = {'times': times, 'cold': curves['cold']}
        seal = dict(updates=updates, shot=70323, index=21, sha256='new',
                    archive_seal_sha256='old-seal', outcomes_used_in_prediction=False,
                    startup_archived=False)
        barrier = {'seals': [{}] * 20 + [{'shot': 70323, 'sha256': 'new'}],
                   'receipt_sha256': [''] * 20 + ['seal']}
        old_barrier = {'seals': [{}] * 20 + [{'shot': 70323, 'sha256': 'old'}]}
        monkeypatch.setattr(replay.audit, 'verified_npz', lambda _p, digest: saved if digest == 'new' else old)
        monkeypatch.setattr(replay.audit, 'read_json', lambda _p: seal)

        def digest(path):
            if str(path).endswith('_seal.json'):
                return 'old-seal' if str(path).startswith('/fake/old/') else 'seal'
            if str(path).endswith('.npz'):
                return 'old' if str(path).startswith('/fake/old/') else 'new'
            return 'raw'

        monkeypatch.setattr(replay.audit, 'sha', digest)
        monkeypatch.setattr(replay.runner.io23, 'path_for', lambda *_: Path('/fake/raw'))
        monkeypatch.setattr(replay.runner.simple_data, 'read_shot', lambda *_a, **_k: {'times': times, **signal})
        monkeypatch.setattr(replay.runner, 'apply_input_contract', lambda raw: raw)
        monkeypatch.setattr(replay.runner, 'transform', lambda observed, _mapping: observed)
        result = replay.replay_shot(Path('/fake/run'), Path('/fake/old'), 20, 70323,
                                    barrier, old_barrier, {'70323': 'raw'}, {}, net, None, policies)
    assert result['receipts_exact'] and len(result['curves_max_abs_error']) == 22
    assert max(result['curves_max_abs_error'].values()) == 0


def test_replay_orchestration_checks_all_seals_before_selection_and_rechecks_provenance(
        monkeypatch, order, comparisons):
    """Mock CUDA and every artifact; this exercises ordering only, never a GPU."""
    events = []
    directory, previous, reference = Path('/fake/run'), Path('/fake/old'), Path('/fake/reference')
    complete = {'comparisons': comparisons, 'artifact_sha256': {
        'protocol.json': 'protocol-hash', 'all_predictions_sealed.json': 'barrier-hash'}}
    protocol = {'order': order, 'score_batch_per_arm': 32, 'padded_rows_per_variant': 9,
                'previous': str(previous), 'policy': '/fake/policy', 'policy_hashes': {'p': 'p-hash'},
                'reference_hashes': {name: 'hash' for name in
                                     ('order.json', 'input_map.json', 'input_hashes.json', 'models/cold.pt')},
                'threshold': .5, 'gpu': 'original', 'torch': 'original'}
    barrier, old_barrier = {}, {}

    def gate(path):
        assert path == directory
        events.append('barrier')
        return complete, protocol, barrier

    def provenance(*_args):
        events.append('provenance')
        return reference, old_barrier, {}

    def seals(*_args):
        assert events == ['barrier', 'provenance']
        events.append('all-seals')

    def digest(path):
        return {'complete.json': 'complete-hash', 'protocol.json': 'protocol-hash',
                'all_predictions_sealed.json': 'barrier-hash'}[Path(path).name]

    def read(path):
        return {'protocol.json': {}, 'input_map.json': {}, 'input_hashes.json': {}}[Path(path).name]

    def load_policies(_path, _old, device):
        assert device == 'cuda' and events[-1] == 'all-seals'
        policies = {v: dict(controller=torch.nn.Linear(1, 1).eval().requires_grad_(False),
                            mean_context=torch.zeros(2), scale={'center': torch.zeros(2)}, radius=.1)
                    for v in stream.VARIANTS}
        return policies, {'p': 'p-hash'}

    class MockCudaNet(torch.nn.Linear):
        def cuda(self):
            return self

    def one_shot(_directory, _previous, index, sid, *_args):
        assert events[-1] in ('all-seals', 'prediction')
        with pytest.raises(AssertionError, match='outcome'):
            replay.runner.io23.read_outcome('EAST', sid)
        events.append('prediction')
        return dict(shot=sid, index=index + 1, times_max_abs_error=0.,
                    curves_max_abs_error={name: 0. for name in replay.NAMES}, receipt_max_abs_error=0.)

    monkeypatch.setattr(replay.audit, 'require_completed_barrier', gate)
    monkeypatch.setattr(replay.audit, 'verify_provenance', provenance)
    monkeypatch.setattr(replay.audit, 'verify_all_seals', seals)
    monkeypatch.setattr(replay.audit, 'sha', digest)
    monkeypatch.setattr(replay.audit, 'read_json', read)
    monkeypatch.setattr(replay.runner, 'load_policies', load_policies)
    monkeypatch.setattr(replay.runner, 'load_parent', lambda *_: (MockCudaNet(1, 1).eval(), {'threshold': .5}))
    monkeypatch.setattr(replay.runner, 'MatrixAttention', lambda net: net)
    monkeypatch.setattr(replay.torch, 'load', lambda *_a, **_k: {'slow': torch.zeros(1)})
    monkeypatch.setattr(replay.torch, 'manual_seed', lambda *_: None)
    monkeypatch.setattr(replay.torch, 'set_num_threads', lambda *_: None)
    monkeypatch.setattr(replay.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(replay.torch.cuda, 'get_device_name', lambda: 'mocked CPU')
    monkeypatch.setattr(replay, 'replay_shot', one_shot)
    result = replay.replay(directory)
    assert result['shots'] == [70323, 1000, 1001, 1004]
    assert result['arm_count'] == 22 and result['frozen_parameters_unchanged'] is True
    assert result['curves_max_abs_error'] == result['times_max_abs_error'] == 0
    assert result['selection_uses_completed_paired_report'] is True
    assert result['io23_outcome_and_label_access_blocked'] is True
    assert events == ['barrier', 'provenance', 'all-seals'] + ['prediction'] * 4 + ['provenance']
    events.clear()
    del protocol['reference_hashes']['input_map.json']
    with pytest.raises(ValueError, match='reference.*hash'):
        replay.replay(directory)
    assert events == ['barrier', 'provenance', 'all-seals']
