"""Replay fixed EAST360 with sealed cold20 and bounded signal-only transport."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import fixed_test_protocol, io23, simple_data
from ..arrival_stream_20261030.run import OutcomeGate, comparisons
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_likelihood_20261029.run import read
from ..latent_completion_20261028.stream import compact
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .stream import NAMES, RADIUS, STATE_SIZE, predict


PREVIOUS = io23.ROOT / 'probe_outputs/arrival_reference_east360_v1'
PREVIOUS_SHA = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'
POLICY = io23.ROOT / 'probe_outputs/operational_transport_cold20_v2'
PLAN = io23.ROOT / 'OPERATIONAL_TRANSPORT_PLAN_20261030.md'


def validate_policy_metadata(complete, protocol, payload, previous, plan_sha):
    """Reject mismatched completed teaching runs before using any policy."""
    try:
        valid = (complete['complete'] is True and complete['source_unchanged'] is True
                 and complete['full_stream_run'] is False and complete['shots'] == 20
                 and complete['examples'] > 0 and complete['disruptive_examples'] > 0
                 and complete['safe_examples'] > 0
                 and complete['examples'] == complete['disruptive_examples'] + complete['safe_examples']
                 and len(complete['policy_sha256']) == 64
                 and protocol['scope'] == 'ENDED_COLD20_MATCHED_QK_GRADIENT_TEACHING'
                 and protocol['warmup_ids'] == previous['order'][:20]
                 and protocol['reference'] == previous['reference']
                 and protocol['reference_hashes'] == previous['reference_hashes']
                 and Path(protocol['previous']) == PREVIOUS
                 and protocol['previous_sha256'] == PREVIOUS_SHA
                 and protocol['source_sha256'] == PARENT_SHA256
                 and protocol['threshold'] == previous['threshold']
                 and protocol['ridge'] == .1 and protocol['radius'] == RADIUS
                 and protocol['source_updates'] == 0 and protocol['suffix_labels_accessed'] is False
                 and protocol['plan_sha256'] == plan_sha and payload['protocol'] == protocol
                 and set(payload['rules']) == {'operational', 'point', 'shuffled'}
                 and len(payload['pairs_sha256']) == 64)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('completed policy and fixed cold20 protocol do not match')
    try:
        for rule in payload['rules'].values():
            if (rule['ridge'] != .1
                    or rule['keys'].shape != (complete['examples'], STATE_SIZE)
                    or rule['coefficients'].shape != rule['keys'].shape
                    or rule['center'].shape != (STATE_SIZE,) or rule['mean'].shape != (STATE_SIZE,)
                    or any(not isinstance(rule[k], torch.Tensor) or not rule[k].is_floating_point()
                           or not torch.isfinite(rule[k]).all()
                           for k in ('keys', 'coefficients', 'center', 'mean'))):
                raise ValueError('invalid learned rule')
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError('completed policy has invalid rule tensors') from exc


def load_policy(policy_dir, previous):
    """Verify completion, teaching code and all serialized policy provenance."""
    complete_path, protocol_path = policy_dir/'complete.json', policy_dir/'protocol.json'
    policy_path = policy_dir/'policy.pt'
    hashes = {name: sha(policy_dir/name) for name in
              ('complete.json', 'protocol.json', 'policy.pt', 'pairs.pt')}
    complete, protocol = read(complete_path), read(protocol_path)
    if hashes['policy.pt'] != complete.get('policy_sha256'):
        raise ValueError('policy differs from completed teaching artifact')
    payload = torch.load(policy_path, map_location='cpu', weights_only=True)
    validate_policy_metadata(complete, protocol, payload, previous, sha(PLAN))
    if payload['pairs_sha256'] != hashes['pairs.pt']:
        raise ValueError('teaching pairs differ from policy provenance')
    for name, digest in protocol['code_sha256'].items():
        path = (io23.ROOT / name).resolve()
        if not path.is_relative_to(io23.ROOT.resolve()) or sha(path) != digest:
            raise ValueError('teaching code changed: ' + name)
    if any(sha(policy_dir/name) != digest for name, digest in hashes.items()):
        raise ValueError('policy artifacts changed while being loaded')
    return payload['rules'], hashes


def evaluate(rows, threshold):
    """Use the existing locked full-test evaluator and strict-30 compact metrics."""
    predictions = {name: fixed_test_protocol.score_fixed_test('EAST', group, threshold)['predictions']
                   for name, group in rows.items()}
    results = {name: {section: compact(group, threshold) for section, group in (
        ('full360', values), ('cold20', values[:20]), ('suffix340', values[20:]),
        ('high_proxy90', values[-90:]))} for name, values in rows.items()}
    return results, predictions, comparisons(predictions)


def code_files():
    files = {Path(__file__), Path(__file__).with_name('stream.py'), PLAN,
             io23.ROOT/'tests/test_operational_transport_stream.py',
             io23.ROOT/'tests/test_operational_transport_gradients.py'}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    return files


def check_cold_archive(cold, archived, threshold):
    error = float(np.max(np.abs(cold - archived)))
    disagreements = int(np.sum((cold >= threshold) != (archived >= threshold)))
    if not np.array_equal(cold, archived):
        raise ValueError(f'cold replay is not exactly archived: max_error={error}, '
                         f'threshold_disagreements={disagreements}')
    return error, disagreements


def run(output, policy_dir=POLICY):
    output, policy_dir = Path(output).absolute(), Path(policy_dir).absolute()
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('fresh direct worktree output required')
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError('at least 1 GiB free required; no automatic deletion')
    if sha(PREVIOUS/'complete.json') != PREVIOUS_SHA:
        raise ValueError('fixed archived360 changed')
    previous = read(PREVIOUS/'protocol.json')
    ids, reference = previous['order'], io23.ROOT / previous['reference']
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('fixed full360 order required')
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('cold parent reference changed: ' + name)
    if [row['shot'] for row in read(reference/'order.json')] != ids:
        raise ValueError('archived order does not match the hash-locked cold parent')
    previous_barrier = read(PREVIOUS/'all_predictions_sealed.json')
    if (previous_barrier['count'] != 360
            or [item['shot'] for item in previous_barrier['seals']] != ids):
        raise ValueError('archived full360 prediction barrier changed')
    previous_barrier_hash = sha(PREVIOUS/'all_predictions_sealed.json')
    mapping, raw_hashes = read(reference/'input_map.json'), read(reference/'input_hashes.json')
    if mapping['calibration_ids'] != ids[:20]:
        raise ValueError('different cold20 mapping')
    rules, policy_hashes = load_policy(policy_dir, previous)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    threshold = float(parent['threshold'])
    if threshold != previous['threshold']:
        raise ValueError('threshold changed')
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    slow_before = slow.clone()
    rules = {name: {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in rule.items()}
             for name, rule in rules.items()}
    output.mkdir()
    (output/'shots').mkdir()
    files = code_files()
    protocol = dict(scope='DEVELOPMENT_COMPLETE360_OPERATIONAL_TRANSPORT_INTRA_ONLY',
        order=ids, arms=NAMES, candidate='operational', warmup=20, qk_radius=RADIUS,
        horizon=25, first_origin=127, update_cadence=25, score_batch_per_arm=32,
        feedback_fit='all_ended_cold20', policy=str(policy_dir), policy_hashes=policy_hashes,
        source_sha256=PARENT_SHA256, threshold=threshold, reference=str(reference),
        reference_hashes=previous['reference_hashes'], previous=str(PREVIOUS),
        previous_complete_sha256=PREVIOUS_SHA, previous_barrier_sha256=previous_barrier_hash,
        plan_sha256=sha(PLAN), reset_each_shot=True,
        reset_each_feedback=True, source_updates=0, inter_updates_after_cold20=0,
        alarm_updates=0, suffix_labels_after_all_predictions=True,
        startup_reused_from_sealed_archive=True, independent_confirmation=False,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    started, receipts = time.monotonic(), []
    gate = OutcomeGate(ids, io23.read_outcome)

    def deny(*args, **kwargs):
        raise AssertionError('outcomes and class manifest unavailable before global seal')

    with patch.object(io23, 'read_outcome', side_effect=deny), \
            patch.object(fixed_test_protocol, 'load_test_contract', side_effect=deny):
        for i, sid in enumerate(ids):
            if sha(io23.path_for('EAST', sid)) != raw_hashes[str(sid)]:
                raise ValueError('raw input changed')
            name = f'{i+1:04d}_{sid}'
            old_path = PREVIOUS/'shots'/(name+'.npz')
            old_digest = sha(old_path)
            if (old_digest != read(PREVIOUS/'shots'/(name+'_seal.json'))['sha256']
                    or old_digest != previous_barrier['seals'][i]['sha256']):
                raise ValueError('archived curve seal changed')
            with np.load(old_path, allow_pickle=False) as old:
                old_cold, times = old['cold'], old['times']
            if i < 20:
                curves = {arm: old_cold.copy() for arm in NAMES}
                info = []
            else:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                if not np.array_equal(times, raw['times']):
                    raise ValueError('raw timeline changed')
                signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
                curves, info = predict(net, signal, slow, rules)
            if set(curves) != set(NAMES):
                raise ValueError('all nine preset arms required')
            for values in curves.values():
                if (values.shape != times.shape or not np.isfinite(values).all()
                        or (values < 0).any() or (values > 1).any()):
                    raise ValueError('one finite probability per timestamp required')
            np.testing.assert_array_equal(curves['cold'], curves['identical'])
            error, disagreements = check_cold_archive(curves['cold'], old_cold, threshold)
            path = output/'shots'/(name+'.npz')
            with path.open('xb') as handle:
                np.savez_compressed(handle, times=times, archive=old_cold, **curves)
                handle.flush()
                os.fsync(handle.fileno())
            digest = sha(path)
            receipt = dict(shot=sid, index=i+1, sha256=digest, updates=info,
                           archive_max_abs_error=error, archive_crossing_disagreements=disagreements,
                           startup_archived=i < 20, outcomes_used_in_prediction=False)
            _durable_json(output/'shots'/(name+'_seal.json'), receipt)
            gate.seal(sid, digest)
            receipts.append({k: v for k, v in receipt.items() if k != 'updates'})
            if (i+1) % 10 == 0:
                progress = dict(shots=i+1, total=360, seconds=time.monotonic()-started,
                                suffix_outcomes_read=0)
                _durable_json(output/'progress.json', progress)
                print(progress, flush=True)
    if len(gate.seals) != 360 or gate.accesses:
        raise AssertionError('all predictions must be sealed before any outcome')
    _durable_json(output/'all_predictions_sealed.json', dict(
        seals=gate.seals, count=len(gate.seals), policy_sha256=policy_hashes['policy.pt']))
    rows = {name: [] for name in (*NAMES, 'archive')}
    updates = {name: 0 for name in NAMES}
    applied = {name: 0 for name in NAMES}
    applied_points = {name: 0 for name in NAMES}
    max_bound = 0.
    for i, sid in enumerate(ids):
        name = f'{i+1:04d}_{sid}'
        path = output/'shots'/(name+'.npz')
        if sha(path) != gate.seals[i]['sha256']:
            raise ValueError('sealed prediction modified before evaluation')
        with np.load(path, allow_pickle=False) as saved:
            flag, down = gate.read_outcome('EAST', sid)
            _, valid = io23.labels_at(saved['times'], flag, down, 125.)
            for arm in rows:
                rows[arm].append(dict(shot=sid, times=saved['times'][valid], scores=saved[arm][valid],
                                      is_disrupt=flag, down=down))
        for entry in read(output/'shots'/(name+'_seal.json'))['updates']:
            max_bound = max(max_bound, entry['max_bound'])
            for arm in NAMES:
                nonzero = entry['state_norm'][arm] > 0
                updates[arm] += int(nonzero)
                applied[arm] += int(nonzero and entry['apply_count'] > 0)
                applied_points[arm] += entry['apply_count'] if nonzero else 0
    results, predictions, paired = evaluate(rows, threshold)
    if any(not torch.equal(net.state_dict()[k].detach().cpu(), v) for k, v in frozen.items()):
        raise AssertionError('frozen source parameter changed')
    if not torch.equal(slow, slow_before) or sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('fixed parent changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('experiment code changed during run: ' + name)
    for name, digest in protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise AssertionError('parent artifact changed: ' + name)
    for name, digest in policy_hashes.items():
        if sha(policy_dir/name) != digest:
            raise AssertionError('policy artifact changed during run: ' + name)
    if sha(PREVIOUS/'complete.json') != PREVIOUS_SHA:
        raise AssertionError('archived completion changed')
    if sha(PREVIOUS/'all_predictions_sealed.json') != previous_barrier_hash:
        raise AssertionError('archived full360 barrier changed')
    candidate, baseline = results['operational']['full360'], results['cold']['full360']
    complete = dict(complete=True, results=results, comparisons=paired,
        seconds=time.monotonic()-started, update_counts=updates, applied_update_counts=applied,
        applied_point_counts=applied_points, max_state_norm=max_bound,
        operational_intra_gain=bool(candidate['tpr'] > baseline['tpr'] and candidate['fpr'] <= .20),
        fpr_cap_status={arm: {str(cap): results[arm]['full360']['fpr'] <= cap
                              for cap in (.10, .15, .20)} for arm in NAMES},
        source_unchanged=True, fixed_cold_unchanged=True, source_parameter_updates=0,
        scientific_goal_achieved=False, independent_confirmation=False,
        archive_max_abs_error=max(r['archive_max_abs_error'] for r in receipts),
        archive_crossing_disagreements=sum(r['archive_crossing_disagreements'] for r in receipts))
    _durable_json(output/'per_shot.json', predictions)
    _durable_json(output/'receipts.json', receipts)
    _durable_json(output/'accesses.json', gate.accesses)
    _durable_json(output/'complete.json', complete)
    print({arm: values['full360'] for arm, values in results.items()}, flush=True)
    print(paired, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=POLICY)
    args = parser.parse_args()
    run(args.output, args.policy)
