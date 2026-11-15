"""Diagnostic full360 replay of all four failed-screen context policies."""

import argparse
import math
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
from ..operational_transport_20261030.run import check_cold_archive, evaluate
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ContextTarget
from .scaling import transform_context
from .stream import ARMS, NAMES, VARIANTS, predict


PREVIOUS = io23.ROOT / 'probe_outputs/arrival_reference_east360_v1'
PREVIOUS_SHA = '57dbd7085fba526fb78acee65d14f1033bafc1e3e25d0d54dd7041413e641a35'
POLICY = io23.ROOT / 'probe_outputs/context_vector_factorial_v1'
PLAN = io23.ROOT / 'CONTEXT_VECTOR_FULLSTREAM_DIAGNOSTIC_PLAN_20261030.md'


def load_policies(policy_dir, previous, device='cuda'):
    """Load all four unchanged completed checkpoints, with their common provenance."""
    policy_dir = Path(policy_dir)
    names = ['complete.json', 'protocol.json'] + [
        f'{variant}/{name}' for variant in VARIANTS for name in ('complete.json', 'foldall.pt')]
    hashes = {name: sha(policy_dir / name) for name in names}
    complete, protocol = read(policy_dir / 'complete.json'), read(policy_dir / 'protocol.json')
    if (complete.get('complete') is not True or complete.get('source_unchanged') is not True
            or complete.get('full_stream_run') is not False or complete.get('suffix_labels_read') != 0
            or complete.get('selected_for_full_stream') is not None
            or set(complete.get('variants', {})) != set(VARIANTS)
            or any(complete['variants'][v].get('full_stream_gate') is not False for v in VARIANTS)):
        raise ValueError('factorial completion must retain all four failed component gates')
    if (protocol.get('scope') != 'ENDED_COLD20_SCALE_RADIUS_FACTORIAL'
            or protocol.get('variants') != list(VARIANTS)
            or protocol.get('warmup_ids') != previous['order'][:20]
            or protocol.get('reference') != previous['reference']
            or protocol.get('reference_hashes') != previous['reference_hashes']
            or protocol.get('full_stream_run') is not False or protocol.get('suffix_labels_read') != 0):
        raise ValueError('policy common protocol differs from the fixed archive')
    predecessor = Path(protocol['previous'])
    predecessor_complete = predecessor / 'complete.json'
    predecessor_protocol = predecessor / 'protocol.json'
    if sha(predecessor_complete) != protocol['previous_complete_sha256']:
        raise ValueError('policy predecessor completion changed')
    origin = read(predecessor_protocol)
    teacher_path = Path(origin['previous']) / 'protocol.json'
    teacher = read(teacher_path)
    if (origin.get('source_sha256') != PARENT_SHA256
            or origin.get('warmup_ids') != previous['order'][:20]
            or origin.get('reference_hashes') != previous['reference_hashes']
            or teacher.get('previous_sha256') != PREVIOUS_SHA
            or teacher.get('source_sha256') != PARENT_SHA256
            or teacher.get('threshold') != previous['threshold']
            or teacher.get('warmup_ids') != previous['order'][:20]):
        raise ValueError('policy teaching provenance differs from the fixed archive')
    for path in (predecessor_complete, predecessor_protocol, teacher_path):
        hashes[str(path)] = sha(path)
    for name, digest in protocol['code_sha256'].items():
        path = (io23.ROOT / name).resolve()
        if not path.is_relative_to(io23.ROOT.resolve()) or sha(path) != digest:
            raise ValueError('policy teaching code changed: ' + name)
        hashes[str(path)] = digest
    policies = {}
    for variant in VARIANTS:
        entry = read(policy_dir / variant / 'complete.json')
        if ({k: v for k, v in entry.items() if k != 'policy_sha256'} != complete['variants'][variant]
                or entry.get('full_stream_gate') is not False
                or entry.get('policy_sha256') != hashes[f'{variant}/foldall.pt']):
            raise ValueError('completed policy or component gate differs: ' + variant)
        model = torch.load(policy_dir / variant / 'foldall.pt', map_location='cpu', weights_only=True)
        if (model['protocol'] != protocol or model['fit_ids'] != previous['order'][:20]
                or not isinstance(model['radius'], (float, int))
                or not math.isfinite(model['radius']) or model['radius'] < 0
                or (variant.endswith('_fixed') and model['radius'] != .1)):
            raise ValueError('policy checkpoint provenance or radius differs: ' + variant)
        mean, scale = model['mean_context'], model['scale']
        if (not isinstance(mean, torch.Tensor) or mean.shape != (358,)
                or not mean.is_floating_point() or not torch.isfinite(mean).all()):
            raise ValueError('policy mean context invalid: ' + variant)
        # This also checks finite, positive CPU scaling statistics.
        transform_context(torch.zeros(1, 358, dtype=mean.dtype), scale)
        controller = ContextTarget()
        controller.load_state_dict(model['state'], strict=True)
        if any(not torch.isfinite(p).all() for p in controller.parameters()):
            raise ValueError('policy contains nonfinite controller weights: ' + variant)
        policies[variant] = dict(controller=controller.to(device).eval().requires_grad_(False),
                                 mean_context=mean.to(device), scale=scale, radius=float(model['radius']))
    if any(sha(policy_dir / name) != digest for name, digest in hashes.items()):
        raise ValueError('policy artifacts changed during loading')
    return policies, hashes


class GlobalOutcomeGate(OutcomeGate):
    """Tighten the existing gate: even cold outcomes wait for a durable barrier."""

    def __init__(self, ids, reader):
        super().__init__(ids, reader)
        self.released = False

    def release(self, barrier_path):
        barrier = read(barrier_path)
        if (len(self.seals) != 360 or self.accesses or barrier.get('count') != 360
                or barrier.get('seals') != self.seals):
            raise AssertionError('complete durable global barrier required')
        self.released = True

    def read_outcome(self, device, sid):
        if not self.released:
            raise AssertionError('outcome blocked until the global barrier')
        return super().read_outcome(device, sid)


def validate_receipts(receipts, length, policies):
    """Validate stream timing and bounded directions, including zero actions."""
    events = list(range(152, length, 25))
    if len(receipts) != len(events):
        raise ValueError('receipt count does not match matured feedback timing')
    for event, item in zip(events, receipts, strict=True):
        if ((item['origin'], item['maturity'], item['first_affected'], item['apply_count'])
                != (event - 25, event, event + 1, min(25, length - event - 1))
                or set(item['state_norm']) != set(NAMES) or set(item['cos_real']) != set(NAMES)):
            raise ValueError('receipt timing or curve keys differ')
        norms = item['state_norm']
        if any(not math.isfinite(v) or v < 0 for v in norms.values()):
            raise ValueError('receipt has invalid action norm')
        if (norms['cold'] != 0 or norms['identical'] != 0
                or not math.isfinite(item['max_bound'])
                or not math.isclose(item['max_bound'], max(norms.values()), abs_tol=1e-6)):
            raise ValueError('receipt bound or zero-action control differs')
        for name, value in item['cos_real'].items():
            if (value is not None and (not math.isfinite(value) or abs(value) > 1 + 1e-5)):
                raise ValueError('receipt cosine is invalid')
            if norms[name] == 0 and value is not None:
                raise ValueError('receipt zero direction must have undefined cosine')
        for variant in VARIANTS:
            if any(norms[f'{variant}__{arm}'] > policies[variant]['radius'] + 1e-6
                   for arm in ARMS[1:6]):
                raise ValueError('receipt action exceeds its frozen bound')
        if item.get('feedback_supported') is False and any(norms.values()):
            raise ValueError('receipt without feedback must replace actions with zero')


def load_archived_shot(archive, index, sid, barrier):
    name = f'{index + 1:04d}_{sid}'
    path, seal_path = archive / 'shots' / (name + '.npz'), archive / 'shots' / (name + '_seal.json')
    seal, digest = read(seal_path), sha(path)
    if (seal['shot'] != sid or seal['index'] != index + 1 or seal['sha256'] != digest
            or barrier['seals'][index] != {'shot': sid, 'sha256': digest}):
        raise ValueError('archived shot seal differs from the global barrier')
    with np.load(path, allow_pickle=False) as old:
        return old['times'].copy(), old['cold'].copy(), sha(seal_path)


def seal_predictions(output, ids, gate, archive, barrier, raw_hashes, mapping,
                     net, slow, policies, threshold):
    """Seal all predictions before allowing either outcome or class-manifest reads."""
    receipts = []

    def deny(*_args, **_kwargs):
        raise AssertionError('outcome, label and class manifest blocked during prediction')

    with patch.object(io23, 'read_outcome', side_effect=deny), \
            patch.object(io23, 'labels_at', side_effect=deny), \
            patch.object(fixed_test_protocol, 'load_test_contract', side_effect=deny):
        for i, sid in enumerate(ids):
            if sha(io23.path_for('EAST', sid)) != raw_hashes[str(sid)]:
                raise ValueError('raw input changed')
            times, cold, old_seal_hash = load_archived_shot(archive, i, sid, barrier)
            if i < 20:
                curves, info = {name: cold.copy() for name in NAMES}, []
            else:
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                if not np.array_equal(times, raw['times']):
                    raise ValueError('raw timeline changed')
                signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
                curves, info = predict(net, signal, slow, policies)
                validate_receipts(info, len(times), policies)
            if set(curves) != set(NAMES):
                raise ValueError('all 22 preset curves required')
            for values in curves.values():
                if (values.shape != times.shape or not np.isfinite(values).all()
                        or (values < 0).any() or (values > 1).any()):
                    raise ValueError('one finite probability per original timestamp required')
            if not np.array_equal(curves['cold'], curves['identical']):
                raise ValueError('cold and identical controls differ')
            error, disagreements = check_cold_archive(curves['cold'], cold, threshold)
            name = f'{i + 1:04d}_{sid}'
            path = output / 'shots' / (name + '.npz')
            with path.open('xb') as handle:
                np.savez_compressed(handle, times=times, **curves)
                handle.flush()
                os.fsync(handle.fileno())
            digest = sha(path)
            seal = dict(shot=sid, index=i + 1, sha256=digest, updates=info,
                        archive_seal_sha256=old_seal_hash, archive_max_abs_error=error,
                        archive_crossing_disagreements=disagreements,
                        startup_archived=i < 20, outcomes_used_in_prediction=False)
            seal_path = output / 'shots' / (name + '_seal.json')
            _durable_json(seal_path, seal)
            gate.seal(sid, digest)
            receipts.append({**{k: v for k, v in seal.items() if k != 'updates'},
                             'seal_sha256': sha(seal_path)})
            if (i + 1) % 10 == 0:
                print(dict(shots=i + 1, total=360, suffix_outcomes_read=0), flush=True)
        if len(gate.seals) != 360 or gate.accesses:
            raise AssertionError('all 360 predictions must precede any outcome')
        for i, item in enumerate(receipts):
            name = f"{i + 1:04d}_{item['shot']}"
            if (sha(output / 'shots' / (name + '.npz')) != item['sha256']
                    or sha(output / 'shots' / (name + '_seal.json')) != item['seal_sha256']):
                raise ValueError('sealed prediction or receipt changed before barrier')
        barrier_path = output / 'all_predictions_sealed.json'
        _durable_json(barrier_path, dict(count=360, seals=gate.seals,
                                         receipt_sha256=[r['seal_sha256'] for r in receipts]))
    gate.release(barrier_path)
    return receipts


def code_files():
    root = io23.ROOT.resolve()
    files = {Path(__file__).resolve(), Path(__file__).with_name('stream.py').resolve(), PLAN.resolve()}
    for name in ('test_context_vector_run_full.py', 'test_context_vector_stream.py'):
        path = root / 'tests' / name
        if path.is_file():
            files.add(path)
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(root) and path.is_file():
                files.add(path)
    return files


def report_predictions(rows, threshold):
    """Reuse the locked evaluator; retain first alarms and paired shot changes."""
    results, predictions, _ = evaluate(rows, threshold)
    for arm in NAMES:
        results[arm]['last90'] = results[arm].pop('high_proxy90')
        for row, record in zip(rows[arm], predictions[arm], strict=True):
            hit = np.flatnonzero(row['scores'] >= threshold)
            record['first_alarm_time_seconds'] = float(row['times'][hit[0]]) if len(hit) else None
    sections = {'full360': slice(None), 'cold20': slice(0, 20),
                'suffix340': slice(20, None), 'last90': slice(-90, None)}
    paired = {section: comparisons({arm: records[part] for arm, records in predictions.items()})
              for section, part in sections.items()}
    controls = {}
    for variant in VARIANTS:
        real = predictions[f'{variant}__real']
        controls[variant] = {}
        for arm in ('reverse', 'mean_feedback', 'constant', 'risk_axis'):
            other = predictions[f'{variant}__{arm}']
            controls[variant][arm] = {
                section: comparisons({'cold': other[part], 'real': real[part]})['real']
                for section, part in sections.items()}
    caps = {arm: {section: {str(cap): (None if metrics['fpr'] is None else metrics['fpr'] <= cap)
                            for cap in (.10, .15, .20)}
                  for section, metrics in values.items()} for arm, values in results.items()}
    return results, predictions, paired, controls, caps


def run(output):
    output = Path(output).absolute()
    if output.exists() or output.parent != io23.ROOT / 'probe_outputs':
        raise ValueError('fresh direct worktree probe_outputs destination required')
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError('at least 1 GiB free required; no automatic deletion')
    if sha(PREVIOUS / 'complete.json') != PREVIOUS_SHA:
        raise ValueError('fixed archived360 completion changed')
    previous_path, barrier_path = PREVIOUS / 'protocol.json', PREVIOUS / 'all_predictions_sealed.json'
    previous, previous_barrier = read(previous_path), read(barrier_path)
    previous_hashes = {'protocol.json': sha(previous_path), 'all_predictions_sealed.json': sha(barrier_path)}
    ids = previous['order']
    if (len(ids) != 360 or len(set(ids)) != 360 or previous['source_sha256'] != PARENT_SHA256
            or previous_barrier['count'] != 360
            or [item['shot'] for item in previous_barrier['seals']] != ids):
        raise ValueError('fixed ordered full360 archive required')
    # The worktree deliberately symlinks the archived runs to the source tree.
    reference = io23.ROOT / previous['reference']
    for name, digest in previous['reference_hashes'].items():
        if sha(reference / name) != digest:
            raise ValueError('fixed reference changed: ' + name)
    if [row['shot'] for row in read(reference / 'order.json')] != ids:
        raise ValueError('fixed reference order differs')
    mapping, raw_hashes = read(reference / 'input_map.json'), read(reference / 'input_hashes.json')
    if mapping['calibration_ids'] != ids[:20]:
        raise ValueError('fixed cold20 input mapping differs')

    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required; no CPU full-stream execution')
    policies, policy_hashes = load_policies(POLICY, previous, device='cuda')
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda()).eval()
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    frozen_policies = {v: {k: x.detach().cpu().clone() for k, x in p['controller'].state_dict().items()}
                       for v, p in policies.items()}
    frozen_means = {v: p['mean_context'].detach().cpu().clone() for v, p in policies.items()}
    frozen_scales = {v: {k: x.clone() for k, x in p['scale'].items()} for v, p in policies.items()}
    frozen_radii = {v: p['radius'] for v, p in policies.items()}
    threshold = float(parent['threshold'])
    if threshold != previous['threshold'] or sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise ValueError('fixed source or alarm threshold changed')
    slow = torch.load(reference / 'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    slow_before = slow.clone()

    output.mkdir()
    (output / 'shots').mkdir()
    files = code_files()
    code_hashes = {str(p.relative_to(io23.ROOT.resolve())): sha(p) for p in sorted(files)}
    protocol = dict(scope='DIAGNOSTIC_FULL360_AFTER_FAILED_COMPONENT_GATE',
        order=ids, curves=NAMES, variants=VARIANTS, update_arms=ARMS[1:6],
        component_gate_status={v: 'FAILED' for v in VARIANTS}, promoted=False,
        policy=str(POLICY), policy_hashes=policy_hashes, source_sha256=PARENT_SHA256,
        threshold=threshold, lead_ms=30., lead_comparison='strict greater than',
        reference=str(reference), reference_hashes=previous['reference_hashes'],
        previous=str(PREVIOUS), previous_complete_sha256=PREVIOUS_SHA,
        previous_hashes=previous_hashes, plan_sha256=sha(PLAN), code_sha256=code_hashes,
        warmup=20, first_origin=127, horizon=25, update_cadence=25,
        first_affected='maturity + 1', score_batch_per_arm=32, padded_rows_per_variant=9,
        reset_each_shot=True, replace_each_feedback=True, no_feedback_replaces_with_zero=True,
        startup_reused_from_sealed_archive=True, source_updates=0, inter_updates_after_cold20=0,
        alarm_updates=0, outcomes_after_all_predictions=True, independent_confirmation=False,
        torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(output / 'protocol.json', protocol)
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT.resolve())))
    fixed_outputs = {name: sha(output / name) for name in ('protocol.json', 'code_snapshot.zip')}
    started = time.monotonic()
    gate = GlobalOutcomeGate(ids, io23.read_outcome)
    receipts = seal_predictions(output, ids, gate, PREVIOUS, previous_barrier, raw_hashes,
                                mapping, net, slow, policies, threshold)
    fixed_outputs['all_predictions_sealed.json'] = sha(output / 'all_predictions_sealed.json')
    rows = {name: [] for name in NAMES}
    updates = {name: 0 for name in NAMES}
    applied_updates = {name: 0 for name in NAMES}
    applied_points = {name: 0 for name in NAMES}
    max_bound = 0.
    for i, sid in enumerate(ids):
        name = f'{i + 1:04d}_{sid}'
        path, seal_path = output / 'shots' / (name + '.npz'), output / 'shots' / (name + '_seal.json')
        if sha(path) != gate.seals[i]['sha256'] or sha(seal_path) != receipts[i]['seal_sha256']:
            raise ValueError('sealed prediction or receipt changed before evaluation')
        with np.load(path, allow_pickle=False) as saved:
            flag, down = gate.read_outcome('EAST', sid)
            _, valid = io23.labels_at(saved['times'], flag, down, 125.)
            for arm in NAMES:
                rows[arm].append(dict(shot=sid, times=saved['times'][valid], scores=saved[arm][valid],
                                      is_disrupt=flag, down=down))
        for item in read(seal_path)['updates']:
            max_bound = max(max_bound, item['max_bound'])
            for arm in NAMES:
                nonzero = item['state_norm'][arm] > 0
                updates[arm] += int(nonzero)
                applied_updates[arm] += int(nonzero and item['apply_count'] > 0)
                applied_points[arm] += item['apply_count'] if nonzero else 0
    results, predictions, paired, controls, caps = report_predictions(rows, threshold)
    if (any(not torch.equal(net.state_dict()[k].detach().cpu(), v) for k, v in frozen.items())
            or not torch.equal(slow, slow_before) or sha(DEFAULT_PARENT) != PARENT_SHA256):
        raise AssertionError('frozen source or slow parameters changed')
    for variant, policy in policies.items():
        if (any(not torch.equal(policy['controller'].state_dict()[k].detach().cpu(), value)
                for k, value in frozen_policies[variant].items())
                or policy['controller'].training or policy['radius'] != frozen_radii[variant]
                or not torch.equal(policy['mean_context'].detach().cpu(), frozen_means[variant])
                or any(not torch.equal(policy['scale'][k], value)
                       for k, value in frozen_scales[variant].items())):
            raise AssertionError('frozen policy changed: ' + variant)
    for name, digest in code_hashes.items():
        if sha(io23.ROOT / name) != digest:
            raise AssertionError('experiment code changed: ' + name)
    if code_files() != files:
        raise AssertionError('new project code imported after the snapshot')
    for name, digest in previous['reference_hashes'].items():
        if sha(reference / name) != digest:
            raise AssertionError('fixed reference changed: ' + name)
    for name, digest in policy_hashes.items():
        if sha(POLICY / name) != digest:
            raise AssertionError('policy artifact changed: ' + name)
    if (sha(PREVIOUS / 'complete.json') != PREVIOUS_SHA
            or any(sha(PREVIOUS / name) != digest for name, digest in previous_hashes.items())):
        raise AssertionError('archived completion, protocol or barrier changed')
    for i, sid in enumerate(ids):
        name = f'{i + 1:04d}_{sid}'
        if (sha(io23.path_for('EAST', sid)) != raw_hashes[str(sid)]
                or sha(PREVIOUS / 'shots' / (name + '.npz')) != previous_barrier['seals'][i]['sha256']
                or sha(PREVIOUS / 'shots' / (name + '_seal.json')) != receipts[i]['archive_seal_sha256']
                or sha(output / 'shots' / (name + '.npz')) != receipts[i]['sha256']
                or sha(output / 'shots' / (name + '_seal.json')) != receipts[i]['seal_sha256']):
            raise AssertionError('raw, archived or newly sealed shot artifact changed')
    for name, value in (('per_shot.json', predictions), ('receipts.json', receipts),
                        ('accesses.json', gate.accesses)):
        _durable_json(output / name, value)
        fixed_outputs[name] = sha(output / name)
    if any(sha(output / name) != digest for name, digest in fixed_outputs.items()):
        raise AssertionError('output artifact changed before completion')
    complete = dict(complete=True, scope=protocol['scope'], results=results, comparisons=paired,
        real_vs_controls=controls, fpr_cap_status=caps, update_counts=updates,
        applied_update_counts=applied_updates, applied_point_counts=applied_points,
        max_state_norm=max_bound, seconds=time.monotonic() - started,
        component_gate_status=protocol['component_gate_status'], promoted=False,
        scientific_goal_achieved=False, independent_confirmation=False,
        source_unchanged=True, fixed_cold_unchanged=True, source_parameter_updates=0,
        archive_max_abs_error=max(r['archive_max_abs_error'] for r in receipts),
        archive_crossing_disagreements=sum(r['archive_crossing_disagreements'] for r in receipts),
        artifact_sha256=fixed_outputs)
    _durable_json(output / 'complete.json', complete)
    print({arm: values['full360'] for arm, values in results.items()}, flush=True)
    return complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
