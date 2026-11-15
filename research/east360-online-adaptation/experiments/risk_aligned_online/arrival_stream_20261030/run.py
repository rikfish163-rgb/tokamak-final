"""Fixed all-cold20 feedback fit followed by sealed intra-only EAST360 replay."""

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
from ..arrival_increment_20261030.learning import fit_pair
from ..arrival_increment_20261030.run import CACHE, CACHE_HASHES
from ..arrival_reference_20261030.learning import fit_reference
from ..arrival_reference_20261030.run import ARMS
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_forecast_20261023.learning import misplaced_labels
from ..conditional_likelihood_20261029.run import (
    COMPLETE_SHA, PREVIOUS, read, tensor_tree, training_rows,
)
from ..latent_completion_20261028.stream import compact
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

ALL_ARMS = ('cold', *ARMS)


class OutcomeGate:
    """Cold labels after own seal; all suffix labels after the global seal."""

    def __init__(self, ids, reader):
        if len(ids) != 360 or len(set(ids)) != 360:
            raise ValueError('complete distinct360 required')
        self.ids, self.reader = list(ids), reader
        self.positions = {sid: i for i, sid in enumerate(ids)}
        self.seals, self.accesses = [], []

    def seal(self, sid, digest):
        if (len(self.seals) == 360 or sid != self.ids[len(self.seals)]
                or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('ordered unique seal with actual SHA256 required')
        self.seals.append(dict(shot=sid, sha256=digest))

    def read_outcome(self, device, sid):
        i = self.positions.get(sid)
        if (device != 'EAST' or i is None or i >= len(self.seals)
                or (i >= 20 and len(self.seals) != 360)):
            raise AssertionError('outcome is not authorized at this sealing stage')
        self.accesses.append(dict(shot=int(sid), sealed_count=len(self.seals)))
        return self.reader(device, sid)


def fit_feedback(output, ids, gate):
    for name, digest in CACHE_HASHES.items():
        if sha(CACHE/name) != digest:
            raise ValueError('fixed observation cache changed')
    with np.load(CACHE/'observations.npz', allow_pickle=False) as saved:
        data = dict(saved)
    if set(data['ids']) != set(ids[:20]):
        raise ValueError('only same ended cold20 observations can fit feedback')
    outcomes = {s: gate.read_outcome('EAST', s) for s in ids[:20]}
    take, labels = training_rows(data, outcomes)
    pairs, refs = {}, {}
    for arm, y in (('real', labels), ('misplaced', misplaced_labels(data['ids'][take], labels))):
        pairs[arm] = fit_pair(data['x'][take], data['target'][take], data['valid'][take],
                              data['ids'][take], y, device='cuda')
        refs[arm] = fit_reference(pairs[arm], data['x'][take], data['target'][take],
                                  data['valid'][take], data['ids'][take], y, device='cuda')
    state = dict(pairs=pairs, references=refs, fit_ids=ids[:20],
                 cache_sha256=CACHE_HASHES['observations.npz'], labels_horizon_ms=125.)
    torch.save(tensor_tree(state), output/'feedback.pt')
    return pairs, refs


def comparisons(predictions):
    result = {}
    for arm, records in predictions.items():
        rescued, harmed = [], []
        for old, new in zip(predictions['cold'], records, strict=True):
            if old['shot'] != new['shot']:
                raise AssertionError('paired shot order changed')
            old_good = old['timely'] if old['is_disrupt'] else not old['alarm']
            new_good = new['timely'] if new['is_disrupt'] else not new['alarm']
            if new_good and not old_good:
                rescued.append(old['shot'])
            if old_good and not new_good:
                harmed.append(old['shot'])
        result[arm] = dict(rescued=rescued, harmed=harmed)
    return result


def run(output):
    from .learning import predict
    if output.exists() or output.absolute().parent != io23.ROOT/'probe_outputs':
        raise ValueError('fresh direct worktree output required')
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError('at least1GiB free required; no automatic deletion')
    if sha(PREVIOUS/'complete.json') != COMPLETE_SHA:
        raise ValueError('fixed archived360 changed')
    previous = read(PREVIOUS/'protocol.json')
    ids, reference = previous['order'], Path(previous['reference'])
    if len(ids) != 360 or len(set(ids)) != 360:
        raise ValueError('fixed full360 order required')
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('cold parent reference changed: '+name)
    mapping, raw_hashes = read(reference/'input_map.json'), read(reference/'input_hashes.json')
    if mapping['calibration_ids'] != ids[:20]:
        raise ValueError('different cold20 mapping')
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
    output.mkdir()
    (output/'shots').mkdir()
    plan = io23.ROOT/'ARRIVAL_STREAM_PLAN_20261030.md'
    files = {Path(__file__), plan, io23.ROOT/'tests/test_arrival_stream_run.py',
             io23.ROOT/'tests/test_arrival_stream_learning.py'}
    files.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='DEVELOPMENT_COMPLETE360_ARRIVAL_REFERENCE_INTRA_ONLY',
        order=ids, arms=ALL_ARMS, candidate='real', warmup=20, qk_radius=.05,
        horizon=25, first_origin=127, update_cadence=25, feedback_fit='all_ended_cold20',
        source_sha256=PARENT_SHA256, threshold=threshold, reference=str(reference),
        reference_hashes=previous['reference_hashes'], previous=str(PREVIOUS),
        previous_complete_sha256=COMPLETE_SHA, cache=str(CACHE), cache_hashes=CACHE_HASHES,
        plan_sha256=sha(plan), reset_each_shot=True, reset_each_feedback=True,
        source_updates=0, inter_updates_after_cold20=0, alarm_updates=0,
        suffix_labels_after_all_predictions=True, startup_reused_from_sealed_archive=True,
        independent_confirmation=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(files)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    start, receipts = time.monotonic(), []
    gate = OutcomeGate(ids, io23.read_outcome)

    def deny_manifest(*args, **kwargs):
        raise AssertionError('class-grouped manifest unavailable before global seal')

    pairs = refs = None
    with patch.object(io23, 'read_outcome', side_effect=gate.read_outcome), \
            patch.object(fixed_test_protocol, 'load_test_contract', side_effect=deny_manifest):
        for i, sid in enumerate(ids):
            if sha(io23.path_for('EAST', sid)) != raw_hashes[str(sid)]:
                raise ValueError('raw input changed')
            name = f'{i+1:04d}_{sid}'
            old_path = PREVIOUS/'shots'/(name+'.npz')
            if sha(old_path) != read(PREVIOUS/'shots'/(name+'_seal.json'))['sha256']:
                raise ValueError('archived curve seal changed')
            with np.load(old_path, allow_pickle=False) as old:
                old_cold, times = old['cold'], old['times']
            if i < 20:
                curves = {a: old_cold.copy() for a in ALL_ARMS}
                info = []
            else:
                if pairs is None or refs is None:
                    raise AssertionError('feedback fit absent')
                raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
                if not np.array_equal(times, raw['times']):
                    raise ValueError('raw timeline changed')
                signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
                curves, info = predict(net, signal, slow, pairs['real'], pairs['misplaced'],
                                       refs['real'], refs['misplaced'])
            if set(curves) != set(ALL_ARMS):
                raise ValueError('all preset arms required')
            for values in curves.values():
                if (values.shape != times.shape or not np.isfinite(values).all()
                        or (values < 0).any() or (values > 1).any()):
                    raise ValueError('one finite probability per original timestamp required')
            np.testing.assert_array_equal(curves['cold'], curves['identical'])
            path = output/'shots'/(name+'.npz')
            with path.open('xb') as handle:
                np.savez_compressed(handle, times=times, archive=old_cold, **curves)
                handle.flush()
                os.fsync(handle.fileno())
            digest = sha(path)
            receipt = dict(shot=sid, index=i+1, sha256=digest, updates=info,
                           archive_max_abs_error=float(np.max(np.abs(curves['cold']-old_cold))),
                           archive_crossing_disagreements=int(np.sum((curves['cold'] >= threshold)
                                                                     != (old_cold >= threshold))),
                           startup_archived=i < 20, outcomes_used_in_prediction=False)
            _durable_json(output/'shots'/(name+'_seal.json'), receipt)
            gate.seal(sid, digest)
            receipts.append({k: v for k, v in receipt.items() if k != 'updates'})
            if i == 19:
                pairs, refs = fit_feedback(output, ids, gate)
                feedback_hash = sha(output/'feedback.pt')
            if (i+1) % 10 == 0:
                progress = dict(shots=i+1, total=360, seconds=time.monotonic()-start,
                                suffix_outcomes_read=0)
                _durable_json(output/'progress.json', progress)
                print(progress, flush=True)
    _durable_json(output/'all_predictions_sealed.json', dict(
        seals=gate.seals, count=len(gate.seals), feedback_sha256=feedback_hash))
    rows = {a: [] for a in (*ALL_ARMS, 'archive')}
    updates = {a: 0 for a in ARMS}
    applied_updates = {a: 0 for a in ARMS}
    for i, sid in enumerate(ids):
        name = f'{i+1:04d}_{sid}'
        path = output/'shots'/(name+'.npz')
        if sha(path) != gate.seals[i]['sha256']:
            raise ValueError('sealed prediction modified before evaluation')
        with np.load(path, allow_pickle=False) as saved:
            flag, down = gate.read_outcome('EAST', sid)
            _, valid = io23.labels_at(saved['times'], flag, down, 125.)
            for a in rows:
                rows[a].append(dict(shot=sid, times=saved['times'][valid], scores=saved[a][valid],
                                    is_disrupt=flag, down=down))
        # Receipt key layout is part of the streaming core contract.
        info = read(output/'shots'/(name+'_seal.json'))['updates']
        for receipt in info:
            for a in ARMS:
                updates[a] += int(receipt['state_norm'][a] > 0)
                applied_updates[a] += int(receipt['state_norm'][a] > 0 and receipt['apply_count'] > 0)
    predictions = {a: fixed_test_protocol.score_fixed_test('EAST', r, threshold)['predictions']
                   for a, r in rows.items()}
    results = {a: {k: compact(v, threshold) for k, v in (
        ('full360', r), ('cold20', r[:20]), ('suffix340', r[20:]), ('high_proxy90', r[-90:]))}
        for a, r in rows.items()}
    for name, old in frozen.items():
        if not torch.equal(net.state_dict()[name].detach().cpu(), old):
            raise AssertionError('frozen source parameter changed: '+name)
    if not torch.equal(slow, slow_before) or sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('fixed parent changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('experiment code changed during run: '+name)
    for name, digest in protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise AssertionError('parent artifact changed')
    if sha(output/'feedback.pt') != feedback_hash:
        raise AssertionError('feedback component changed during suffix')
    candidate, baseline = results['real']['full360'], results['cold']['full360']
    completed = dict(complete=True, results=results, comparisons=comparisons(predictions),
        seconds=time.monotonic()-start, update_counts=updates, applied_update_counts=applied_updates,
        operational_intra_gain=bool(candidate['tpr'] > baseline['tpr'] and candidate['fpr'] <= .20),
        fpr_cap_status={str(c): candidate['fpr'] <= c for c in (.1, .15, .2)},
        J_gain=bool(candidate['J'] > baseline['J']), source_unchanged=True,
        fixed_cold_unchanged=True, source_parameter_updates=0,
        scientific_goal_achieved=False, independent_confirmation=False,
        archive_max_abs_error=max(r['archive_max_abs_error'] for r in receipts),
        archive_crossing_disagreements=sum(r['archive_crossing_disagreements'] for r in receipts))
    _durable_json(output/'per_shot.json', predictions)
    _durable_json(output/'receipts.json', receipts)
    _durable_json(output/'accesses.json', gate.accesses)
    _durable_json(output/'complete.json', completed)
    print({a: r['full360'] for a, r in results.items()}, flush=True)
    print(completed['comparisons'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.absolute())
