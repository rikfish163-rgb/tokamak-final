"""H25 arrival-information screen with actual temporary attention updates.

Auxiliary logits produce feedback potentials, NOT deployed risk predictions
or calibrated likelihood ratios. Independent episodes are not a full stream.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_forecast_20261023.learning import misplaced_labels
from ..conditional_likelihood_20261029.neural import finite_step
from ..conditional_likelihood_20261029.run import (
    evaluation_labels, read, summarize, tensor_tree, training_rows,
)
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

CACHE = io23.ROOT/'probe_outputs/conditional_likelihood_cold20_v2'
CACHE_HASHES = {
    'observations.npz': '8e21a7e6a7716ec00578b0ed8232a6c232de1bfca97b57d477ca417da15c4e1d',
    'protocol.json': 'ef80c02530abc44bb4ff245af8fb6f8a3b6731b8d33a0e2bc43abee28779d8e7',
    'completed.json': '69bcbe17f9ea004bc5165607fa1806e1bb9c57c992955e7f9cb4fe1c769909e6',
}
ARMS = ('real', 'paired', 'constant', 'reverse', 'misplaced', 'identical')


def require_same_parent(prediction, reference):
    # The original cache used batch32; finite_step used batch8 and produced
    # up to 4.77e-6 differences already in the archived v2 experiment. Compare
    # against that SAME batch8 reference exactly, rather than relaxing a bound.
    for name in ('origin_logit', 'baseline_logits'):
        if not np.array_equal(prediction[name], reference[name]):
            raise AssertionError('matched frozen parent changed: '+name)


def neural_scores(real, reversed_arrival, misplaced):
    """Map estimated feedback increments to two unnormalized evidence potentials."""
    values = dict(real=real['arrival_increment'], paired=real['paired_increment'],
                  constant=real['constant_logit']-real['q0_logit'],
                  reverse=reversed_arrival['arrival_increment'],
                  misplaced=misplaced['arrival_increment'],
                  identical=np.zeros_like(real['arrival_increment']))
    return {arm: np.column_stack((np.zeros_like(v), v)) for arm, v in values.items()}


def evaluate_shot(data, sid, fold, flag, down):
    selected = data['ids'] == sid
    times = data['future_times'][selected]
    y, valid = evaluation_labels(times, flag, down)
    before = data['baseline_logits'][selected].astype(float)
    if flag:
        timely = valid & (y == 1) & (np.round((down-times)*1000, 9) > 30)
        origins = data['origin_time'][selected]
        arrivals = data['arrival_time'][selected]
        origin_in_window = (np.round((down-origins)*1000, 9) <= 125) & (origins <= down)
        arrival_timely = np.round((down-arrivals)*1000, 9) > 30
        masks = dict(danger_timely=timely,
                     danger_origin_intersection=timely & (origin_in_window & arrival_timely)[:, None],
                     early_normal=valid & (y == 0),
                     danger_late=valid & (y == 1) & ~timely)
    else:
        masks = {'safe': valid}
    before_loss = np.logaddexp(0, before)-y*before
    rows = []
    for arm in ARMS:
        after = data[arm+'_logits'][selected].astype(float)
        if after.shape != before.shape or not np.isfinite(after).all():
            raise ValueError('finite paired future neural predictions required')
        if arm == 'identical' and not np.array_equal(after, before):
            raise AssertionError('uninformative control changed predictions')
        after_loss = np.logaddexp(0, after)-y*after
        for group, use in masks.items():
            if not use.any():
                continue  # This group is absent, not a zero-loss observation.
            rows.append(dict(shot=int(sid), fold=int(fold), group=group, arm=arm,
                             windows=int(use.any(1).sum()), timepoints=int(use.sum()),
                             before_bce=float(before_loss[use].mean()),
                             after_bce=float(after_loss[use].mean()),
                             delta_bce=float((after_loss-before_loss)[use].mean()),
                             mean_logit_change=float((after-before)[use].mean())))
    return rows


def gate(groups):
    reasons = []
    for name in ('safe', 'danger_timely', 'danger_origin_intersection'):
        real = groups.get('real', {}).get(name)
        if not (real and real['shots'] >= 2 and real['mean_delta_bce'] < 0
                and 2*real['improved'] >= real['shots']):
            reasons.append(name+': real did not improve both mean and shot count')
            continue
        for control in ('reverse', 'misplaced', 'constant'):
            other = groups.get(control, {}).get(name)
            if not (other and other['shots'] == real['shots']
                    and real['mean_delta_bce'] < other['mean_delta_bce']):
                reasons.append(name+': real did not exceed '+control)
    early = groups.get('real', {}).get('early_normal')
    if not early or early['mean_delta_bce'] > 0:
        reasons.append('early_normal: absent or mean loss worsened')
    return not reasons, reasons


def run(output):
    from .learning import fit_pair, score_pair
    if output.exists() or output.absolute().parent != io23.ROOT/'probe_outputs':
        raise ValueError('fresh output directly under worktree probe_outputs required')
    for name, digest in CACHE_HASHES.items():
        if sha(CACHE/name) != digest:
            raise ValueError('frozen observation cache changed: '+name)
    previous = read(CACHE/'protocol.json')
    reference = Path(previous['reference'])
    for name, digest in previous['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent/map changed: '+name)
    specs = previous['folds']
    order = sorted({s for spec in specs for s in spec['held']})
    with np.load(CACHE/'observations.npz', allow_pickle=False) as archive:
        data = dict(archive)
    if set(data['ids']) != set(order) or len(order) != 20:
        raise ValueError('exact original cold20 required')
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env CUDA required')
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = snapshot(net)
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = read(reference/'input_map.json')
    output.mkdir()
    plan = io23.ROOT/'ARRIVAL_INCREMENT_PLAN_20261030.md'
    paths = {Path(__file__), Path(__file__).with_name('learning.py'),
             Path(__file__).with_name('verify.py'), plan,
             io23.ROOT/'tests/test_arrival_increment_run.py',
             io23.ROOT/'tests/test_arrival_increment_learning.py'}
    paths.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    protocol = dict(scope='COLD20_ARRIVAL_INFORMATION_COMPONENT_DIAGNOSTIC',
                    folds=specs, cache=str(CACHE), cache_hashes=CACHE_HASHES,
                    reference=str(reference), reference_hashes=previous['reference_hashes'],
                    source_sha256=PARENT_SHA256, threshold=parent['threshold'],
                    arms=ARMS, l2=.1, horizon=25, next_prediction_offsets=[26, 50],
                    qk_radius=.05, episode_reset=True, timing_contract='see pre-result plan',
                    plan_sha256=sha(plan), source_parameter_updates=0, later340_access=False,
                    calibrated_likelihood_ratio=False, independent_confirmation=False,
                    gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(paths)})
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    start = time.monotonic()
    signals, timelines, hashes = {}, {}, {}
    signal_reader, outcome_reader = io23.read_signals, io23.read_outcome

    def guarded_signals(device, sid, *args, **kwargs):
        if device != 'EAST' or sid not in order:
            raise AssertionError('only cold20 signals permitted')
        return signal_reader(device, sid, *args, **kwargs)

    def deny_outcome(*args, **kwargs):
        raise AssertionError('no outcomes during observation scoring or temporary update')

    with patch.object(io23, 'read_signals', side_effect=guarded_signals), \
            patch.object(io23, 'read_outcome', side_effect=deny_outcome):
        for sid in order:
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            signals[sid] = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            timelines[sid] = raw['times']
            hashes[str(sid)] = sha(io23.path_for('EAST', sid))
    collection = read(CACHE/'collection.json')
    if hashes != collection['input_hashes']:
        raise ValueError('raw signals changed since observation cache')
    accesses = []
    for spec in specs:
        directory = output/f"fold{spec['fold']}"
        directory.mkdir()
        old_directory = CACHE/f"fold{spec['fold']}"
        if sha(old_directory/'neural.npz') != read(old_directory/'neural_seal.json')['sha256']:
            raise ValueError('matched neural reference changed')
        with np.load(old_directory/'neural.npz', allow_pickle=False) as archive:
            old_neural = dict(archive)
        outcomes = {sid: outcome_reader('EAST', sid) for sid in spec['fit']}
        accesses.extend(dict(fold=spec['fold'], shot=sid, role='fit') for sid in spec['fit'])
        take, labels = training_rows(data, outcomes)
        held = np.isin(data['ids'], spec['held'])
        if set(data['ids'][take]) & set(data['ids'][held]):
            raise AssertionError('fit/held overlap')
        sealed = {k: data[k][held] for k in ('ids', 'ends', 'origin_time', 'arrival_time', 'logit')}
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            fitted = {}
            for name, y in (('real', labels), ('misplaced', misplaced_labels(data['ids'][take], labels))):
                fitted[name] = fit_pair(data['x'][take], data['target'][take], data['valid'][take],
                                        data['ids'][take], y, device='cuda')
                torch.save(tensor_tree(fitted[name]), directory/(name+'_pair.pt'))
            arguments = (data['x'][held], data['target'][held], data['valid'][held])
            real = score_pair(fitted['real'], *arguments)
            reverse = score_pair(fitted['real'], arguments[0], arguments[1][:, ::-1].copy(), arguments[2])
            misplaced = score_pair(fitted['misplaced'], *arguments)
            scores = neural_scores(real, reverse, misplaced)
            count = data['valid'][held].sum(1)
            for arm in ARMS:
                scores[arm][count == 0] = 0.
                sealed[arm+'_potentials'] = scores[arm]
            sealed['observed_channels'] = count
            for name, value in real.items():
                sealed['probe_'+name] = value
        with (directory/'sealed.npz').open('xb') as handle:
            np.savez_compressed(handle, **sealed)
        _durable_json(directory/'sealed.json', dict(sha256=sha(directory/'sealed.npz'),
                      fit_ids=spec['fit'], held_ids=spec['held'], held_outcome_reads=0))
        records, exclusions = [], []
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            for sid in spec['held']:
                belongs = sealed['ids'] == sid
                complete = sealed['ends']+50 < len(timelines[sid])
                selected = belongs & complete
                exclusions.append(dict(shot=sid, incomplete_future=int((belongs & ~complete).sum())))
                if not selected.any():
                    continue  # Missing complete future episode is explicitly logged.
                ends = sealed['ends'][selected]
                prediction = finite_step(net, signals[sid], ends, slow,
                    {a: sealed[a+'_potentials'][selected] for a in ARMS},
                    {a: count[selected] for a in ARMS})
                old_selected = old_neural['ids'] == sid
                if not np.array_equal(ends, old_neural['ends'][old_selected]):
                    raise AssertionError('matched episode origins changed')
                require_same_parent(prediction, {k: old_neural[k][old_selected]
                                                for k in ('origin_logit', 'baseline_logits')})
                records.append(dict(ids=np.full(len(ends), sid, np.int64), ends=ends,
                    origin_time=sealed['origin_time'][selected], arrival_time=sealed['arrival_time'][selected],
                    future_times=timelines[sid][ends[:, None]+np.arange(26, 51)], **prediction))
                print(json.dumps(dict(stage='neural_scored', fold=spec['fold'], shot=sid,
                                      episodes=len(ends))), flush=True)
        if not records:
            raise ValueError('no complete held episodes')
        neural = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
        with (directory/'neural.npz').open('xb') as handle:
            np.savez_compressed(handle, **neural)
        _durable_json(directory/'neural_seal.json', dict(sha256=sha(directory/'neural.npz'),
                      held_outcome_reads=0, exclusions=exclusions))
    _durable_json(output/'training_access.json', accesses)
    rows, counts = [], {a: 0 for a in ARMS}
    for spec in specs:
        directory = output/f"fold{spec['fold']}"
        if sha(directory/'neural.npz') != read(directory/'neural_seal.json')['sha256']:
            raise ValueError('neural predictions changed after sealing')
        with np.load(directory/'neural.npz', allow_pickle=False) as archive:
            neural = dict(archive)
        for arm in ARMS:
            counts[arm] += int(np.count_nonzero(neural[arm+'_state_norm']))
        for sid in spec['held']:
            flag, down = outcome_reader('EAST', sid)
            accesses.append(dict(fold=spec['fold'], shot=sid, role='evaluate_after_all_seals'))
            rows.extend(evaluate_shot(neural, sid, spec['fold'], flag, down))
    groups = summarize(rows)
    passed, reasons = gate(groups)
    current = snapshot(net)
    if any(not torch.equal(v, current[k]) for k, v in frozen.items()):
        raise AssertionError('frozen model changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(
            sha(io23.path_for('EAST', int(sid))) != digest for sid, digest in hashes.items()):
        raise AssertionError('source or raw data changed')
    _durable_json(output/'per_shot.json', rows)
    _durable_json(output/'accesses.json', accesses)
    result = dict(groups=groups, direction_screen_passed=passed, reasons=reasons,
                  nonzero_episode_updates=counts, seconds=time.monotonic()-start,
                  source_unchanged=True, source_parameter_updates=0, full_stream_run=False,
                  raw_hashes_unchanged=True, calibrated_likelihood_ratio=False,
                  warning='Independent cold20 Q/K episodes, not continuous online alarms or independent validation.')
    _durable_json(output/'completed.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.absolute())
