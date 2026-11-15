"""Fixed-parent temporary Q/K updates relative to a negative-stage reference."""

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
from ..arrival_increment_20261030.learning import score_pair, summarize_arrival
from ..arrival_increment_20261030.run import CACHE, CACHE_HASHES, require_same_parent
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_forecast_20261023.learning import misplaced_labels
from ..conditional_likelihood_20261029.neural import finite_step
from ..conditional_likelihood_20261029.run import evaluation_labels, read, summarize, tensor_tree, training_rows
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

PRIOR = io23.ROOT/'probe_outputs/arrival_increment_cold20_v2'
PRIOR_SHA = '0ccd89978ff2535e0181cc4a5dec914edd9203f48b9247c319fc2201f1397753'
ARMS = ('real', 'mean', 'zero', 'reverse', 'misplaced', 'reference_only', 'paired', 'identical')


def masks_for(data, sid, flag, down):
    selected = data['ids'] == sid
    times = data['future_times'][selected]
    y, valid = evaluation_labels(times, flag, down)
    if not flag:
        return y, {'safe': valid}
    timely = valid & (y == 1) & (np.round((down-times)*1000, 9) > 30)
    origins, arrivals = data['origin_time'][selected], data['arrival_time'][selected]
    intersect = ((np.round((down-origins)*1000, 9) <= 125) & (origins <= down)
                 & (np.round((down-arrivals)*1000, 9) > 30))
    return y, dict(danger_timely=timely, danger_origin_intersection=timely & intersect[:, None],
                   early_normal=valid & (y == 0), danger_late=valid & (y == 1) & ~timely)


def score_shot(data, sid, fold, flag, down):
    y, masks = masks_for(data, sid, flag, down)
    selected = data['ids'] == sid
    before = data['baseline_logits'][selected].astype(float)
    loss0 = np.logaddexp(0, before)-y*before
    rows = []
    for arm in ARMS:
        after = data[arm+'_logits'][selected].astype(float)
        if after.shape != before.shape or not np.isfinite(after).all():
            raise ValueError('paired finite neural scores required')
        if arm == 'identical' and not np.array_equal(after, before):
            raise AssertionError('zero feedback changed model prediction')
        loss1 = np.logaddexp(0, after)-y*after
        for group, mask in masks.items():
            if not mask.any():
                continue  # No eligible point is an absent group, not zero loss.
            rows.append(dict(shot=int(sid), fold=int(fold), arm=arm, group=group,
                             windows=int(mask.any(1).sum()), timepoints=int(mask.sum()),
                             before_bce=float(loss0[mask].mean()), after_bce=float(loss1[mask].mean()),
                             delta_bce=float((loss1-loss0)[mask].mean()),
                             mean_logit_change=float((after-before)[mask].mean())))
    return rows


def gate(groups):
    reasons = []
    for name in ('safe', 'danger_timely', 'danger_origin_intersection'):
        real = groups.get('real', {}).get(name)
        if not (real and real['shots'] >= 2 and real['mean_delta_bce'] < 0
                and 2*real['improved'] >= real['shots']):
            reasons.append(name+': real did not improve mean and shot count')
            continue
        for control in ('mean', 'reverse', 'misplaced', 'reference_only'):
            other = groups.get(control, {}).get(name)
            if not (other and other['shots'] == real['shots']
                    and real['mean_delta_bce'] < other['mean_delta_bce']):
                reasons.append(name+': real did not exceed '+control)
    early = groups.get('real', {}).get('early_normal')
    if not early or early['mean_delta_bce'] > 0:
        reasons.append('early_normal: absent or worsened')
    return not reasons, reasons


def make_scores(pair, wrong_pair, reference, wrong_reference, x, target, valid):
    from .learning import feedback, predict_reference
    supported = np.asarray(reference['supported']) & np.asarray(wrong_reference['supported'])
    mask = valid & supported
    context = predict_reference(reference, pair, x, mask, mode='context')
    mean = predict_reference(reference, pair, x, mask, mode='mean')
    wrong = predict_reference(wrong_reference, wrong_pair, x, mask, mode='context')
    e, counts = feedback(pair, target, mask, context, supported)
    values = dict(real=e, mean=feedback(pair, target, mask, mean, supported)[0],
                  zero=feedback(pair, target, mask, np.zeros_like(context), supported)[0],
                  reverse=feedback(pair, target[:, ::-1].copy(), mask, context, supported)[0],
                  misplaced=feedback(wrong_pair, target, mask, wrong, supported)[0],
                  reference_only=feedback(pair, np.zeros_like(target), mask, context, supported)[0],
                  paired=score_pair(pair, x, target, mask)['paired_increment'],
                  identical=np.zeros(len(x)))
    scores = {arm: np.column_stack((np.zeros(len(x)), value)) for arm, value in values.items()}
    return scores, counts, dict(context_reference=context, mean_reference=mean,
                               wrong_reference=wrong, common_mask=mask, common_supported=supported)


def run(output):
    from .learning import fit_reference
    if output.exists() or output.absolute().parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    if sha(PRIOR/'completed.json') != PRIOR_SHA:
        raise ValueError('previous diagnostic changed')
    for name, digest in CACHE_HASHES.items():
        if sha(CACHE/name) != digest:
            raise ValueError('observation cache changed')
    old_protocol = read(PRIOR/'protocol.json')
    reference = Path(old_protocol['reference'])
    for name, digest in old_protocol['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('parent reference changed')
    specs = old_protocol['folds']
    order = sorted({s for spec in specs for s in spec['held']})
    with np.load(CACHE/'observations.npz', allow_pickle=False) as archive:
        data = dict(archive)
    if len(order) != 20 or set(order) != set(data['ids']):
        raise ValueError('original twenty cold shots required')
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
    plan = io23.ROOT/'ARRIVAL_REFERENCE_PLAN_20261030.md'
    paths = {Path(__file__), Path(__file__).with_name('learning.py'), plan,
             io23.ROOT/'tests/test_arrival_reference_run.py', io23.ROOT/'tests/test_arrival_reference_learning.py'}
    paths.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    models = [PRIOR/f"fold{s['fold']}"/(arm+'_pair.pt') for s in specs for arm in ('real', 'misplaced')]
    protocol = dict(scope='COLD20_NEGATIVE_STAGE_REFERENCE_DIAGNOSTIC', folds=specs, arms=ARMS,
                    prior=str(PRIOR), prior_completed_sha256=PRIOR_SHA, cache=str(CACHE), cache_hashes=CACHE_HASHES,
                    fixed_pairs={str(p): sha(p) for p in models}, reference=str(reference),
                    reference_hashes=old_protocol['reference_hashes'], source_sha256=PARENT_SHA256,
                    threshold=parent['threshold'], ridge=.1, horizon=25, qk_radius=.05,
                    plan_sha256=sha(plan), code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(paths)},
                    prefix_conditioning_only=True, independent_confirmation=False, later340_access=False,
                    source_updates=0, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    start = time.monotonic()
    signal_reader, outcome_reader = io23.read_signals, io23.read_outcome

    def guarded_signals(device, sid, *args, **kwargs):
        if device != 'EAST' or sid not in order:
            raise AssertionError('only current cold20 signals permitted')
        return signal_reader(device, sid, *args, **kwargs)

    def deny_outcome(*args, **kwargs):
        raise AssertionError('labels cannot enter scoring or temporary updates')

    signals, times, hashes = {}, {}, {}
    with patch.object(io23, 'read_signals', side_effect=guarded_signals), \
            patch.object(io23, 'read_outcome', side_effect=deny_outcome):
        for sid in order:
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            signals[sid] = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            times[sid] = raw['times']
            hashes[str(sid)] = sha(io23.path_for('EAST', sid))
    if hashes != read(CACHE/'collection.json')['input_hashes']:
        raise AssertionError('raw observations changed')
    accesses = []
    for spec in specs:
        folder = output/f"fold{spec['fold']}"
        folder.mkdir()
        old_folder = PRIOR/f"fold{spec['fold']}"
        if sha(old_folder/'neural.npz') != read(old_folder/'neural_seal.json')['sha256']:
            raise ValueError('old neural baseline changed')
        with np.load(old_folder/'neural.npz', allow_pickle=False) as archive:
            old_neural = dict(archive)
        outcomes = {s: outcome_reader('EAST', s) for s in spec['fit']}
        accesses.extend(dict(fold=spec['fold'], shot=s, role='fit') for s in spec['fit'])
        take, labels = training_rows(data, outcomes)
        held = np.isin(data['ids'], spec['held'])
        assert not set(data['ids'][take]) & set(data['ids'][held])
        pairs = {a: torch.load(old_folder/(a+'_pair.pt'), map_location='cpu', weights_only=True)
                 for a in ('real', 'misplaced')}
        refs = {}
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            for arm, y in (('real', labels), ('misplaced', misplaced_labels(data['ids'][take], labels))):
                refs[arm] = fit_reference(pairs[arm], data['x'][take], data['target'][take],
                                         data['valid'][take], data['ids'][take], y, device='cuda')
                torch.save(tensor_tree(refs[arm]), folder/(arm+'_reference.pt'))
            scores, counts, info = make_scores(pairs['real'], pairs['misplaced'], refs['real'], refs['misplaced'],
                                               data['x'][held], data['target'][held], data['valid'][held])
        sealed = {k: data[k][held] for k in ('ids', 'ends', 'origin_time', 'arrival_time')}
        sealed.update(info, observed_channels=counts)
        sealed.update({a+'_potentials': scores[a] for a in ARMS})
        with (folder/'sealed.npz').open('xb') as handle:
            np.savez_compressed(handle, **sealed)
        _durable_json(folder/'sealed.json', dict(sha256=sha(folder/'sealed.npz'), held_outcome_reads=0))
        records, exclusions = [], []
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            for sid in spec['held']:
                belongs = sealed['ids'] == sid
                selected = belongs & (sealed['ends']+50 < len(times[sid]))
                exclusions.append(dict(shot=sid, incomplete_future=int((belongs & ~selected).sum())))
                if not selected.any():
                    continue  # Missing complete future interval, logged above.
                ends = sealed['ends'][selected]
                result = finite_step(net, signals[sid], ends, slow,
                                     {a: scores[a][selected] for a in ARMS}, {a: counts[selected] for a in ARMS})
                old_selected = old_neural['ids'] == sid
                np.testing.assert_array_equal(ends, old_neural['ends'][old_selected])
                require_same_parent(result, {k: old_neural[k][old_selected] for k in ('origin_logit', 'baseline_logits')})
                records.append(dict(ids=np.full(len(ends), sid, np.int64), ends=ends,
                    origin_time=sealed['origin_time'][selected], arrival_time=sealed['arrival_time'][selected],
                    future_times=times[sid][ends[:, None]+np.arange(26, 51)], **result))
                print(json.dumps(dict(stage='neural_scored', fold=spec['fold'], shot=sid, episodes=len(ends))), flush=True)
        if not records:
            raise ValueError('no complete held episode')
        neural = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
        with (folder/'neural.npz').open('xb') as handle:
            np.savez_compressed(handle, **neural)
        _durable_json(folder/'neural_seal.json', dict(sha256=sha(folder/'neural.npz'), exclusions=exclusions,
                                                   held_outcome_reads=0))
    rows, quality, updates = [], [], {a: 0 for a in ARMS}
    for spec in specs:
        folder = output/f"fold{spec['fold']}"
        if sha(folder/'neural.npz') != read(folder/'neural_seal.json')['sha256']:
            raise ValueError('prediction seal changed')
        with np.load(folder/'neural.npz', allow_pickle=False) as archive:
            neural = dict(archive)
        with np.load(folder/'sealed.npz', allow_pickle=False) as archive:
            sealed = dict(archive)
        pair = torch.load(PRIOR/f"fold{spec['fold']}"/'real_pair.pt', map_location='cpu', weights_only=True)
        held = np.isin(data['ids'], spec['held'])
        actual = summarize_arrival(data['target'][held], data['valid'][held]).reshape(-1, 23, 4)
        actual /= np.asarray(pair['arrival_scale']).reshape(23, 4)
        for arm in ARMS:
            updates[arm] += int(np.count_nonzero(neural[arm+'_state_norm']))
        for sid in spec['held']:
            flag, down = outcome_reader('EAST', sid)
            accesses.append(dict(fold=spec['fold'], shot=sid, role='evaluate_after_all_seals'))
            rows.extend(score_shot(neural, sid, spec['fold'], flag, down))
            select = sealed['ids'] == sid
            y, valid = io23.labels_at(sealed['origin_time'], flag, down, 125.)
            _, arrived = io23.labels_at(sealed['arrival_time'], flag, down, 125.)
            use = select & valid & arrived & (y == 0)
            mask = sealed['common_mask'] & use[:, None]
            n = mask.sum(1)
            eligible = n > 0
            if not eligible.any():
                continue  # No normal-label target; no fabricated zero error.
            item = dict(shot=sid, fold=spec['fold'], windows=int(eligible.sum()))
            for name in ('context', 'mean'):
                error = np.where(mask[:, :, None], (actual-sealed[name+'_reference'])**2, 0).sum((1, 2))
                item[name+'_mse'] = float(np.mean(error[eligible]/(4*n[eligible])))
            quality.append(item)
    groups = summarize(rows)
    passed, reasons = gate(groups)
    current = snapshot(net)
    assert all(torch.equal(current[k], v) for k, v in frozen.items())
    assert sha(DEFAULT_PARENT) == PARENT_SHA256
    assert all(sha(io23.path_for('EAST', int(s))) == h for s, h in hashes.items())
    assert all(sha(Path(p)) == h for p, h in protocol['fixed_pairs'].items())
    _durable_json(output/'per_shot.json', rows)
    _durable_json(output/'reference_quality.json', quality)
    _durable_json(output/'accesses.json', accesses)
    result = dict(groups=groups, direction_screen_passed=passed, reasons=reasons,
                  nonzero_episode_updates=updates, seconds=time.monotonic()-start,
                  source_unchanged=True, raw_hashes_unchanged=True, auxiliary_classifiers_unchanged=True,
                  full_stream_run=False, source_parameter_updates=0,
                  reference_quality=dict(shots=len(quality), context_mean_mse=float(np.mean([r['context_mse'] for r in quality])),
                      mean_reference_mse=float(np.mean([r['mean_mse'] for r in quality])),
                      improved_shots=sum(r['context_mse'] < r['mean_mse'] for r in quality)))
    _durable_json(output/'completed.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.absolute())
