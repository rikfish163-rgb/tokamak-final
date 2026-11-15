"""Fixed-cold20 conditional-density and temporary neural-step diagnostics."""

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
from ..conditional_forecast_20261023.learning import direct_targets, misplaced_labels
from ..delayed_recurrent_memory_20261025.learning import batch
from ..qk_matched_head_20261029.run import snapshot
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform

PREVIOUS = io23.ROOT / 'probe_outputs/local_factorial_east360_v1'
COMPLETE_SHA = '8898224f755c1129411de2942789f32c0a9e979edc7f418f20b92593a23d56e2'
ARMS = ('real', 'misplaced', 'identical')


def read(path):
    return json.loads(Path(path).read_text())


def folds(order):
    if len(order) != 20 or len(set(order)) != 20:
        raise ValueError('exactly twenty distinct cold shots required')
    return [dict(fold=f, fit=[s for i, s in enumerate(order) if i % 5 != f],
                 held=[s for i, s in enumerate(order) if i % 5 == f]) for f in range(5)]


@torch.no_grad()
def encode_features(net, signal, ends, slow):
    if set(signal) != {'values', 'available'} or not len(ends) or min(ends) < 127:
        raise ValueError('complete signal-only past windows required')
    features, logits = [], []
    for lo in range(0, len(ends), 32):
        x, m = batch(signal, ends[lo:lo+32], net.base.center.device)
        captured = []
        handle = net.base.forecast_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            output = net(x, m, x.new_zeros(len(x), 8192), slow, 'qk')
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError('one actual forecast readout expected')
        features.append(captured[0].reshape(len(x), 17, 64)[:, -1].cpu().numpy())
        logits.append(output['logit'].cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)


def signal_windows(net, signal, ends):
    target, mask = direct_targets(signal, ends, net.base.center.cpu().numpy(),
                                  net.base.scale.cpu().numpy(), 25)
    mask = mask.all(1)
    mask[:, 10] = False  # Uncertified mode-number extraction, not inferred units.
    return target, mask


def training_rows(data, outcomes):
    take = np.zeros(len(data['ids']), dtype=bool)
    labels = np.zeros(len(take), dtype=np.int64)
    for sid, (flag, down) in outcomes.items():
        selected = data['ids'] == sid
        y, valid = io23.labels_at(data['origin_time'][selected], flag, down, 125.)
        _, arrived_valid = io23.labels_at(data['arrival_time'][selected], flag, down, 125.)
        take[selected] = valid & arrived_valid
        labels[selected] = y.astype(np.int64)
    return take, labels[take]


def summarize(rows):
    result = {}
    for arm in sorted({r['arm'] for r in rows}):
        result[arm] = {}
        for group in sorted({r['group'] for r in rows if r['arm'] == arm}):
            values = [r for r in rows if r['arm'] == arm and r['group'] == group]
            delta = np.asarray([r['delta_bce'] for r in values])
            result[arm][group] = dict(shots=len(values), windows=sum(r['windows'] for r in values),
                mean_delta_bce=float(delta.mean()), median_delta_bce=float(np.median(delta)),
                improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()),
                unchanged=int((delta == 0).sum()))
    return result


def direction_gate(result):
    gate, reasons = True, []
    for group in ('safe', 'danger_timely'):
        real = result.get('real', {}).get(group)
        control = result.get('misplaced', {}).get(group)
        ok = (real is not None and control is not None and real['shots'] >= 2
              and real['mean_delta_bce'] < 0 and 2*real['improved'] >= real['shots']
              and real['mean_delta_bce'] < control['mean_delta_bce'])
        gate &= ok
        if not ok:
            reasons.append(group+' did not pass paired direction screen')
    return bool(gate), reasons


def evaluation_labels(times, flag, down):
    # The archived safe-shot label routine accepts a one-dimensional timeline.
    # Flatten/restore here, preserving its established labels and validity.
    times = np.asarray(times)
    y, valid = io23.labels_at(times.reshape(-1), flag, down, 125.)
    return y.reshape(times.shape), valid.reshape(times.shape)


def evaluate_neural(output, fold_specs):
    rows, exclusions, counts = [], [], {a: 0 for a in ARMS}
    for spec in fold_specs:
        directory = output / f"fold{spec['fold']}"
        if sha(directory/'neural.npz') != read(directory/'neural_seal.json')['sha256']:
            raise ValueError('neural predictions changed after sealing')
        with np.load(directory/'neural.npz', allow_pickle=False) as archive:
            data = dict(archive)
        for arm in ARMS:
            counts[arm] += int(np.count_nonzero(data[arm+'_state_norm']))
        for sid in spec['held']:
            flag, down = io23.read_outcome('EAST', sid)
            selected = data['ids'] == sid
            times = data['future_times'][selected]
            y, valid = evaluation_labels(times, flag, down)
            masks = {'safe': valid} if not flag else {
                'danger_timely': valid & (y == 1) & (np.round((down-times)*1000, 9) > 30),
                'early_normal': valid & (y == 0),
                'danger_late': valid & (y == 1) & (np.round((down-times)*1000, 9) <= 30),
            }
            before = data['baseline_logits'][selected].astype(np.float64)
            loss = np.logaddexp(0, before)-y*before
            for arm in ARMS:
                after = data[arm+'_logits'][selected].astype(np.float64)
                if arm == 'identical' and not np.array_equal(after, before):
                    raise AssertionError('uninformative control changed frozen predictions')
                after_loss = np.logaddexp(0, after)-y*after
                for group, mask in masks.items():
                    if not mask.any():
                        exclusions.append(dict(shot=sid, arm=arm, group=group, reason='no eligible future point'))
                        continue
                    rows.append(dict(shot=sid, fold=spec['fold'], group=group, arm=arm,
                        windows=int(mask.any(1).sum()), timepoints=int(mask.sum()),
                        before_bce=float(loss[mask].mean()), after_bce=float(after_loss[mask].mean()),
                        delta_bce=float((after_loss-loss)[mask].mean()),
                        mean_logit_change=float((after-before)[mask].mean()),
                        max_abs_logit_change=float(np.abs(after-before)[mask].max())))
    groups = summarize(rows)
    gate, reasons = direction_gate(groups)
    _durable_json(output/'neural_per_shot.json', rows)
    _durable_json(output/'neural_exclusions.json', exclusions)
    return dict(groups=groups, eligible_for_full_stream=gate, reasons=reasons,
                actual_nonzero_episode_updates=counts,
                warning='Independent one-step prefix episodes, not a continuous stream or alarm TPR/FPR.')


def evaluate(output, fold_specs):
    rows, exclusions, outcomes, accesses = [], [], {}, []
    # None of the held labels below was supplied to that fold's fit/scoring.
    for spec in fold_specs:
        path = output / f"fold{spec['fold']}" / 'sealed.npz'
        seal = read(path.with_suffix('.json'))
        if sha(path) != seal['sha256']:
            raise ValueError('held scores changed after sealing')
        with np.load(path, allow_pickle=False) as archive:
            data = dict(archive)
        for sid in spec['held']:
            flag, down = io23.read_outcome('EAST', sid)
            accesses.append(dict(fold=spec['fold'], shot=sid, role='evaluation_after_all_seals'))
            outcomes[str(sid)] = dict(disruptive=flag, down=down)
            selected = data['ids'] == sid
            z = data['logit'][selected].astype(np.float64)
            times = data['origin_time'][selected]
            arrivals = data['arrival_time'][selected]
            y, valid = io23.labels_at(times, flag, down, 125.)
            _, valid_arrival = io23.labels_at(arrivals, flag, down, 125.)
            valid &= valid_arrival
            masks = {'safe': valid} if not flag else {
                'danger_timely': valid & (y == 1) & (np.round((down-arrivals)*1000, 9) > 30),
                'early_normal': valid & (y == 0),
                'danger_late': valid & (y == 1) & (np.round((down-arrivals)*1000, 9) <= 30),
            }
            for arm in ARMS:
                logs = data[arm+'_logs'][selected]
                counts = data[arm+'_counts'][selected]
                evidence = logs[:, 1] - logs[:, 0]
                after = z + evidence
                # Stable BCE from logits; no clipping/temperature or probability floor.
                before_loss = np.logaddexp(0, z) - y*z
                after_loss = np.logaddexp(0, after) - y*after
                for group, mask in masks.items():
                    eligible = mask & (counts > 0)
                    if not eligible.any():
                        exclusions.append(dict(shot=sid, fold=spec['fold'], arm=arm,
                            group=group, reason='no eligible observed target', windows=int(mask.sum())))
                        continue
                    delta = after_loss[eligible]-before_loss[eligible]
                    rows.append(dict(shot=sid, fold=spec['fold'], arm=arm, group=group,
                        windows=int(eligible.sum()), delta_bce=float(delta.mean()),
                        before_bce=float(before_loss[eligible].mean()), after_bce=float(after_loss[eligible].mean()),
                        mean_log_bayes_factor=float(evidence[eligible].mean()),
                        median_log_bayes_factor=float(np.median(evidence[eligible])),
                        max_abs_log_bayes_factor=float(np.abs(evidence[eligible]).max()),
                        posterior_extreme_fraction=float((np.abs(after[eligible]) > np.log(999.)).mean()),
                        direction_correct_fraction=float(((2*y[eligible]-1)*evidence[eligible] > 0).mean()),
                        observed_channels_min=int(counts[eligible].min())))
            exclusions.append(dict(shot=sid, fold=spec['fold'], reason='postterminal target',
                                   windows=int((~valid).sum())))
    result = summarize(rows)
    gate, reasons = direction_gate(result)
    for group in result.get('identical', {}).values():
        if group['mean_delta_bce'] != 0:
            raise AssertionError('identical classes must supply no risk evidence')
    _durable_json(output/'per_shot.json', rows)
    _durable_json(output/'exclusions.json', exclusions)
    _durable_json(output/'outcomes.json', outcomes)
    _durable_json(output/'evaluation_access.json', accesses)
    return dict(groups=result, posterior_direction_screen=bool(gate), reasons=reasons,
                warning='Posterior direction diagnostic only; not updated risk or TPR/FPR.')


def tensor_tree(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    return value


def density_fitter(mode):
    if mode == 'fitted_residual':
        from .density import fit_density
        return fit_density
    if mode == 'leave_shot_out':
        from .crossfit import fit_crossfit_density
        return fit_crossfit_density
    raise ValueError('unknown density scale mode: '+str(mode))


def run(output, density_scale='fitted_residual'):
    from .density import log_densities
    from .neural import finite_step
    fit_density = density_fitter(density_scale)
    if output.exists() or output.absolute().parent != io23.ROOT/'probe_outputs':
        raise ValueError('fresh worktree output required')
    if sha(PREVIOUS/'complete.json') != COMPLETE_SHA:
        raise ValueError('preceding immutable experiment changed')
    prior = read(PREVIOUS/'protocol.json')
    reference = io23.ROOT/prior['reference']
    for name, digest in prior['reference_hashes'].items():
        if sha(reference/name) != digest:
            raise ValueError('fixed parent artifact changed: '+name)
    order = prior['order'][:20]
    specifications = folds(order)
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('rl_env local CUDA required')
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = snapshot(net)
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping = read(reference/'input_map.json')
    output.mkdir()
    start = time.monotonic()
    paths = {Path(__file__), Path(__file__).with_name('density.py'), Path(__file__).with_name('neural.py')}
    paths.update(Path(m.__file__) for n, m in list(sys.modules.items())
                 if n.startswith('experiments.risk_aligned_online.') and getattr(m, '__file__', None))
    tests = [io23.ROOT/'tests'/name for name in (
        'test_conditional_likelihood_run.py', 'test_conditional_likelihood_density.py',
        'test_conditional_likelihood_neural.py')]
    paths.update(tests)
    plan = io23.ROOT/'CONDITIONAL_LIKELIHOOD_PLAN.md'
    if density_scale == 'leave_shot_out':
        paths.add(io23.ROOT/'tests/test_conditional_likelihood_crossfit.py')
        plan = io23.ROOT/'CONDITIONAL_CROSSFIT_PLAN_20261030.md'
    paths.add(plan)
    protocol = dict(scope='COLD20_CONDITIONAL_COMPONENT_DIAGNOSTIC', folds=specifications,
        source_sha256=PARENT_SHA256, threshold=parent['threshold'], first_origin=127,
        stride=25, horizon=25, reference=str(reference), reference_hashes=prior['reference_hashes'],
        arms=ARMS, df=4, ridge=.1, covariance_shrinkage=.5, covariance_ridge=1e-6,
        plan_sha256=sha(plan), density_scale=density_scale,
        code_sha256={str(p.relative_to(io23.ROOT)): sha(p) for p in sorted(paths)},
        note='Fixed parent/map used all20; feedback component held-shot only, not clean transfer CV.',
        outcome_timing='fit labels allowed per fold; held evaluation only after all score seals',
        later340_access=False, independent_confirmation=False, source_risk_updates=0,
        episode_qk_radius=.05, episode_reset=True, feedback_maturity=25, next_prediction_length=25,
        torch=torch.__version__, gpu=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json', protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(io23.ROOT)))
    records, receipts, hashes, signals, times_by_shot = [], [], {}, {}, {}
    signal_reader, outcome_reader = io23.read_signals, io23.read_outcome

    def guarded_signals(device, sid, *args, **kwargs):
        if device != 'EAST' or sid not in order:
            raise AssertionError('only declared EAST cold20 signals allowed')
        return signal_reader(device, sid, *args, **kwargs)

    def deny_outcome(*args, **kwargs):
        raise AssertionError('no current outcomes while encoding/forecast scoring')

    with patch.object(io23, 'read_signals', side_effect=guarded_signals), \
            patch.object(io23, 'read_outcome', side_effect=deny_outcome):
        for sid in order:
            raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
            signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
            signals[sid], times_by_shot[sid] = signal, raw['times']
            ends = np.arange(127, len(raw['times'])-25, 25)
            if not len(ends):
                raise ValueError(f'no complete windows for required cold shot {sid}')
            features, logits = encode_features(net, signal, ends, slow)
            targets, valid = signal_windows(net, signal, ends)
            records.append(dict(x=features, target=targets, valid=valid, logit=logits,
                ids=np.full(len(ends), sid, np.int64), ends=ends,
                origin_time=raw['times'][ends], arrival_time=raw['times'][ends+25]))
            receipts.append(dict(shot=sid, windows=len(ends), first_origin=int(ends[0]),
                last_origin=int(ends[-1]), observations=int(valid.sum())))
            hashes[str(sid)] = sha(io23.path_for('EAST', sid))
            print(json.dumps(dict(stage='collected', shot=sid, windows=len(ends))), flush=True)
    data = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    with (output/'observations.npz').open('xb') as handle:
        np.savez_compressed(handle, **data)
    _durable_json(output/'collection.json', dict(receipts=receipts, input_hashes=hashes,
        observations_sha256=sha(output/'observations.npz'), outcome_reads=0))
    accesses = []
    for spec in specifications:
        directory = output/f"fold{spec['fold']}"
        directory.mkdir()
        outcomes = {}
        for sid in spec['fit']:
            outcomes[sid] = outcome_reader('EAST', sid)
            accesses.append(dict(fold=spec['fold'], shot=sid, role='fit'))
        take, labels = training_rows(data, outcomes)
        held = np.isin(data['ids'], spec['held'])
        if set(data['ids'][take]) & set(data['ids'][held]):
            raise AssertionError('fit/held overlap')
        sealed = {k: data[k][held] for k in ('ids', 'ends', 'origin_time', 'arrival_time', 'logit')}
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            for arm, y in (('real', labels), ('misplaced', misplaced_labels(data['ids'][take], labels))):
                fit = fit_density(data['x'][take], data['target'][take], data['valid'][take],
                                  data['ids'][take], y, device='cuda')
                torch.save(tensor_tree(fit), directory/(arm+'_density.pt'))
                logs, counts = log_densities(fit, data['x'][held], data['target'][held], data['valid'][held])
                if logs.shape != (held.sum(), 2) or not np.isfinite(logs).all():
                    raise FloatingPointError('invalid held-out density')
                sealed[arm+'_logs'], sealed[arm+'_counts'] = logs, counts
                print(json.dumps(dict(stage='fit_scored', fold=spec['fold'], arm=arm,
                    fitting_windows=int(take.sum()), held_windows=int(held.sum()))), flush=True)
        sealed['identical_logs'] = np.repeat(sealed['real_logs'][:, :1], 2, axis=1)
        sealed['identical_counts'] = sealed['real_counts'].copy()
        with (directory/'sealed.npz').open('xb') as handle:
            np.savez_compressed(handle, **sealed)
        _durable_json(directory/'sealed.json', dict(sha256=sha(directory/'sealed.npz'),
            fit_ids=spec['fit'], held_ids=spec['held'], held_outcome_reads=0))
        neural_records, neural_excluded = [], []
        with patch.object(io23, 'read_outcome', side_effect=deny_outcome):
            for sid in spec['held']:
                belongs = sealed['ids'] == sid
                complete = sealed['ends']+50 < len(times_by_shot[sid])
                selected = belongs & complete
                neural_excluded.append(dict(shot=sid, reason='incomplete next25', windows=int((belongs & ~complete).sum())))
                if not selected.any():
                    continue  # Explicit missing episode; logged, never assigned a zero score.
                ends = sealed['ends'][selected]
                prediction = finite_step(net, signals[sid], ends, slow,
                    {arm: sealed[arm+'_logs'][selected] for arm in ARMS},
                    {arm: sealed[arm+'_counts'][selected] for arm in ARMS})
                neural_records.append(dict(ids=np.full(len(ends), sid, np.int64), ends=ends,
                    future_times=times_by_shot[sid][ends[:, None]+np.arange(26, 51)], **prediction))
                print(json.dumps(dict(stage='neural_scored', fold=spec['fold'], shot=sid,
                                      episodes=len(ends))), flush=True)
        if not neural_records:
            raise ValueError('no held one-step episode in fold')
        neural_data = {k: np.concatenate([v[k] for v in neural_records]) for k in neural_records[0]}
        with (directory/'neural.npz').open('xb') as handle:
            np.savez_compressed(handle, **neural_data)
        _durable_json(directory/'neural_seal.json', dict(sha256=sha(directory/'neural.npz'),
            fit_ids=spec['fit'], held_ids=spec['held'], held_outcome_reads=0, exclusions=neural_excluded))
    _durable_json(output/'training_access.json', accesses)
    result = dict(posterior=evaluate(output, specifications), neural=evaluate_neural(output, specifications))
    current = snapshot(net)
    if any(not torch.equal(current[k], v) for k, v in frozen.items()):
        raise AssertionError('frozen model changed')
    if sha(DEFAULT_PARENT) != PARENT_SHA256:
        raise AssertionError('source file changed')
    result.update(seconds=time.monotonic()-start, source_unchanged=True, source_parameter_updates=0,
        full_stream_run=False, raw_input_hashes_unchanged=all(
            sha(io23.path_for('EAST', int(sid))) == digest for sid, digest in hashes.items()))
    _durable_json(output/'completed.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--density-scale', choices=('fitted_residual', 'leave_shot_out'), default='fitted_residual')
    args = parser.parse_args()
    run(args.output.absolute(), args.density_scale)
