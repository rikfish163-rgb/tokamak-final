"""Post-hoc source fit and causal-feedback information diagnostics, not a policy."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import warnings

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha

GROUPS = ('safe', 'danger', 'early')


def read(path):
    return json.loads(path.read_text())


def balanced_shot_weights(records):
    """Unit total weight: equal binary groups, then equal shots within group."""
    counts = defaultdict(int)
    for r in records:
        if r['group'] not in ('safe', 'danger'):
            raise ValueError('only safe and danger are supervised probe groups')
        counts[r['group'], r['shot']] += 1
    shots = {g: len({sid for kind, sid in counts if kind == g}) for g in ('safe', 'danger')}
    if not all(shots.values()):
        raise ValueError('both safe and danger groups required')
    return np.array([.5/shots[r['group']]/counts[r['group'], r['shot']] for r in records])


def standardize(train, held, weights):
    train, held, weights = (np.asarray(v, dtype=np.float64) for v in (train, held, weights))
    if (train.ndim != 2 or held.ndim != 2 or train.shape[1] != held.shape[1]
            or weights.shape != (len(train),) or not np.all(weights > 0)
            or not all(np.isfinite(v).all() for v in (train, held, weights))):
        raise ValueError('finite matching features and positive training weights required')
    center = np.average(train, axis=0, weights=weights)
    variance = np.average((train-center)**2, axis=0, weights=weights)
    scale = np.where(variance > 0, np.sqrt(variance), 1.)
    return (train-center)/scale, (held-center)/scale, center, scale


def feedback_controls(context, mean):
    context, mean = np.asarray(context), np.asarray(mean)
    if context.ndim != 2 or context.shape[1] != 358 or mean.shape != (358,):
        raise ValueError('358 dimensional context required')
    reverse, average = context.copy(), context.copy()
    reverse[:, 128:] = context[:, 128:].reshape(-1, 5, 46)[:, ::-1].reshape(-1, 230)
    average[:, 128:] = mean[128:]
    return reverse, average


def grouped_losses(records, before, after):
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if (before.shape != (len(records),) or after.shape != before.shape
            or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError('finite matching prediction vectors required')
    labels = np.array([r['label'] for r in records])
    loss0 = np.logaddexp(0., before)-labels*before
    loss1 = np.logaddexp(0., after)-labels*after
    buckets = defaultdict(list)
    for i, r in enumerate(records):
        buckets[r['group'], r['shot']].append(i)
    rows = [dict(group=g, shot=int(s), windows=len(ix), before_bce=float(loss0[ix].mean()),
        after_bce=float(loss1[ix].mean()), delta_bce=float((loss1[ix]-loss0[ix]).mean()),
        logit_delta=float((after[ix]-before[ix]).mean())) for (g, s), ix in sorted(buckets.items())]
    groups = {}
    for group in GROUPS:
        selected = [r for r in rows if r['group'] == group]
        groups[group] = dict(shots=len(selected), windows=sum(r['windows'] for r in selected),
            **{k: float(np.mean([r[k] for r in selected])) if selected else None
               for k in ('before_bce', 'after_bce', 'delta_bce', 'logit_delta')},
            improved=sum(r['delta_bce'] < -1e-6 for r in selected),
            worsened=sum(r['delta_bce'] > 1e-6 for r in selected))
    return dict(groups=groups, per_shot=rows)


def collect(net, controller, records, ids, advanced, old_hashes):
    import torch
    from .. import simple_data
    from ..delayed_recurrent_memory_20261025.learning import batch
    from .observation import observation
    from .prepare import adapted_state
    from .source_run import source_gate

    by_shot = defaultdict(list)
    for r in records:
        by_shot[r['shot']].append(r)
    if [r for sid in ids for r in by_shot[sid]] != records:
        raise ValueError('saved grouped role order changed')
    values = defaultdict(list)
    with source_gate(ids):
        for index, sid in enumerate(ids):
            path = io23.path_for('J-TEXT', sid)
            if sha(path) != old_hashes[str(sid)]:
                raise ValueError('raw source changed')
            if by_shot[sid]:
                shot = simple_data.attach_outcome(simple_data.read_shot('J-TEXT', sid), 'J-TEXT', advanced)
                signal = {k: shot[k] for k in ('values', 'available')}
                for r in by_shot[sid]:
                    if r['label'] != float(shot['y'][r['query']]):
                        raise ValueError('query label changed relative to saved experiment')
                    obs = observation(net, signal, r['origin'])
                    action = adapted_state(net, controller, obs).detach()
                    x, a = batch(signal, [r['origin'], r['maturity'], r['query']], 'cuda')
                    with torch.no_grad():
                        zeros = x.new_zeros(3, 8192)
                        base = net(x, a, zeros, None, 'qk', forecast=False)['logit']
                        changed = net(x[2:], a[2:], action, None, 'qk', forecast=False)['logit']
                        mature, _, _ = net.first(x[1:2], a[1:2], zeros[1:2], 'qk')
                        target = controller(obs['context'], obs['support'])
                    for name, tensor in dict(context=obs['context'],
                        maturity=torch.cat((mature[:, -1], mature[:, -2]), dim=1), target=target,
                        origin_logit=base[0:1], maturity_logit=base[1:2], query_logit=base[2:3],
                        real_logit=changed, action_norm=action.norm(dim=1)).items():
                        values[name].append(tensor.detach().cpu().numpy())
            if (index+1) % 100 == 0 or index+1 == len(ids):
                print(dict(stage='diagnostic_collect', shots=index+1, total=len(ids),
                           windows=len(values['context'])), flush=True)
    result = {k: np.concatenate(v) for k, v in values.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite diagnostics')
    return result


def probes(train, val, train_records, val_records):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    train_ix = np.array([r['group'] != 'early' for r in train_records])
    val_ix = np.array([r['group'] != 'early' for r in val_records])
    tr = [r for r in train_records if r['group'] != 'early']
    va = [r for r in val_records if r['group'] != 'early']
    w, vw = balanced_shot_weights(tr), balanced_shot_weights(va)
    y, vy = np.array([r['label'] for r in tr]), np.array([r['label'] for r in va])
    context = train['context'][train_ix]
    mean = np.average(context, axis=0, weights=w)
    reverse, average = feedback_controls(val['context'][val_ix], mean)
    results, learned, predictions = {}, {}, {}
    specs = {'origin': ('context', slice(0, 128)), 'feedback': ('context', slice(128, 358)),
             'combined': ('context', slice(None)), 'maturity': ('maturity', slice(None))}
    for name, (key, features) in specs.items():
        x, v, center, scale = standardize(train[key][train_ix, features], val[key][val_ix, features], w)
        model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x, y, sample_weight=w*len(w))
        learned[name] = dict(center=center.tolist(), scale=scale.tolist(),
            coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(),
            iterations=model.n_iter_.tolist(), sample_weight_sum=float((w*len(w)).sum()))
        queries = {name: v}
        if name == 'combined':
            queries.update(combined_reverse=(reverse-center)/scale, combined_mean=(average-center)/scale)
        for arm, query in queries.items():
            pred = model.decision_function(query)
            predictions[arm] = pred
            losses = np.logaddexp(0., pred)-vy*pred
            results[arm] = dict(balanced_bce=float(vw@losses),
                weighted_auc=float(roc_auc_score(vy, pred, sample_weight=vw)),
                by_group=grouped_losses(va, np.zeros(len(va)), pred)['groups'])
    for name in ('origin_logit', 'maturity_logit', 'query_logit'):
        pred = val[name][val_ix].astype(float)
        predictions[name] = pred
        results[name] = dict(balanced_bce=float(vw@(np.logaddexp(0., pred)-vy*pred)),
            weighted_auc=float(roc_auc_score(vy, pred, sample_weight=vw)),
            query_logit_is_future_reference=name == 'query_logit')
    return dict(results=results, fit=learned, val_record_indices=np.flatnonzero(val_ix).tolist(),
                not_online_policy=True, weights_sum_for_fit='N, each binary group N/2',
                regularization_not_tuned=True), predictions


def target_geometry(train, val, records):
    # Reference is the fixed training mean; norms are not taken as evidence of physics.
    mean = train['target'].astype(float).mean(0)
    result = {}
    for group in GROUPS:
        ix = np.array([r['group'] == group for r in records])
        x = val['target'][ix].astype(float)
        if not len(x):
            result[group] = dict(windows=0)
            continue
        denominator = np.linalg.norm(x, axis=1)*np.linalg.norm(mean)
        cosine = np.divide(x@mean, denominator, out=np.zeros(len(x)), where=denominator > 0)
        energy = np.sum(x*x)
        result[group] = dict(windows=len(x), mean_norm=float(np.linalg.norm(x, axis=1).mean()),
            cosine_training_mean_median=float(np.median(cosine)), zero_norm_rows=int((denominator == 0).sum()),
            centered_energy_fraction=float(np.sum((x-x.mean(0))**2)/energy) if energy else None)
    return result


def run(previous, output):
    import sklearn
    import torch
    from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
    from ..attention_basis_20261029.learning import MatrixAttention
    from .learning import ContextTarget

    previous, output = previous.absolute(), output.absolute()
    if output.exists() or output.parent != io23.ROOT/'probe_outputs':
        raise ValueError('new worktree probe output required')
    complete, protocol = read(previous/'complete.json'), read(previous/'protocol.json')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise ValueError('frozen dependency changed: '+name)
    for name in ('controller.pt', 'train_records.json', 'val_records.json', 'validation_arrays.npz'):
        if sha(previous/name) != complete['artifact_sha256'][name]:
            raise ValueError('previous artifact changed: '+name)
    roles, advanced = io23.source_roles(), io23.advanced_times()
    if roles['train'] != protocol['train_ids'] or roles['val'] != protocol['val_ids']:
        raise ValueError('role IDs changed')
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, _ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda()).eval().requires_grad_(False)
    original = {k: v.detach().clone() for k, v in net.state_dict().items()}
    policy = torch.load(previous/'controller.pt', weights_only=True, map_location='cuda')
    controller = ContextTarget().cuda().eval().requires_grad_(False)
    controller.load_state_dict(policy['state'])
    output.mkdir()
    started = time.monotonic()
    code_paths = [Path(__file__), io23.ROOT/'tests/test_context_feedback_diagnosis.py',
                  io23.ROOT/'CONTEXT_FEEDBACK_DIAGNOSIS_PLAN_20261030.md']
    hashes = {str(p.relative_to(io23.ROOT)): sha(p) for p in code_paths}
    metadata = dict(original_mapping_sha256=sha(io23.JTEXT_ADVANCED_TIME_PATH),
        metadata_preprocessing_parsed_test_entries=len(set(roles['test']) & set(advanced)),
        test_values_used_for_fit=False, original_run_mapping_hash_not_available=True,
        effective_horizon={role: {str(s): io23.effective_horizon_ms(s, advanced)[0] for s in roles[role]}
                           for role in ('train', 'val')})
    _durable_json(output/'protocol.json', dict(previous=str(previous), source_sha256=PARENT_SHA256,
        previous_protocol_sha256=sha(previous/'protocol.json'), code_sha256=hashes,
        metadata=metadata, test_hdf5_read=False, east_hdf5_read=False, not_online_performance=True,
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), sklearn=sklearn.__version__))
    data, records, results = {}, {}, {}
    for role in ('train', 'val'):
        records[role] = read(previous/f'{role}_records.json')
        data[role] = collect(net, controller, records[role], roles[role],
            {s: advanced[s] for s in roles[role] if s in advanced}, read(previous/f'{role}_input_hashes.json'))
        np.savez_compressed(output/f'{role}.npz', **data[role])
        results[role] = grouped_losses(records[role], data[role]['query_logit'], data[role]['real_logit'])
        _durable_json(output/f'{role}_fit.json', results[role])
    # Changed batching may affect float32 kernels: record differences, never relabel a tolerance as exact.
    with np.load(previous/'validation_arrays.npz', allow_pickle=False) as old:
        differences = {name: float(np.max(np.abs(data['val'][key]-old[name])))
            for key, name in [('query_logit', 'baseline'), ('real_logit', 'real_logit'), ('action_norm', 'real_norm')]}
    if max(differences.values()) > 1e-5:
        raise AssertionError(('new diagnostic disagrees with original validation', differences))
    probe, prediction = probes(data['train'], data['val'], records['train'], records['val'])
    _durable_json(output/'probes.json', probe)
    np.savez_compressed(output/'probe_predictions.npz', **prediction)
    geometry = target_geometry(data['train'], data['val'], records['val'])
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(net.state_dict()[k], v) for k, v in original.items()):
        raise AssertionError('source changed')
    if any(sha(io23.ROOT/k) != v for k, v in hashes.items()):
        raise AssertionError('diagnostic code changed')
    if any(sha(io23.ROOT/k) != v for k, v in protocol['code_sha256'].items()):
        raise AssertionError('frozen dependency changed during diagnosis')
    if sha(io23.JTEXT_ADVANCED_TIME_PATH) != metadata['original_mapping_sha256']:
        raise AssertionError('label metadata changed during diagnostic')
    result = dict(complete=True, scientific_goal_achieved=False, source_unchanged=True,
        seconds=time.monotonic()-started, validation_recompute_max_abs=differences,
        source_fit={role: value['groups'] for role, value in results.items()},
        target_geometry=geometry, probes=probe['results'], no_new_east_results=True,
        artifact_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    _durable_json(output/'complete.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.previous, args.output)
