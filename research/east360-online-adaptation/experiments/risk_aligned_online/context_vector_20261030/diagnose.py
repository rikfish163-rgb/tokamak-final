"""Independent saved-logit recount and an explicitly label-aware capacity probe."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..causal_input import apply_input_contract
from ..conditional_likelihood_20261029.run import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import representation_step


def collect(output):
    protocol, result = read(output/'protocol.json'), read(output/'complete.json')
    records, specs = read(output/'records.json'), read(output/'folds.json')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest:
            raise AssertionError('captured code changed: '+name)
    pred, covered = {}, []
    for spec in specs:
        path = output/f"fold{spec['fold']}_held.npz"
        if sha(path) != read(output/f"fold{spec['fold']}_seal.json")['sha256']:
            raise AssertionError('held predictions changed')
        with np.load(path, allow_pickle=False) as archive:
            index = archive['index']
            if {records[i]['shot'] for i in index} != set(spec['held']):
                raise AssertionError('fold ownership mismatch')
            covered.extend(index.tolist())
            for key in archive.files:
                if key == 'index':
                    continue
                pred.setdefault(key, np.empty(len(records), dtype=archive[key].dtype))[index] = archive[key]
    if sorted(covered) != list(range(len(records))):
        raise AssertionError('held rows do not form an exact partition')
    summary, geometry, rounding = {}, {}, []
    for arm in result['groups']:
        summary[arm], geometry[arm] = {}, {}
        for group in ('safe', 'danger_timely'):
            rows, cosine, norms = [], [], []
            for sid in protocol['warmup_ids']:
                take = np.array([r['shot'] == sid and (not r['disruptive'] if group == 'safe'
                                 else r['disruptive'] and r['point_label'] == 1.) for r in records])
                if not take.any():
                    continue
                y = float(group == 'danger_timely')
                before, after = pred['baseline'][take].astype(float), pred[arm+'_logit'][take].astype(float)
                delta = np.mean(np.logaddexp(0., after)-y*after-(np.logaddexp(0., before)-y*before))
                rows.append(float(delta))
                cosine.extend(pred[arm+'_cos_real'][take].tolist())
                norms.extend(pred[arm+'_norm'][take].tolist())
            actual = dict(shots=len(rows), mean_delta_bce=float(np.mean(rows)),
                          improved=sum(x < 0 for x in rows), worsened=sum(x > 0 for x in rows))
            expected = result['groups'][arm][group]
            difference = abs(actual['mean_delta_bce']-expected['mean_delta_bce'])
            # The independent pass sums shots in cold-stream order; the saved
            # pass sums fold/shot order. Observed discrepancy: 5.6e-17 float64.
            # Counts remain exact; 1e-14 bounds summation roundoff, not efficacy.
            if (difference > 1e-14 or any(actual[k] != expected[k]
                                         for k in ('shots', 'improved', 'worsened'))):
                raise AssertionError(f'independent group recount mismatch: {arm}/{group}: '
                                     f'{actual} versus {result["groups"][arm][group]}')
            rounding.append(difference)
            summary[arm][group] = actual
            geometry[arm][group] = dict(median_cosine_real=float(np.median(cosine)),
                                        median_norm=float(np.median(norms)), max_norm=max(norms))
    audit = dict(complete=True, scope='SAVED_HELD_QUERY_RECOUNT_NOT_STREAM_METRICS',
                 rows=len(records), groups=summary, geometry=geometry,
                 full_stream_run=False, max_aggregation_roundoff=max(rounding),
                 complete_sha256=sha(output/'complete.json'))
    _durable_json(output/'independent_audit.json', audit)
    return protocol, records, pred


def capacity(output):
    protocol, records, pred = collect(output)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base, parent = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda())
    frozen = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    reference = Path(protocol['reference'])
    slow = torch.load(reference/'models/cold.pt', map_location='cuda', weights_only=True)['slow']
    mapping, hashes = read(reference/'input_map.json'), read(reference/'input_hashes.json')
    threshold_logit = math.log(parent['threshold']/(1-parent['threshold']))
    rows = []
    for sid in protocol['warmup_ids']:
        if sha(io23.path_for('EAST', sid)) != hashes[str(sid)]:
            raise AssertionError('raw data changed')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = transform({k: raw[k] for k in ('values', 'available')}, mapping)
        indices = [i for i, r in enumerate(records) if r['shot'] == sid
                   and (not r['disruptive'] or r['point_label'] == 1.)]
        for lo in range(0, len(indices), 16):
            selected = indices[lo:lo+16]
            x, mask = batch(signal, [records[i]['query'] for i in selected], base.center.device)
            y = x.new_tensor([records[i]['point_label'] for i in selected])
            sign = 2*y-1
            state = x.new_zeros(len(x), 8192)
            with torch.no_grad():
                before = net(x, mask, state, slow, 'qk', forecast=False)['logit']
            # Different diagnostic batches can round differently; report, do not
            # rewrite archived held logits or claim a bitwise replay here.
            saved = x.new_tensor(pred['baseline'][selected])
            difference = (before-saved).abs()
            if difference.max() > 1e-4:
                raise AssertionError('diagnostic baseline differs beyond FP32 batching tolerance')
            best, best_iteration = before.clone(), torch.zeros_like(y, dtype=torch.long)
            initial_norm = None
            for iteration in range(1, 9):
                state = state.detach().requires_grad_()
                z = net(x, mask, state, slow, 'qk', forecast=False)['logit']
                gradient, = torch.autograd.grad((sign*z).sum(), state)
                magnitude = gradient.norm(dim=1, keepdim=True)
                if initial_norm is None:
                    initial_norm = magnitude[:, 0].detach()
                state = state.detach()+.025*gradient.detach()/magnitude.detach().clamp_min(1e-30)
                state *= (.1/state.norm(dim=1, keepdim=True).clamp_min(.1))
                with torch.no_grad():
                    after = net(x, mask, state, slow, 'qk', forecast=False)['logit']
                    better = sign*after > sign*best
                    best = torch.where(better, after, best)
                    best_iteration = torch.where(better, iteration, best_iteration)
            origin_x, origin_mask = batch(signal, [records[i]['origin'] for i in selected], base.center.device)
            target_raw = x.new_zeros(len(x), 64, requires_grad=True)
            optimizer = torch.optim.Adam([target_raw], lr=.1)
            target_best = before.clone()
            for iteration in range(30):
                zero = x.new_zeros(len(x), 8192, requires_grad=True)
                h, _, _ = net.first(origin_x, origin_mask, zero, 'qk')
                step = representation_step(h[:, -1], h[:, -1].detach(), target_raw.tanh(), zero, create_graph=True)
                z = net(x, mask, step, slow, 'qk', forecast=False)['logit']
                with torch.no_grad():
                    target_best = torch.where(sign*z > sign*target_best, z.detach(), target_best)
                optimizer.zero_grad()
                (-sign*z).sum().backward()
                optimizer.step()
            for j, i in enumerate(selected):
                rows.append(dict(index=i, shot=sid, query=records[i]['query'], y=float(y[j]),
                    cold_logit=float(before[j]), learned_logit=float(pred['real_logit'][i]),
                    label_aware_best_logit=float(best[j]), best_iteration=int(best_iteration[j]),
                    label_aware_target_best_logit=float(target_best[j]),
                    initial_query_risk_gradient_norm=float(initial_norm[j]),
                    local_linear_boundary_distance=float(abs(float(before[j])-threshold_logit)/max(float(initial_norm[j]), 1e-30)),
                    batch_baseline_difference=float(difference[j])))
    summary = {}
    for y, name in ((0., 'safe'), (1., 'danger_timely')):
        selected = [r for r in rows if r['y'] == y]
        counts = {}
        for mode in ('cold_logit', 'learned_logit', 'label_aware_best_logit', 'label_aware_target_best_logit'):
            counts[mode] = sum((r[mode] >= threshold_logit) == bool(y) for r in selected)
        cold_wrong = [r for r in selected if (r['cold_logit'] >= threshold_logit) != bool(y)]
        summary[name] = dict(windows=len(selected), correct_query_decisions=counts,
            label_aware_repaired_cold_wrong=sum((r['label_aware_best_logit'] >= threshold_logit) == bool(y) for r in cold_wrong),
            target_oracle_repaired_cold_wrong=sum((r['label_aware_target_best_logit'] >= threshold_logit) == bool(y) for r in cold_wrong),
            cold_wrong=len(cold_wrong), local_linear_distance_median=float(np.median([r['local_linear_boundary_distance'] for r in cold_wrong])) if cold_wrong else None)
    if sha(DEFAULT_PARENT) != PARENT_SHA256 or any(not torch.equal(v.cpu(), frozen[k]) for k, v in net.state_dict().items()):
        raise AssertionError('frozen source changed')
    artifact = dict(scope='POSTHOC_LABEL_AWARE_QUERY_FEASIBILITY_DIAGNOSTIC_NOT_ONLINE',
                    radius=.1, iterations=8, normalized_step=.025, summary=summary, rows=rows,
                    target_oracle_iterations=30, target_oracle_adam_lr=.1,
                    not_a_certified_upper_bound=True, uses_query_signal_and_true_historical_label=True,
                    full_stream_run=False, suffix_labels_read=0, source_unchanged=True,
                    code_sha256=sha(Path(__file__)))
    _durable_json(output/'query_capacity_diagnostic.json', artifact)
    print(dict(scope=artifact['scope'], summary=summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--capacity', action='store_true')
    args = parser.parse_args()
    if args.capacity:
        capacity(args.output.absolute())
    else:
        protocol, records, _ = collect(args.output.absolute())
        print(dict(scope=protocol['scope'], verified_records=len(records)), flush=True)
