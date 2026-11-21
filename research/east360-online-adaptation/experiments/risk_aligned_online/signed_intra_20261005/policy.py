"""One history32 fit/audit, then at most one adjacent32 H/P/L experiment.

This is a bounded diagnostic from reports17/20, not the all2953 online runner.
Source classifier/direction/threshold fixed; no extra trials after failed gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import simple_data, simple_train
from ..simple_online import _durable_json, _hash, seal_predictions
from ..redesign_20261004.intra import _batch_windows
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, classifier_state, assert_classifier_unchanged,
)
from ..ffn_intra_20261005.model import FfnModel
from .context import VisibleContext, advance_q, ACTIONS
from .finite_probe import reference_order, whole_cost, choose_historical_action, bias_fit


def plan(signals, rule, center, scale, *, arm='H', intervention=None):
    """Generate causal action trace only; risk must use q at the same index.

    No outcome or valid-evaluation mask is read. Precomputing a deterministic
    prefix-local rule is an offline acceleration, not access to future inputs.
    Tests compare every prefix against a separately truncated signal stream.
    intervention=(event-list-index, action) is HISTORICAL counterfactual only.
    """
    if signals['device'] != 'EAST': raise ValueError('EAST signal stream required')
    x, mask = np.asarray(signals['values']), np.asarray(signals['available'])
    if x.shape != mask.shape or x.ndim != 2 or x.shape[1] != 23 or mask.dtype != bool:
        raise ValueError('flat23 signal and availability required')
    state = VisibleContext(arm); q = np.empty(len(x), np.float64); events = []; start = 0
    used_intervention = False
    for origin in range(0, len(x), 26):
        for offset, segment in ((5, 'early'), (25, 'late')):
            end = origin + offset
            if end >= len(x): continue
            q[start:end+1] = state.q; start = end+1
            first = origin+1 if segment == 'early' else origin+5
            truth = torch.as_tensor(x[first:end+1].copy(), dtype=rule.dtype)
            observed = torch.as_tensor(mask[first:end+1].copy(), dtype=torch.bool)
            before = state.q
            context, action = state.step(truth, observed, center, scale, segment, rule)
            if context is None: continue
            if intervention is not None and len(events) == intervention[0]:
                action = intervention[1]; state.q = advance_q(before, action, segment)
                used_intervention = True
            events.append(dict(end=end, effect=end+1 if end+1 < len(x) else None,
                segment=segment, context=context.tolist(), action=action,
                q_before=before, q_after=state.q))
    q[start:] = state.q
    if intervention is not None and not used_intervention:
        raise ValueError('counterfactual must name an existing valid historical event')
    return q, events


def history_partition(rows):
    fit, audit = [], []
    for row in rows:
        bucket = int(hashlib.sha256(f'131:{int(row["shot"])}'.encode()).hexdigest(), 16) % 5
        (audit if bucket == 0 else fit).append(row)
    return fit, audit


def guard(candidate, old):
    return bool(candidate['tp'] >= old['tp'] and candidate['fp'] <= old['fp']
        and all(candidate['risks'][k] is not None and old['risks'][k] is not None
                and candidate['risks'][k] <= old['risks'][k] + 1e-6 for k in ('0', '1'))
        and any(candidate['risks'][k] < old['risks'][k] - 1e-6 for k in ('0', '1')))


def score_plan(model, signals, direction, q, *, past=None, start=0):
    logits = np.empty(len(q), np.float32); scores = np.empty(len(q), np.float32)
    if start:
        if past is None: raise ValueError('past emitted traces required')
        logits[:start], scores[:start] = past[0][:start], past[1][:start]
    while start < len(q):
        stop = start+1
        while stop < min(len(q), start+256) and q[stop] == q[start]: stop += 1
        x, a = _batch_windows(simple_data, {k: signals[k] for k in ('values', 'available')},
                               range(start, stop), model, model.center.device)
        delta = (float(q[start])*direction).detach().requires_grad_(True)
        out = model(x, a, delta)['logit'].detach()
        if not torch.isfinite(out).all(): raise FloatingPointError('nonfinite signed policy risk')
        logits[start:stop] = out.cpu().numpy(); scores[start:stop] = out.sigmoid().cpu().numpy()
        start = stop
    return logits, scores


def summarize(rows, costs, threshold):
    m = simple_train.metrics(rows, threshold, lead=30.)
    risks = {str(k): [cost for row, cost in zip(rows, costs)
                      if int(row['is_disrupt']) == k and cost is not None] for k in (0, 1)}
    return dict(m, risks={k: float(np.mean(v)) if v else None for k, v in risks.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', type=Path, required=True)
    p.add_argument('--history', type=Path, required=True, help='completed first32 finite probe, zero arm only')
    p.add_argument('--static', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131); rng = np.random.default_rng(131)
    started = time.monotonic(); order = reference_order(args.static)
    asset_summary = json.loads((args.direction.parent/'summary.json').read_text())
    if _hash(args.direction) != asset_summary['direction_sha256']: raise ValueError('direction changed')
    asset = torch.load(args.direction, weights_only=True, map_location='cpu')
    history_summary = json.loads((args.history/'summary.json').read_text())
    if (history_summary['protocol']['shot_ids'] != [r['shot'] for r in order[:32]]
            or history_summary['protocol']['direction_sha256'] != _hash(args.direction)
            or history_summary['source_reference_max_abs'] != 0
            or asset['protocol']['parent_sha256'] != PARENT_SHA256):
        raise ValueError('same fixed first32 cold-start evidence required')
    base, checkpoint = load_parent(DEFAULT_PARENT); base.cuda(); model = FfnModel(base)
    before = classifier_state(base); direction = asset['direction'].to(model.center)
    threshold = float(checkpoint['threshold']); center = model.center.detach().cpu().double()
    scale = model.scale.detach().cpu().double()
    args.output.mkdir(parents=True, exist_ok=False)
    historical = []; cold_rows = []
    # Every historical label is read only after verifying that shot's immutable
    # original zero-state prediction. This is postshot fitting, not support110.
    for index, item in enumerate(order[:32], 1):
        sid = item['shot']; stem = f'{index:04d}_{sid}'
        r = json.loads((args.history/'0'/f'{stem}_seal.json').read_text())
        if (_hash(args.history/'0'/r['npz']) != r['sha256']
                or _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']):
            raise ValueError('historical seal/raw input changed')
        signals = simple_data.read_shot('EAST', sid, rate_contract='nominal')
        shot = simple_data.attach_outcome(signals, 'EAST', {}); historical.append(shot)
        with np.load(args.history/'0'/r['npz'], allow_pickle=False) as saved:
            if not np.array_equal(saved['times'], shot['times']): raise ValueError('historical times changed')
            cold_rows.append(whole_cost(shot, saved['scores'], threshold)[0])
    fit, audit = history_partition(historical)
    if any({int(s['is_disrupt']) for s in group} != {0, 1} for group in (fit, audit)):
        raise ValueError('first32 lacks both fit/audit classes; do not select a later successful block')
    root = Path(__file__).parents[1]
    files = list(Path(__file__).parent.glob('*.py')) + [root/name for name in
        ('simple23.py','simple_data.py','io23.py','simple_online.py','simple_train.py',
         'ffn_intra_20261005/model.py','redesign_20261004/intra.py','redesign_20261004/prepare_prior.py')]
    files = [f for f in files if not f.name.startswith('test_')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f, str(f.relative_to(root)))
    protocol = dict(parent_sha256=PARENT_SHA256, direction_sha256=_hash(args.direction),
        history_summary_sha256=_hash(args.history/'summary.json'), threshold=threshold, seed=131,
        fit_ids=[s['shot'] for s in fit], audit_ids=[s['shot'] for s in audit],
        future_ids=[r['shot'] for r in order[32:64]], future_access='only after one historical guard passes',
        steps=32, pairs='one safe + one disruptive each step, chronological cyclic within class',
        lr=3e-5, optimizer='Adam, no weight decay', training_target='single-event three-action historical counterfactual; error then whole risk',
        rules='zero B before first32; only candidate after first32; reject once and stop if audit fails',
        information_arms=['S','H','P','L'], source_frozen=True, direction_frozen=True,
        final_score_correction=False, source_only_initialization=True,
        evidence='DEVELOPMENT_ONLY; historical rule training is supervised, current shot action inputs unlabeled',
        code_sha256={str(f.resolve()): _hash(f) for f in files})
    _durable_json(args.output/'protocol.json', protocol)
    rule = torch.zeros(3, 117, dtype=torch.float64, requires_grad=True)
    original = rule.detach().clone(); optimizer = torch.optim.Adam([rule], lr=3e-5)
    by_class = [[s for s in fit if int(s['is_disrupt']) == k] for k in (0, 1)]
    log = []
    for iteration in range(32):
        contexts, targets, examples = [], [], []
        frozen = rule.detach().clone()
        for k in (0, 1):
            shot = by_class[k][iteration % len(by_class[k])]
            # Only signal fields cross the plan boundary; outcome used below.
            signals = {key: shot[key] for key in ('device','values','available','times')}
            q, events = plan(signals, frozen, center, scale)
            possible = [i for i, e in enumerate(events) if e['effect'] is not None]
            if not possible: raise ValueError('historical fit shot has no effective update event')
            event_index = int(rng.choice(possible)); event = events[event_index]
            baseline = score_plan(model, signals, direction, q)
            costs = {}; traces = {}
            for action in ACTIONS:
                altered_q, altered_events = plan(signals, frozen, center, scale, intervention=(event_index, action))
                if not np.array_equal(altered_q[:event['effect']], q[:event['effect']]):
                    raise RuntimeError('historical intervention changed already emitted parameter state')
                result = baseline if np.array_equal(altered_q, q) else score_plan(
                    model, signals, direction, altered_q, past=baseline, start=event['effect'])
                _, costs[action], prediction = whole_cost(shot, result[1], threshold)
                traces[str(action)] = dict(error=costs[action][0], risk=costs[action][1],
                                           prediction=prediction)
            target = choose_historical_action(costs) if all(r is not None for _, r in costs.values()) else 0
            contexts.append(event['context']); targets.append(ACTIONS.index(target))
            examples.append(dict(shot=shot['shot'], label=k, event_index=event_index, event=event,
                                 chosen_action=target, counterfactual=traces))
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.tensor(contexts, dtype=rule.dtype)
        loss = F.cross_entropy(inputs @ rule.T, torch.tensor(targets))
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite action loss')
        loss.backward(); optimizer.step()
        log.append(dict(iteration=iteration+1, loss=float(loss.detach()), examples=examples))
        if (iteration+1) % 4 == 0:
            _durable_json(args.output/'training.json', log)
            print(json.dumps(dict(stage='history_rule_fit', pair=iteration+1, total=32,
                                  seconds=time.monotonic()-started)), flush=True)
    candidate = rule.detach().clone()
    torch.save(dict(rule=candidate, protocol=protocol, admitted=False), args.output/'candidate.pt')
    audits = {}
    for name, b in (('old', original), ('candidate', candidate)):
        rows, costs = [], []
        for shot in audit:
            signals = {key: shot[key] for key in ('device','values','available','times')}
            q, _ = plan(signals, b, center, scale)
            _, scores = score_plan(model, signals, direction, q)
            row, cost, _ = whole_cost(shot, scores, threshold); rows.append(row); costs.append(cost[1])
        audits[name] = summarize(rows, costs, threshold)
    admitted = guard(audits['candidate'], audits['old'])
    _durable_json(args.output/'audit.json', dict(admitted=admitted, reports=audits))
    result = dict(protocol=protocol, audit=audits, admitted=admitted, future_run=False,
                  cold_start=simple_train.metrics(cold_rows, threshold, lead=30.))
    print(json.dumps(dict(stage='historical_guard', admitted=admitted, reports={k:
        {f: v[f] for f in ('tp','fp','n_disrupt','n_safe','risks')} for k,v in audits.items()})), flush=True)
    if admitted:
        torch.save(dict(rule=candidate, protocol=protocol, admitted=True), args.output/'admitted.pt')
        future_rows = {arm: [] for arm in ('S','H','P','L')}; future_costs = {arm: [] for arm in future_rows}
        bias_pairs = []; records = []
        for arm in future_rows: (args.output/arm).mkdir()
        for index, item in enumerate(order[32:64], 33):
            sid = item['shot']; signals = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            if _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']:
                raise ValueError('next-block input changed')
            traces = {}
            for arm in future_rows:
                q, events = plan(signals, original if arm == 'S' else candidate, center, scale,
                                  arm='H' if arm == 'S' else arm)
                logits, scores = score_plan(model, signals, direction, q)
                receipt = seal_predictions(args.output/arm, index, sid, signals['times'], scores,
                                           arm=arm, threshold=threshold)
                np.savez_compressed(args.output/arm/f'{index:04d}_{sid}_states.npz', q=q, logits=logits)
                _durable_json(args.output/arm/f'{index:04d}_{sid}_events.json', events)
                traces[arm] = dict(q=q, logits=logits, scores=scores, receipt=receipt)
            shot = simple_data.attach_outcome(signals, 'EAST', {})
            predictions = {}
            for arm, trace in traces.items():
                row, cost, prediction = whole_cost(shot, trace['scores'], threshold)
                future_rows[arm].append(row); future_costs[arm].append(cost[1]); predictions[arm] = prediction
            keep = np.asarray(shot['valid'], bool)
            bias_pairs.append((traces['H']['q'][keep], (traces['H']['logits']-traces['S']['logits'])[keep]))
            records.append(dict(index=index, shot=sid, predictions=predictions))
            if index % 8 == 0: print(json.dumps(dict(stage='next32_information', done=index-32)), flush=True)
        reports = {arm: summarize(rows, future_costs[arm], threshold) for arm, rows in future_rows.items()}
        h = reports['H']
        passed = all(h['tp'] >= reports[a]['tp'] and h['fp'] <= reports[a]['fp']
                     and (h['tp'] > reports[a]['tp'] or h['fp'] < reports[a]['fp']) for a in ('S','P','L'))
        result.update(future_run=True, information_gate=passed, future=reports, records=records,
            cold_plus_future={arm: simple_train.metrics(cold_rows+rows, threshold, lead=30.) for arm, rows in future_rows.items()},
            bias_diagnostic=bias_fit(bias_pairs))
    assert_classifier_unchanged(base, before)
    if _hash(DEFAULT_PARENT) != PARENT_SHA256: raise RuntimeError('qualified parent file changed')
    result.update(classifier_unchanged=True, elapsed_seconds=time.monotonic()-started)
    _durable_json(args.output/'summary.json', result)
    print(json.dumps(dict(stage='history_experiment_complete', admitted=admitted,
        future_run=result['future_run'], information_gate=result.get('information_gate'),
        seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
