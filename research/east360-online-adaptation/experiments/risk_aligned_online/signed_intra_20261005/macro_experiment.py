"""Bounded first32 historical cost table, then conditional adjacent32 replay.

No warm-up errors are removed: first32 retain their sealed static predictions.
The nine historical action pairs are supervised computational counterfactuals,
not nine independent shots and not deployed hindsight-optimal predictions.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import simple_data, simple_train
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json, _hash, seal_predictions
from ..redesign_20261004.prepare_prior import (
    DEFAULT_PARENT, PARENT_SHA256, load_parent, classifier_state, assert_classifier_unchanged)
from ..ffn_intra_20261005.model import FfnModel
from .context import ACTIONS
from .finite_probe import reference_order, whole_cost
from .policy import score_plan
from .macro_credit import macro_plan, choose_pair, fit_rule

PAIRS = list(itertools.product(range(3), repeat=2))


def statistics(records):
    nd = sum(bool(r['disruptive']) for r in records); ns = len(records)-nd
    tp = sum(bool(r['disruptive']) and r['prediction']['timely'] for r in records)
    fp = sum(not r['disruptive'] and r['prediction']['alarm'] for r in records)
    return dict(n_disrupt=nd, n_safe=ns, tp=tp, fp=fp,
                tpr=tp/nd if nd else None, fpr=fp/ns if ns else None)


def evaluate_tables(tables, rule):
    rows = []
    for t in tables:
        a,b = choose_pair(t, rule)
        rows.append(dict(shot=t['shot'], disruptive=t['disruptive'], pair=[a,b],
                         prediction=t['predictions'][a][b]))
    return dict(metrics=statistics(rows), rows=rows)


def static_rows(tables):
    return [dict(shot=t['shot'], disruptive=t['disruptive'], prediction=t['predictions'][0][0])
            for t in tables]


def rescue_gate(tables, selected=None):
    rescued = {0: [], 1: []}
    for i,t in enumerate(tables):
        costs = np.asarray(t['costs'])
        candidate = int(costs.min()) if selected is None else int(costs[tuple(selected[i]['pair'])])
        if costs[0,0] == 1 and candidate == 0:
            rescued[int(t['disruptive'])].append(t['shot'])
    return dict(passed=all(rescued[k] for k in (0,1)), rescued=rescued)


def global_pair(tables):
    counts = [sum(int(t['disruptive']) == k for t in tables) for k in (0,1)]
    cost = sum(np.asarray(t['costs'], float)/(2*counts[int(t['disruptive'])]) for t in tables)
    return tuple(map(int, np.unravel_index(cost.argmin(), (3,3))))


def make_table(model, direction, signals, *, index, output, reference, threshold):
    center, scale = model.center.detach().cpu(), model.scale.detach().cpu()
    traces, plans = {}, {}
    for a,b in PAIRS:
        q, decisions, events = macro_plan(signals, center, scale, pair=(ACTIONS[a], ACTIONS[b]))
        _, scores = score_plan(model, signals, direction, q)
        folder = output/'teachers'/f'{a}{b}'; folder.mkdir(parents=True, exist_ok=True)
        receipt = seal_predictions(folder, index, signals['shot'], signals['times'], scores,
                                   arm=f'historical_macro_{a}{b}', threshold=threshold)
        traces[a,b] = scores; plans[a,b] = decisions
        _durable_json(folder/f'{index:04d}_{signals["shot"]}_events.json',
                      dict(decisions=decisions, events=events, seal_sha256=receipt['sha256']))
    difference = float(np.max(np.abs(traces[0,0]-reference)))
    if difference > 3e-6 or np.any((traces[0,0] >= threshold) != (reference >= threshold)):
        raise RuntimeError('historical hold/hold does not reproduce frozen static reference')
    # All nine full predictions durable before this function reads outcomes.
    shot = simple_data.attach_outcome(signals, 'EAST', {})
    costs = np.zeros((3,3), int); predictions = [[None]*3 for _ in range(3)]
    for a,b in PAIRS:
        _, cost, prediction = whole_cost(shot, traces[a,b], threshold)
        costs[a,b] = cost[0]; predictions[a][b] = prediction
    c1 = None; c2 = [None]*3
    for a,b in PAIRS:
        for decision in plans[a,b]:
            if decision['effect'] is None:
                continue  # No later prediction exists: no actionable training example.
            c = decision['context']
            if decision['slot'] == 0:
                if c1 is not None and c1 != c: raise RuntimeError('first context depends on teacher choice')
                c1 = c
            else:
                if c2[a] is not None and c2[a] != c: raise RuntimeError('second context depends on second action')
                c2[a] = c
    table = dict(index=index, shot=shot['shot'], disruptive=bool(shot['is_disrupt']),
        context_1=c1, context_2=c2, costs=costs.tolist(), predictions=predictions,
        static_max_abs_difference=difference)
    _durable_json(output/'tables'/f'{index:04d}_{shot["shot"]}.json', table)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction', type=Path, required=True)
    p.add_argument('--static', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-guard', action='store_true',
        help='New development variant: accept only strict class-Pareto joint-history improvement')
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('local CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131); started = time.monotonic()
    order = reference_order(args.static)
    summary = json.loads((args.direction.parent/'summary.json').read_text())
    if _hash(args.direction) != summary['direction_sha256']: raise ValueError('source direction changed')
    asset = torch.load(args.direction, map_location='cpu', weights_only=True)
    if asset['protocol']['parent_sha256'] != PARENT_SHA256 or not asset['protocol']['source_only']:
        raise ValueError('source-only fixed direction required')
    base, state = load_parent(DEFAULT_PARENT); model = FfnModel(base.cuda())
    before = classifier_state(base); threshold = float(state['threshold'])
    direction = asset['direction'].to(model.center)
    if direction.shape != (model.delta_count,) or abs(float(direction.double().norm())-1) > 1e-6:
        raise ValueError('fixed unit FFN direction required')
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'tables').mkdir()
    root = Path(__file__).parents[1]
    files = [f for f in Path(__file__).parent.glob('*.py') if not f.name.startswith('test_')]
    files += [root/n for n in ('simple23.py','simple_data.py','io23.py','causal_input.py',
        'simple_online.py','simple_train.py','ffn_intra_20261005/model.py',
        'redesign_20261004/prepare_prior.py','redesign_20261004/intra.py')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in files: archive.write(f, str(f.relative_to(root)))
    protocol = dict(parent_sha256=PARENT_SHA256, direction_sha256=_hash(args.direction),
        threshold=threshold, history_ids=[r['shot'] for r in order[:32]],
        future_ids=[r['shot'] for r in order[32:64]], decision_valid_events=[1,21],
        events_per_macro=20, max_events=40, actions=list(ACTIONS), steps=[.02,.08],
        q_bounds=[-1.,1.], ridge=1e-3, fit_iterations=3, history_old_audit_is_now_training=True,
        historical_guard=args.historical_guard,
        guard_contract='both class errors nonincreasing, one strictly lower; stop on first rejection',
        guard_is_posthoc_development_variant=args.historical_guard,
        future_access='only after historical capacity and common-student gates',
        hypothesis='historical full costs can teach early signal-conditional internal actions',
        label_access='natural postshot reveal, no query budget',
        scope='DEVELOPMENT inter-trained intra, NOT pure source-only intra or new physics',
        source_and_alarm_frozen=True, target_statistics_fitted=False,
        input_contract='nominal EAST + unavailable MNM + pointwise CIII sentinel mask',
        max_teacher_replays=567, max_active_future_replays=96,
        code_sha256={str(f.resolve()):_hash(f) for f in files}, gpu=torch.cuda.get_device_name())
    _durable_json(args.output/'protocol.json', protocol)

    def load_signal(index):
        item = order[index-1]; sid = int(item['shot']); stem = f'{index:04d}_{sid}'
        receipt = json.loads((args.static/f'{stem}_seal.json').read_text())
        npz = args.static/receipt['npz']
        if (_hash(npz) != receipt['sha256'] or receipt['threshold'] != threshold
                or receipt['shot'] != sid
                or _hash(Path(item['provenance']['path'])) != item['provenance']['sha256']):
            raise ValueError('static or input provenance changed')
        signals = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        with np.load(npz, allow_pickle=False) as saved:
            if not np.array_equal(saved['times'], signals['times']): raise ValueError('time grid changed')
            return signals, saved['scores'].copy()

    tables = []
    for index in range(1,33):
        signals, scores = load_signal(index)
        tables.append(make_table(model,direction,signals,index=index,output=args.output,
                                 reference=scores,threshold=threshold))
        if index % 4 == 0:
            print(json.dumps(dict(stage='nine_macro_history', shots=index,
                                  seconds=time.monotonic()-started)),flush=True)
    cold = static_rows(tables); capacity = rescue_gate(tables)
    result = dict(protocol=protocol, cold_start=statistics(cold), capacity=capacity,
                  student_admitted=False, future_run=False)
    rule = torch.zeros(3,117,dtype=torch.float64)
    if capacity['passed']:
        rule, logs = fit_rule(tables,rule,conservative=args.historical_guard)
        historical = evaluate_tables(tables,rule); gate = rescue_gate(tables,historical['rows'])
        old, new = result['cold_start'], historical['metrics']
        admitted = bool(gate['passed'] and new['tp']>=old['tp'] and new['fp']<=old['fp'])
        result.update(student_admitted=admitted, historical_student=historical,
                      fit_log=logs, student_rescues=gate)
        torch.save(dict(rule=rule,protocol=protocol,admitted=admitted),args.output/'history_rule.pt')
    print(json.dumps(dict(stage='history_gate',capacity=capacity,
        admitted=result['student_admitted'],metrics=result.get('historical_student',{}).get('metrics'))),flush=True)
    if result['student_admitted']:
        future = {arm:[] for arm in ('S','H','P','G')}; updates=[]
        for index in range(33,65):
            signals, reference = load_signal(index); sid = signals['shot']; gp = global_pair(tables)
            traces={}; rule_hash = _hash(args.output/'history_rule.pt') if index==33 else _hash(args.output/f'rule_after_{index-1}.pt')
            for arm in future:
                folder=args.output/arm; folder.mkdir(exist_ok=True)
                if arm=='S':
                    scores=reference; q=np.zeros(len(scores)); decisions=[]; events=[]
                else:
                    kwargs = dict(pair=tuple(ACTIONS[a] for a in gp)) if arm=='G' else dict(rule=rule,arm=arm)
                    q,decisions,events=macro_plan(signals,model.center,model.scale,**kwargs)
                    _,scores=score_plan(model,signals,direction,q)
                receipt=seal_predictions(folder,index,sid,signals['times'],scores,arm=arm,threshold=threshold)
                np.savez_compressed(folder/f'{index:04d}_{sid}_q.npz',q=q)
                _durable_json(folder/f'{index:04d}_{sid}_events.json',dict(decisions=decisions,events=events,
                    rule_file_sha256=rule_hash,history_end=index-1,global_pair=list(gp)))
                traces[arm]=scores
            # No candidate accesses current outcome until all four predictions are sealed.
            shot=simple_data.attach_outcome(signals,'EAST',{})
            for arm,scores in traces.items():
                _,_,prediction=whole_cost(shot,scores,threshold)
                future[arm].append(dict(shot=sid,disruptive=bool(shot['is_disrupt']),prediction=prediction))
            if index<64:
                tables.append(make_table(model,direction,signals,index=index,output=args.output,
                                         reference=reference,threshold=threshold))
                rule,logs=fit_rule(tables,rule,conservative=args.historical_guard)
                torch.save(dict(rule=rule,history_end=index),args.output/f'rule_after_{index}.pt')
                updates.append(dict(index=index,fit_log=logs))
            print(json.dumps(dict(stage='progressive_next32',done=index-32,seconds=time.monotonic()-started)),flush=True)
        metrics={arm:statistics(rows) for arm,rows in future.items()}; h=metrics['H']
        passed=bool(h['fpr'] is not None and h['fpr']<=.20 and all(
            h['tp']>=metrics[a]['tp'] and h['fp']<=metrics[a]['fp'] and
            (h['tp']>metrics[a]['tp'] or h['fp']<metrics[a]['fp']) for a in ('S','P','G')))
        result.update(future_run=True,future=metrics,future_records=future,updates=updates,
            information_gate=passed,cold_plus_future={arm:statistics(cold+rows) for arm,rows in future.items()})
    assert_classifier_unchanged(base,before)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256: raise RuntimeError('source artifact changed')
    result.update(source_unchanged=True,elapsed_seconds=time.monotonic()-started)
    _durable_json(args.output/'summary.json',result)
    print(json.dumps(dict(stage='macro_complete',student_admitted=result['student_admitted'],
        future_run=result['future_run'],future=result.get('future'),information_gate=result.get('information_gate'),
        seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    main()
