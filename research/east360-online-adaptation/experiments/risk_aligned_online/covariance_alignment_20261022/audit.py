"""Re-read signals, reconstruct maps, and independently score sealed traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent
from ..warmstart_alignment_20261022.run import predict_many
from ..warmstart_alignment_20261022 import alignment as scalar
from ..warmstart_alignment_20261022.audit import counts, flips
from .alignment import summarize, combine, fit_maps, transform
from .run import stability


def compare_replay(saved, replay, threshold):
    if saved.shape != replay.shape or not np.isfinite(replay).all():
        raise AssertionError('replay shape or probability failure')
    error = float(np.max(np.abs(saved-replay)))
    if error > 2e-6 or not np.array_equal(saved>=threshold, replay>=threshold):
        raise AssertionError('map-to-prediction replay mismatch')
    return error


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    if (args.run/'verification.json').exists(): raise FileExistsError('already verified')
    protocol = json.loads((args.run/'protocol.json').read_text())
    done = json.loads((args.run/'completed.json').read_text())
    declared = json.loads((args.run/'per_shot.json').read_text())
    contract = load_test_contract('EAST'); order = contract['shot_ids']; threshold = protocol['threshold']
    if protocol['contract'] != contract: raise AssertionError('test changed')
    if sha(protocol['source_checkpoint']) != protocol['source_sha256']: raise AssertionError('source changed')
    for name, digest in protocol['code_sha256'].items():
        if sha(io23.ROOT/name) != digest: raise AssertionError(f'code changed: {name}')
    if sha(protocol['plan']) != protocol['plan_sha256']: raise AssertionError('plan changed')
    center, scale = protocol['coordinate']['center'], protocol['coordinate']['scale']
    records = [{k:s[k] for k in scalar.SIGNAL_KEYS} for s in
        (simple_data.read_shot('J-TEXT', sid) for sid in io23.source_roles()['train'])]
    summaries = [summarize(r, center=center, scale=scale) for r in records]
    source = combine(summaries, device='J-TEXT', allowed_ids=io23.source_roles()['train'])
    ss = scalar.fit_statistics(records, device='J-TEXT', allowed_ids=io23.source_roles()['train'])
    for name, obj in (('source_shot_statistics', summaries), ('source_statistics', source), ('source_scalar_statistics', ss)):
        if obj != json.loads((args.run/(name+'.json')).read_text()): raise AssertionError(name+' mismatch')
    del records, summaries
    receipts = json.loads((args.run/'prediction_receipts.json').read_text())
    if [r['shot'] for r in receipts] != order: raise AssertionError('receipt IDs mismatch')
    warm = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in order[:20]]
    records = [{k:r[k] for k in scalar.SIGNAL_KEYS} for r in warm]
    summaries = [summarize(r, center=center, scale=scale) for r in records]
    maps = {}
    for name, (n, policy) in protocol['arms'].items():
        if policy == 'density_gain':
            m = scalar.fit_prefix_map(ss, records[:n], order=order, count=n, policy=policy)
        else:
            m = fit_maps(source, summaries[:n], order=order, count=n, policy=policy)
            m['stability'] = stability(source, summaries[:n], m)
        m['preceding_prediction_seals'] = {str(r['shot']):r['prediction_sha256'] for r in receipts[:n]}
        if m != json.loads((args.run/'maps'/f'{name}.json').read_text()): raise AssertionError(name+' map mismatch')
        maps[name] = m
    arms = ['static', *protocol['arms']]; rows = {name:[] for name in arms}; aggregate = {}
    # Proportionate end-to-end replay, declared independently of outcomes.
    # All 3960 records are re-scored; these 7*11 records are ALSO re-inferred.
    # Do not misdescribe sampled inference replay as full inference reproduction.
    replay_indices = {0, 9, 10, 19, 20, 179, 359}; replay_audit = []
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for matched numerical replay')
    torch.set_num_threads(2); torch.manual_seed(131)
    base, parent = load_parent(protocol['source_checkpoint'])
    if float(parent['threshold']) != threshold: raise AssertionError('threshold is not source threshold')
    model = PairedFeedback(base, np.ones(23)).cuda(); model.set_stage('frozen')
    for index, sid in enumerate(order):
        path = args.run/'shots'/f'{sid}.npz'
        if sha(path) != receipts[index]['prediction_sha256']: raise AssertionError('trace hash mismatch')
        active = sorted(k for k,(n,_) in protocol['arms'].items() if index >= n)
        if receipts[index]['active_maps'] != active: raise AssertionError('noncausal map use')
        raw = apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal'))
        signal = {k:raw[k] for k in ('values', 'available')}
        inputs = dict(static=signal)
        for n in (10, 20):
            if index>=n: inputs[f'density_gain_n{n}'] = scalar.transform(signal, maps[f'density_gain_n{n}'])
        for name in active:
            n, policy = protocol['arms'][name]
            if policy == 'density_gain': continue
            density = inputs[f'density_gain_n{n}']
            inputs[name], audit = transform(density, maps[name], center=center, scale=scale)
            if audit != receipts[index]['mapping_audit'][name]: raise AssertionError('input audit mismatch')
            for block, vals in audit.items():
                ag = aggregate.setdefault(name, {}).setdefault(block, {})
                for key, value in vals.items():
                    if type(value) is int: ag[key] = ag.get(key, 0)+value
        replay = predict_many(model, inputs, device='cuda') if index in replay_indices else None
        if replay is not None:
            for name in arms:
                if name not in replay: replay[name] = replay['static'].copy()
        d, down = io23.read_outcome('EAST', sid)
        valid = np.ones(len(raw['times']), bool) if not d else (down-raw['times'])*1000 >= -1e-6
        with np.load(path, allow_pickle=False) as trace:
            if set(trace.files) != {'times', *arms}: raise AssertionError('trace columns mismatch')
            if not np.array_equal(trace['times'], raw['times']): raise AssertionError('clock mismatch')
            for name in arms:
                q = trace[name]
                if replay is not None:
                    replay_audit.append(dict(shot=sid, arm=name, max_abs_error=compare_replay(q, replay[name], threshold)))
                if q.shape != raw['times'].shape or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
                    raise AssertionError('invalid risk trace')
                if name != 'static' and index < protocol['arms'][name][0]:
                    if not np.array_equal(q, trace['static']): raise AssertionError('warmup revised')
                crossings = np.flatnonzero((q>=threshold)&valid); alarm = bool(len(crossings))
                lead = float(round((down-raw['times'][crossings[0]])*1000., 9)) if d and alarm else None
                rows[name].append(dict(shot=sid, is_disrupt=bool(d), alarm=alarm, lead_ms=lead,
                                       timely=bool(lead is not None and lead>30.)))
    if aggregate != json.loads((args.run/'mapping_audit.json').read_text()): raise AssertionError('aggregate input audit')
    for name in arms:
        if rows[name] != declared[name]['predictions']: raise AssertionError('prediction metric mismatch')
        for part, subset in (('full360', rows[name]), ('common_suffix340', rows[name][20:])):
            independent = counts(subset)
            for key in ('tp','fp','n_disrupt','n_safe','tpr','fpr','J'):
                if independent[key] != done['metrics'][name][part][key]: raise AssertionError('metric mismatch')
    result = dict(status='VERIFIED', verified_prediction_records=len(order)*len(arms),
        source_and_target_statistics_recomputed=True, unchanged_source_sha256=protocol['source_sha256'],
        end_to_end_replay=dict(indices=sorted(replay_indices), records=replay_audit,
                              scope='sampled 77 complete shot-arm traces, not full3960 inference replay'),
        full360={name:counts(rs) for name,rs in rows.items()},
        common_suffix340={name:counts(rs[20:]) for name,rs in rows.items()},
        versus_density={name:flips(rows[f'density_gain_n{n}'][20:], rows[name][20:])
                        for name,(n,p) in protocol['arms'].items() if p!='density_gain'},
        independent_confirmation=False, mapping_audit=aggregate,
        interval_caveat='descriptive Wilson intervals; serial dependence and development selection remain')
    _durable_json(args.run/'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','verified_prediction_records','full360')}, indent=2))


if __name__ == '__main__': main()
