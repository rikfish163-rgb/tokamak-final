"""Independent alarm reconstruction for the aligned in-shot experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from .alignment import SIGNAL_KEYS, fit_prefix_map
from .audit import counts, flips


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    output = args.run/'verification.json'
    if output.exists(): raise FileExistsError(output)
    protocol = json.loads((args.run/'protocol.json').read_text())
    done = json.loads((args.run/'completed.json').read_text())
    original = json.loads((args.run/'per_shot.json').read_text())
    receipts = json.loads((args.run/'prediction_receipts.json').read_text())
    contract = load_test_contract('EAST'); order = contract['shot_ids']
    if protocol['contract'] != contract or [r['shot'] for r in receipts] != order:
        raise AssertionError('fixed test identity mismatch')
    for filename, expected in protocol['code_sha256'].items():
        if sha(io23.ROOT/filename) != expected: raise AssertionError(f'code changed: {filename}')
    for pathkey, hashkey in (('source_checkpoint','source_sha256'), ('checkpoint','checkpoint_sha256')):
        if sha(protocol[pathkey]) != protocol[hashkey]: raise AssertionError('checkpoint changed')
    reference = Path(protocol['alignment_reference'])
    if sha(reference/'verification.json') != protocol['alignment_verification_sha256']:
        raise AssertionError('alignment evidence changed')
    if sha(reference/'source_statistics.json') != protocol['source_statistics_sha256']:
        raise AssertionError('source statistics changed')
    source = json.loads((reference/'source_statistics.json').read_text())
    records = [apply_input_contract(simple_data.read_shot('EAST', sid, rate_contract='nominal')) for sid in order[:20]]
    fitted = fit_prefix_map(source, [{k:r[k] for k in SIGNAL_KEYS} for r in records],
                            order=order, count=20, policy='density_gain')
    fitted['preceding_prediction_seals'] = {str(r['shot']):r['prediction_sha256'] for r in receipts[:20]}
    if fitted != json.loads((args.run/'mapping.json').read_text()): raise AssertionError('prefix fit mismatch')
    rows = {k:[] for k in ('static','intra')}; early_fp={k:[] for k in rows}; distances=[]; updates=0
    for index, sid in enumerate(order):
        file = args.run/'shots'/f'{sid}.npz'; infofile=args.run/'shots'/f'{sid}_updates.json'
        receipt=receipts[index]
        if sha(file) != receipt['prediction_sha256'] or sha(infofile) != receipt['update_sha256']:
            raise AssertionError('sealed evidence changed')
        if receipt['mapped'] != (index >= 20): raise AssertionError('noncausal mapping activation')
        info = json.loads(infofile.read_text()); updates += len(info['updates'])
        if info['inter_updates'] != 0 or len(info['updates']) != receipt['update_count']:
            raise AssertionError('update count mismatch')
        for u in info['updates']:
            if u['last_observation'] < 127 or u['first_affected_prediction'] != u['last_observation']+1:
                raise AssertionError('noncausal update')
            if u['steps'] not in (0,2) or u['weights'] != [1.]*23:
                raise AssertionError('uniform optimizer control changed')
            distances.append(u['delta'])
        raw = simple_data.read_shot('EAST',sid,rate_contract='nominal')
        d, down = io23.read_outcome('EAST',sid)
        valid = np.ones(len(raw['times']),bool) if not d else (down-raw['times'])*1000 >= -1e-6
        with np.load(file,allow_pickle=False) as trace, np.load(reference/'shots'/f'{sid}.npz',allow_pickle=False) as ref:
            if set(trace.files) != {'times','static','intra'} or not np.array_equal(trace['times'],raw['times']):
                raise AssertionError('incomplete or shifted observations')
            if not np.array_equal(trace['static'],ref['density_gain_n20']): raise AssertionError('static mismatch')
            if not np.array_equal(trace['static'][:128],trace['intra'][:128]): raise AssertionError('past rewritten')
            for arm in rows:
                q=trace[arm]
                if q.shape!=raw['times'].shape or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
                    raise AssertionError('invalid probability')
                cross=np.flatnonzero((q>=protocol['threshold'])&valid); alarm=bool(len(cross))
                lead=float(round((down-raw['times'][cross[0]])*1000,9)) if d and alarm else None
                rows[arm].append(dict(shot=sid,is_disrupt=bool(d),alarm=alarm,lead_ms=lead,
                                      timely=bool(lead is not None and lead>30.)))
                if not d and alarm and cross[0]<128: early_fp[arm].append(sid)
    if updates != done['recorded_updates']: raise AssertionError('update count changed')
    for arm in rows:
        if rows[arm] != original[arm]['predictions']: raise AssertionError('per-shot alarm mismatch')
        for subset, values in (('full360',rows[arm]),('common_suffix340',rows[arm][20:]),('warmup20',rows[arm][:20])):
            for k,v in counts(values).items():
                if k.endswith('_wilson95'): continue
                if done['metrics'][arm][subset][k] != v: raise AssertionError('aggregate mismatch')
    result=dict(status='VERIFIED',protocol_sha256=sha(args.run/'protocol.json'),
        completed_sha256=sha(args.run/'completed.json'),verified_predictions=len(order)*2,
        full360={k:dict(**counts(v),**flips(rows['static'],v)) for k,v in rows.items()},
        common_suffix340={k:dict(**counts(v[20:]),**flips(rows['static'][20:],v[20:])) for k,v in rows.items()},
        recorded_updates=updates,update_delta_quantiles=np.quantile(distances,[0,.25,.5,.75,1.]).tolist(),
        static_traces_bitwise_identical=True,target_statistics_recomputed_from_exact_ended_prefix=True,
        false_alarms_before_first_update=early_fp,inter_updates=0,independent_confirmation=False,
        uncertainty='descriptive Wilson intervals; development-selected mapping; no independent confirmation')
    _durable_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
