"""Read-only independent score/count verifier for the completed prefix pilot.

Uses HDF5 outcomes and raw NPZ curves, not the training metric helper. Writes
one NEW sidecar; never modifies a curve, rule, protocol or existing summary.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .io23 import read_outcome
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    output = args.run/'independent_verification.json'
    if output.exists():
        p.error('new independent receipt required')
    protocol = json.loads((args.run/'protocol.json').read_text())
    order = json.loads((args.run/'order.json').read_text())
    comparison = json.loads((args.run/'comparison.json').read_text())
    training = json.loads((args.run/'learned_after30/rule_training.json').read_text())
    if not (training['steps'] == len(training['records']) == protocol['rule_steps']
            and training['fit_ids'] == [s['shot'] for s in order[:30]]):
        raise AssertionError('executed rule budget or fit identities differ from declaration')
    if digest(DEFAULT_PARENT) != PARENT_SHA256 or digest(args.run/'code_snapshot.zip') != protocol['code_snapshot_sha256']:
        raise AssertionError('source or code archive changed')
    rows = {}
    verified_seals = 0
    for arm in ('static', 'uniform', 'learned_after30'):
        records = []
        for index, item in enumerate(order, 1):
            sid = item['shot']
            receipt = json.loads((args.run/arm/f'{index:04d}_{sid}_seal.json').read_text())
            path = args.run/arm/receipt['npz']
            if digest(path) != receipt['sha256']:
                raise AssertionError('whole-shot seal hash mismatch')
            with np.load(path, allow_pickle=False) as trace:
                times, scores = trace['times'], trace['scores']
                if int(trace['shot']) != sid or times.shape != scores.shape or not np.isfinite(scores).all():
                    raise AssertionError('trace identity or shape mismatch')
                disrupt, down = read_outcome('EAST', sid)
                eligible = times <= down if disrupt else np.ones(len(times), bool)
                fired = np.flatnonzero(eligible & (scores >= protocol['threshold']))
                lead = round(float((down-times[fired[0]])*1000.), 9) if disrupt and len(fired) else None
                records.append(dict(shot=sid, is_disrupt=disrupt, alarm=bool(len(fired)),
                    timely=bool(lead is not None and lead>30.), lead_ms=lead))
            verified_seals += 1
        rows[arm] = records
        for period, a, b in (('all', 0, len(order)), ('first30', 0, 30), ('after30', 30, len(order))):
            selected = records[a:b]
            counts = dict(tp=sum(r['timely'] for r in selected),
                fp=sum(r['alarm'] and not r['is_disrupt'] for r in selected),
                n_disrupt=sum(r['is_disrupt'] for r in selected),
                n_safe=sum(not r['is_disrupt'] for r in selected))
            recorded = comparison[arm][period]
            if any(counts[k] != recorded[k] for k in counts):
                raise AssertionError('raw trace counts differ from summary')
            if counts['tp']/counts['n_disrupt'] != recorded['tpr'] or counts['fp']/counts['n_safe'] != recorded['fpr']:
                raise AssertionError('raw metric differs from summary')
    pairs = {}
    for parent, candidate in (('static', 'uniform'), ('uniform', 'learned_after30'), ('static', 'learned_after30')):
        deltas = {}
        for period, start in (('all', 0), ('after30', 30)):
            rescues, losses, removed_fp, new_fp = [], [], [], []
            for a, b in zip(rows[parent][start:], rows[candidate][start:]):
                if a['is_disrupt']:
                    if not a['timely'] and b['timely']: rescues.append(a['shot'])
                    if a['timely'] and not b['timely']: losses.append(a['shot'])
                else:
                    if a['alarm'] and not b['alarm']: removed_fp.append(a['shot'])
                    if not a['alarm'] and b['alarm']: new_fp.append(a['shot'])
            deltas[period] = dict(rescued=rescues, lost=losses,
                                 removed_fp=removed_fp, new_fp=new_fp)
        pairs[f'{parent}_to_{candidate}'] = deltas
    receipt = dict(verified_seals=verified_seals, counts_from_hdf5_and_raw_npz=True,
        source_sha256=PARENT_SHA256, declared_rule_steps=protocol['rule_steps'],
        executed_rule_steps=len(training['records']), paired_changes=pairs,
        rule_fit_ids=training['fit_ids'], interpretation='development pilot; no physical-cause or independent-validation claim')
    _durable_json(output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
