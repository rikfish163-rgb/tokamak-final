"""Read-only raw HDF5/NPZ count audit; adds one new receipt, never scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data
from .ended_decoder_experiment import ARMS, curve_error
from .io23 import read_outcome
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash


def classify(times, scores, valid, *, threshold, disruptive, down):
    eligible = np.asarray(valid, bool).copy()
    if disruptive:
        eligible &= times <= down
    fired = np.flatnonzero(eligible & (scores >= threshold))
    lead = round(float((down-times[fired[0]])*1000.), 9) if disruptive and len(fired) else None
    return dict(is_disrupt=disruptive, alarm=bool(len(fired)), lead_ms=lead,
                timely=bool(lead is not None and lead > 30.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = args.run/'independent_verification.json'
    if output.exists():
        parser.error('new verification receipt required')
    protocol = json.loads((args.run/'protocol.json').read_text())
    order = json.loads((args.run/'order.json').read_text())
    comparison = json.loads((args.run/'comparison.json').read_text())
    if (_hash(DEFAULT_PARENT) != PARENT_SHA256 or
            _hash(args.run/'code_snapshot.zip') != protocol['code_snapshot_sha256']):
        raise AssertionError('source or code archive identity changed')
    rows, seals = {}, 0
    for arm in ARMS:
        records = []
        for index, item in enumerate(order, 1):
            sid = item['shot']
            raw = simple_data.read_shot('EAST', sid, rate_contract='nominal')
            disruptive, down = read_outcome('EAST', sid)
            receipt = json.loads((args.run/arm/f'{index:04d}_{sid}_seal.json').read_text())
            audit = json.loads((args.run/arm/f'{index:04d}_{sid}_audit.json').read_text())
            if (audit['seal_sha256'] != receipt['sha256'] or
                    _hash(args.run/arm/receipt['npz']) != receipt['sha256']):
                raise AssertionError('whole-shot seal hash mismatch')
            with np.load(args.run/arm/receipt['npz'], allow_pickle=False) as trace:
                times, scores = trace['times'], trace['scores']
                if (int(trace['shot']) != sid or not np.array_equal(times, raw['times'])
                        or scores.shape != times.shape or not np.isfinite(scores).all()):
                    raise AssertionError('trace identity, time or score shape mismatch')
                # read_shot deliberately contains signals only, NOT valid/y
                # outcome fields. Safe records count their whole emitted grid;
                # disrupted records count only times through DownTime below.
                records.append(dict(shot=sid, **classify(times, scores, np.ones(len(times), bool),
                    threshold=protocol['threshold'], disruptive=disruptive, down=down)))
            if not audit['fast_discarded']:
                raise AssertionError('temporary FFN state retained')
            seals += 1
        rows[arm] = records
        for period, start, stop in (('all', 0, len(order)), ('first30', 0, 30), ('after30', 30, len(order))):
            selected = records[start:stop]
            counts = dict(tp=sum(r['timely'] for r in selected),
                fp=sum(r['alarm'] and not r['is_disrupt'] for r in selected),
                n_disrupt=sum(r['is_disrupt'] for r in selected),
                n_safe=sum(not r['is_disrupt'] for r in selected),
                early_over_3000=sum(r['timely'] and r['lead_ms'] > 3000. for r in selected))
            expected = comparison[arm][period]
            if any(counts[k] != expected[k] for k in counts):
                raise AssertionError('raw count differs from recorded metric')
            if counts['tp']/counts['n_disrupt'] != expected['tpr'] or counts['fp']/counts['n_safe'] != expected['fpr']:
                raise AssertionError('raw rate differs from recorded metric')
    pair_changes = {}
    for left, right in (('static', 'original_intra'), ('original_intra', 'signal_after30'),
                        ('original_intra', 'meta_after30'), ('signal_after30', 'meta_after30'),
                        ('static', 'meta_after30')):
        delta = {}
        for period, start in (('all', 0), ('after30', 30)):
            pairs = list(zip(rows[left][start:], rows[right][start:]))
            delta[period] = dict(
                rescued=[a['shot'] for a,b in pairs if a['is_disrupt'] and not a['timely'] and b['timely']],
                lost=[a['shot'] for a,b in pairs if a['is_disrupt'] and a['timely'] and not b['timely']],
                removed_fp=[a['shot'] for a,b in pairs if not a['is_disrupt'] and a['alarm'] and not b['alarm']],
                new_fp=[a['shot'] for a,b in pairs if not a['is_disrupt'] and not a['alarm'] and b['alarm']])
        pair_changes[f'{left}_to_{right}'] = delta
    trained = [json.loads((args.run/arm/'head_training.json').read_text())
               for arm in ('signal_after30', 'meta_after30', 'meta_static')]
    if any(t['fit_ids'] != [r['shot'] for r in order[:30]] or
           t['steps'] != len(t['records']) or t['steps'] != protocol['head_steps'] for t in trained):
        raise AssertionError('actual fit manifest or budget differs from declaration')
    heads = [torch.load(args.run/arm/'ended_decoder.pt', map_location='cpu', weights_only=True)['decoder']
             for arm in ('meta_after30', 'meta_static')]
    if any(not torch.equal(heads[0][key], heads[1][key]) for key in heads[0]):
        raise AssertionError('meta-static and meta-intra auxiliary heads differ')
    previous = Path(__file__).parent/'runs/ended_context120_v2_20261010'
    control_errors = {new: curve_error(args.run/new, previous/old, order)
                      for new,old in (('static', 'static'), ('original_intra', 'uniform'))}
    if any(control_errors.values()):
        raise AssertionError('archived baseline curve changed after extension hooks')
    _durable_json(output, dict(verified_seals=seals, counts_from_raw_hdf5_npz=True,
        paired_changes=pair_changes, archived_baseline_max_errors=control_errors,
        source_sha256_unchanged=True, identical_meta_auxiliary_heads=True,
        executed_head_steps=[len(t['records']) for t in trained],
        interpretation='single-parent development pilot; no independent or physical-law claim'))
    print(json.dumps(json.loads(output.read_text())), flush=True)


if __name__ == '__main__':
    main()
