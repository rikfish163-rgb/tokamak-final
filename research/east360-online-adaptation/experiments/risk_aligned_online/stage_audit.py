"""Descriptive stages of the existing paired stream; no training or tuning.

Stage boundaries are indices, NOT physical operating-regime labels. Outcomes
are retrospective evaluation data. No hypothesis test or independent-validation
claim is made for this repeatedly developed stream.
"""
import argparse
import json
import math
from pathlib import Path

from .score_shift_audit import sha


def analyze_rows(rows, threshold, ends):
    if set(rows) != {'static', 'fixed'} or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('paired static/fixed and probability threshold required')
    n = len(rows['static'])
    if not n or len(rows['fixed']) != n:
        raise ValueError('nonempty complete pair required')
    previous = -1
    for a, b in zip(rows['static'], rows['fixed']):
        if (type(a['shot']) is not int or a['shot'] <= previous or a['shot'] != b['shot']
                or type(a['disruptive']) is not bool or type(b['disruptive']) is not bool
                or a['disruptive'] != b['disruptive']):
            raise ValueError('identical increasing shot order and outcomes required')
        previous = a['shot']
        for row in (a, b):
            peak = row['peak']
            if peak is None:
                if not row['disruptive']:
                    raise ValueError('safe shot must have a peak')
            elif not math.isfinite(peak) or not 0 <= peak <= 1:
                raise ValueError('finite probability required')
    if (not ends or any(type(e) is not int for e in ends) or ends[-1] != n
            or any(b <= a for a, b in zip([0] + ends[:-1], ends))):
        raise ValueError('increasing inclusive ends must partition the entire stream')

    def outcome(row):
        return row['peak'] is not None and row['peak'] >= threshold

    def segment(start, stop):
        out = dict(indices=[start + 1, stop],
                   shot_bounds=[rows['static'][start]['shot'], rows['static'][stop-1]['shot']])
        for arm in rows:
            selected = rows[arm][start:stop]
            nd = sum(r['disruptive'] for r in selected)
            ns = len(selected) - nd
            tp = sum(r['disruptive'] and outcome(r) for r in selected)
            fp = sum(not r['disruptive'] and outcome(r) for r in selected)
            tpr, fpr = tp / nd if nd else None, fp / ns if ns else None
            out[arm] = dict(n_disrupt=nd, n_safe=ns, tp=tp, fp=fp,
                            tpr=tpr, fpr=fpr, J=tpr-fpr if nd and ns else None)
        flips = dict(tp_gained=0, tp_lost=0, fp_removed=0, fp_added=0)
        for a, b in zip(rows['static'][start:stop], rows['fixed'][start:stop]):
            old, new = outcome(a), outcome(b)
            if old != new:
                key = ('tp_gained' if new else 'tp_lost') if a['disruptive'] else (
                    'fp_added' if new else 'fp_removed')
                flips[key] += 1
        out['flips'] = flips
        return out

    stages = [segment(start, stop) for start, stop in zip([0] + ends[:-1], ends)]
    overall = segment(0, n)
    for arm in rows:
        for key in ('n_disrupt', 'n_safe', 'tp', 'fp'):
            if sum(s[arm][key] for s in stages) != overall[arm][key]:
                raise AssertionError('stage accounting does not conserve counts')
    return dict(overall=overall, stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operating-point', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous diagnostic')
    op = json.loads(args.operating_point.read_text())
    paired = json.loads(args.paired.read_text())
    if (op['paired_sha256'] != sha(args.paired)
            or not paired['verification']['full2953_verified']
            or op['source_sha256'] != paired['source_sha256']
            or len(op['rows']['static']) != 2953):
        raise ValueError('verified completed full2953 provenance required')
    receipts_checked = 0
    for arm, path in paired['runs'].items():
        receipts = [json.loads(p.read_text()) for p in Path(path).glob('*_seal.json')]
        recorded = {int(r['shot']) for r in receipts}
        indexed = {r['shot']: r for r in op['rows'][arm]}
        if recorded != set(indexed) or len(receipts) != len(indexed):
            raise ValueError('sealed shot roster no longer matches')
        # Receipt identities/hashes tie this reaggregation to the prior full
        # curve audit. NPZ bytes are NOT all re-read in this lightweight audit.
        for receipt in receipts:
            row = indexed[int(receipt['shot'])]
            if receipt['sha256'] != row['sealed_sha256'] or receipt['threshold'] != paired['threshold']:
                raise ValueError('sealed receipt changed')
            receipts_checked += 1
    report = analyze_rows(op['rows'], paired['threshold'], [30, 120, 600, 1500, 2953])
    for arm in ('static', 'fixed'):
        for key in ('n_disrupt', 'n_safe', 'tp', 'fp', 'tpr', 'fpr', 'J'):
            if report['overall'][arm][key] != paired['segments']['overall'][arm][key]:
                raise ValueError('full-stream counts disagree with previous verified audit')
    report.update(scope='RETROSPECTIVE_DEVELOPMENT_DESCRIPTIVE_STAGE_AUDIT',
        threshold=paired['threshold'], parent_sha256=paired['source_sha256'],
        inputs={str(args.paired): sha(args.paired), str(args.operating_point): sha(args.operating_point)},
        code_sha256=sha(__file__), receipts_checked=receipts_checked, npz_bytes_rehashed_this_run=False,
        source_weights_changed=False, new_predictions=False, threshold_selection=False,
        ordering='ascending shot ID; not verified collection time or low-to-high hazard',
        interpretation='No inter-shot state in either arm. Stage differences do not establish '
                       'catastrophic forgetting, operating-regime cause, or learning over shots.')
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
