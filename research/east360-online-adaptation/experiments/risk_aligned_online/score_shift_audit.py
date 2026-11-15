"""Offline paired diagnosis of sealed EAST scores; never trains or calibrates.

All public numeric time arguments below are milliseconds. HDF5 and sealed NPZ
times are seconds and are converted explicitly in main. Labels are accessed
only for retrospective evaluation of previously sealed predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import io23


def summarize_pair(times, baseline, candidate, *, disruptive, down, threshold):
    times, baseline, candidate = [np.asarray(x, dtype=np.float64)
                                  for x in (times, baseline, candidate)]
    if (times.ndim != 1 or not len(times) or baseline.shape != times.shape or
            candidate.shape != times.shape or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0) or not 0 < threshold <= 1):
        raise ValueError('invalid paired time series or threshold')
    for score in (baseline, candidate):
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError('invalid probability')
    if disruptive and (down is None or not np.isfinite(down)):
        raise ValueError('disruption time required')
    if disruptive:
        lead = np.round(down-times, 9)
        valid = times <= down
        groups = {'early_disruptive': valid & (lead > 125. + 1e-6),
                  'precursor': valid & (lead <= 125. + 1e-6),
                  'actionable_precursor': valid & (lead <= 125. + 1e-6) & (lead > 30.)}
    else:
        valid = np.ones(len(times), bool)
        groups = {'safe_shot': valid}
    result = {'is_disrupt': bool(disruptive), 'groups': {}}
    for name, mask in groups.items():
        if not mask.any():
            continue  # Empty intervals have no average; omitted, not reported as zero.
        before, after = baseline[mask], candidate[mask]
        result['groups'][name] = dict(
            points=int(mask.sum()), baseline_mean_probability=float(before.mean()),
            candidate_mean_probability=float(after.mean()),
            mean_probability_delta=float((after-before).mean()),
            baseline_exceedance_fraction=float(np.mean(before >= threshold)),
            candidate_exceedance_fraction=float(np.mean(after >= threshold)))
    for name, scores in [('baseline', baseline), ('candidate', candidate)]:
        alarm = np.flatnonzero(valid & (scores >= threshold))
        result[f'{name}_alarm'] = bool(len(alarm))
        result[f'{name}_timely'] = bool(disruptive and len(alarm) and lead[alarm[0]] > 30.)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_seals(directory):
    rows = {}
    receipts = sorted(Path(directory).glob('*_seal.json'))
    if not receipts:
        raise ValueError('no sealed predictions')
    for path in receipts:
        receipt = json.loads(path.read_text())
        sid = int(receipt['shot'])
        if sid in rows:
            raise ValueError('duplicate shot in sealed run')
        npz = path.parent / receipt['npz']
        if sha(npz) != receipt['sha256']:
            raise ValueError(f'prediction hash mismatch: {sid}')
        with np.load(npz, allow_pickle=False) as data:
            if int(data['shot']) != sid:
                raise ValueError('sealed shot identity mismatch')
            rows[sid] = dict(times=data['times'].copy(), scores=data['scores'].copy(),
                             threshold=float(receipt['threshold']), sha256=receipt['sha256'])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing audit')
    base, candidate = read_seals(args.baseline), read_seals(args.candidate)
    if set(base) != set(candidate):
        raise ValueError('paired evaluation requires identical shot sets')
    rows = []
    for sid in base:
        a, b = base[sid], candidate[sid]
        if not np.array_equal(a['times'], b['times']) or a['threshold'] != b['threshold']:
            raise ValueError('paired time axis or threshold differs')
        disruptive, down = io23.read_outcome('EAST', sid)
        result = summarize_pair(a['times']*1000., a['scores'], b['scores'],
                                disruptive=disruptive,
                                down=down*1000. if disruptive else None,
                                threshold=a['threshold'])
        rows.append(dict(shot=sid, **result))
    groups = {}
    for group in ('safe_shot', 'early_disruptive', 'precursor', 'actionable_precursor'):
        selected = [r['groups'][group] for r in rows if group in r['groups']]
        if selected:
            groups[group] = dict(shots=len(selected), points=sum(s['points'] for s in selected),
                                 **{key: float(np.mean([s[key] for s in selected]))
                                    for key in selected[0] if key != 'points'})
    flips = {}
    for disruptive, field in [(False, 'alarm'), (True, 'timely')]:
        selected = [r for r in rows if r['is_disrupt'] == disruptive]
        flips['disruptive' if disruptive else 'safe'] = {
            f'{int(before)}_to_{int(after)}': sum(
                r[f'baseline_{field}'] == before and r[f'candidate_{field}'] == after
                for r in selected)
            for before in (False, True) for after in (False, True)}
    report = dict(scope='RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC; no model or threshold updates',
                  aggregation='equal weight per shot within each nonempty group',
                  baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()),
                  code_sha256=sha(__file__), groups=groups, flips=flips, rows=rows,
                  seals={str(s): dict(baseline=base[s]['sha256'], candidate=candidate[s]['sha256'])
                         for s in base})
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(groups=groups, flips=flips), indent=2))


if __name__ == '__main__':
    main()
