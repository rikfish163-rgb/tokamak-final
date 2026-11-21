"""Read-only dataset/existing-prediction audit; never changes evaluation masks."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import io23, simple_data, simple_train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(scope='offline descriptive diagnostic; no training, no shortened evaluation',
                  code_sha256=simple_train.sha(__file__), rows=[], summary={})
    for domain, ids in [('J-TEXT', simple_data.source_roles()['val']),
                        ('EAST', io23.cohort_ids('EAST'))]:
        advanced = simple_data.advanced_times() if domain == 'J-TEXT' else {}
        for sid in ids:
            s = simple_data.attach_outcome(simple_data.read_shot(domain, sid), domain, advanced)
            row = dict(domain=domain, shot=int(sid), disruptive=bool(s['is_disrupt']),
                       samples=len(s['values']), valid_samples=int(s['valid'].sum()),
                       positive_samples=int(s['y'].sum()), recorded_start=float(s['times'][0]))
            if domain == 'EAST' and not row['disruptive']:
                matches = list(args.static_dir.glob(f'*_{sid}_seal.json'))
                if len(matches) != 1:
                    raise ValueError(f'expected one seal for {sid}')
                receipt = json.loads(matches[0].read_text())
                path = args.static_dir / receipt['npz']
                if simple_train.sha(path) != receipt['sha256']:
                    raise ValueError(f'seal hash mismatch {sid}')
                with np.load(path) as z:
                    if int(z['shot']) != sid or not np.array_equal(z['times'], s['times']):
                        raise ValueError(f'prediction/data mismatch {sid}')
                    scores = z['scores']
                    alarm = np.flatnonzero(scores >= receipt['threshold'])
                    edges = np.diff(np.r_[False, scores >= receipt['threshold'], False].astype(int))
                    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                    row.update(first_alarm_sample=int(alarm[0]) if len(alarm) else None,
                               longest_exceeding_run=int(lengths.max()) if len(lengths) else 0,
                               exceeding_fraction=float(np.mean(scores >= receipt['threshold'])),
                               prediction_sha256=receipt['sha256'])
            result['rows'].append(row)
        for label in (False, True):
            rows = [r for r in result['rows'] if r['domain'] == domain and r['disruptive'] == label]
            result['summary'][f'{domain}_{label}'] = dict(n=len(rows),
                sample_quantiles=np.quantile([r['samples'] for r in rows], [0,.25,.5,.75,1]).tolist(),
                positive_quantiles=np.quantile([r['positive_samples'] for r in rows], [0,.25,.5,.75,1]).tolist())
    safe = [r for r in result['rows'] if r['domain'] == 'EAST' and not r['disruptive']]
    alarms = [r['first_alarm_sample'] for r in safe if r['first_alarm_sample'] is not None]
    result['summary']['east_safe_errors'] = dict(n=len(safe), fp=len(alarms),
        within_first128_samples=sum(t < 128 for t in alarms),
        within_first300_samples=sum(t < 300 for t in alarms),
        within_first1000_samples=sum(t < 1000 for t in alarms),
        continuous_exceedance_counts={str(n): sum(r['longest_exceeding_run'] >= n for r in safe)
                                      for n in (1, 5, 10, 25, 50, 100, 300, 1000)},
        mean_exceeding_fraction=float(np.mean([r['exceeding_fraction'] for r in safe])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
