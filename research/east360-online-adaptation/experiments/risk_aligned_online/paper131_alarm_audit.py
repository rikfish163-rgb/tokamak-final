"""Read-only model audit: re-score sealed curves, never change online alarms.

The thesis specifies 10 points/1 ms in chapter 3 (0.1 ms sampling), but does
not restate the persistence rule in chapter 4 (1 ms sampling). All diagnostic
interpretations are reported, not selected for performance or called a replica.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import io23


RULES = {'single_point': {}, 'elapsed_1ms': {'duration_ms': 1.},
         'consecutive_10_points': {'points': 10}, 'elapsed_10ms': {'duration_ms': 10.}}


def alarm_endpoints(times, scores, threshold, *, points=1, duration_ms=None):
    """Return (run start, actual confirmation), or None; times are seconds."""
    t, p = np.asarray(times, dtype=float), np.asarray(scores, dtype=float)
    if (t.ndim != 1 or p.shape != t.shape or not np.isfinite(t).all()
            or not np.isfinite(p).all() or np.any(np.diff(t) <= 0)
            or np.any((p < 0) | (p > 1)) or not 0 < threshold <= 1):
        raise ValueError('invalid probability trace')
    if (type(points) is not int or points < 1 or
            (duration_ms is not None and
             (points != 1 or not np.isfinite(duration_ms) or duration_ms < 0))):
        raise ValueError('specify either positive point count or nonnegative duration')
    edges = np.diff(np.r_[False, p >= threshold, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    qualifies = (ends-starts+1 >= points if duration_ms is None else
                 np.round((t[ends]-t[starts])*1000., 9) >= duration_ms)
    if not qualifies.any():
        return None
    k = np.flatnonzero(qualifies)[0]
    start, end = int(starts[k]), int(ends[k])
    confirm = (start+points-1 if duration_ms is None else
               start+int(np.flatnonzero(np.round((t[start:end+1]-t[start])*1000., 9)
                                       >= duration_ms)[0]))
    return start, confirm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    nd = sum(r['is_disrupt'] for r in rows)
    ns = len(rows)-nd
    out = {}
    for name in RULES:
        fp = sum(not r['is_disrupt'] and r['rules'][name] is not None for r in rows)
        leads = [r['rules'][name] for r in rows
                 if r['is_disrupt'] and r['rules'][name] is not None]
        tp = sum(v['confirmation_lead_ms'] > 30. for v in leads)
        tp_backdated = sum(v['start_lead_ms'] > 30. for v in leads)
        out[name] = dict(n_disrupt=nd, n_safe=ns, tp=tp, fp=fp,
                         tpr=tp/nd, fpr=fp/ns,
                         tp_using_run_start_not_deployable=tp_backdated,
                         early_over_3s=sum(v['confirmation_lead_ms'] > 3000. for v in leads))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior audit')
    paired = json.loads(args.paired.read_text())
    cohort = json.loads(io23.EAST_COHORT.read_text())['test']
    local_ids = set(cohort['disruption_shots']) | set(cohort['non_disruption_shots'])
    assert len(local_ids) == 360
    result = dict(scope='RETROSPECTIVE DEVELOPMENT DIAGNOSTIC, not neural gain or paper replication',
                  model_updates=False, threshold_updates=False, gpu_used=False,
                  rules=RULES, alarm_time='confirmation, never retroactively at run start',
                  threshold=paired['threshold'], paired_sha256=sha(args.paired),
                  source_sha256=paired['source_sha256'], code_sha256=sha(__file__),
                  local360_manifest=str(io23.EAST_COHORT),
                  local360_manifest_sha256=sha(io23.EAST_COHORT),
                  local360_warning='local HDF5-authoritative 180+180 subset, not certified thesis shot identities',
                  arms={})
    for arm, directory in paired['runs'].items():
        rows = []
        for receipt_path in sorted(Path(directory).glob('*_seal.json')):
            receipt = json.loads(receipt_path.read_text())
            path = receipt_path.parent / receipt['npz']
            assert sha(path) == receipt['sha256']
            assert receipt['threshold'] == paired['threshold']
            sid = int(receipt['shot'])
            disruptive, down = io23.read_outcome('EAST', sid)
            with np.load(path, allow_pickle=False) as sealed:
                assert int(sealed['shot']) == sid
                t, p = sealed['times'].copy(), sealed['scores'].copy()
            if disruptive:
                p, t = p[t <= down], t[t <= down]
            assert len(t) > 0
            if len(t) > 1:
                assert np.allclose(np.diff(t), np.median(np.diff(t)), rtol=0, atol=1e-9)
            rules = {}
            for name, kwargs in RULES.items():
                alarm = alarm_endpoints(t, p, paired['threshold'], **kwargs)
                if alarm is None:
                    rules[name] = None
                else:
                    start, confirm = alarm
                    rules[name] = dict(start_time_s=float(t[start]), confirmation_time_s=float(t[confirm]),
                        start_lead_ms=float(round((down-t[start])*1000., 9)) if disruptive else None,
                        confirmation_lead_ms=float(round((down-t[confirm])*1000., 9)) if disruptive else None)
            rows.append(dict(shot=sid, is_disrupt=disruptive, npz_sha256=receipt['sha256'], rules=rules))
        assert len(rows) == len({r['shot'] for r in rows}) == paired['expected_count']
        assert local_ids <= {r['shot'] for r in rows}
        overall = summarize(rows)
        original = paired['segments']['overall'][arm]
        assert all(overall['single_point'][k] == original[k] for k in ('tp', 'fp', 'n_disrupt', 'n_safe'))
        subset = [r for r in rows if r['shot'] in local_ids]
        assert sum(r['is_disrupt'] for r in subset) == 180
        result['arms'][arm] = dict(overall=overall, local360=summarize(subset), rows=rows)
        print(json.dumps(dict(arm=arm, overall=overall, local360=summarize(subset))), flush=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
