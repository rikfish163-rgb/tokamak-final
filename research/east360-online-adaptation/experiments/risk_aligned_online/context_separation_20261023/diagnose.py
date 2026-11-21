"""Posthoc crossing diagnostics; never change the deployed alarm criterion."""
import argparse
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ, NAMES


def diagnose(root):
    verified = READ(root/'verification.json')
    assert verified['status'] == 'VERIFIED'
    assert verified['completed_sha256'] == sha(root/'completed.json')
    p = READ(root/'protocol.json'); threshold = p['threshold']
    seals = {r['shot']:r for r in READ(root/'prediction_receipts.json')}
    rows = []
    for sid in sorted(map(int, verified['changed_shot_details'])):
        path = root/'shots'/f'{sid}.npz'
        assert sha(path) == seals[sid]['prediction_sha256']
        disrupt, down = io23.read_outcome('EAST', sid)
        with np.load(path, allow_pickle=False) as z:
            times = z['times']
            valid = (down-times)*1000 >= -1e-6 if disrupt else np.ones(len(times),bool)
            timely = np.round((down-times)*1000,9)>30 if disrupt else valid
            result = dict(shot=sid,is_disrupt=disrupt,curves={})
            for a in NAMES:
                score = z[a]; hit = valid & (score>=threshold)
                crossings = np.flatnonzero(hit)
                first = int(crossings[0]) if len(crossings) else None
                peak = int(np.flatnonzero(timely)[np.argmax(score[timely])])
                run = 0
                if first is not None:
                    tail = hit[first:]; stops = np.flatnonzero(~tail)
                    run = int(stops[0]) if len(stops) else len(tail)
                result['curves'][a] = dict(first_index=first,
                    first_time_s=float(times[first]) if first is not None else None,
                    first_lead_ms=round(float((down-times[first])*1000),9) if disrupt and first is not None else None,
                    first_run_samples=run,
                    first_run_observed_span_ms=float((times[first+run-1]-times[first])*1000) if run else None,
                    timely_crossing_samples=int((hit & timely).sum()),
                    timely_peak=float(score[peak]),timely_peak_margin=float(score[peak])-threshold,
                    timely_peak_time_s=float(times[peak]))
            rows.append(result)
    result = dict(scope='Posthoc changed-shot diagnostics only, not a selection or new alarm rule',
        diagnostic_sha256=sha(Path(__file__)),verification_sha256=sha(root/'verification.json'),
        threshold=threshold,rows=rows,
        caveat='Consecutive samples are correlated; peak/episode duration is descriptive, not mechanistic evidence.')
    _durable_json(root/'changed_crossings.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    diagnose(args.root)
