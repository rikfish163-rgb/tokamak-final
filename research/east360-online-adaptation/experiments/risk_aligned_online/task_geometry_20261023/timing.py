"""Posthoc temporal opportunity audit, not outcome-driven online scheduling."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ


def feedback_timing(times, down, counts):
    times = np.asarray(times); counts = np.asarray(counts,dtype=int)
    if (times.ndim != 1 or len(times)<2 or not np.isfinite(times).all() or
            not np.isfinite(down) or (np.diff(times)<=0).any() or
            (counts<1).any() or (counts>=len(times)).any()):
        raise ValueError('finite increasing actual time grid and valid update indices required')
    lead_first = np.round((down-times[counts])*1000,9)
    lead_last = np.round((down-times[counts-1])*1000,9)
    usable = lead_first > 30.
    positive = usable & (lead_last<=125.)
    return dict(median_sample_ms=float(np.round(np.median(np.diff(times))*1000,9)),
        median_update_ms=float(np.round(np.median(np.diff(times[counts]))*1000,9)) if len(counts)>1 else None,
        before_deadline=counts[usable].tolist(),within_positive_window_before_deadline=counts[positive].tolist(),
        last_observed_lead_ms_before_deadline=float(lead_last[usable][-1]) if usable.any() else None)


def main(root):
    p,full,verified = READ(root/'protocol.json'),READ(root/'per_shot.json'),READ(root/'verification.json')
    assert verified['status'] == 'VERIFIED' and verified['completed_sha256'] == sha(root/'completed.json')
    rows = []
    for r in full['static']['predictions']:
        if not r['is_disrupt']:
            continue
        sid = r['shot']; disruptive,down = io23.read_outcome('EAST',sid); assert disruptive
        updates = READ(root/'shots'/f'{sid}_updates.json')['importance']
        counts = [u['first_affected_prediction'] for u in updates if u['gradient_steps']]
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            timing = feedback_timing(z['times'],down,counts)
        category = 'timely' if r['timely'] else 'late' if r['alarm'] else 'never_alarm'
        rows.append(dict(shot=sid,static_category=category,**timing))
    summary = {}
    for category in ('all','timely','late','never_alarm'):
        group = rows if category == 'all' else [r for r in rows if r['static_category'] == category]
        summary[category] = dict(shots=len(group),no_update_with_positive_window_signal_before_deadline=
            sum(not r['within_positive_window_before_deadline'] for r in group))
    result = dict(scope='POSTHOC DATA-TIMESTAMP AUDIT; no method or schedule changed',
        code_sha256=sha(Path(__file__)),verification_sha256=sha(root/'verification.json'),summary=summary,shots=rows,
        median_sample_ms_values=sorted({r['median_sample_ms'] for r in rows}),
        median_update_ms_values=sorted({r['median_update_ms'] for r in rows if r['median_update_ms'] is not None}),
        warning='125ms denotes the training positive window, NOT a proven physical precursor onset; earlier signals may still aid adaptation.',
        opportunity_not_sufficiency='An update inside this window is neither necessary nor sufficient for correct early prediction.',
        excludes_compute_latency=True)
    _durable_json(root/'temporal_feedback_diagnostic.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'shots'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root',type=Path); main(p.parse_args().root)
