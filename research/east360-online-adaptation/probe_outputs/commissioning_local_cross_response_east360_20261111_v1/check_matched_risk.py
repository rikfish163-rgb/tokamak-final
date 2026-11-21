"""Retrospective risk diagnostic; no weights or alarm thresholds are fitted."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.risk_aligned_online import io23


LOCAL = Path(__file__).resolve().parent
OUTPUTS = LOCAL.parent
FIXED = OUTPUTS/'commissioning_cross_response_east360_20261111_v1'
ALARM = OUTPUTS/'commissioning_direction_east360_20261111_v1'
CALIPER = np.array([.5, .5, .125, .5, np.log(2.)])
FLOOR = 1.e-6


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(d, s, shot_d, shot_s, columns):
    if not len(d):
        return dict(pairs=0, disruptive_shots=0, safe_shots=0)
    win = (d > s) + .5*(d == s)
    per = np.stack([win[shot_d == i].mean(0) for i in np.unique(shot_d)])
    return dict(pairs=len(d), disruptive_shots=len(per), safe_shots=len(np.unique(shot_s)),
                concordance=dict(zip(columns, per.mean(0).tolist(), strict=True)))


def run():
    refs = {}
    for root, name in ((LOCAL,'local_error_seal.npz'), (FIXED,'issued_error_seal.npz')):
        refs[str(root/name)] = sha(root/name)
        assert refs[str(root/name)] == json.loads((root/'result.json').read_text())['score_seal_sha256']
    p = json.loads((LOCAL/'protocol.json').read_text())
    order, startup = p['order'], p['startup']
    a = json.loads((ALARM/'protocol.json').read_text())
    alarm_seal = json.loads((ALARM/'seal.json').read_text())['curve_sha256']
    assert a['order'] == order and a['startup'] == startup
    fpr = json.loads((ALARM/'per_shot.json').read_text())['0.2']
    by_old = {r['shot']: r for r in fpr['legacy_current4']}
    d = 'magnetic_to_motion_'
    with np.load(LOCAL/'local_error_seal.npz', allow_pickle=False) as local:
        shot, q, release, error, active = (local[d+k] for k in
                                            ('shot','q','release','huber','update_before_issue'))
    with np.load(FIXED/'issued_error_seal.npz', allow_pickle=False) as static:
        np.testing.assert_array_equal(shot,static[d+'shot'])
        np.testing.assert_array_equal(q,static[d+'q'])
        ctx = static[d+'context']
    cutoffs = np.empty(len(shot),dtype=np.float64)
    cutoffs.fill(np.nan)
    innovation = np.log(error[:,1:]+FLOOR)-np.log(error[:,[0]]+FLOOR)
    relative = np.full(len(shot),np.nan)
    for rank,sid in enumerate(order,1):
        if rank <= startup:
            continue
        ix = np.flatnonzero(shot==sid)
        name = f'{rank:04d}_{sid}.npz'
        curve = ALARM/'shots'/name
        assert sha(curve)==alarm_seal[name]
        with np.load(curve,allow_pickle=False) as data:
            np.testing.assert_array_equal(data['times'][q[ix]+26],release[ix])
            prob = data['fixed'][q[ix]+26].astype(np.float64).clip(1.e-6,1.-1.e-6)
        cutoffs[ix] = np.log(prob)-np.log1p(-prob)
        for j in range(4,len(ix)):
            earlier = ix[j-4:j]
            if active[earlier,0].all():
                relative[ix[j]] = innovation[ix[j],0]-np.median(innovation[earlier,0])
    # Freeze all evaluated scores before reading raw outcomes for matching.
    columns = ('own_relative_error','own_recent_surprise','joint_relative_error',
               'lagged_source_relative_error','delayed_feedback_relative_error',
               'own_minus_joint','fixed_logit','persistence_amplitude')
    amp = np.log(error[:,0]+FLOOR)
    scores = np.column_stack((innovation[:,0],relative,innovation[:,1:],
                              innovation[:,0]-innovation[:,1],cutoffs,amp))
    covariates = np.column_stack((ctx,cutoffs,amp))
    available = np.isfinite(scores).all(1) & active[:,0]
    safe = np.zeros(len(shot),bool)
    early = safe.copy()
    before_match = 0
    for sid in order[startup:]:
        ix = np.flatnonzero(shot==sid)
        with h5py.File(io23.path_for('EAST',sid),'r') as f:
            y = bool(np.asarray(f['meta/IsDisrupt'][()]).item())
            down = float(np.asarray(f['meta/DownTime'][()]).item()) if y else None
            ip = f['data/ip']
            rate = float(np.asarray(ip.attrs['SampleRate']).item())
            start = float(np.asarray(ip.attrs['StartTime']).item())
        np.testing.assert_array_equal(release[ix], start+(q[ix]+26)/rate)
        assert y == by_old[sid]['disruptive']
        if y:
            lead = (down-release[ix])*1000
            early[ix] = (lead > 30.+1.e-6) & (lead <= 125.+1.e-6)
            before_match += int(early[ix].sum())
        else:
            safe[ix] = True
    assert before_match == 511
    d_rows, s_rows, distances = [], [], []
    safe_ix = np.flatnonzero(safe&available)
    for r in np.flatnonzero(early&available):
        scaled = (covariates[safe_ix]-covariates[r])/CALIPER
        match = np.flatnonzero(np.max(abs(scaled),axis=1)<=1.)
        if len(match):
            nearest = match[np.argmin(np.square(scaled[match]).sum(1))]
            d_rows.append(r)
            s_rows.append(safe_ix[nearest])
            distances.append(float(np.linalg.norm(scaled[nearest])))
    d_rows, s_rows = np.asarray(d_rows,dtype=int), np.asarray(s_rows,dtype=int)
    missed = np.array([not by_old[int(s)]['timely'] for s in shot[d_rows]])
    result = dict(
        status='CONDITIONAL_RISK_ASSOCIATION_DIAGNOSTIC_NOT_DEPLOYABLE',
        hashes=refs, cutoff_logit_caliper=.5, activity_ratio_caliper=2.,
        context_calipers=CALIPER[:3].tolist(), safe_matches_can_repeat=True,
        forecast_origin_and_release_clock_audited=bool(before_match),
        all_timely_windows=before_match,
        available_timely_windows=int((early&available).sum()),
        old_missed_disruptive_shots=int(sum(r['disruptive'] and not r['timely']
                                           for r in by_old.values() if r['shot'] in order[startup:])),
        median_normalized_match_distance=float(np.median(distances)) if distances else None,
        all=stats(scores[d_rows],scores[s_rows],shot[d_rows],shot[s_rows],columns),
        old_misses=stats(scores[d_rows][missed],scores[s_rows][missed],
                         shot[d_rows][missed],shot[s_rows][missed],columns))
    with (LOCAL/'conditional_risk_diagnostic.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(result)


if __name__ == '__main__':
    run()
