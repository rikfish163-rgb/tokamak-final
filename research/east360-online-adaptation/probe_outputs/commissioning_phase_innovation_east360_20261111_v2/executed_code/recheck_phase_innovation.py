from collections import Counter, defaultdict
from pathlib import Path
import h5py
import numpy as np
from scipy.special import logit
from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write

out = Path('probe_outputs/commissioning_phase_innovation_east360_20261111_v2')
p, s, per, res, lead, h = (read(out / f) for f in (
    'protocol.json', 'seal.json', 'per_shot.json', 'result.json', 'lead_sensitivity.json', 'historical.json'))
guard = Path(p['previous_guard'])
gp, gs = (read(guard / f) for f in ('protocol.json', 'seal.json'))
parent = Path(gp['previous_teacher'])
pp, ps = (read(parent / f) for f in ('protocol.json', 'seal.json'))
assert sha(guard/'seal.json') == p['previous_seal']
assert sha(parent/'seal.json') == gp['previous_seal']
assert sha(out/'historical.json') == p['historical_sha256']
src = Path('experiments/risk_aligned_online')
for f, v in p['code_sha256'].items():
    snap = out/'executed_code'/f if f == 'commissioning_phase_innovation_20261111.py' else src/f
    assert sha(snap) == v
assert len(p['order']) == len(set(p['order'])) == len(s['curve_sha256']) == 360
assert len(s['prefix_witnesses']) == 3
assert p['persistent_updates'] == 0 and p['source_sha256'] == gp['source_sha256']
roles = io23.source_roles()
assert p['fit_roles'] == {'J-TEXT/train': roles['train'], 'EAST/cold': p['order'][:20]}
assert p['held_source'] == roles['val']
for role in ('train', 'cold', 'val'):
    allowed = roles[role] if role != 'cold' else p['order'][:20]
    assert len(h[role]) == len({r['shot'] for r in h[role]})
    for r in h[role]:
        assert r['role'] == role and r['shot'] in allowed
        assert p['horizon']+1 <= r['action']-r['origin'] < p['horizon']+1+p['stride']

counts = {b: {k: Counter() for k in p['arms']} for b in p['calibration']}
coverage = Counter()
forecast = defaultdict(list)
near_miss = []
new, original_parent = p['new'], 'legacy_current4'
for rank, sid in enumerate(p['order'], 1):
    name = f'{rank:04d}_{sid}.npz'
    npath, gpath, opath = (r/'shots'/name for r in (out, guard, parent))
    for f, a in ((npath, s), (gpath, gs), (opath, ps)):
        assert sha(f) == a['curve_sha256'][name]
    with h5py.File(io23.path_for('EAST', sid), 'r') as h5:
        flag = np.asarray(h5['meta/IsDisrupt'][()]).item()
        assert flag in (0, 1)
        d = bool(flag)
        down = float(np.asarray(h5['meta/DownTime'][()]).item()) if d else None
        ip = h5['data/ip']
        t = float(np.asarray(ip.attrs['StartTime']).item()) + np.arange(len(ip))/float(
            np.asarray(ip.attrs['SampleRate']).item())
    leads = np.round((down-t)*1000, 9) if d else None
    legal = (down-t)*1000 >= -1e-6 if d else np.ones(len(t), bool)
    with np.load(npath, allow_pickle=False) as n, np.load(gpath, allow_pickle=False) as g, np.load(
            opath, allow_pickle=False) as o:
        for f in (n, g, o):
            np.testing.assert_array_equal(t, f['times'])
        y = {k: o[k] for k in pp['arms']} | {k: g[k] for k in gp['new']} | {k: n[k] for k in new}
        assert set(y) == set(p['arms'])
        for band, tau in p['calibration'].items():
            tau = p['original'] if rank <= 20 else tau
            for k, prob in y.items():
                hit = np.flatnonzero(legal & (prob >= tau))
                first = float(leads[hit[0]]) if d and len(hit) else None
                row = dict(shot=sid, disruptive=d, down_s=down, alarm=bool(len(hit)), lead_ms=first,
                           timely=bool(first is not None and first > 30.))
                assert row == per[band][k][rank-1], (rank, band, k)
                c = counts[band][k]
                c.update(D=int(d), S=int(not d), FP=int(not d and bool(len(hit))))
                for limit in (15, 20, 25, 30):
                    c[str(limit)] += int(first is not None and first > limit)
                    c[str(limit)+'_bounded'] += int(first is not None and limit < first <= 3000.)
        if rank <= 20:
            for k in new:
                np.testing.assert_array_equal(y[k], o['fixed'])
            continue
        q, r = n['release'], n['radius']
        support, changed, error = n['supported'], n['changed'], n['forecast_mse']
        assert r.shape == (len(q), len(new)) and error.shape == (len(q), 3)
        np.testing.assert_array_equal(q, g['release'])
        assert ((q-p['first_effect'])%p['stride']==0).all()
        assert (q < len(t)).all() and (np.diff(q)>0).all()
        assert (q[support] >= p['first_effect'] + 2*p['stride']).all() and not changed[~support].any()
        np.testing.assert_allclose(r[:, :3], np.repeat(r[:, :1], 3, axis=1), atol=3e-7, rtol=1e-6)
        slot = np.searchsorted(q, np.arange(len(t)), side='right')
        for i, k in enumerate(new):
            unused = np.r_[True, r[:,i]==0.][slot]
            np.testing.assert_array_equal(y[k][unused], y[original_parent][unused])
        comparable = np.r_[True, ~changed][slot]
        np.testing.assert_allclose(y[new[0]][comparable], y[new[1]][comparable], atol=2e-6, rtol=0.)
        coverage.update(released=len(q), reference_ready=int(support.sum()), donors_changed=int(changed.sum()),
                        shots_with_changed=int(changed.any()), nonzero_updates=int((r[:,0]>0.).sum()))
        masks = {'all_ready': support, 'changed': changed,
                 'safe_ready': support if not d else np.zeros(len(q), bool)}
        if d:
            near = (leads > 30.) & (leads <= 125.)
            masks['near_disruption_ready'] = support & near[q]
            masks['near_disruption_changed'] = changed & near[q]
            if near.any() and not per['0.2'][original_parent][rank-1]['timely']:
                z = {k: logit(np.clip(y[k][near].astype(np.float64), 1e-12, 1.-1e-12))
                     for k in (original_parent, *new)}
                near_miss.append(dict(shot=sid, ready=bool((near[q] & support).any()),
                    changed=bool((near[q] & changed).any()), margin=float(z[original_parent].max()-logit(p['calibration']['0.2'])),
                    uplift={k: float((z[k]-z[original_parent]).max()) for k in new},
                    peak_vs_current=float(z[new[1]].max()-z[new[0]].max())))
        for name, mask in masks.items():
            if mask.any():
                forecast[name].append((int(mask.sum()), error[mask].mean(axis=0, dtype=np.float64)))

for b, groups in counts.items():
    for k, v in groups.items():
        assert v['D'] == v['S'] == 180
        assert (v['30'], v['FP']) == (res['metrics'][b][k]['tp'], res['metrics'][b][k]['fp'])
        for t in (15,20,25,30):
            l = lead['metrics'][b][k]['by_deadline'][str(t)]
            assert (l['tp'], l['tp_at_most_3000ms']) == (v[str(t)], v[str(t)+'_bounded'])
summary = {}
for name, blocks in forecast.items():
    size, values = zip(*blocks)
    x = np.stack(values)
    summary[name] = dict(shots=len(x), releases=sum(size), shot_weighted_mse=x.mean(0).tolist(),
                         window_weighted_mse=np.average(x, weights=size, axis=0).tolist(),
                         better_stage_shots=int((x[:,2]<x[:,1]).sum()),
                         worse_stage_shots=int((x[:,2]>x[:,1]).sum()))
diag = dict(shots=len(near_miss), ready=sum(r['ready'] for r in near_miss),
            changed=sum(r['changed'] for r in near_miss),
            median_gap=float(np.median([r['margin'] for r in near_miss])),
            median_uplift={k:float(np.median([r['uplift'][k] for r in near_miss])) for k in new},
            positive_peak_shift=sum(r['peak_vs_current']>1e-6 for r in near_miss))
outcome = dict(status='INDEPENDENT_PHASE_INNOVATION_HDF5_AUDIT_PASS',
    checked_records=360*len(p['arms'])*len(p['calibration']), coverage=dict(coverage),
    signature_forecast=summary, near_miss=diag, near_miss_shots=near_miss,
    deadlines={b:{k:{q:int(v[q]) for q in ('15','20','25','30','30_bounded','FP')}
                  for k,v in g.items()} for b,g in counts.items()})
write(out/'hdf5_recheck.json', outcome)
print({k:v for k,v in outcome.items() if k not in ('deadlines','near_miss_shots')}, flush=True)
