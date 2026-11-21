"""Quantify source/target support without fitting or changing any threshold."""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.history_gradient_metric_20261102.run import read, sha, write


out=Path('probe_outputs/commissioning_excursion_order_east360_20261111_v1')
p, seal, r = (read(out/f) for f in ('protocol.json','seal.json','result.json'))
for f in ('source_train.npz','east_windows.npz'):
    assert sha(out/f) == seal['assets'][f]

def rows(file, ids, device, minimum):
    advanced=io23.advanced_times() if device=='J-TEXT' else {}
    with np.load(out/file,allow_pickle=False) as f:
        shot, t, x = (f[k] for k in ('shot','time','match'))
    keep=np.zeros(len(shot),bool)
    lead=np.zeros(len(shot))
    for sid in ids:
        d,down=io23.read_outcome(device,sid)
        if d:
            h=io23.effective_horizon_ms(sid,advanced)[0] if device=='J-TEXT' else 125.
            m=shot==sid
            lead[m]=np.round((down-t[m])*1000,9)
            keep[m]=(lead[m]>minimum)&(lead[m]<=h+1e-6)
    return shot[keep], x[keep], lead[keep]

source, sx, lead = rows('source_train.npz', p['fit'], 'J-TEXT', 10.)
target, tx, _ = rows('east_windows.npz', p['east_order'][p['startup']:], 'EAST', 30.)
old = read(Path(p['reference'])/'per_shot.json')['0.2']['legacy_current4'][p['startup']:]
missed=np.isin(target,[m['shot'] for m in old if m['disruptive'] and not m['timely']])
assert int(missed.sum())==r['comparisons']['east_legacy_misses']['positive_windows']
assert len(tx)==r['comparisons']['east_suffix']['positive_windows']
keys=p['caliper_keys']
tol=np.array(p['calipers'])
answer={}
for name, include in (('source_official_gt10',np.ones(len(sx),bool)),('source_official_gt30',lead>30)):
    a=sx[include]
    info=dict(source_windows=len(a),source_shots=len(np.unique(source[include])))
    for view, cols in (('physical_and_amplitude',[1,2,3,4,5]),('without_absolute_Ip_Bt',[3,4,5])):
        d,_=cKDTree(a[:,cols]/tol[cols]).query(tx[:,cols]/tol[cols],p=np.inf)
        support=d<=1.+1e-12
        info[view]={part:dict(windows=int(mask.sum()),in_support=int(support[mask].sum()),
                              shots=len(np.unique(target[mask])),
                              shots_with_support=len(np.unique(target[mask & support])),
                              median_box_distance=float(np.median(d[mask])))
                    for part,mask in (('all_disruptive',np.ones(len(tx),bool)),('legacy_misses',missed))}
    info['marginal_outside_source_range_with_caliper']={
        keys[j]:int(((tx[:,j]<a[:,j].min()-tol[j])|(tx[:,j]>a[:,j].max()+tol[j])).sum())
        for j in range(1,len(keys))}
    answer[name]=info
write(out/'source_support_postseal.json',dict(scope='POSTSEAL_OVERLAP_NOT_ONLINE_POLICY',
    same_frozen_input_coordinates=True, diagnostic_only=True, report=answer))
print(answer,flush=True)
