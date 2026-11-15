"""Bound alarm outcomes for pointwise selection among the already sealed EAST curves."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .commissioning_frozen_extension_20261113 import group
from .history_gradient_metric_20261102.run import read, sha, write


def run(folder, output):
    p,s,r = (read(folder/f'{k}.json') for k in ('protocol','seal','result'))
    if r['status']!='COMPLETE_FROZEN_EXTENDED_EAST_DEVELOPMENT' or len(s['curve_sha256'])!=len(p['order']):
        raise ValueError('complete extension required')
    per = read(folder/'per_shot.json')
    menus = (('current_5','earlier_5'),('bias_only','current_5','earlier_5'),tuple(p['arms']))
    counts = {b:[Counter() for _ in menus] for b in ('0.2','0.15')}
    for i,row in enumerate(p['order'],1):
        sid = row['shot']
        name = f'{i:04d}_{sid}.npz'
        path = folder/'shots'/name
        if sha(path)!=s['curve_sha256'][name]:
            raise ValueError('changed sealed curve')
        y,d,t = group.hdf._raw_truth(sid)
        with np.load(path,allow_pickle=False) as f:
            np.testing.assert_array_equal(t,f['times'])
            data = {k:f[k] for k in p['arms']}
        timely = np.round((d-t)*1000.,9)>30. if y else None
        for j,arm in enumerate(menus):
            lower = np.minimum.reduce([data[k] for k in arm])
            upper = np.maximum.reduce([data[k] for k in arm])
            for b,c in counts.items():
                tau = p['calibration'][b]
                old = per[b]['current_5'][i-1]
                if (old['shot'],old['disruptive'],old['down_s'])!=(sid,y,d):
                    raise ValueError('archived score and original HDF5 disagree')
                if y:
                    hit = bool(((upper>=tau)&timely).any())
                    c[j]['n_disruptive'] += 1
                    c[j]['timely_upper'] += hit
                    c[j]['beyond_current_5'] += bool(hit and not old['timely'])
                else:
                    hit = bool((lower>=tau).any())
                    c[j]['n_safe'] += 1
                    c[j]['false_alarm_lower'] += hit
                    c[j]['false_alarm_lower_old835' if row['old835'] else 'false_alarm_lower_other'] += hit
    write(output,dict(scope='POSTSEAL_PER_ENDPOINT_MENU_BOUNDS_NOT_A_POLICY',menu=menus,
        threshold={k:p['calibration'][k] for k in counts},bounds=counts,
        seal_sha256=sha(folder/'seal.json')))
    print(counts,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('extension',type=Path)
    p.add_argument('out',type=Path)
    a = p.parse_args()
    run(a.extension.resolve(),a.out.resolve())
