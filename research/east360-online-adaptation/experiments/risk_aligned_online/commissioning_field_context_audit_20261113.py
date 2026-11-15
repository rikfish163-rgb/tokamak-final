"""Describe other observed-input differences across the raw EAST field strata."""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from . import commissioning_hidden_background_20261113 as bg
from . import io23
from .causal_input import CIII_MISSING_ATOL, CIII_MISSING_VALUE
from .commissioning_extension_order_budget_20261113 import PREFIX
from .history_gradient_metric_20261102.run import read, sha, write


def run(output):
    root = bg.PARENT
    balance = read(root/'postseal_bt_scale_balance.json')
    seal = sha(root/'seal.json')
    p = read(root/'protocol.json')
    if balance['seal_sha256']!=seal or len(p['order'])!=2953:
        raise ValueError('sealed input-strata audit required')
    s = read(root/'per_shot.json')['0.2']
    cold = p['startup']
    a = read(PREFIX/'order.json')
    v = {r['shot']:r['initial_abs_Bt_raw'] for r in a}
    if sha(PREFIX/'order.json')!=read(PREFIX/'roster.json')['order_sha256']:
        raise ValueError('raw prefix reference changed')
    fields = ('CIII','Bt','ip','qa_proxy')
    raw = defaultdict(list)
    source = []
    for sid in io23.source_roles()['train']:
        with h5py.File(io23.path_for('J-TEXT',sid),'r') as f:
            source.append(float(np.median(abs(f['data/Bt'][:100]))))
    for i,row in enumerate(p['order']):
        sid = row['shot']
        low = v[sid]<balance['raw_split']
        with h5py.File(io23.path_for('EAST',sid),'r') as f:
            c = f['data/CIII'][:]
            mask = np.isfinite(c)&~np.isclose(c,CIII_MISSING_VALUE,atol=CIII_MISSING_ATOL,rtol=0.)
            avail = 'all' if mask.all() else 'partial' if mask.any() else 'none'
            bt,ip,q = (np.asarray(f[f'data/{k}'][:128],dtype=np.float64) for k in fields[1:])
        with np.errstate(divide='ignore',invalid='ignore'):
            ratios = q*ip/bt
        ratio = float(np.median(ratios[np.isfinite(ratios)]))
        label = s['fixed'][i]['disruptive']
        calls = {}
        for a in ('fixed','current_5','current_no_states'):
            z = s[a][i]
            if (sid,label)!=(z['shot'],z['disruptive']):
                raise ValueError('scored shot identity changed')
            calls[a] = z['timely' if label else 'alarm']
        co = row['cohort'] if i>=cold else 'cold20'
        raw[f'{co}/{"smaller" if low else "larger"}/{avail}'].append((sid,label,ratio,calls))
    table = {}
    for k,data in raw.items():
        table[k] = dict(n_disruptive=sum(r[1] for r in data),n_safe=sum(not r[1] for r in data),
                        ratio_p10_p50_p90=np.quantile([r[2] for r in data],[.1,.5,.9]).tolist(),
                        arms={a:[sum(r[1] and r[3][a] for r in data),
                                 sum(not r[1] and r[3][a] for r in data)] for a in data[0][3]})
    write(output,dict(scope='POSTSEAL_RAW_FIELD_AND_CIII_AVAILABILITY',bands=table,seal_sha256=seal,
        ciii_contract={'missing_value':CIII_MISSING_VALUE,'atol':CIII_MISSING_ATOL},
        ratio_field='median over available first 128 points of qa_proxy * Ip / Bt',
        source_train=dict(shots=len(source),min=min(source),median=float(np.median(source)),max=max(source))))
    print(table,flush=True)


if __name__=='__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('out',type=Path)
    run(a.parse_args().out.resolve())
