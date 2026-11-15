"""Verify the sealed field intervention and separate peak shifts by original magnitude."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from . import commissioning_field_scale_20261113 as field
from .causal_input import CIII_MISSING_ATOL, CIII_MISSING_VALUE
from .history_gradient_metric_20261102.run import read, sha, write
from .operating_point_audit import best_at_count, timely_auc


def run(folder,out):
    p,s,r,rows = (read(folder/f'{k}.json') for k in ('protocol','seal','result','per_shot'))
    parent = field.bg.PARENT
    pp = read(parent/'protocol.json')
    if (p['order']!=pp['order'] or len(s['curve_sha256'])!=2953 or len(s['direction'])!=2953
            or p['parent_seal']!=sha(parent/'seal.json')):
        raise ValueError('complete static field replay required')
    paths = {c:Path(v['path']) for c,v in pp['references'].items()}
    seals = {c:read(d/'seal.json') for c,d in paths.items()}
    masks = read(parent/'postseal_bt_scale_balance.json')
    matched = set(masks['matched_ids'])
    arms = ('fixed',*field.VIEWS)
    by = {k:{a:[] for a in arms} for k in ('extension','smaller','larger','raw_scale_balance')}
    unchanged_prefix = {a:0 for a in arms}
    ciii = defaultdict(lambda:dict(n=Counter(),arm={a:Counter() for a in arms}))
    for i,row in enumerate(p['order'][360:],361):
        sid,rank = (row[k] for k in ('shot','rank'))
        d,down,t = field.group.hdf._raw_truth(sid)
        m = np.round((down-t)*1000.,9)>30. if d else np.ones(len(t),bool)
        phase = 'smaller' if s['direction'][i-1]>0 else 'larger'
        if not s['direction'][i-1]:
            raise ValueError('extension needs valid first 100 samples')
        subsets = ('extension',phase)+(('raw_scale_balance',) if sid in matched else ())
        with h5py.File(field.io23.path_for('EAST',sid),'r') as f:
            z = f['data/CIII'][:]
        valid = np.isfinite(z)&~np.isclose(z,CIII_MISSING_VALUE,atol=CIII_MISSING_ATOL,rtol=0.)
        availability = 'all' if valid.all() else 'partial' if valid.any() else 'none'
        stratum = ciii[f'{phase}/{availability}']
        label = 'disruptive' if d else 'safe'
        stratum['n'][label] += 1
        src = field.par.bind(paths['extension'],f'{rank:04d}_{sid}.npz',seals['extension'])
        cur = field.par.bind(folder,f'{i:04d}_{sid}.npz',s)
        with np.load(src,allow_pickle=False) as f,np.load(cur,allow_pickle=False) as g:
            np.testing.assert_array_equal(f['times'],t)
            base = f['fixed']
            for a in arms:
                v = base if a=='fixed' else g[a]
                np.testing.assert_array_equal(v[:p['first']],base[:p['first']])
                hit = bool((v[m]>=p['calibration']['0.2']).any())
                archived = rows['0.2'][a][i-1]
                if (archived['shot'],archived['disruptive'],archived['down_s'],archived['timely' if d else 'alarm'])!=(sid,d,down,hit):
                    raise ValueError('fixed threshold peak does not reproduce the sealed first alarm')
                if not d and (v[:p['first']]>=p['calibration']['0.2']).any():
                    unchanged_prefix[a] += 1
                stratum['arm'][a][label] += hit
                peak = dict(disruptive=d,peak=float(v[m].max()) if m.any() else None)
                for k in subsets:
                    by[k][a].append(peak)
    groups = {}
    for k,subset in by.items():
        groups[k] = {}
        n = sum(not z['disruptive'] for z in subset['fixed'])
        base = subset['fixed']
        for a,values in subset.items():
            caps = {str(c):{key:v[key] for key in ('tp','fp','max_fp')}
                    for c,v in ((c,best_at_count(values,int(c*n))) for c in (.10,.15,.20))}
            delta = {kind:[b['peak']-x['peak'] for x,b in zip(base,values,strict=True)
                          if b['disruptive']==y and x['peak'] is not None and b['peak'] is not None]
                     for kind,y in (('disruptive',True),('safe',False))}
            groups[k][a] = dict(matched_fp=caps,auc=timely_auc(values),
                delta_peak_p10_p50_p90={d:np.quantile(v,[.1,.5,.9]).tolist() for d,v in delta.items()})
            lookup = 'small' if k=='smaller' else 'large' if k=='larger' else k
            if a!='fixed' and caps!=r['groups'][lookup]['postseal_matched_fp'][a]:
                raise ValueError('independent matched-count computation differs')
    write(out,dict(scope='POSTSEAL_FIELD_PEAK_RANK_AND_PREFIX',groups=groups,
                   unchanged_early_safe=unchanged_prefix,field_and_ciii=ciii,seal_sha256=sha(folder/'seal.json')))
    print({k:{a:(v['matched_fp']['0.2'],round(v['auc'],4)) for a,v in g.items()}
           for k,g in groups.items()},flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    run(a.folder.resolve(),a.output.resolve())
