"""Independent distances, one-fold neural replay, and post-hoc feature audit."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..external65_20261023.run import models_for
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import PLAN,VIEWS,fold,summarize,reference_indices,weights


def independent_distances(q,reference,w):
    w=np.asarray(w,dtype=np.float64);w/=w.sum();reference=reference.astype(np.float64)
    mean=np.einsum('n,nd->d',w,reference);r=reference-mean
    c=np.einsum('n,ni,nj->ij',w,r,r);t=max(np.trace(c)/c.shape[0],1e-12)
    c=.5*c+.5*t*np.eye(c.shape[0]);eig,u=np.linalg.eigh(c)
    differences=(reference-q)@u
    return float(np.min(np.sum(differences**2/eig,axis=1)))


def main(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for group,base in (('artifact_sha256',root),('code_sha256',io23.ROOT)):
        for name,h in (done if group=='artifact_sha256' else p)[group].items():assert sha(base/name)==h
    for group in ('checkpoints','input_sha256'):
        for name,h in p[group].items():assert sha(name)==h
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    torch.set_num_threads(2);torch.manual_seed(131)
    outcomes={s:io23.read_outcome('EAST',s) for s in p['shot_ids']}
    raw={s:apply_input_contract(simple_data.read_shot('EAST',s,rate_contract='nominal')) for s in p['shot_ids']}
    mode=[]
    for sid,s in raw.items():
        j=io23.RAW23.index('mode_number_n');v=s['values'][s['available'][:,j],j]
        mode.append(dict(shot=sid,is_disrupt=outcomes[sid][0],n=len(v),available_fraction=float(s['available'][:,j].mean()),
            zero_fraction=float((v==0).mean()) if len(v) else None,
            quantiles=np.quantile(v,[0,.01,.5,.99,1]).tolist() if len(v) else None))
    distance_checks=0;max_error=0.;orthogonal={};summary={}
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';records=READ(folder/'records.json');details=READ(folder/'distances.json')
        probes=READ(folder/'probes.json');model=models_for(f)['prepared'];frozen=state_cpu(model)
        with np.load(folder/'features.npz') as z:arrays={k:z[k].copy() for k in z.files}
        lookup={(r['shot'],r['end']):i for i,r in enumerate(records)}
        orth=arrays['risk_state'].astype(np.float64);w=model.base.risk_head.weight.detach().cpu().numpy().flatten().astype(np.float64)
        w/=np.linalg.norm(w);orth=orth-(orth@w)[:,None]*w
        orthr=[]
        for r in details:
            i=lookup[r['shot'],r['end']]
            result=dict(shot=r['shot'],end=r['end'],group=r['group'],fixed=r['fixed'],early_fp=r['early_fp'],distances={})
            for kind in ('prefix','full'):
                idx=reference_indices(records,kind,r['shot'] if r['group']=='fit_safe' else None)
                assert all(records[j]['shot'] in plan['fit'] and not outcomes[records[j]['shot']][0] for j in idx)
                ww=weights([records[j] for j in idx])
                for view in VIEWS:
                    d=independent_distances(arrays[view][i],arrays[view][idx],ww)
                    expected=r['distances'][kind][view];max_error=max(max_error,abs(d-expected))
                    np.testing.assert_allclose(d,expected,rtol=1e-9,atol=1e-9);distance_checks+=1
                result['distances'][kind]=independent_distances(orth[i],orth[idx],ww)
            orthr.append(result)
        orthogonal[str(f)]=orthr
        assert summarize(details,probes,p['threshold'])==done['summary'][str(f)]
        if f==0:
            mapping=READ(REFERENCE/f'fold{f}'/'input_map.json')
            signals={sid:dict(times=s['times'],**transform({k:s[k] for k in ('values','available')},mapping)) for sid,s in raw.items()}
            def deny(*args,**kwargs):raise AssertionError('outcome access during replay')
            with patch.object(io23,'read_outcome',side_effect=deny):rr,aa,dd,pp,_=fold(model,signals,outcomes,plan,p['threshold'])
            assert rr==records and dd==details and pp==probes
            for k in arrays:np.testing.assert_array_equal(aa[k],arrays[k])
        assert_unchanged(model,frozen)
        summary[str(f)]={g:float(np.median([r['distances']['full'] for r in orthr if r['fixed'] and r['group']=='external_safe' and r['early_fp']==flag]))
            for g,flag in (('early_fp',True),('not_early_fp',False))}
        print(dict(fold=f,distances=distance_checks,seconds=round(time.monotonic()-start,2)),flush=True)
    _durable_json(root/'mode_number_audit.json',mode);_durable_json(root/'risk_orthogonal.json',orthogonal)
    result=dict(status='VERIFIED_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        distance_checks=distance_checks,max_absolute_distance_difference=max_error,
        exact_replayed_fold=0,windows_replayed=len(READ(root/'fold0'/'records.json')),
        risk_orthogonal_full_reference_medians=summary,
        posthoc_diagnostics_sha256={name:sha(root/name) for name in ('mode_number_audit.json','risk_orthogonal.json')},
        seconds=time.monotonic()-start,source_unchanged=sha(DEFAULT_PARENT)==PARENT_SHA256,test360_accessed=False)
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
