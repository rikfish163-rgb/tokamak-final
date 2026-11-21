"""Independent count/loss checks and exact GPU replay of declared case shots."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import ARMS,predict,eligible


def main(root):
    done=READ(root/'completed.json');assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    p=READ(root/'protocol.json')
    for name,digest in p['input_sha256'].items():assert sha(Path(name))==digest,name
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['parent_sha256']
    assert sha(REFERENCE/'input_map.json')==p['mapping_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replayed=[]
    def deny(*args,**kwargs):raise AssertionError('no outcome access for replay')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in (89272,89284,89584):
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates=predict(model,s)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for a in ('off',)+ARMS:np.testing.assert_array_equal(curves[a],z[a])
            assert updates==READ(root/'updates'/f'{sid}.json');replayed.append(sid)
    outcomes={r['shot']:r for r in READ(root/'outcomes.json')};metrics={};geometry=[];counts={}
    for a in ('off',)+ARMS:
        tp=fp=early=0;danger=[];safe=[]
        for sid in p['shot_ids']:
            o=outcomes[sid]
            with np.load(root/'shots'/f'{sid}.npz') as z:
                lead=np.round((o['down']-z['times'])*1000,9) if o['is_disrupt'] else None
                valid=lead>=-1e-6 if o['is_disrupt'] else np.ones(len(z['times']),bool)
                score=z[a][valid].astype(np.float64);alarms=np.flatnonzero(score>=p['threshold'])
                if o['is_disrupt']:
                    timely=bool(len(alarms) and lead[valid][alarms[0]]>30)
                    tp+=int(timely);early+=int(timely and lead[valid][alarms[0]]>3000)
                    danger.append(float(-np.log(score[(lead[valid]>30)&(lead[valid]<=125)].clip(1e-12,1)).mean()))
                else:
                    fp+=int(bool(len(alarms)));safe.append(float(-np.log1p(-score.max().clip(0,1-1e-12))))
        metrics[a]=dict(tp=tp,fp=fp,very_early_over3000=early,danger_loss=float(np.mean(danger)),
            safe_loss=float(np.mean(safe)),balanced_loss=float((np.mean(danger)+np.mean(safe))/2))
        for k,v in metrics[a].items():np.testing.assert_allclose(v,done['results'][a][k],atol=1e-12,rtol=1e-10)
    for sid in p['shot_ids']:
        updates=READ(root/'updates'/f'{sid}.json')
        for a,rows in updates.items():
            counts[a]=counts.get(a,0)+sum(r['gradient_steps'] for r in rows)
            for i,r in enumerate(rows):
                assert r['first_affected_prediction']==r['last_observation']+1==256+128*i
                assert r['actual_step_norm']<=.001001
                if r['gradient_steps']:np.testing.assert_allclose(r['requested_step_norm'],.001,atol=1e-9,rtol=1e-6)
                np.testing.assert_allclose(r['losses']['full'],r['losses']['level']+r['losses']['shape'],atol=1e-8,rtol=1e-6)
        for r in updates['shape']:
            geometry.append(dict(shot=sid,count=r['first_affected_prediction'],
                raw_level_cosine=r['raw_level_cosine'],raw_shape_cosine=r['raw_shape_cosine'],
                shape_hold_cosine=r['shape_hold_cosine'],shape_shuffle_cosine=r['shape_shuffle_cosine'],
                level_loss_fraction=r['losses']['level']/r['losses']['full'] if r['losses']['full'] else None,
                shape_to_level_gradient_norm=r['gradient_norms']['shape']/r['gradient_norms']['level'] if r['gradient_norms']['level'] else None,
                gradient_decomposition_relative_error=r['gradient_decomposition_relative_error']))
    summary={}
    for k in list(geometry[0])[2:]:
        vals=[r[k] for r in geometry if r[k] is not None]
        summary[k]=dict(n=len(vals),median=float(np.median(vals)),min=float(np.min(vals)),max=float(np.max(vals))) if vals else dict(n=0)
    own_loss={a:dict(updates=0,strictly_improved=0,unchanged=0,worsened=0) for a in ARMS}
    for sid in p['shot_ids']:
        for a,rows in READ(root/'updates'/f'{sid}.json').items():
            for r in rows:
                own_loss[a]['updates']+=1
                key='strictly_improved' if r['loss_after']<r['loss_before'] else 'worsened' if r['loss_after']>r['loss_before'] else 'unchanged'
                own_loss[a][key]+=1
    assert eligible(done['results'])==done['eligible_for360']
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'gradient_geometry.json',geometry)
    result=dict(status='passed',completed_sha256=sha(root/'completed.json'),complete_shots_replayed=replayed,
        independent_metrics=metrics,gradient_updates=counts,gradient_geometry_summary=summary,
        own_loss_changes=own_loss,input_files_rehashed=len(p['input_sha256']),source_and_parent_unchanged=True,
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
