"""Post-hoc attribution only: predicted-shape energy vs arrived-target cross term.

These signed components are NOT update objectives offered for deployment. In
particular, minimizing the cross term alone is unbounded, so it is not a fix.
"""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23
from ..clock_channel_loss_20261024.learning import future_groups,_risk_gradient
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,state_cpu,assert_unchanged,
    signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..history_tangent_20261023.learning import flat
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import decompose


def expand_shape(prediction,target,mask):
    if (prediction.ndim!=3 or prediction.shape!=target.shape or mask.shape!=target.shape
            or mask.dtype!=torch.bool or not torch.isfinite(prediction[mask]).all()
            or not torch.isfinite(target[mask]).all()):
        raise ValueError('finite observed arrays and matching mask required')
    n=mask.sum((0,1),keepdim=True).clamp_min(1);total=mask.sum().clamp_min(1)
    p=prediction.masked_fill(~mask,0);y=target.detach().masked_fill(~mask,0)
    p=(p-p.sum((0,1),keepdim=True)/n).masked_fill(~mask,0)
    y=(y-y.sum((0,1),keepdim=True)/n).masked_fill(~mask,0)
    return dict(prediction_energy=p.square().sum()/(2*total),
                observed_cross=-(p*y).sum()/total,target_energy=y.square().sum()/(2*total))


def cosine(a,b):
    den=a.double().norm()*b.double().norm()
    return float(a.double()@b.double()/den) if den else None


def main(previous,root):
    if root.exists():raise FileExistsError(root)
    done=READ(previous/'completed.json');assert READ(previous/'audit.json')['status']=='passed'
    for name,digest in done['artifact_sha256'].items():assert sha(previous/name)==digest,name
    protocol=READ(previous/'protocol.json');outcomes={r['shot']:r for r in READ(previous/'outcomes.json')}
    torch.set_num_threads(2);torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();model=models_for_calibration(parent)['mapped_fixed'];original=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');root.mkdir(parents=True);start=time.monotonic()
    _durable_json(root/'protocol.json',dict(scope='POSTHOC_GRADIENT_ATTRIBUTION_NO_POLICY_CHANGE',
        previous_completed_sha256=sha(previous/'completed.json'),code_sha256=sha(Path(__file__)),
        shot_ids=protocol['shot_ids'],new_outcomes_accessed=0,new360=False,
        purpose='separate dependence on arrived targets from predicted temporal variation'))
    rows=[];risk_rows=[]
    def deny(*args,**kwargs):raise AssertionError('diagnosis uses archived labels only')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for sid in protocol['shot_ids']:
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            old=READ(previous/'updates'/f'{sid}.json')['shape'];outcome=outcomes[sid]
            for record in old:
                count=record['first_affected_prediction'];fast=initial_encoder(model)
                x,m,y,ym,_=forecast_examples({k:v[:count] for k,v in s.items()},count,model.cfg,'cuda')
                mask=target_mask(x,m,y,ym)
                target=torch.asinh((torch.where(mask,y,model.base.center)-model.base.center)/model.base.scale)
                pred=model.predict_fast(x,m,fast)['forecast'];terms=expand_shape(pred,target,mask)
                shape=decompose(pred,target,mask)['shape']
                torch.testing.assert_close(shape,sum(terms.values()),atol=1e-7,rtol=1e-5)
                gradients={k:flat(torch.autograd.grad(v,tuple(fast.values()),retain_graph=True)).detach()
                           for k,v in dict(shape=shape,prediction=terms['prediction_energy'],observed=terms['observed_cross']).items()}
                diff=(gradients['shape']-gradients['prediction']-gradients['observed']).double().norm()
                scale=sum(g.double().norm() for g in gradients.values());error=float(diff/scale) if scale else 0.
                if error>3e-5:raise AssertionError('shape expansion derivative inconsistent')
                assert float(shape.detach())==record['losses']['shape']
                assert float(gradients['shape'].double().norm())==record['gradient_norms']['shape']
                norms={k:float(g.double().norm()) for k,g in gradients.items()}
                rows.append(dict(shot=sid,count=count,terms={k:float(v.detach()) for k,v in terms.items()},norms=norms,
                    observed_prediction_norm_ratio=norms['observed']/norms['prediction'] if norms['prediction'] else None,
                    shape_prediction_cosine=cosine(gradients['shape'],gradients['prediction']),
                    shape_observed_cosine=cosine(gradients['shape'],gradients['observed']),relative_error=error))
                for group,ends in future_groups(raw['times'],count,outcome['is_disrupt'],outcome['down']).items():
                    if not len(ends):continue
                    r,receipt=_risk_gradient(model,s,ends,fast,float(group=='danger'),device='cuda')
                    risk_rows.append(dict(shot=sid,count=count,group=group,base=receipt,
                        # Sign of the first-order task loss change for one
                        # common .001-norm direction, NOT measured gain.
                        **{k:float(-.001*r.double()@g.double()/g.double().norm()) if g.double().norm()>0 else 0.
                           for k,g in gradients.items()}))
            print(dict(stage='gradient_attribution',shot=sid,seconds=round(time.monotonic()-start,2)),flush=True)
    per_shot=[];summary={}
    for group in ('safe','early','danger'):
        take=[r for r in risk_rows if r['group']==group];ids=sorted({r['shot'] for r in take})
        local=[]
        for sid in ids:
            rs=[r for r in take if r['shot']==sid]
            local.append(dict(group=group,shot=sid,blocks=len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in ('shape','prediction','observed')}))
        per_shot+=local
        summary[group]=dict(shots=len(ids),blocks=len(take),**{k:float(np.mean([r[k] for r in local])) for k in ('shape','prediction','observed')})
    geometry={}
    for k in ('observed_prediction_norm_ratio','shape_prediction_cosine','shape_observed_cosine','relative_error'):
        values=[r[k] for r in rows if r[k] is not None]
        geometry[k]=dict(n=len(values),median=float(np.median(values)),min=float(np.min(values)),max=float(np.max(values)))
    _durable_json(root/'rows.json',rows);_durable_json(root/'risk_rows.json',risk_rows);_durable_json(root/'per_shot.json',per_shot)
    assert_unchanged(model,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='completed',rows=len(rows),risk_rows=len(risk_rows),seconds=time.monotonic()-start,
        geometry=geometry,first_order_risk_summary=summary,source_and_parent_unchanged=True,
        scope='POSTHOC_NOT_A_NEW_DEPLOYABLE_POLICY_OR_UPPER_BOUND',
        artifact_sha256={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file()})
    _durable_json(root/'completed.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.previous,args.out)
