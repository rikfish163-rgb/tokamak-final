"""Post-hoc directional feasibility; true outcomes NEVER enter deployment.

Inspect the first four added FPs and first four lost timely alarms in sorted
shot order, without selecting attractive examples. For the initial inner
step, ask whether ANY allowed positive channel weighting could lower the
true endpoint loss to first order. This is neither a nonlinear guarantee nor
a deployable gate; the endpoint outcome is available only in this audit.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..masked_state_20261022.model import masked_views
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_train import sha
from ..simple_online import _durable_json
from .model import update_observed,channel_losses
from .train import model_from_warm


def main():
    torch.set_num_threads(2)
    root=io23.ROOT/'experiments/risk_aligned_online/runs'
    stream=root/'context_feedback_east_event_20261022'
    checkpoint=root/'context_feedback_event_20261022/last.pt'
    config=json.loads((stream/'protocol.json').read_text())
    if sha(checkpoint)!=config['checkpoint_sha256']: raise AssertionError('checkpoint changed')
    ck=torch.load(checkpoint,map_location='cuda',weights_only=True)
    model,threshold=model_from_warm();model.load_state_dict(ck['model']);model.freeze_base()
    done=json.loads((stream/'completed.json').read_text())
    cases=[(sid,False,'intra') for sid in sorted(done['paired_flips']['false_alarm_added'])[:4]]
    cases += [(sid,True,'static') for sid in sorted(done['paired_flips']['lost'])[:4]]
    result=[]
    for sid,desired,alarm_arm in cases:
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        sealed=np.load(stream/'shots'/f'{sid}.npz')
        d,down=io23.read_outcome('EAST',sid)
        if d!=desired: raise AssertionError('case outcome mismatch')
        _,valid=io23.labels_at(s['times'],d,down,125.)
        query=int(np.flatnonzero(valid & (sealed[alarm_arm]>=threshold))[0])
        lo=(query//32)*32;hi=min(lo+32,len(s['values']));now=lo-1
        if now<127: raise AssertionError('changed case cannot precede first update')
        x,m=[torch.as_tensor(v,device='cuda')[None] for v in simple_data.window(s,now,model.cfg.window)]
        examples=[simple_data.window(s,t,model.cfg.window) for t in range(lo,hi)]
        current,available=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples)]
        origin=initial_encoder(model);params=tuple(origin.values())
        before=model.predict_fast(current,available,origin)['logit'][query-lo]
        risk=F.softplus(-before if desired else before)
        risk_grad=torch.autograd.grad(risk,params)
        xx,mm,held=masked_views(x,m)
        predictions=model.predict_fast(xx,mm,origin,mode='reconstruct')['reconstruction']
        losses,counts=channel_losses(model,predictions,xx,held)
        # d_c already includes each channel's fraction of all observed targets.
        components=losses*counts/counts.sum();dots=[]
        for c in range(23):
            g=torch.autograd.grad(components[c],params,retain_graph=c<22)
            dots.append(float(sum((a.double()*b.double()).sum() for a,b in zip(risk_grad,g))))
        fast,info=update_observed(model,x,m,use_event=True)
        after=model.predict_fast(current,available,fast)['logit'][query-lo]
        actual=(float(before.detach().sigmoid()),float(after.detach().sigmoid()))
        expected=(float(sealed['static'][query]),float(sealed['intra'][query]))
        if max(abs(a-b) for a,b in zip(actual,expected))>2e-6:
            raise AssertionError('actual emitted endpoint replay differs')
        d_c=np.asarray(dots)
        max_numerator=float(np.where(d_c>0,1.5,.5)@d_c)
        learned_dot=float(np.asarray(info['weights'])@d_c)
        result.append(dict(shot=sid,desired_label=int(desired),endpoint_index=query,
            last_observation_index=now,before_probability=actual[0],after_probability=actual[1],
            risk_loss_before=float(risk.detach()),
            risk_loss_after=float(F.softplus(-after if desired else after).detach()),
            uniform_risk_gradient_dot=float(d_c.sum()),learned_risk_gradient_dot=learned_dot,
            maximum_allowed_weight_numerator=max_numerator,
            some_bounded_positive_weights_can_help_to_first_order=max_numerator>0,
            replay_max_difference=max(abs(a-b) for a,b in zip(actual,expected)),
            channels=[dict(name=name,gradient_dot=float(dot),weight=weight)
                      for name,dot,weight in zip(io23.RAW23,d_c,info['weights'])]))
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/context_feedback_gradient_cases.json'
    if output.exists(): raise FileExistsError(output)
    _durable_json(output,dict(status='VERIFIED',case_rule='first4 addedFP + first4 lostTP, sorted shot ID',
        cases=result,scope='offline true-outcome gradient, not online policy; first-order only; not physical causation'))
    for row in result: print(json.dumps({k:v for k,v in row.items() if k!='channels'}))


if __name__=='__main__':main()
