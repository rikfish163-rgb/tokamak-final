"""Post-hoc decomposition at real changed safe-shot alarms; never an online gate.

For one current safe endpoint, d BCE / d encoder is available ONLY for this
offline diagnostic. Attribute its dot product with the observed completion
gradient to each diagnostic target. These are loss-task contributions, not
causal physical importance or an attribution of the whole nonlinear update.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..event_adapt_20261022.train import load_source
from ..masked_state_20261022.model import masked_views
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder, update_observed
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha


def main():
    torch.set_num_threads(2)
    root=Path(__file__).parents[1]/'runs'
    path=root/'joint_feedback_meta_weighted_20261022/last.pt'
    stream=root/'joint_feedback_val_meta_weighted_20261022'
    protocol=json.loads((stream/'protocol.json').read_text())
    if sha(path)!=protocol['checkpoint_sha256']: raise AssertionError('diagnostic checkpoint changed')
    ck=torch.load(path,map_location='cpu',weights_only=True)
    base,_=load_parent(DEFAULT_PARENT)
    model=PairedFeedback(base,ck['model']['level_scale'])
    model.load_state_dict(ck['model']);model.cuda();model.prepare_decoder()
    done=json.loads((stream/'completed.json').read_text())
    threshold=done['matched_static_threshold']['static']['threshold']
    traces=np.load(stream/'traces.npz'); rows=[]
    for sid,alarm_arm in ((1054177,'intra'),(1052909,'static')):
        s=load_source([sid],io23.advanced_times())[0]
        if s['is_disrupt']: raise AssertionError('this audit is explicitly two safe shots')
        query=int(np.flatnonzero(s['valid'] & (traces[f'{sid}_{alarm_arm}']>=threshold))[0])
        lo=(query//8)*8; hi=min(lo+8,len(s['values'])); now=lo-1
        if now<127: raise AssertionError('cannot audit an update before first legal adaptation')
        xx,mm=simple_data.window(s,now,model.cfg.window)
        x=torch.as_tensor(xx,device='cuda')[None];m=torch.as_tensor(mm,device='cuda')[None]
        windows=[simple_data.window(s,k,model.cfg.window) for k in range(lo,hi)]
        current,available=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
        origin=initial_encoder(model);parameters=tuple(origin.values())
        before=model.predict_fast(current,available,origin)['logit'][query-lo]
        risk=F.softplus(before)  # safe outcome; used for OFFLINE analysis only
        risk_grad=torch.autograd.grad(risk,parameters)
        mx,ma,held=masked_views(x,m)
        prediction=model.predict_fast(mx,ma,origin,mode='reconstruct')['reconstruction']
        target=torch.asinh((torch.where(held,mx,model.base.center)-model.base.center)/model.base.scale)
        losses=F.smooth_l1_loss((prediction-target)/model.level_scale,
                              torch.zeros_like(prediction),reduction='none').masked_fill(~held,0.)
        channel_losses=losses.sum((0,1))/held.sum().clamp_min(1)
        total_grad=torch.autograd.grad(channel_losses.sum(),parameters,retain_graph=True)
        dot=sum((a.double()*b.double()).sum() for a,b in zip(risk_grad,total_grad))
        contributions=[]
        for c,name in enumerate(io23.RAW23):
            gc=torch.autograd.grad(channel_losses[c],parameters,retain_graph=c<22)
            product=sum((a.double()*b.double()).sum() for a,b in zip(risk_grad,gc))
            contributions.append(dict(channel=name,completion=float(channel_losses[c].detach()),
                risk_gradient_dot=float(product),predicted_first_step_risk_change=float(-.01*product)))
        summed=sum(r['risk_gradient_dot'] for r in contributions)
        if not np.isclose(summed,float(dot),rtol=2e-5,atol=2e-6):
            raise AssertionError('per-channel gradients do not reconstruct full gradient')
        fast,info=update_observed(model,x,m,learning_rate=.01,steps=2)
        after=model.predict_fast(current,available,fast)['logit'][query-lo]
        before_p=float(before.detach().sigmoid());after_p=float(after.detach().sigmoid())
        expected=(float(traces[f'{sid}_static'][query]),float(traces[f'{sid}_intra'][query]))
        error=max(abs(before_p-expected[0]),abs(after_p-expected[1]))
        if error>2e-6: raise AssertionError('actual emitted alarm-state replay differs')
        rows.append(dict(shot=sid,current_index=query,last_observed_update_index=now,
            update_affects_current_after_observations=True,before_probability=before_p,
            after_probability=after_p,threshold=threshold,replay_max_difference=error,
            before_risk_loss=float(risk.detach()),after_risk_loss=float(F.softplus(after).detach()),
            gradient_dot=float(dot),predicted_first_step_risk_change=float(-.01*dot),
            actual_two_step_risk_change=float((F.softplus(after)-risk).detach()),
            observations=info['observations'],decomposition_error=abs(summed-float(dot)),
            contributions=sorted(contributions,key=lambda r:r['predicted_first_step_risk_change'],reverse=True)))
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/joint_feedback_gradient_cases.json'
    if output.exists(): raise FileExistsError(output)
    result=dict(status='VERIFIED',partition='J-TEXT VAL only; diagnostic',
        checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),cases=rows,
        scope='post-hoc true-safe gradient; not deployable query/update criterion; not a physical cause claim')
    _durable_json(output,result)
    for r in rows:
        print(json.dumps({**{k:v for k,v in r.items() if k!='contributions'},
            'largest_risk_raising':r['contributions'][:4],
            'largest_risk_lowering':r['contributions'][-4:]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
