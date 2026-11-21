"""Post-hoc mechanism diagnostic, NOT an online accept/reject policy.

Select newly false-alarming safe shots after completed evaluation. For alarms
immediately after an update, replay the label-free prefix and hold the next
signal window fixed while withholding ONLY that most recent weight update.
This does not erase earlier sealed alarms or claim to rescue the entire shot.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import simple_data
from .causal_input import apply_input_contract
from .cross_channel_prefix import CrossChannelModel
from .fast_update import normalized_step
from .redesign_20261004.intra import _batch_windows
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .simple_online import _durable_json,_hash


def replay_to(model,shot,end):
    if end<7 or (end-7)%32:raise ValueError('declared actual update clock required')
    if {'is_disrupt','down','y','valid'} & set(shot):raise ValueError('signal-only replay required')
    raw={k:shot[k] for k in ('values','available')};device=model.center.device
    fast=model.initial_fast(1).requires_grad_();origin=fast.detach().clone()
    for index,t in enumerate(range(7,end+1,32)):
        before=fast.detach().clone()
        x,a=_batch_windows(simple_data,raw,[t],model,device)
        loss=model.reconstruction_loss(x,a,fast,phase=index%4)
        grad,=torch.autograd.grad(loss,fast)
        candidate=normalized_step(fast.detach(),origin,grad.detach()).requires_grad_()
        after=model.reconstruction_loss(x,a,candidate,phase=index%4)
        if bool(after<loss.detach()):fast=candidate
        fast=fast.detach().requires_grad_()
    return before,fast,dict(loss_before=float(loss.detach()),loss_after=float(after.detach()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--heads',type=Path,required=True)
    p.add_argument('--arm',choices=('signal_embedding','risk_embedding'),required=True)
    args=p.parse_args();out=args.run/f'{args.arm}_counterfactual.json'
    if out.exists():p.error('new diagnostic output required')
    torch.set_num_threads(4)
    protocol=json.loads((args.run/'protocol.json').read_text());threshold=protocol['threshold']
    old=json.loads((args.run/'static/verification.json').read_text())['decisions']
    new=json.loads((args.run/args.arm/'verification.json').read_text())['decisions']
    if [r['shot'] for r in old]!=[r['shot'] for r in new]:raise ValueError('shot identity mismatch')
    base,_=load_parent(DEFAULT_PARENT);m=CrossChannelModel(base,site='embedding').cuda().eval()
    branch='signal' if args.arm=='signal_embedding' else 'risk_embedding'
    headpath=args.heads/f'{branch}_head.pt'
    if _hash(headpath)!=protocol['head_sha256'][headpath.name]:raise ValueError('head provenance mismatch')
    checkpoint=torch.load(headpath,map_location='cpu',weights_only=True)
    if checkpoint['parent_sha256']!=PARENT_SHA256:raise ValueError('parent mismatch')
    m.decoder.load_state_dict(checkpoint['decoder']);m.requires_grad_(False)
    initial={k:v.detach().clone() for k,v in m.state_dict().items()};rows=[];new_fp=0;direct=0
    for i,(b,r) in enumerate(zip(old,new),1):
        if r['is_disrupt'] or b['alarm'] or not r['alarm']:continue
        new_fp+=1;sid=r['shot'];stem=f'{i:04d}_{sid}'
        stream=args.run/args.arm/'stream'
        with np.load(stream/f'{stem}_sealed.npz',allow_pickle=False) as saved:
            j=int(np.flatnonzero(saved['scores']>=threshold)[0]);score=float(saved['scores'][j])
        if j<8 or (j-8)%32:continue
        direct+=1;s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        before,after,loss=replay_to(m,s,j-1)
        audit=json.loads((stream/f'{stem}_audit.json').read_text())
        step=next(a for a in audit['attempts'] if a['update_index']==j-1)
        if m.summarize_fast(after)!=step['fast_after']:raise AssertionError('temporary state replay differs')
        x,a=_batch_windows(simple_data,{k:s[k] for k in ('values','available')},[j],m,'cuda')
        with torch.no_grad():
            without=float(m(x,a,before)['logit'].sigmoid())
            actual=float(m(x,a,after)['logit'].sigmoid())
        if abs(actual-score)>2e-6:raise AssertionError('sealed alarm does not reproduce')
        rows.append(dict(shot=sid,first_alarm_index=j,update_index=j-1,**loss,
            without_latest_update=without,with_latest_update=actual,sealed_probability=score,
            only_this_alarm_removed=bool(without<threshold),
            full_shot_rescue_not_tested=True,temporary_state_hash_matches=True))
    for k,v in m.state_dict().items():torch.testing.assert_close(v,initial[k],atol=0,rtol=0)
    _durable_json(out,dict(scope='POSTHOC SELECTED-ERROR COUNTERFACTUAL, NOT A DEPLOYABLE POLICY',
        arm=args.arm,new_fp=new_fp,immediate_update_alarms=direct,
        below_threshold_without_last_update=sum(r['only_this_alarm_removed'] for r in rows),
        threshold=threshold,selection_uses_completed_outcomes=True,no_new_target_training=True,
        changed_only_latest_update_not_signal=True,rows=rows,code_sha256=_hash(__file__)))
    print(json.dumps(dict(arm=args.arm,new_fp=new_fp,immediate=direct,
        below_without=sum(r['only_this_alarm_removed'] for r in rows))),flush=True)


if __name__=='__main__':main()
