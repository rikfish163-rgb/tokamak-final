"""Posthoc per-channel feedback directions, never a deployable selection policy."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import AttentionAdapter


def channel_losses(base,prediction,truth,mask):
    if prediction.shape!=truth.shape or mask.shape!=truth.shape or mask.dtype!=torch.bool:
        raise ValueError('matching predictions, true observations and masks required')
    mask=mask & torch.isfinite(truth)
    target=torch.asinh((torch.where(mask,truth.detach(),base.center)-base.center)/base.scale)
    error=F.smooth_l1_loss(prediction,target,reduction='none').masked_fill(~mask,0.)
    return error.sum((0,1))/mask.sum().clamp_min(1)


def choose_channel(gradients,risk_gradient):
    if (gradients.ndim!=2 or risk_gradient.shape!=gradients.shape[1:]
            or not torch.isfinite(gradients).all() or not torch.isfinite(risk_gradient).all()):
        raise ValueError('finite channel-by-parameter and risk gradients required')
    norm=gradients.norm(dim=1)*risk_gradient.norm()
    cosine=(gradients@risk_gradient)/norm.clamp_min(1e-30)
    cosine=torch.where(norm>0,cosine,torch.zeros_like(cosine))
    index=int(cosine.argmax())
    # Numerically flat directions are reported, not promoted into evidence of
    # useful first-order alignment. This is a diagnostic tolerance, not policy.
    return (index if cosine[index]>1e-6 else -1),cosine


def feedback_clock(end):
    if type(end) is not int or end<0:raise ValueError('nonnegative integer endpoint required')
    count=end//128*128
    return count if count>=256 else None


def describe(rows):
    if not rows:return dict(n=0)
    return dict(n=len(rows),all_channels_risk_improved=sum(r['risk_loss_change']['all']<0 for r in rows),
        oracle_channel_risk_improved=sum(r['risk_loss_change']['oracle_channel']<0 for r in rows),
        any_positive_channel=sum(r['oracle_channel']>=0 for r in rows),
        all_channels_positive_gradient=sum(r['all_cosine']>1e-6 for r in rows),
        median_all_risk_change=float(np.median([r['risk_loss_change']['all'] for r in rows])),
        median_reverse_gradient_cosine=float(np.median([r['reverse_gradient_cosine'] for r in rows])))


def run(output,points,stream):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree-local output required')
    p=read(points,'protocol.json');old=read(points,'records.json')['records']
    a=read(points,'audit.json');sa=read(stream,'audit.json');sp=read(stream,'protocol.json')
    if (not a['complete'] or a['complete_sha256']!=sha(points/'complete.json') or not sa['complete']
            or sa['complete_sha256']!=sha(stream/'complete.json') or p['order']!=sp['order']
            or p['reference_hashes']!=sp['reference_hashes']):raise ValueError('audited same-parent references required')
    ref=Path(p['reference']);mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json')
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold parent changed')
    selected=[r for r in old if not r['flag'] or 'positive_window_peak' in r['reasons']]
    if len(selected)!=340 or len({r['shot'] for r in selected})!=340:raise ValueError('exact one point per suffix shot required')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=AttentionAdapter(base)
    frozen={k:v.cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    output.mkdir();start=time.monotonic()
    protocol=dict(scope='POSTHOC_LABEL_INFORMED_CHANNEL_GRADIENT_NOT_ONLINE',
        source_sha256=sha(DEFAULT_PARENT),points=str(points),points_sha256=sha(points/'records.json'),
        stream=str(stream),stream_audit_sha256=sha(stream/'audit.json'),reference=str(ref),
        reference_hashes=p['reference_hashes'],order=p['order'],threshold=parent['threshold'],
        step=.05,plan_sha256=sha(io23.ROOT/'CHANNEL_DIRECTION_PLAN.md'),
        code_sha256={str(Path(__file__).relative_to(io23.ROOT)):sha(Path(__file__)),
                     str(Path(__file__).with_name('model.py').relative_to(io23.ROOT)):sha(Path(__file__).with_name('model.py'))},
        channels=list(io23.RAW23),terminal_labels_used=True,temporary_state_reset_for_each_point=True)
    _durable_json(output/'protocol.json',protocol)
    rows=[];excluded=[]
    for j,r in enumerate(selected):
        count=feedback_clock(r['end']);sid=r['shot']
        if count is None:
            excluded.append(dict(shot=sid,end=r['end'],flag=r['flag'],reason='endpoint_before_first_matured_update'))
            continue
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        flag,down=io23.read_outcome('EAST',sid)
        if flag!=r['flag'] or (flag and not 30<round((down-raw['times'][r['end']])*1000,9)<=125):
            raise ValueError('endpoint label does not match fixed125ms training contract')
        signal=transform({k:raw[k] for k in ('values','available')},mapping)
        prefix={k:v[:count] for k,v in signal.items()}
        x,mask,truth,available,origins=forecast_examples(prefix,count,base.cfg,'cuda')
        available=available & torch.isfinite(truth) & mask[:,-1:] & torch.isfinite(x[:,-1:]);available[:,:,10]=False
        if not available.any():
            excluded.append(dict(shot=sid,end=r['end'],flag=flag,reason='no_observed_regression_targets'));continue
        state=x.new_zeros(1,512,requires_grad=True)
        prediction=net(x,mask,state.expand(len(x),-1),slow,'qk')['forecast']
        terms=channel_losses(base,prediction,truth,available)
        gradients=torch.stack([torch.autograd.grad(t,state,retain_graph=True)[0][0] for t in terms])
        total=torch.autograd.grad(base.forecast_loss(prediction,truth,available),state,retain_graph=True)[0][0]
        if not torch.allclose(gradients.sum(0),total,atol=2e-7,rtol=2e-4):raise ValueError('channel gradient decomposition differs')
        reverse_loss=base.forecast_loss(prediction,truth.flip(1),available.flip(1))
        reverse=torch.autograd.grad(reverse_loss,state)[0][0]
        qx,qa=batch(signal,[r['end']],'cuda');query=net(qx,qa,state,slow,'qk',forecast=False)['logit']
        target=query.new_tensor([float(flag)]);risk_loss=F.binary_cross_entropy_with_logits(query,target)
        risk_gradient=torch.autograd.grad(risk_loss,state)[0][0]
        index,cosine=choose_channel(gradients,risk_gradient)
        norm=lambda g:g/g.norm().clamp_min(1e-12)
        states=torch.stack((state[0].detach(),-.05*norm(total),
                            -.05*norm(gradients[index]) if index>=0 else state[0].detach(),-.05*norm(reverse)))
        with torch.no_grad():
            logits=net(qx.repeat(4,1,1),qa.repeat(4,1,1),states,slow,'qk',forecast=False)['logit']
            losses=F.binary_cross_entropy_with_logits(logits,target.expand(4),reduction='none')
        if abs(float(logits[0].sigmoid())-r['sealed_probability'])>2e-6:raise ValueError('zero-state endpoint mismatch')
        rows.append(dict(shot=sid,end=r['end'],flag=flag,cold_alarm=r['cold_alarm'],cold_timely=r['cold_timely'],
            count=count,origins=origins,target_count=int(available.sum()),oracle_channel=index,
            channel_cosine=cosine.detach().cpu().tolist(),channel_gradient_norm=gradients.norm(dim=1).detach().cpu().tolist(),
            channel_dot=(gradients@risk_gradient).detach().cpu().tolist(),channel_loss=terms.detach().cpu().tolist(),
            all_cosine=float(F.cosine_similarity(total[None],risk_gradient[None])),
            reverse_gradient_cosine=float(F.cosine_similarity(total[None],reverse[None])),
            initial_logit=float(logits[0]),risk_loss_change={k:float(losses[i]-losses[0]) for i,k in enumerate(('zero','all','oracle_channel','reverse'))},
            logit_change={k:float(logits[i]-logits[0]) for i,k in enumerate(('zero','all','oracle_channel','reverse'))}))
        if (j+1)%40==0:print(dict(points=j+1,seconds=time.monotonic()-start),flush=True)
    if not all(torch.equal(v.cpu(),frozen[k]) for k,v in net.state_dict().items()) or not torch.equal(slow,slow_before):
        raise ValueError('persistent model changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during diagnostic')
    groups={k:describe(v) for k,v in dict(all=rows,
        disruptive_positive=[r for r in rows if r['flag']],safe=[r for r in rows if not r['flag']],
        cold_missed=[r for r in rows if r['flag'] and not r['cold_timely']],
        cold_fp=[r for r in rows if not r['flag'] and r['cold_alarm']]).items()}
    _durable_json(output/'records.json',dict(records=rows,excluded=excluded))
    result=dict(complete=True,scope=protocol['scope'],groups=groups,coverage=340,excluded=excluded,
        records_sha256=sha(output/'records.json'),protocol_sha256=sha(output/'protocol.json'),
        seconds=time.monotonic()-start,scientific_goal_achieved=False,online_success=False)
    _durable_json(output/'complete.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',type=Path,required=True);p.add_argument('--stream',type=Path,required=True)
    a=p.parse_args();run(a.output,a.points,a.stream)
