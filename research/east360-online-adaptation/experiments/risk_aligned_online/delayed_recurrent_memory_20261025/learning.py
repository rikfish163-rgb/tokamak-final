"""Held-shot task teaching and signal-only causal replay of diagnostic memory."""
import numpy as np
import torch

from .. import simple_data
from ..operational_bag_loss_20261025.learning import bag_loss, candidate_ends
from .model import ARMS, patch_signal, endpoint_corrections


def batch(signal, ends, device):
    pairs = [simple_data.window(signal,int(e),128) for e in ends]
    return [torch.as_tensor(np.stack(v),device=device) for v in zip(*pairs)]


@torch.no_grad()
def original_predictions(base, signal):
    """The actual frozen parent, not functional_call with cloned parameters.

    CUDA regression on shot89284's final11 windows found a2.38e-7 probability
    difference solely with the old cloned-parameter comparator. Direct base
    forward and an unchanged-parameter wrapper agree exactly with off.
    All arms use the original frozen tensors; no tolerance is substituted.
    """
    result=[]
    for lo in range(0,len(signal['values']),32):
        x,m=batch(signal,range(lo,min(lo+32,len(signal['values']))),base.center.device)
        result.append(base(x,m)['logit'].sigmoid().cpu().numpy())
    return np.concatenate(result)


def prepare_example(adapter, signal, times, flag, down, threshold):
    ends = candidate_ends(times,flag,down)
    scores=[]
    with torch.no_grad():
        for lo in range(0,len(ends),32):
            x,m=batch(signal,ends[lo:lo+32],adapter.base.center.device)
            h,valid=adapter.tokens(x,m)
            scores.append(adapter.risk(h,valid,h.new_zeros(len(h),92)).cpu().numpy())
    z=np.concatenate(scores)
    chosen=np.unique(np.concatenate((ends[np.linspace(0,len(ends)-1,min(48,len(ends)),dtype=int)],
                                     ends[np.argsort(z,kind='stable')[-16:]])))
    hs,vs=[],[]
    with torch.no_grad():
        for lo in range(0,len(chosen),32):
            x,m=batch(signal,chosen[lo:lo+32],adapter.base.center.device)
            h,v=adapter.tokens(x,m);hs.append(h);vs.append(v)
    h,valid=torch.cat(hs),torch.cat(vs)
    # Risk chunks remain32 in training and replay, avoiding changed GEMM sizes.
    with torch.no_grad():
        base=torch.cat([adapter.risk(h[lo:lo+32],valid[lo:lo+32],h.new_zeros(len(h[lo:lo+32]),92))
                        for lo in range(0,len(h),32)])
    means,complete=patch_signal(signal,adapter.base.center,adapter.base.scale)
    return dict(adapter=adapter,means=means,complete=complete,h=h,valid=valid,
                ends=torch.as_tensor(chosen,device=h.device),flag=bool(flag),
                baseline=float(bag_loss(base.max(),flag,threshold)),
                frozen_peak_full=float(z.max()),frozen_peak_selected=float(base.max()),
                selected_ends=chosen.tolist(),full_candidate_count=len(ends))


def risk_from_trace(example,trace):
    h,v=example['h'],example['valid']
    delta=endpoint_corrections(trace['delta'],example['ends'])
    return torch.cat([example['adapter'].risk(h[lo:lo+32],v[lo:lo+32],delta[lo:lo+32])
                      for lo in range(0,len(h),32)])


def train(memory,examples,threshold,*,progress=None):
    optimizer=torch.optim.Adam(memory.parameters(),lr=.001)
    counts={f:sum(e['flag']==f for e in examples.values()) for f in (False,True)}
    if not all(counts.values()):raise ValueError('both historical classes required')
    best=None
    for epoch in range(11):
        optimizer.zero_grad(set_to_none=True)
        rows=[];objective=0.;auxiliary=0.
        for sid,e in examples.items():
            trace=memory.trace(e['means'],e['complete'],mode='memory')
            logits=risk_from_trace(e,trace)
            loss=bag_loss(logits.max(),e['flag'],threshold)
            delta=loss-e['baseline'];weight=.5/counts[e['flag']]
            risk=delta+delta.relu();aux=trace['forecast_loss']
            total=(risk+.02*aux)*weight
            if not torch.isfinite(total):raise FloatingPointError('nonfinite historical loss')
            total.backward()
            objective+=float(risk.detach())*weight;auxiliary+=float(aux.detach())*weight
            rows.append(dict(shot=sid,flag=e['flag'],loss=float(loss.detach()),delta=float(delta.detach()),
                             forecast_loss=float(aux.detach()),memory_updates=trace['updates'],
                             maximum_memory_norm=float(trace['norm'].max().detach())))
        gradients={k:float(p.grad.norm()) for k,p in memory.named_parameters()}
        norm=float(torch.nn.utils.clip_grad_norm_(memory.parameters(),1.,error_if_nonfinite=True))
        record=dict(epoch=epoch,objective=objective,forecast_loss=auxiliary,per_shot=rows,
                    gradient_norm=norm,parameter_gradient_norms=gradients,eta=float(memory.eta.detach()))
        if best is None or objective<best['objective']:
            best=dict(record,state={k:v.detach().cpu().clone() for k,v in memory.state_dict().items()})
        if progress:progress(record)
        if epoch==10:break
        optimizer.step()
    memory.load_state_dict(best['state'])
    return best


@torch.no_grad()
def predict(adapter,signal,*,arms=('off',)+ARMS):
    if set(signal)!={'values','available'}:raise ValueError('signal-only replay required')
    memory=adapter.memory
    means,complete=patch_signal(signal,adapter.base.center,adapter.base.scale)
    main=memory.trace(means,complete,mode='memory')
    traces={a:main if a=='memory' else memory.trace(means,complete,mode=a,reference_norm=main['norm'])
            for a in arms}
    curves={a:[] for a in arms};raw={a:[] for a in arms}
    for lo in range(0,len(signal['values']),32):
        ends=range(lo,min(lo+32,len(signal['values'])))
        x,m=batch(signal,ends,adapter.base.center.device)
        h,valid=adapter.tokens(x,m)
        indices=torch.as_tensor(list(ends),device=h.device)
        for a,t in traces.items():
            z=adapter.risk(h,valid,endpoint_corrections(t['delta'],indices))
            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite risk replay')
            raw[a].append(z.cpu().numpy());curves[a].append(z.sigmoid().cpu().numpy())
    receipts={a:dict(updates=t['updates'],norm=t['norm'].cpu().tolist(),
                    zero_direction=t['zero_direction'].cpu().tolist(),forecast_loss=float(t['forecast_loss']),
                    first_affected_index=40,first_target_last_index=39,
                    issue_patch_end_indices=(np.arange(len(means))*8+7).tolist()) for a,t in traces.items()}
    return {a:np.concatenate(v) for a,v in curves.items()},receipts,{a:np.concatenate(v) for a,v in raw.items()}


def eligible(results,history_passed):
    main,off=results['memory'],results['off']
    return bool(history_passed and main['tp']>=off['tp'] and main['fp']<=off['fp']
                and main['fpr']<=.20 and main['very_early_over3000']<=off['very_early_over3000']
                and main['bag_disrupt_loss']<=off['bag_disrupt_loss']
                and main['bag_safe_loss']<=off['bag_safe_loss']
                and all(main['bag_balanced_loss']<results[a]['bag_balanced_loss']-1e-6
                        for a in ('off','local','hold','reverse')))
