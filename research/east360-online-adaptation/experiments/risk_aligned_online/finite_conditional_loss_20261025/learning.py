"""Low-capacity policies trained through actual finite internal-encoder updates."""
import numpy as np
import torch
from torch import nn

from .. import simple_data
from ..clock_channel_loss_20261024.learning import bounded_step,_risk_gradient,future_groups
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..feedback_cone_feasibility_20261024.stream import signal_parts
from ..history_tangent_20261023.learning import flat,logit,apply_step,matched_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder

ARMS=('linear','finite_global','finite_context','uniform','permuted','hold')


class LossPolicy(nn.Module):
    def __init__(self,kind,*,device):
        super().__init__()
        if kind not in ('global','context'):raise ValueError('declared policy kind required')
        self.kind=kind
        self.logits=nn.Parameter(torch.zeros(23,device=device,dtype=torch.float64))
        if kind=='context':self.slope=nn.Parameter(torch.zeros_like(self.logits))
        else:self.register_buffer('slope',torch.zeros_like(self.logits))
        self.strength=nn.Parameter(torch.zeros((),device=device,dtype=torch.float64))

    def forward(self,context,observed):
        if context.shape!=(23,) or observed.shape!=(23,) or observed.dtype!=torch.bool or not torch.isfinite(context).all():
            raise ValueError('finite23 arrived residual context and observed mask required')
        if not observed.any():
            # Explicit no-observation behavior, preserving an exactly zero
            # derivative for every policy parameter. No replacement update.
            return torch.zeros_like(self.logits)*sum(p.sum() for p in self.parameters())
        scores=self.logits+self.slope*context.to(self.logits)
        return self.strength.sigmoid()*scores.masked_fill(~observed,-torch.inf).softmax(0)

    def export(self):
        return dict(kind=self.kind,state={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})


def load_policy(record,*,device):
    policy=LossPolicy(record['kind'],device=device)
    policy.load_state_dict({k:torch.as_tensor(v,device=device,dtype=torch.float64) for k,v in record['state'].items()})
    return policy


def feedback_context(model,signal,count,fast,*,hold,device):
    if set(signal)!={'values','available'} or not 256<=count<=len(signal['values']):
        raise ValueError('signal-only arrived prefix at least256 points required')
    x,m,truth,tm,_=forecast_examples({k:v[:count] for k,v in signal.items()},count,model.cfg,device)
    valid=target_mask(x,m,truth,tm)
    target=x[:,-1:].expand_as(truth) if hold else truth
    prediction=model.predict_fast(x,m,fast)['forecast'].detach()
    scaled=torch.asinh((torch.where(valid,target,model.base.center)-model.base.center)/model.base.scale)
    error=(scaled-prediction).masked_fill(~valid,0)
    n=valid.sum((0,1));context=(error.sum((0,1))/n.clamp_min(1)).tanh()
    return context.detach().double(),n>0


def policy_step(parts,coeff):
    if parts.shape[0]!=23 or coeff.shape!=(23,) or not torch.isfinite(parts).all() or not torch.isfinite(coeff).all():
        raise ValueError('finite channel parts and coefficients required')
    scale=parts.double().norm(dim=1).max()
    if scale==0:return torch.zeros_like(parts[0])
    return (.01*(coeff.double()@parts.double())/scale).to(parts)


def outer_gradient(policy,context,observed,parts,scale,risk_gradient):
    if float(scale)<=0:raise ValueError('positive historical gradient scale required')
    coefficients=policy(context,observed)
    # r is evaluated at theta AFTER the actual finite update. G and M are
    # constants because the inner step always starts at the frozen parent.
    d_coeff=-.01*(parts.double()@risk_gradient.detach().double())/scale
    surrogate=(coefficients*d_coeff.detach()).sum()
    return flat(torch.autograd.grad(surrogate,tuple(policy.parameters()))).detach()


def aggregate_objective(rows):
    result=[];total=0.;gradient=None
    for group in ('safe','early','danger'):
        take=[r for r in rows if r['group']==group];ids=sorted({r['shot'] for r in take})
        if not ids:raise ValueError(f'missing historical group {group}')
        for sid in ids:
            local=[r for r in take if r['shot']==sid]
            delta=float(np.mean([r['delta'] for r in local]))
            g=np.mean([r['gradient'] for r in local],axis=0)
            if not np.isfinite(delta) or not np.isfinite(g).all():raise FloatingPointError('nonfinite outer objective')
            factor=2. if delta>0 else 1.
            total+=(delta+max(delta,0))/(3*len(ids))
            term=factor*g/(3*len(ids));gradient=term if gradient is None else gradient+term
            result.append(dict(group=group,shot=sid,blocks=len(local),delta=delta,harm_multiplier=factor))
    return total,gradient,result


def evaluate_policy(model,policy,cache,shots,*,device='cuda'):
    records=[]
    for item in cache:
        sid,count=item['shot'],item['count'];signal=shots[sid]['signal']
        parts=item['parts'].to(device);context=item['context'].to(device);mask=item['observed'].to(device)
        fast=initial_encoder(model)
        coeff=policy(context,mask).detach();step=policy_step(parts,coeff);changed=apply_step(fast,step)
        for r in item['rows']:
            ends=future_groups(shots[sid]['times'],count,shots[sid]['flag'],shots[sid]['down'])[r['group']]
            risk,receipt=_risk_gradient(model,signal,ends,changed,r['label'],device=device)
            gradient=outer_gradient(policy,context,mask,parts,item['scale'],risk).cpu().numpy()
            records.append(dict(shot=sid,group=r['group'],delta=receipt['loss']-r['risk']['loss'],gradient=gradient))
    return aggregate_objective(records)


def train_policy(model,kind,cache,shots,*,device='cuda',progress=None):
    policy=LossPolicy(kind,device=device);optimizer=torch.optim.Adam(policy.parameters(),lr=.05)
    best=None;best_value=float('inf');logs=[]
    for epoch in range(11):
        loss,gradient,detail=evaluate_policy(model,policy,cache,shots,device=device)
        if loss<best_value:best_value=loss;best=policy.export()
        record=dict(epoch=epoch,objective=loss,per_shot=detail,strength=float(policy.strength.detach().sigmoid()),
                    gradient_norm=float(np.linalg.norm(gradient)),policy=policy.export())
        logs.append(record)
        if progress is not None:progress(kind,record)
        if epoch==10:break
        offset=0
        for p in policy.parameters():
            p.grad=torch.as_tensor(gradient[offset:offset+p.numel()],device=device,dtype=p.dtype).reshape_as(p).clone()
            offset+=p.numel()
        assert offset==len(gradient)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
    return best,logs


def predict(model,signal,records,linear_weights,*,device='cuda'):
    if set(signal)!={'values','available'}:raise ValueError('signal-only input required')
    policies={k:load_policy(v,device=device) for k,v in records.items()}
    if set(policies)!={'global','context'}:raise ValueError('both fitted policies required')
    states={a:initial_encoder(model) for a in ('off',)+ARMS}
    curves={a:[] for a in states};receipts={a:[] for a in ARMS};n=len(signal['values'])
    perm=torch.as_tensor(np.random.default_rng(131).permutation(23),device=device)
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            windows=[simple_data.window(signal,e,128) for e in range(start,min(hi,start+32))]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            with torch.no_grad():
                for arm in states:curves[arm].append(logit(model,x,m,states[arm]).sigmoid().cpu().numpy())
        if hi<256 or hi==n:continue
        fast=initial_encoder(model)
        real,rr=signal_parts(model,signal,hi,fast,hold=False,device=device)
        fake,fr=signal_parts(model,signal,hi,fast,hold=True,device=device)
        c,mask=feedback_context(model,signal,hi,fast,hold=False,device=device)
        fc,fmask=feedback_context(model,signal,hi,fast,hold=True,device=device)
        with torch.no_grad():
            coeff={k:p(c,mask) for k,p in policies.items()};hold=policies['context'](fc,fmask)
        zero=torch.zeros(sum(v.numel() for v in fast.values()),device=device)
        main=zero if real is None else policy_step(real,coeff['context']);length=float(main.double().norm())
        for arm in ARMS:
            if arm=='finite_context':step=main
            elif arm=='finite_global':step=zero if real is None else policy_step(real,coeff['global'])
            elif arm=='linear':step=zero if real is None else bounded_step(real,linear_weights)
            elif arm=='uniform':step=zero if real is None else matched_step(real.sum(0),length)
            elif arm=='permuted':step=zero if real is None else matched_step((coeff['context'][perm]@real.double()).to(real),length)
            else:step=zero if fake is None else matched_step((hold@fake.double()).to(fake),length)
            states[arm]=apply_step(fast,step)
            actual=flat([states[arm][k].detach()-v.detach() for k,v in fast.items()])
            r=dict(fr if arm=='hold' else rr)
            r.update(actual_step_norm=float(actual.double().norm()),requested_step_norm=float(step.double().norm()),
                main_requested_norm=length,gradient_steps=int(bool(step.any())),reset='parent',
                control_zero_direction=bool(not step.any() and length>0))
            if arm in ('finite_global','finite_context'):
                r['coefficients']=coeff[arm.removeprefix('finite_')].cpu().tolist()
            receipts[arm].append(r)
    return {a:np.concatenate(v) for a,v in curves.items()},receipts


def eligible(results):
    main,off=results['finite_context'],results['off']
    return (main['tp']>=off['tp'] and main['fp']<=off['fp']
        and main['danger_loss']<=off['danger_loss'] and main['safe_loss']<=off['safe_loss']
        and all(main['balanced_loss']<results[a]['balanced_loss']-1e-6 for a in ('off','linear','finite_global','uniform','permuted','hold')))
