"""Source-only risk teaching through consecutive matured signal updates."""

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

import run_joint_trajectory_20261107 as j
from probe_risk_masked_utility_20261107 import masked_view
from risk_meta_adam_20261108 import adam_step, direction_step
from risk_meta_step_20261107 import WEIGHT_KEY


class _SignalObjective(nn.Module):
    """Expose the existing signal loss to functional_call without copying it."""

    def __init__(self,model):
        super().__init__()
        self.model=model

    def forward(self,episode):
        signal=self.model.signal_loss(episode)
        prior=(self.model.weight-self.model.initial_weight).square().sum()
        return dict(signal=signal,prior=prior,objective=signal+j.PRIOR_WEIGHT*prior)


def teacher_sequence(model,episodes,query,query_available,labels,*,credit='full',step_rule='adam'):
    """Unroll actual protected AdamW, then apply source future-label credit.

    Caller uses MATH attention for mixed derivatives. Only the final outer-risk
    link is stopped in the paired control: every inner parameter/moment value
    and the signal-maintenance loss remain identical at matched decoder state.
    Labels never enter inner feedback. The original model is not mutated.
    """
    if credit not in ('full','last','none'):
        raise ValueError('risk credit must be full, last, or none')
    if step_rule not in ('adam','momentum_direction'):
        raise ValueError('step rule must be adam or momentum_direction')
    stepper=adam_step if step_rule=='adam' else direction_step
    objective=_SignalObjective(model)
    weight=model.weight
    history=()
    steps=[];signals=[];priors=[]
    for episode in episodes:
        previous_weight,previous_history=weight,history
        losses=functional_call(objective,{'model.risk.'+WEIGHT_KEY:weight},(episode,))
        gradient,=torch.autograd.grad(losses['objective'],weight,create_graph=True)
        step=stepper(weight,gradient,history,lr=j.INTRA_LR)
        if step_rule=='momentum_direction' and step['momentum_norm']==0:
            raise FloatingPointError('zero momentum is not a supported higher-order teaching state')
        weight,history=step['weight'],step['history']
        steps.append(step);signals.append(losses['signal']);priors.append(losses['prior'])
    fast=weight
    if credit=='none':
        fast=model.weight+(fast-model.weight).detach()
    elif credit=='last':
        incoming=previous_weight.detach().requires_grad_(True)
        incoming_history=tuple(g.detach() for g in previous_history)
        last_loss=functional_call(objective,{'model.risk.'+WEIGHT_KEY:incoming},(episodes[-1],))
        gradient,=torch.autograd.grad(last_loss['objective'],incoming,create_graph=True)
        fast=stepper(incoming,gradient,incoming_history,lr=j.INTRA_LR)['weight']
    logit=functional_call(model.risk,{WEIGHT_KEY:fast},(query,query_available))['logit']
    return dict(steps=steps,priors=priors,signal_loss=torch.stack(signals).mean(),logit=logit,
                risk_loss=F.binary_cross_entropy_with_logits(logit,labels))


def sequence_episode(shots,pools,rng,group,device):
    """Four real H25 forecasts released 32 points apart; no autoregression."""
    choices=pools[group]
    i,starts=choices[int(rng.integers(len(choices)))]
    first=int(rng.choice(starts));s=shots[i]
    origins=[first+32*k for k in range(4)];q=origins[-1]+26
    note=dict(shot=s.get('shot',i),origins=origins,query=q,label_time=q+25,group=group)
    return sequence_from_note(s,note,device)


def sequence_from_note(s,note,device):
    """Rebuild the same four-step source episode from its recorded identity."""
    origins=note['origins'];q=note['query'];t=note['label_time']
    if (len(origins)!=4 or origins[0]<127 or any(b-a!=32 for a,b in zip(origins,origins[1:]))
            or q!=origins[-1]+26 or t!=q+25 or t>=len(s['values'])
            or not s['valid'][origins[0]-127:t+1].all()):
        raise ValueError('complete source sequence and its recorded clock required')
    expected=tuple(map(float,note['group'].split('_to_')))
    if (float(s['y'][q]),float(s['y'][t]))!=expected:
        raise ValueError('recorded group no longer matches native source labels')
    episodes=[]
    qx,qa=j.simple_data.window(s,q,128)
    qx,qa=torch.as_tensor(qx[None],device=device),torch.as_tensor(qa[None],device=device)
    for origin in origins:
        x,a=j.simple_data.window(s,origin,128)
        episode={k:torch.as_tensor(v[None],device=device) for k,v in
            dict(x=x,a=a,truth=s['values'][origin+1:origin+26],mask=s['available'][origin+1:origin+26]).items()}
        episode,masked_query=masked_view(episode,qa)
        episodes.append(episode)
    label=torch.as_tensor(s['y'][q+25:q+26],device=device)
    return episodes,qx,masked_query,label,{k:note[k] for k in ('shot','origins','query','label_time','group')}


def probe(out):
    """Real source-only four-step parity and mixed-gradient evidence."""
    from copy import deepcopy
    from pathlib import Path
    import numpy as np
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from risk_feature_intra_20261107 import FeatureIntra,update
    from risk_meta_step_20261107 import MetaOrderedTrajectory
    from probe_risk_masked_utility_20261107 import PARENT
    from run_joint_motion_20261107 import FIT
    from run_risk_teacher_20261107 import GROUPS,teacher_pools

    j.r.configure()
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    state=deepcopy(base.state_dict())
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(FIT/'source_predictor.pt',map_location='cuda',weights_only=True))
    source,_=j.load_source_train()
    excluded=set(v['shot'] for v in j.r.read(PARENT)['records'])
    held=[s for s in source if s['shot'] in excluded]
    pools=teacher_pools(held,steps=4)
    rng=np.random.default_rng(139);records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(12):
            es,qx,qa,label,note=sequence_episode(held,pools,rng,GROUPS[i%3],'cuda')
            model=FeatureIntra(base,decoder);model.decoder.requires_grad_(True)
            native=FeatureIntra(base,decoder)
            optimizer=torch.optim.AdamW([native.weight],lr=j.INTRA_LR,weight_decay=0.,foreach=False)
            full=teacher_sequence(model,es,qx,qa,label)
            parameters=tuple(model.decoder.parameters())
            gf=torch.autograd.grad(full['risk_loss'],parameters)
            last=teacher_sequence(model,es,qx,qa,label,credit='last')
            gl=torch.autograd.grad(last['risk_loss'],parameters)
            control=teacher_sequence(model,es,qx,qa,label,credit='none')
            gc=torch.autograd.grad(control['risk_loss'],parameters,allow_unused=True)
            torch.testing.assert_close(full['logit'],last['logit'],atol=0,rtol=0)
            torch.testing.assert_close(full['logit'],control['logit'],atol=0,rtol=0)
            if any(g is not None and torch.count_nonzero(g) for g in gc):
                raise ValueError('stopped risk credit reached the diagnostic decoder')
            if any(not torch.isfinite(g).all() for g in (*gf,*gl)):
                raise FloatingPointError('nonfinite real four-step mixed derivative')
            errors=[]
            for k,e in enumerate(es):
                update(native,optimizer,e)
                a=full['steps'][k];actual=optimizer.state[native.weight]
                torch.testing.assert_close(a['weight'],native.weight,atol=2e-7,rtol=0)
                for key in ('exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(a[key],actual[key],atol=2e-7,rtol=1e-4)
                errors.append(float((a['weight']-native.weight).abs().max().detach()))
            with torch.no_grad():
                native_logit=native.risk(qx,qa)['logit']
            torch.testing.assert_close(full['logit'],native_logit,atol=3e-6,rtol=0)
            nf=float(torch.stack([v.square().sum() for v in gf]).sum().sqrt())
            nl=float(torch.stack([v.square().sum() for v in gl]).sum().sqrt())
            if min(nf,nl)<=0:
                raise ValueError('both full and last-step source credit must reach the decoder')
            records.append(dict(**note,full_outer_norm=nf,last_outer_norm=nl,
                gradient_difference_norm=float(torch.stack([(a-b).square().sum() for a,b in zip(gf,gl,strict=True)]).sum().sqrt()),
                maximum_weight_error=max(errors),maximum_logit_error=float((full['logit']-native_logit).abs().max().detach()),
                prior_by_step=[float(v.detach()) for v in full['priors']]))
    if any(not torch.equal(v,state[k]) for k,v in base.state_dict().items()):
        raise ValueError('qualified source risk state changed')
    summary=dict(n=len(records),four_steps=True,full_last_none_forward_identical=True,
        all_mixed_gradients_finite_nonzero=True,
        maximum_weight_error=max(v['maximum_weight_error'] for v in records),
        maximum_logit_error=max(v['maximum_logit_error'] for v in records))
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'result.json',dict(status='SOURCE_FOUR_STEP_PATH_VERIFIED',summary=summary,
        records=records,eligible_held_shots={k:len(v) for k,v in pools.items()},seed=139,
        source_TRAIN_only=True,source_classifier_unchanged=True,no_EAST_data=True,
        unroll_scope='four updates including clipping, accumulated moments, and .01 parameter prior',
        not_full_shot_training=True,source_files={str(p):p.read_text() for p in
            [Path(__file__),j.r.ROOT/'risk_meta_adam_20261108.py']}))
    print(summary,flush=True)


def utility(out,fit_root):
    """Actual four-step updates on fixed TRAIN shots excluded from label teaching."""
    import numpy as np
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from risk_feature_intra_20261107 import FeatureIntra,update
    from risk_meta_step_20261107 import MetaOrderedTrajectory
    from run_risk_teacher_20261107 import GROUPS,teacher_pools

    j.r.configure()
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    protocol=j.r.read(fit_root/'protocol.json')
    if protocol['sequence_steps']!=4:
        raise ValueError('four-step fitted teacher required')
    source,_=j.load_source_train()
    excluded=set(protocol['excluded_outer_shots'])
    held=[s for s in source if s['shot'] in excluded]
    pools=teacher_pools(held,steps=4)
    models={}
    for phase in ('source','prepared'):
        for name in ('control','last_step','taught'):
            decoder=MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
            decoder.load_state_dict(torch.load(fit_root/f'{phase}_{name}.pt',map_location='cuda',weights_only=True))
            models[f'{phase}_{name}']=decoder
    rng=np.random.default_rng(139);records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for i in range(48):
            group=GROUPS[i%3]
            es,qx,qa,label,note=sequence_episode(held,pools,rng,group,'cuda')
            current=torch.full_like(label,float(group.split('_to_')[0]))
            with torch.no_grad():
                before=base(qx,qa)['logit']
                future_before=F.binary_cross_entropy_with_logits(before,label)
                current_before=F.binary_cross_entropy_with_logits(before,current)
            rows={}
            for name,decoder in models.items():
                model=FeatureIntra(base,decoder)
                optimizer=torch.optim.AdamW([model.weight],lr=j.INTRA_LR,weight_decay=0.)
                for episode in es:
                    update(model,optimizer,episode)
                with torch.no_grad():
                    after=model.risk(qx,qa)['logit']
                    future_after=F.binary_cross_entropy_with_logits(after,label)
                    current_after=F.binary_cross_entropy_with_logits(after,current)
                rows[name]=dict(future_before=float(future_before),future_after=float(future_after),
                    current_before=float(current_before),current_after=float(current_after),
                    weight_displacement=float((model.weight-model.initial_weight).norm().detach()))
            records.append(dict(**note,variants=rows))
    summary={}
    for group in GROUPS:
        selected=[v for v in records if v['group']==group]
        summary[group]={}
        for name in models:
            rows=[v['variants'][name] for v in selected]
            summary[group][name]=dict(n=len(rows),
                future_improved=sum(v['future_after']<v['future_before'] for v in rows),
                current_improved=sum(v['current_after']<v['current_before'] for v in rows),
                mean_future_delta=float(np.mean([v['future_after']-v['future_before'] for v in rows])))
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'result.json',dict(status='SOURCE_FOUR_STEP_EXCLUDED_UTILITY',summary=summary,
        records=records,seed=139,source_TRAIN_only=True,excluded_from_outer_label_fitting=True,
        original_signal_pretraining_saw_all_TRAIN=True,not_independent_confirmation=True,
        prepared_models_used_ended20_EAST_signals_only=True,not_EAST_efficacy=True))
    print(summary,flush=True)


if __name__=='__main__':
    probe(j.r.ROOT/'probe_outputs/risk_teacher_sequence_numerical_20261108_v1')
