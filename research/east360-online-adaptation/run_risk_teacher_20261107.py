"""Matched source teaching of signal-driven, shot-local risk representation updates."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

import run_joint_trajectory_20261107 as j
from probe_risk_masked_utility_20261107 import PARENT,masked_view,measure
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_step_20261107 import MetaOrderedTrajectory,teacher_step
from run_joint_motion_20261107 import FIT as INITIAL
from run_risk_feature_intra_20261107 import feature_curves


FIT=j.r.ROOT/'probe_outputs/risk_teacher_source_20261107_v1'
GROUPS=('0_to_0','0_to_1','1_to_1')
ARMS=('static','control_real','taught_real','taught_wrong_time','taught_hold')


def teacher_pools(shots,*,steps=1,require_all_groups=True):
    """Complete causal sequences, optionally partial for incremental EOS replay."""
    if not isinstance(steps,int) or steps<1:
        raise ValueError('positive integer feedback count required')
    offset=(steps-1)*32
    width=179+offset
    pools={k:[] for k in GROUPS}
    for i,s in enumerate(shots):
        valid=np.asarray(s['valid'],bool)
        if len(valid)<width:
            continue
        starts=np.flatnonzero(np.convolve(valid.astype(np.int32),np.ones(width,np.int32),'valid')==width)
        origins=starts+127
        for group in GROUPS:
            current,future=map(int,group.split('_to_'))
            selected=origins[(s['y'][origins+offset+26]==current)&(s['y'][origins+offset+51]==future)]
            if len(selected):
                pools[group].append((i,selected))
    if require_all_groups and any(not v for v in pools.values()):
        raise ValueError('all three native current/future source classes need complete episodes')
    return pools


def teacher_episode(shots,pools,rng,group,device):
    choices=pools[group]
    i,origins=choices[int(rng.integers(len(choices)))]
    o=int(rng.choice(origins));q=o+26;s=shots[i]
    x,a=j.simple_data.window(s,o,128)
    qx,qa=j.simple_data.window(s,q,128)
    e={k:torch.as_tensor(v[None],device=device) for k,v in
       dict(x=x,a=a,truth=s['values'][o+1:o+26],mask=s['available'][o+1:o+26]).items()}
    qx,qa=torch.as_tensor(qx[None],device=device),torch.as_tensor(qa[None],device=device)
    e,qa=masked_view(e,qa)
    label=torch.as_tensor(s['y'][q+25:q+26],device=device)
    return e,qx,qa,label,dict(shot=s.get('shot',i),origin=o,query=q,label_time=q+25,group=group)


def fit(out,*,sequence_steps=1):
    if sequence_steps not in (1,4):
        raise ValueError('this experiment compares one or four source feedback steps')
    started=time.monotonic();j.r.configure()
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    original=deepcopy(base.state_dict())
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(INITIAL/'source_predictor.pt',map_location='cuda',weights_only=True))
    names=('control','taught') if sequence_steps==1 else ('control','last_step','taught')
    models={k:FeatureIntra(base,decoder) for k in names}
    for model in models.values():
        model.decoder.requires_grad_(True)
    opts={k:torch.optim.AdamW(v.decoder.parameters(),lr=3e-4,weight_decay=0.) for k,v in models.items()}
    source,ids=j.load_source_train()
    excluded=set(v['shot'] for v in j.r.read(PARENT)['records'])
    outer_shots=[s for s in source if s['shot'] not in excluded]
    pools=teacher_pools(outer_shots,steps=sequence_steps)
    if sequence_steps==4:
        from risk_meta_sequence_20261108 import sequence_episode,teacher_sequence
    rng=np.random.default_rng(137)
    _,cfg,_,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if warm!=20 or mapping['calibration_ids']!=cfg['order'][:20]:
        raise ValueError('original20 unlabeled preparation and fixed input map required')
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'protocol.json',dict(status='SOURCE_SIGNAL_GRADIENT_TEACHING',
        parent_sha256=j.PARENT_SHA256,initial_predictor=str(INITIAL/'source_predictor.pt'),
        source_train_ids=ids,excluded_outer_shots=sorted(excluded),
        source_steps=512,source_episode_batch=1,balanced_groups=GROUPS,sequence_steps=sequence_steps,
        eligible_shots={k:len(v) for k,v in pools.items()},seed=137,
        outer_objective='after-step future-label BCE + same matured signal energy, equal weights',
        outer_lr=3e-4,outer_plastic='diagnostic decoder only; all source risk weights fixed',
        inner=f'{sequence_steps} accumulated AdamW updates, lr1e-4, clip1, no decay, .01 W prior; current risk first FFN matrix',
        target='native y(query+25), not native y(query); original source label function unchanged',
        domain_task='MNM unavailable at past, matured truth, and query; all values unchanged',
        matched_control='control stops final risk link; last_step stops earlier risk credit; signal loss and forward values matched',
        online_optimizer_scope=f'exact {sequence_steps}-step source unroll; full online shots can contain more steps',
        cold20=cfg['order'][:20],cold_steps=128,cold_batch=32,
        east_labels_for_fitting=False,source_val_or_test_selection=False,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in [Path(__file__),
            j.r.ROOT/'risk_meta_step_20261107.py',j.r.ROOT/'meta_energy_20261107.py']+
            ([j.r.ROOT/'risk_meta_sequence_20261108.py',j.r.ROOT/'risk_meta_adam_20261108.py'] if sequence_steps==4 else [])}))
    history=[]
    with sdpa_kernel(SDPBackend.MATH):
        for step in range(512):
            sample=teacher_episode if sequence_steps==1 else sequence_episode
            e,qx,qa,y,note=sample(outer_shots,pools,rng,GROUPS[step%3],'cuda')
            losses={}
            for name,model in models.items():
                model.zero_grad(set_to_none=True)
                if sequence_steps==1:
                    result=teacher_step(model,e,qx,qa,y,stop_update=name=='control')
                else:
                    credit={'control':'none','last_step':'last','taught':'full'}[name]
                    result=teacher_sequence(model,e,qx,qa,y,credit=credit)
                objective=result['risk_loss']+result['signal_loss']
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite source teaching objective')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
                opts[name].step()
                losses[name]=dict(signal=float(result['signal_loss'].detach()),
                    risk=float(result['risk_loss'].detach()),gradient_norm=float(norm))
            history.append(dict(**note,losses=losses))
            if (step+1)%64==0:
                print('source-risk-teacher',step+1,losses,flush=True)
        for name,model in models.items():
            with (out/f'source_{name}.pt').open('xb') as handle:
                torch.save(model.decoder.state_dict(),handle)
        cold=[]
        with patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EAST outcomes unavailable for preparation')):
            for sid in cfg['order'][:20]:
                cold.append(j.r.prepare_actual(j.raw_for(sid,cfg['raw_sha256']),21,mapping))
            preparation={name:j.fit_predictor(base,model.decoder,cold,128,j.SEED+1,f'ended20-{name}')
                         for name,model in models.items()}
    for model in [base,*[v.risk for v in models.values()]]:
        if any(not torch.equal(v,original[k]) for k,v in model.state_dict().items()):
            raise ValueError('source risk state changed during decoder teaching')
    for name,model in models.items():
        with (out/f'prepared_{name}.pt').open('xb') as handle:
            torch.save(model.decoder.state_dict(),handle)
    j.r.write_new(out/'fit.json',dict(status='SOURCE_TEACHER_FIT_COMPLETE',history=history,
        preparation=preparation,source_classifier_unchanged=True,seconds=time.monotonic()-started))
    print('source teaching completed',round(time.monotonic()-started,2),'s',flush=True)


def comparison_curves(base,decoders,signal,*,warm=False,feedback_controls=True,feedback_plan=None):
    """Real feedback for teaching controls, and three feedbacks for taught."""
    plan=feedback_plan
    if plan is None:
        plan={name:(('real','wrong_time','hold') if name=='taught' and feedback_controls else ('real',))
              for name in decoders}
    if (set(plan)!=set(decoders) or any(not values or len(set(values))!=len(values)
            or set(values)-{'real','wrong_time','hold'} for values in plan.values())):
        raise ValueError('a nonempty unique feedback list is required for each selected decoder')
    traces={};info={}
    for name,decoder in decoders.items():
        feedbacks=tuple(plan[name])
        curves,info[name]=feature_curves(base,decoder,signal,warm=warm,feedbacks=feedbacks)
        if 'static' not in traces:
            traces['static']=curves['static']
        else:
            j.require_parity(curves['static'],traces['static'])
        traces.update({f'{name}_{key}':curves[key] for key in feedbacks})
    return traces,info


def predict(out,*,fit_root=FIT,feedback_controls=True,feedback_plan=None):
    started=time.monotonic();j.r.configure()
    base,parent=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    fitted=j.r.read(fit_root/'fit.json')
    fitted_names=tuple(fitted['history'][0]['losses'])
    if fitted['status']!='SOURCE_TEACHER_FIT_COMPLETE' or fitted_names not in (('control','taught'),('control','last_step','taught'),('last_step',),('recorded_sampling','pooled_sampling')):
        raise ValueError('a completed matched source teaching run is required')
    fit_protocol=j.r.read(fit_root/'protocol.json')
    sequence_steps=fit_protocol['sequence_steps']
    if fit_protocol.get('deterministic_algorithms',False):
        torch.use_deterministic_algorithms(True)
    plan=feedback_plan
    if plan is None:
        plan={name:(('real','wrong_time','hold') if name=='taught' and feedback_controls else ('real',))
              for name in fitted_names}
    if not plan or set(plan)-set(fitted_names):
        raise ValueError('requested decoders must come from the completed source fit')
    names=tuple(plan)
    decoders={k:MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval() for k in names}
    for name,model in decoders.items():
        model.load_state_dict(torch.load(fit_root/f'prepared_{name}.pt',map_location='cuda',weights_only=True))
    arms=('static',*[f'{name}_{feedback}' for name,values in plan.items() for feedback in values])
    reference,cfg,archive,(real,stale,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if warm!=20 or mapping['calibration_ids']!=cfg['order'][:20] or parent['threshold']!=j.r.THRESHOLD:
        raise ValueError('same original20, source and threshold required')
    originals={name:deepcopy(model.state_dict()) for name,model in dict(base=base,**decoders).items()}
    out.mkdir(parents=True,exist_ok=False);(out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='RISK_TEACHER_INTRA360',order=cfg['order'],
        arms=arms,threshold=j.r.THRESHOLD,warm=warm,fit=str(fit_root),parent_sha256=j.PARENT_SHA256,
        source_teacher=f'{sequence_steps}-step source-only teaching; actual online optimizer remains accumulated AdamW',
        deterministic_algorithms=fit_protocol.get('deterministic_algorithms',False),
        feedback_controls_run=any(key!='real' for values in plan.values() for key in values),feedback_plan=plan,
        inner_lr=1e-4,prior_weight=.01,stride=32,first_affected=153,
        prediction='current actual window risk; fixed head, no virtual-risk maximum',
        temporary_reset=True,inter_updates=0,east_outcome_use=False,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in [Path(__file__),
            j.r.ROOT/'run_risk_feature_intra_20261107.py',j.r.ROOT/'risk_feature_intra_20261107.py']}))
    records=[];witnesses=[]
    with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EAST outcomes unavailable before seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid=note['shot'];times,baseline,_,_=j.recorded(reference,note,rank,real,stale)
            raw=j.raw_for(sid,cfg['raw_sha256']);np.testing.assert_array_equal(times,raw['times'])
            signal=j.r.prepare_actual(raw,rank,mapping)
            traces,info=comparison_curves(base,decoders,signal,warm=rank<=warm,feedback_plan=plan)
            j.require_parity(traces['static'],baseline)
            if rank in (21,181,359):
                stop=min(len(times),max(154,len(times)//2));partial={k:signal[k][:stop] for k in ('values','available')}
                short,_=comparison_curves(base,decoders,partial,feedback_plan=plan)
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(short[k],traces[k][:stop]) for k in arms)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**traces)
            records.append(dict(shot=sid,rank=rank,**info))
            if rank%20==0:
                print('risk-teacher',rank,'/360; seconds',round(time.monotonic()-started,1),flush=True)
    for name,model in dict(base=base,**decoders).items():
        if any(not torch.equal(v,originals[name][k]) for k,v in model.state_dict().items()):
            raise ValueError('an immutable source or decoder template changed')
    j.r.write_new(out/'completed.json',dict(status='SEALED_RISK_TEACHER360',records=records,
        witnesses=witnesses,seconds=time.monotonic()-started,templates_unchanged=True))
    print('sealed risk teacher360',flush=True)


def utility(out):
    """Fixed diagnostic episodes excluded from the new outer-label fitting."""
    j.r.configure()
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    models={}
    for phase in ('source','prepared'):
        for name in ('control','taught'):
            model=MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
            model.load_state_dict(torch.load(FIT/f'{phase}_{name}.pt',map_location='cuda',weights_only=True))
            models[f'{phase}_{name}']=model
    cases=j.r.read(PARENT)['records']
    advanced=j.simple_data.advanced_times()
    shots={sid:j.simple_data.attach_outcome(j.simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
           for sid in dict.fromkeys(v['shot'] for v in cases)}
    excluded=set(j.r.read(FIT/'protocol.json')['excluded_outer_shots'])
    if any(v['shot'] not in excluded for v in cases):
        raise ValueError('utility cases must be excluded from outer-label teaching')
    records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for old in cases:
            s=shots[old['shot']];o,q=old['origin'],old['query']
            x,a=j.simple_data.window(s,o,128);qx,qa=j.simple_data.window(s,q,128)
            e={k:torch.as_tensor(v[None],device='cuda') for k,v in
               dict(x=x,a=a,truth=s['values'][o+1:o+26],mask=s['available'][o+1:o+26]).items()}
            qx,qa=torch.as_tensor(qx[None],device='cuda'),torch.as_tensor(qa[None],device='cuda')
            e,qa=masked_view(e,qa)
            rows={name:measure(base,model,e,qx,qa,float(s['y'][q]),float(s['y'][q+25]))
                  for name,model in models.items()}
            records.append(dict(shot=old['shot'],group=old['group'],origin=o,query=q,variants=rows))
    summary={}
    for group in GROUPS:
        selected=[v for v in records if v['group']==group]
        summary[group]={}
        for name in models:
            values=[v['variants'][name] for v in selected]
            summary[group][name]=dict(n=len(values),
                future_improved=sum(v['future_risk_after']<v['future_risk_before'] for v in values),
                current_improved=sum(v['current_risk_after']<v['current_risk_before'] for v in values),
                mean_future_delta=float(np.mean([v['future_risk_after']-v['future_risk_before'] for v in values])),
                median_cosine=float(np.median([v['cosine'] for v in values if v['cosine'] is not None])))
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'result.json',dict(status='SOURCE_TEACHER_EXCLUDED_EPISODE_UTILITY',
        summary=summary,records=records,source_TRAIN_only=True,excluded_from_outer_teaching=True,
        initial_signal_pretraining_used_all_TRAIN=True,not_independent_confirmation=True,
        prepared_models_used_ended20_EAST_signals_only=True))
    print(summary,flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,_)=j.cohort('fixed360')
    if done['status']!='SEALED_RISK_TEACHER360' or protocol['order']!=cfg['order'] or [v['shot'] for v in done['records']]!=cfg['order']:
        raise ValueError('complete sealed original360 required')
    arms=tuple(protocol['arms'])
    rows={k:[] for k in arms}
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];times,_,_,_=j.recorded(reference,note,rank,real,stale)
        truth,clock=j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as data:
            for name in arms:
                rows[name].append(j.score(times,data[name],truth,protocol['threshold']))
    metrics={k:j.counts(v) for k,v in rows.items()}
    if (metrics['static']['tp'],metrics['static']['fp'])!=(116,29):
        raise ValueError('qualified original/static360 changed')
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    comparisons={name:(j.pair(rows[a],rows[b]) if a in rows and b in rows else None)
                 for name,a,b in [('teaching_increment','taught_real','control_real'),
                                  ('continuous_credit_increment','taught_real','last_step_real'),
                                  ('positive_pool_increment','pooled_sampling_real','recorded_sampling_real')]}
    j.r.write_new(out/'result.json',dict(status='HDF5_RISK_TEACHER360',metrics=metrics,rows=rows,
        paired_static={k:j.pair(rows[k],rows['static']) for k in arms[1:]},
        taught_vs_feedback=({k:j.pair(rows['taught_real'],rows[k]) for k in ('taught_wrong_time','taught_hold') if k in rows}
                           if 'taught_real' in rows else {}),
        primary_comparisons_not_run=[name for name,value in comparisons.items() if value is None],
        feedback_controls_not_run=[k for k in ('taught_wrong_time','taught_hold') if k not in rows],
        prefix20={k:j.counts(v[:20]) for k,v in rows.items()},suffix340={k:j.counts(v[20:]) for k,v in rows.items()},
        independent_confirmation=False,**comparisons))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


def compare_controls(out,parent):
    """Pair separately sealed last-step controls with their recorded real arm."""
    a,b=(j.r.read(path/'protocol.json') for path in (out,parent))
    for key in ('order','fit','threshold','warm','parent_sha256'):
        if a[key]!=b[key]:
            raise ValueError('feedback comparisons require the same parent, stream and alarm contract')
    controls,real=(j.r.read(path/'result.json') for path in (out,parent))
    if controls['status']!='HDF5_RISK_TEACHER360' or real['status']!='HDF5_RISK_TEACHER360':
        raise ValueError('both streams must be completely sealed and HDF5 scored')
    if set(controls['rows'])!={'static','last_step_wrong_time','last_step_hold'} or 'last_step_real' not in real['rows']:
        raise ValueError('only the declared last-step followup controls can be paired here')
    for x,y in zip(controls['rows']['static'],real['rows']['static'],strict=True):
        if any(x[key]!=y[key] for key in ('shot','disruptive','down','status','first_index','lead_ms')):
            raise ValueError('same scored static trace and HDF5 outcome required')
    maximum=0.
    for rank,sid in enumerate(a['order'],1):
        filename=f'{rank:04d}_{sid}.npz'
        with np.load(out/'shots'/filename,allow_pickle=False) as x,np.load(parent/'shots'/filename,allow_pickle=False) as y:
            maximum=max(maximum,j.require_parity(x['static'],y['static']))
    metrics={'real':real['metrics']['last_step_real'],
             **{name:controls['metrics'][f'last_step_{name}'] for name in ('wrong_time','hold')}}
    pairs={name:j.pair(real['rows']['last_step_real'],controls['rows'][f'last_step_{name}'])
           for name in ('wrong_time','hold')}
    j.r.write_new(out/'paired_real_controls.json',dict(status='MATCHED_LAST_STEP_REAL_FEEDBACK_CONTROLS',
        parent_run=str(parent),control_run=str(out),metrics=metrics,real_vs_feedback=pairs,
        maximum_static_error=maximum,selection_after_primary_development_result=True,
        no_model_training=True,independent_confirmation=False,
        time_order_caveat='wrong_time reverses truth and mask positions; do not assume identical temporal support'))
    print({k:(v['tp'],v['fp']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','utility','predict','score'))
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    {'fit':fit,'utility':utility,'predict':predict,'score':recount}[args.phase](args.out)
