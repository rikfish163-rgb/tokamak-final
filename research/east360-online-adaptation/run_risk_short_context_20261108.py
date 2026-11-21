"""Widen source auxiliary teaching to real, causally padded short contexts.

The qualified source risk model is not trained. CUDA requires the process-local
CUBLAS_WORKSPACE_CONFIG=:4096:8 setting before Python starts.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_sequence_20261108 import sequence_from_note as full_episode,teacher_sequence
from risk_meta_step_20261107 import MetaOrderedTrajectory
from risk_short_context_20261108 import context_pools,sequence_from_note as context_episode
from run_joint_motion_20261107 import FIT as INITIAL
from run_risk_positive_pool_20261108 import FIT as REFERENCE,census


OUT=j.r.ROOT/'probe_outputs/risk_short_context_source_20261108_v1'
STEPS=512
REFERENCE_STREAM=j.r.ROOT/'probe_outputs/risk_positive_pool360_20261108_v2'
ARMS=('static','control_full','control_early','short_full','short_early')


def short_notes(shots,pools,history,*,seed=145):
    """Keep the native label group at every recorded step, not its old origin."""
    rng=np.random.default_rng(seed);notes=[]
    for previous in history:
        group=previous['group'];choices=pools[group]
        i,origins=choices[int(rng.integers(len(choices)))];first=int(rng.choice(origins))
        q=first+122;s=shots[i]
        expected=tuple(map(float,group.split('_to_')))
        if (float(s['y'][q]),float(s['y'][q+25]))!=expected:
            raise ValueError('sampled source episode changed its native current/future label group')
        notes.append(dict(shot=s['shot'],origins=[first+32*k for k in range(4)],
                          query=q,label_time=q+25,group=group))
    return notes


def fit(out=OUT,*,reference_fit=REFERENCE,initial_root=INITIAL,device='cuda'):
    started=time.monotonic();j.r.configure();torch.use_deterministic_algorithms(True)
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.to(device)
    original=deepcopy(base.state_dict())
    prior,done=(j.r.read(reference_fit/f'{k}.json') for k in ('protocol','fit'))
    if (done['status']!='SOURCE_TEACHER_FIT_COMPLETE' or not prior['deterministic_algorithms']
            or prior['parent_sha256']!=j.PARENT_SHA256 or prior['source_steps']!=STEPS
            or prior['sequence_steps']!=4
            or Path(prior['initial_predictor']).resolve()!=(initial_root/'source_predictor.pt').resolve()):
        raise ValueError('same deterministic four-feedback source reference required')
    source,ids=j.load_source_train()
    if ids!=prior['source_train_ids']:
        raise ValueError('canonical source TRAIN membership changed')
    excluded=set(prior['excluded_outer_shots']);shots=[s for s in source if s['shot'] not in excluded]
    by_id={s['shot']:s for s in shots}
    recorded=[dict(item['recorded_note']) for item in done['history']]
    if len(recorded)!=STEPS or any(v['shot'] not in by_id for v in recorded):
        raise ValueError('complete recorded source schedule with unchanged exclusions required')
    pools=context_pools(shots,minimum_origin=7);expanded=short_notes(shots,pools,recorded)
    full_pools=context_pools(shots,minimum_origin=127)
    positive_coverage={name:len({i for group in ('0_to_1','1_to_1') for i,_ in values[group]})
                       for name,values in (('control',full_pools),('short_context',pools))}
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
    decoder.load_state_dict(torch.load(initial_root/'source_predictor.pt',map_location=device,weights_only=True))
    models={name:FeatureIntra(base,decoder) for name in ('control','short_context')}
    for model in models.values():
        model.decoder.requires_grad_(True)
    optimizers={name:torch.optim.AdamW(model.decoder.parameters(),lr=3e-4,weight_decay=0.)
                for name,model in models.items()}
    _,cfg,_,(_,_,warm)=j.cohort('fixed360');mapping=j.r.read(j.r.COLD/'input_map.json')
    if warm!=20 or mapping['calibration_ids']!=cfg['order'][:20]:
        raise ValueError('unchanged ended20 unlabelled signal preparation required')
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'protocol.json',dict(status='SOURCE_SHORT_CONTEXT_TEACHING',
        source_train_ids=ids,excluded_outer_shots=sorted(excluded),parent_sha256=j.PARENT_SHA256,
        reference_fit=str(reference_fit),initial_predictor=str(initial_root/'source_predictor.pt'),
        source_steps=STEPS,sequence_steps=4,batch=1,lr=3e-4,seed=145,
        origin_control='recorded origins >=127',origin_short_context='uniform shot then valid origin >=7',
        control_census=census(recorded),short_census=census(expanded),
        short_first_origin_below127=sum(v['origins'][0]<127 for v in expanded),
        eligible_shots={k:len(v) for k,v in pools.items()},eligible_future_positive_shots=positive_coverage,
        objective='unchanged four-feedback signal energy + last-step future native-label BCE',
        source_labels='unchanged max(25ms,AdvancedTime); auxiliary outer target y(query+25)',
        intra_steps='same AdamW1e-4, clip1, .01 shot-start W prior',
        cold20=cfg['order'][:20],cold_steps=128,cold_batch=32,
        risk_classifier_training=False,east_label_fitting=False,source_val_or_test_selection=False,
        jtext_online=False,deterministic_algorithms=True,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),
            j.r.ROOT/'risk_short_context_20261108.py',j.r.ROOT/'risk_meta_sequence_20261108.py',
            j.r.ROOT/'risk_meta_adam_20261108.py')}))
    history=[]
    with sdpa_kernel(SDPBackend.MATH):
        for step,(a,b) in enumerate(zip(recorded,expanded,strict=True),1):
            losses={}
            for name,note in (('control',a),('short_context',b)):
                model=models[name];model.zero_grad(set_to_none=True)
                builder=full_episode if name=='control' else context_episode
                episodes,qx,qa,label,_=builder(by_id[note['shot']],note,device)
                result=teacher_sequence(model,episodes,qx,qa,label,credit='last')
                objective=result['risk_loss']+result['signal_loss']
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite short-context source teaching objective')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
                optimizers[name].step()
                losses[name]=dict(signal=float(result['signal_loss'].detach()),
                    risk=float(result['risk_loss'].detach()),gradient_norm=float(norm))
            history.append(dict(step=step,control=a,short_context=b,losses=losses))
            if step%64==0:
                print('short-context source',step,'/',STEPS,losses,flush=True)
        reference_source=torch.load(reference_fit/'source_recorded_sampling.pt',map_location=device,weights_only=True)
        control=models['control'].decoder.state_dict()
        if any(not torch.equal(control[k],v) for k,v in reference_source.items()):
            raise ValueError('full-context decoder no longer reproduces the deterministic source reference')
        for name,model in models.items():
            with (out/f'source_{name}.pt').open('xb') as handle:
                torch.save(model.decoder.state_dict(),handle)
        with patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EAST outcomes unavailable for auxiliary preparation')):
            cold=[j.r.prepare_actual(j.raw_for(sid,cfg['raw_sha256']),21,mapping) for sid in cfg['order'][:20]]
            preparation={name:j.fit_predictor(base,model.decoder,cold,128,j.SEED+1,f'ended20-{name}')
                         for name,model in models.items()}
    reference_prepared=torch.load(reference_fit/'prepared_recorded_sampling.pt',map_location=device,weights_only=True)
    if any(not torch.equal(models['control'].decoder.state_dict()[k],v) for k,v in reference_prepared.items()):
        raise ValueError('full-context decoder no longer reproduces the deterministic ended20 reference')
    for model in [base,*[m.risk for m in models.values()]]:
        if any(not torch.equal(v,original[k]) for k,v in model.state_dict().items()):
            raise ValueError('qualified source risk state changed during auxiliary teaching')
    for name,model in models.items():
        with (out/f'prepared_{name}.pt').open('xb') as handle:
            torch.save(model.decoder.state_dict(),handle)
    j.r.write_new(out/'fit.json',dict(status='SOURCE_SHORT_CONTEXT_FIT_COMPLETE',history=history,
        preparation=preparation,source_control_exact=True,prepared_control_exact=True,
        source_classifier_unchanged=True,seconds=time.monotonic()-started))
    print('short-context fit complete',round(time.monotonic()-started,2),'s',flush=True)


def comparison_curves(base,decoders,signal,*,warm=False):
    """Cross two frozen auxiliary predictors with two shot-local clocks."""
    from risk_early_context_20261108 import early_curves
    from run_risk_feature_intra_20261107 import feature_curves

    if set(decoders)!=set(('control','short')):
        raise ValueError('the fitted control and short-context predictors are required')
    curves={};info={}
    for name,decoder in decoders.items():
        full,full_info=feature_curves(base,decoder,signal,warm=warm,feedbacks=('real',))
        early,early_info=early_curves(base,decoder,signal,warm=warm)
        if 'static' not in curves:
            curves['static']=full['static']
        j.require_parity(full['static'],curves['static'])
        j.require_parity(early['static'],curves['static'])
        curves[name+'_full']=full['real'];info[name+'_full']=full_info
        curves[name+'_early']=early['real'];info[name+'_early']=early_info
    return curves,info


def predict(out,*,fit_root=OUT,reference_run=REFERENCE_STREAM,device='cuda'):
    started=time.monotonic();j.r.configure();torch.use_deterministic_algorithms(True)
    base,metadata=j.r.load_parent(j.r.DEFAULT_PARENT);base.to(device)
    fitted,teaching=(j.r.read(fit_root/f'{k}.json') for k in ('fit','protocol'))
    prior,done=(j.r.read(reference_run/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if (fitted['status']!='SOURCE_SHORT_CONTEXT_FIT_COMPLETE'
            or not fitted['source_control_exact'] or not fitted['prepared_control_exact']
            or not fitted['source_classifier_unchanged'] or teaching['parent_sha256']!=j.PARENT_SHA256
            or teaching['risk_classifier_training'] or teaching['east_label_fitting'] or teaching['jtext_online']
            or not teaching['deterministic_algorithms'] or teaching['cold20']!=cfg['order'][:20]
            or done['status']!='SEALED_RISK_TEACHER360' or prior['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order'] or warm!=20
            or prior['parent_sha256']!=j.PARENT_SHA256 or prior['threshold']!=metadata['threshold']
            or metadata['threshold']!=j.r.THRESHOLD or not prior['deterministic_algorithms']
            or Path(prior['fit']).resolve()!=Path(teaching['reference_fit']).resolve()
            or mapping['calibration_ids']!=cfg['order'][:20]):
        raise ValueError('completed matched source fit and unchanged source/full360/cold20/threshold required')
    for name in ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py'):
        path=j.r.ROOT/name
        if prior['source_files'][str(path)]!=path.read_text():
            raise ValueError('reference full-feedback implementation changed')
    decoders={name:MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
              for name in ('control','short')}
    for name,decoder in decoders.items():
        checkpoint='control' if name=='control' else 'short_context'
        decoder.load_state_dict(torch.load(fit_root/f'prepared_{checkpoint}.pt',map_location=device,weights_only=True))
        decoder.requires_grad_(False)
    originals=[deepcopy(m.state_dict()) for m in (base,*decoders.values())]
    out.mkdir(parents=True,exist_ok=False);(out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='SOURCE_CONTEXT_CLOCK_CROSS360',arms=ARMS,
        order=cfg['order'],threshold=j.r.THRESHOLD,warm=20,parent_sha256=j.PARENT_SHA256,
        reference_run=str(reference_run),fit=str(fit_root),inter_updates=0,east_outcome_use=False,
        origins=dict(full=127,early=31),first_affected=dict(full=153,early=57),
        origin_cadence=32,horizon=25,early_grid='31,63,95 then unchanged 127+32k',
        adaptation='temporary first FFN W only; AdamW1e-4, .01 shot-start prior, clip1',
        update_budget='same full-H25 cadence; count extra matured startup updates explicitly',
        source_classifier_immutable=True,temporary_reset=True,alarm_changes=False,
        deterministic_algorithms=True,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),
            j.r.ROOT/'risk_early_context_20261108.py',j.r.ROOT/'risk_feature_intra_20261107.py',
            j.r.ROOT/'run_risk_feature_intra_20261107.py')}))
    records=[];witnesses=[];maximum_reference_error=0.
    with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcomes unavailable before full-stream seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid=note['shot'];raw=j.raw_for(sid,cfg['raw_sha256'])
            signal=j.r.prepare_actual(raw,rank,mapping)
            curves,info=comparison_curves(base,decoders,signal,warm=rank<=warm)
            with np.load(reference_run/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as old:
                maximum_reference_error=max(maximum_reference_error,
                    j.require_parity(curves['static'],old['static']),
                    j.require_parity(curves['control_full'],old['recorded_sampling_real']))
            if rank in (21,181,359):
                stop=min(len(raw['times']),max(154,len(raw['times'])//2))
                short,_=comparison_curves(base,decoders,
                    {k:signal[k][:stop] for k in ('values','available')},warm=False)
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(short[k],curves[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**curves)
            records.append(dict(shot=sid,rank=rank,adaptation=info))
            if rank%20==0:
                print('source-context cross',rank,'/',len(cfg['order']),
                      'seconds',round(time.monotonic()-started,1),flush=True)
    for model,old in zip((base,*decoders.values()),originals,strict=True):
        if any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()):
            raise ValueError('source or a fixed auxiliary predictor changed')
    j.r.write_new(out/'completed.json',dict(status='SEALED_SOURCE_CONTEXT_CLOCK360',records=records,
        witnesses=witnesses,maximum_reference_error=maximum_reference_error,
        templates_unchanged=True,seconds=time.monotonic()-started))
    print('sealed source-context cross complete',flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,_)=j.cohort('fixed360')
    if (done['status']!='SEALED_SOURCE_CONTEXT_CLOCK360' or protocol['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order']):
        raise ValueError('complete sealed source-context stream is required before raw-HDF5 scoring')
    rows={k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];times,_,_,_=j.recorded(reference,note,rank,real,stale)
        actual,clock=j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as arrays:
            for name in ARMS:
                rows[name].append(j.score(times,arrays[name],actual,protocol['threshold']))
    parent=j.r.read(Path(protocol['reference_run'])/'result.json')
    empty=dict(tp_gained=[],tp_lost=[],fp_added=[],fp_removed=[])
    for name,previous in (('static','static'),('control_full','recorded_sampling_real')):
        if j.pair(rows[name],parent['rows'][previous])!=empty:
            raise ValueError('reference source or original intra-feedback outcomes changed')
    metrics={k:j.counts(v) for k,v in rows.items()}
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    comparisons=(('control_early','control_full'),('short_full','control_full'),
        ('short_early','short_full'),('short_early','control_early'),
        ('short_early','control_full'),('short_early','static'))
    j.r.write_new(out/'result.json',dict(status='HDF5_SOURCE_CONTEXT_CLOCK360',metrics=metrics,rows=rows,
        comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b]) for a,b in comparisons},
        capped_comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b],capped=True) for a,b in comparisons},
        prefix20={k:j.counts(v[:20]) for k,v in rows.items()},
        update_counts={k:sum(v['adaptation'][k]['updates']['real'] for v in done['records']) for k in ARMS[1:]},
        teaching_clock_interaction={k:metrics['short_early'][k]-metrics['short_full'][k]
            -metrics['control_early'][k]+metrics['control_full'][k] for k in ('tp','fp','j')},
        inter_updates=0,independent_confirmation=False))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','predict','score'),nargs='?',default='fit')
    parser.add_argument('out',type=Path,nargs='?')
    args=parser.parse_args()
    if args.phase=='fit':
        fit(args.out or OUT)
    elif args.out is None:
        parser.error('predict and score require an output directory')
    else:
        {'predict':predict,'score':recount}[args.phase](args.out)
