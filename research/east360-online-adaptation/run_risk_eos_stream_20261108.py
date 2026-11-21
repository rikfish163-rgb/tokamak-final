"""End-of-shot teaching of signal gradients on a shared persistent risk model.

CUDA launch requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts.
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
from risk_feature_inter_20261108 import train_inter
from risk_meta_sequence_20261108 import sequence_episode
from risk_meta_step_20261107 import MetaOrderedTrajectory,WEIGHT_KEY
from run_risk_feature_joint_20261108 import FIT,slow_curves
from run_risk_teacher_20261107 import GROUPS,teacher_pools


PARENT=j.r.ROOT/'probe_outputs/risk_feature_joint360_20261108_v1'
DECODERS=('fixed','signal','guided')
ARMS=('inter',*DECODERS)


class EndedTasks:
    """Incrementally index only supplied ended EAST shots, without HDF I/O."""

    def __init__(self,*,seed=143):
        self.views=[];self.pools={g:[] for g in GROUPS};self.rng=np.random.default_rng(seed)

    def add(self,shot,outcome):
        if shot['device']!='EAST':
            raise ValueError('online task replay is EAST-only')
        flag,down=outcome
        y,valid=j.r.io23.labels_at(shot['times'],flag,down,125.)
        view=dict(shot,y=y,valid=valid)
        index=len(self.views);self.views.append(view)
        partial=teacher_pools([view],steps=4,require_all_groups=False)
        for group,entries in partial.items():
            for _,origins in entries:
                if flag:
                    q=origins+3*32+26
                    origins=origins[np.round((down-shot['times'][q])*1000,9)>30]
                if len(origins):
                    self.pools[group].append((index,origins))

    def draw(self,group,device):
        if not self.pools[group]:
            raise ValueError('the requested native risk group has no ended task')
        return sequence_episode(self.views,self.pools,self.rng,group,device)


def parallel_curves(slow,decoders,signal,*,warm=False):
    traces={};info={}
    for name,decoder in decoders.items():
        pair,info[name]=slow_curves(slow,decoder,signal,warm=warm)
        if 'inter' not in traces:
            traces['inter']=pair['inter']
        else:
            j.require_parity(pair['inter'],traces['inter'])
        traces[name]=pair['both']
    return traces,info


def predict(out,*,parent_run=PARENT,fit_root=FIT,device='cuda'):
    from risk_eos_teacher_20261108 import teach_decoder

    started=time.monotonic();j.r.configure();torch.use_deterministic_algorithms(True)
    base,metadata=j.r.load_parent(j.r.DEFAULT_PARENT);base.to(device)
    prior,done=(j.r.read(parent_run/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if (done['status']!='SEALED_RISK_FEATURE_JOINT360' or prior['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order'] or done['eos_updates']!=len(cfg['order'])-20
            or prior['parent_sha256']!=j.PARENT_SHA256 or warm!=20
            or prior['threshold']!=metadata['threshold'] or metadata['threshold']!=j.r.THRESHOLD
            or Path(prior['fit']).resolve()!=fit_root.resolve() or not prior['deterministic_algorithms']
            or mapping['calibration_ids']!=cfg['order'][:20]):
        raise ValueError('completed same-source joint replay and unchanged stream contract required')
    for name in ('run_risk_feature_joint_20261108.py','risk_feature_inter_20261108.py'):
        path=j.r.ROOT/name
        if prior['source_files'][str(path)]!=path.read_text():
            raise ValueError('persistent replay implementation changed')
    decoders={name:MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
              for name in DECODERS}
    state=torch.load(fit_root/'prepared_recorded_sampling.pt',map_location=device,weights_only=True)
    for decoder in decoders.values():
        decoder.load_state_dict(state);decoder.requires_grad_(False)
    optimizers={name:torch.optim.AdamW(decoders[name].parameters(),lr=3e-4,weight_decay=0.)
                for name in ('signal','guided')}
    original=deepcopy(base.state_dict());fixed_original=deepcopy(decoders['fixed'].state_dict())
    slow=deepcopy(base);initial_weight=slow.get_parameter(WEIGHT_KEY).detach().clone()
    out.mkdir(parents=True,exist_ok=False);(out/'shots').mkdir();(out/'eos').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='EOS_SIGNAL_GRADIENT_TEACHING360',arms=ARMS,
        order=cfg['order'],threshold=j.r.THRESHOLD,warm=20,parent_sha256=j.PARENT_SHA256,
        parent_run=str(parent_run),fit=str(fit_root),feedback='natural EOS; not costly-query policy',
        persistent_risk='exact recorded 8x64 batches; raw-logit BCE and .01 source-W prior',
        intra='unchanged H25/stride32/first affected153; fixed risk head; no score correction',
        decoder_EOS='one four-feedback task per EOS, signal energy + native y(query+25) BCE',
        credit={'signal':'none','guided':'last'},decoder_lr=3e-4,decoder_clip=1.,decoder_decay=0.,
        decoder_optimizer='persistent across EOS; initialized at shot21',
        task_seed=143,task_groups=GROUPS,task_source='only ended EAST shots; uniform shot then origin',
        task_queries='strict lead>30ms; outer label at query+25 still uses native125ms definition',
        initial_supervised_warm_steps=0,shared_slow=True,fast_promotion=False,
        independent_confirmation=False,deterministic_algorithms=True,
        source_files={str(p):p.read_text() for p in (Path(__file__),j.r.ROOT/'risk_eos_teacher_20261108.py',
            j.r.ROOT/'risk_meta_sequence_20261108.py',j.r.ROOT/'risk_meta_adam_20261108.py',
            j.r.ROOT/'run_risk_teacher_20261107.py')}))
    bank=EndedTasks(seed=143);raw_cold={};shots={};truth={};records=[];witnesses=[]
    eos_updates=0;maximum_reference_error=0.;maximum_training_error=0.
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];filename=f'{rank:04d}_{sid}.npz'
        with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcome unavailable before seal')):
            raw=j.raw_for(sid,cfg['raw_sha256']);signal=j.r.prepare_actual(raw,rank,mapping)
            curves,info=parallel_curves(slow,decoders,signal,warm=rank<=warm)
            with np.load(parent_run/'shots'/filename,allow_pickle=False) as old:
                maximum_reference_error=max(maximum_reference_error,
                    j.require_parity(curves['inter'],old['inter']),j.require_parity(curves['fixed'],old['both']))
            if rank in (21,181,359):
                stop=min(len(raw['times']),max(154,len(raw['times'])//2))
                short,_=parallel_curves(slow,decoders,{k:signal[k][:stop] for k in ('values','available')})
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(short[k],curves[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/filename,**curves)
        # Teacher tasks and labels are admitted only after this shot is saved.
        truth[sid]=j.r.io23.read_outcome('EAST',sid)
        if rank<=warm:
            raw_cold[sid]=raw
        if rank==warm:
            for key,value in raw_cold.items():
                shots[key]=dict(j.r.prepare_actual(value,21,mapping),times=value['times'],device='EAST')
                bank.add(shots[key],truth[key])
        elif rank>warm:
            shots[sid]=dict(signal,times=raw['times'],device='EAST');bank.add(shots[sid],truth[sid])
            receipt=j.r.read(parent_run/'inter'/f'{rank:04d}_{sid}.json')
            if (receipt['shot']!=sid or receipt['rank']!=rank or receipt['previous_ids']!=cfg['order'][:rank-1]
                    or tuple(receipt['outcome'])!=truth[sid]):
                raise ValueError('recorded EOS identity, history or raw HDF5 outcome changed')
            rows=[];schedule=[]
            for step,selected in enumerate(receipt['selected'],1):
                if len(selected)!=64 or sum(v['shot']==sid for v in selected)!=32:
                    raise ValueError('declared half-current replay budget changed')
                start=len(rows);rows.extend(selected)
                schedule.append(dict(step=step,indices=list(range(start,len(rows)))))
            group=GROUPS[eos_updates%len(GROUPS)]
            with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EOS fitting uses supplied outcomes only')),patch.object(j.r.io23,'read_signals',side_effect=RuntimeError('EOS fitting uses supplied signals only')):
                risk_history=train_inter(slow,shots,rows,schedule,initial_weight,device=device,lr=1e-4)
                es,qx,qa,label,task=bank.draw(group,device)
                teaching={name:teach_decoder(slow,decoders[name],optimizers[name],es,qx,qa,label,
                    credit='none' if name=='signal' else 'last') for name in ('signal','guided')}
            if len(risk_history)!=len(receipt['history']):
                raise ValueError('recorded persistent update count changed')
            error=max(abs(a[k]-b[k]) for a,b in zip(risk_history,receipt['history'],strict=True)
                      for k in ('loss','risk_loss','prior','gradient_norm'))
            maximum_training_error=max(maximum_training_error,error)
            j.r.write_new(out/'eos'/f'{rank:04d}_{sid}.json',dict(shot=sid,rank=rank,task=task,
                task_ended_ids=cfg['order'][:rank],native_future_label=float(label.item()),
                teaching=teaching,persistent_history=risk_history,reference_training_error=error))
            eos_updates+=1
        records.append(dict(shot=sid,rank=rank,adaptation=info))
        if rank%20==0:
            print('EOS-teacher',rank,'/',len(cfg['order']),'updates',eos_updates,
                  'seconds',round(time.monotonic()-started,1),flush=True)
    if any(not torch.equal(v,original[k]) for k,v in base.state_dict().items()):
        raise ValueError('immutable source changed')
    if any(not torch.equal(v,original[k]) for k,v in slow.state_dict().items() if k!=WEIGHT_KEY):
        raise ValueError('persistent risk changed a frozen component')
    if any(not torch.equal(v,fixed_original[k]) for k,v in decoders['fixed'].state_dict().items()):
        raise ValueError('fixed decoder reference changed')
    prior_weight=torch.load(parent_run/'final_slow_weight.pt',map_location=device,weights_only=True)
    exact=torch.equal(slow.get_parameter(WEIGHT_KEY),prior_weight)
    for name in ('signal','guided'):
        with (out/f'final_decoder_{name}.pt').open('xb') as handle:
            torch.save(decoders[name].state_dict(),handle)
    j.r.write_new(out/'completed.json',dict(status='SEALED_EOS_TEACHER360',records=records,
        eos_updates=eos_updates,label_shots=len(truth),witnesses=witnesses,seconds=time.monotonic()-started,
        maximum_reference_error=maximum_reference_error,maximum_training_error=maximum_training_error,
        persistent_final_weight_exact=exact,source_and_fixed_decoder_unchanged=True))
    print('sealed EOS teacher complete',flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,_)=j.cohort('fixed360')
    if (done['status']!='SEALED_EOS_TEACHER360' or protocol['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order'] or done['eos_updates']!=340):
        raise ValueError('complete sealed original360 with declared EOS teacher updates required')
    rows={k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];times,_,_,_=j.recorded(reference,note,rank,real,stale)
        actual,clock=j.raw_truth(j.r.io23.path_for('EAST',sid),sid);np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as values:
            for name in ARMS:
                rows[name].append(j.score(times,values[name],actual,protocol['threshold']))
    parent=j.r.read(Path(protocol['parent_run'])/'result.json')
    if j.pair(rows['inter'],parent['rows']['inter'])!=dict(tp_gained=[],tp_lost=[],fp_added=[],fp_removed=[]):
        raise ValueError('persistent parent prediction outcomes changed')
    if j.pair(rows['fixed'],parent['rows']['both'])!=dict(tp_gained=[],tp_lost=[],fp_added=[],fp_removed=[]):
        raise ValueError('fixed-decoder combined outcomes changed')
    metrics={k:j.counts(v) for k,v in rows.items()}
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    pairs={a+'_vs_'+b:j.pair(rows[a],rows[b]) for a,b in
        (('signal','fixed'),('guided','signal'),('guided','fixed'),('guided','inter'))}
    j.r.write_new(out/'result.json',dict(status='HDF5_EOS_TEACHER360',metrics=metrics,rows=rows,
        comparisons=pairs,capped_comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b],capped=True) for a,b in
            (('signal','fixed'),('guided','signal'),('guided','fixed'),('guided','inter'))},
        reused_source_metrics={k:parent['metrics'][k] for k in ('static','intra')},
        independent_confirmation=False,feedback='natural EOS; not query-budget superiority'))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'));parser.add_argument('out',type=Path)
    args=parser.parse_args();{'predict':predict,'score':recount}[args.phase](args.out)
