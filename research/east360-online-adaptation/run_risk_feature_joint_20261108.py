"""Matched actual-risk intra/inter learning, with natural end-of-shot outcomes.

CUDA launch requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts.
Static and intra curves reuse the unchanged completed same-checkpoint replay.
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
from experiments.risk_aligned_online.joint_lora_20261104.training import eos_schedule
from risk_feature_inter_20261108 import train_inter
from risk_meta_step_20261107 import MetaOrderedTrajectory,WEIGHT_KEY
from run_risk_feature_intra_20261107 import feature_curves


REFERENCE=j.r.ROOT/'probe_outputs/risk_positive_pool360_20261108_v2'
FIT=j.r.ROOT/'probe_outputs/risk_positive_pool_source_20261108_v2'
ARMS=('static','intra','inter','both')


def ended_risk_records(shot,outcome):
    """Include scored startup windows; EOS risk teaching needs no future signal."""
    flag,down=outcome
    labels,valid=j.r.io23.labels_at(shot['times'],flag,down,125.)
    records=[]
    for query in range(0,len(labels),8):
        if not valid[query] or (flag and labels[query]
                and round(float((down-shot['times'][query])*1000),9)<=30):
            continue
        group='safe' if not flag else ('positive' if labels[query] else 'earlier')
        records.append(dict(shot=shot['shot'],query=query,y=float(labels[query]),group=group))
    return records


def slow_curves(base,decoder,signal,*,warm=False):
    """Read both arms from one persistent parent; discard the temporary copy."""
    curves,info=feature_curves(base,decoder,signal,warm=warm,feedbacks=('real',))
    return dict(inter=curves['static'],both=curves['real']),info


def predict(out,*,reference_run=REFERENCE,fit_root=FIT,device='cuda'):
    started=time.monotonic();j.r.configure();torch.use_deterministic_algorithms(True)
    base,parent=j.r.load_parent(j.r.DEFAULT_PARENT);base.to(device)
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
    decoder.load_state_dict(torch.load(fit_root/'prepared_recorded_sampling.pt',map_location=device,weights_only=True))
    reference,cfg,archive,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    cached,complete=(j.r.read(reference_run/f'{k}.json') for k in ('protocol','completed'))
    if (complete['status']!='SEALED_RISK_TEACHER360' or cached['order']!=cfg['order']
            or [v['shot'] for v in complete['records']]!=cfg['order']
            or Path(cached['fit']).resolve()!=fit_root.resolve() or cached['warm']!=warm or warm!=20
            or cached['parent_sha256']!=j.PARENT_SHA256 or cached['threshold']!=parent['threshold']
            or parent['threshold']!=j.r.THRESHOLD or not cached['deterministic_algorithms']
            or mapping['calibration_ids']!=cfg['order'][:20]):
        raise ValueError('completed unchanged parent/intra replay, original20, map and threshold required')
    for name in ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py'):
        path=j.r.ROOT/name
        if cached['source_files'][str(path)]!=path.read_text():
            raise ValueError('cached intra implementation changed; do not reuse its curves')
    original=deepcopy(base.state_dict());decoder_original=deepcopy(decoder.state_dict())
    slow=deepcopy(base);initial_weight=slow.get_parameter(WEIGHT_KEY).detach().clone()
    out.mkdir(parents=True,exist_ok=False);(out/'shots').mkdir();(out/'inter').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='RISK_FEATURE_INTRA_INTER360',arms=ARMS,order=cfg['order'],
        threshold=j.r.THRESHOLD,parent_sha256=j.PARENT_SHA256,warm=20,fit=str(fit_root),
        reference_run=str(reference_run),reused_arms={'static':'static','intra':'recorded_sampling_real'},
        intra='same arrived H25, stride32, first affected153, AdamW1e-4, .01 shot-start W prior',
        inter='from shot21 EOS only; raw-logit BCE + .01 squared distance to immutable source W',
        inter_scope=WEIGHT_KEY,inter_parameters=initial_weight.numel(),inter_lr=1e-4,inter_steps=8,
        inter_batch=64,replay='half current, half earlier; reuse declared shot-uniform strata and seed131+rank',
        initial_supervised_warm_steps=0,inter_optimizer_reset_each_EOS=True,fast_state_promotion=False,
        shared_slow_for_inter_and_both=True,feedback='all natural EOS; not a costly-query policy experiment',
        cold20_labels='replay only after their own seal and completed signal-map preparation',
        label_window_ms=125.,labels='raw HDF5 meta only; late positive points excluded, not relabelled',
        inter_queries='every8 observed points from index0; causal padding; no future-signal requirement',
        deterministic_algorithms=True,independent_confirmation=False,alarm_changes=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),j.r.ROOT/'risk_feature_inter_20261108.py')}))
    raw_ended={};shots={};truth={};history=[];records=[];witnesses=[]
    eos_updates=0
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];filename=f'{rank:04d}_{sid}.npz'
        with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcome unavailable before this shot is sealed')):
            raw=j.raw_for(sid,cfg['raw_sha256']);signal=j.r.prepare_actual(raw,rank,mapping)
            with np.load(reference_run/'shots'/filename,allow_pickle=False) as saved:
                traces=dict(static=saved['static'].copy(),intra=saved['recorded_sampling_real'].copy())
            if any(len(v)!=len(raw['times']) for v in traces.values()):
                raise ValueError('cached curve and actual signal clocks differ')
            extra,info=slow_curves(slow,decoder,signal,warm=rank<=warm);traces.update(extra)
            if rank<=21:
                j.require_parity(traces['inter'],traces['static'])
                j.require_parity(traces['both'],traces['intra'])
            if rank in (21,181,359):
                stop=min(len(raw['times']),max(154,len(raw['times'])//2))
                short,_=slow_curves(slow,decoder,{k:signal[k][:stop] for k in ('values','available')})
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(short[k],traces[k][:stop]) for k in ('inter','both'))))
            j.r.save_arrays(out/'shots'/filename,**traces)
        # Only a physically saved prediction may be followed by this label read.
        truth[sid]=j.r.io23.read_outcome('EAST',sid)
        if rank<=warm:
            raw_ended[sid]=raw
        if rank==warm:
            for key,value in raw_ended.items():
                # The completed cold map may transform past training signals;
                # their already-sealed predictions are never recomputed.
                shots[key]=dict(j.r.prepare_actual(value,21,mapping),times=value['times'],device='EAST')
            history=[row for key in cfg['order'][:warm] for row in ended_risk_records(shots[key],truth[key])]
        elif rank>warm:
            shots[sid]=dict(signal,times=raw['times'],device='EAST')
            current=ended_risk_records(shots[sid],truth[sid])
            rows,schedule=eos_schedule(current,history,seed=131+rank,steps=8,batch_size=64)
            with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('training uses supplied ended outcomes only')),patch.object(j.r.io23,'read_signals',side_effect=RuntimeError('training uses supplied ended signals only')):
                losses=train_inter(slow,shots,rows,schedule,initial_weight,device=device,lr=1e-4)
            selected=[[rows[i] for i in draw['indices']] for draw in schedule]
            j.r.write_new(out/'inter'/f'{rank:04d}_{sid}.json',dict(shot=sid,rank=rank,
                outcome=truth[sid],previous_ids=cfg['order'][:rank-1],selected=selected,history=losses,
                updated_only_after_seal=True,source_weight_displacement=float((slow.get_parameter(WEIGHT_KEY)-initial_weight).norm())))
            history.extend(current);eos_updates+=1
        records.append(dict(shot=sid,rank=rank,adaptation=info))
        if rank%20==0:
            print('feature-joint',rank,'/',len(cfg['order']),'EOS',eos_updates,'seconds',round(time.monotonic()-started,1),flush=True)
    for model,state in ((base,original),(decoder,decoder_original)):
        if any(not torch.equal(v,state[k]) for k,v in model.state_dict().items()):
            raise ValueError('immutable source or decoder template changed')
    if any(not torch.equal(v,original[k]) for k,v in slow.state_dict().items() if k!=WEIGHT_KEY):
        raise ValueError('persistent learning changed a frozen model component')
    with (out/'final_slow_weight.pt').open('xb') as handle:
        torch.save(slow.get_parameter(WEIGHT_KEY).detach().cpu(),handle)
    j.r.write_new(out/'completed.json',dict(status='SEALED_RISK_FEATURE_JOINT360',records=records,
        witnesses=witnesses,eos_updates=eos_updates,label_shots=len(truth),seconds=time.monotonic()-started,
        source_and_decoder_unchanged=True,slow_frozen_parts_unchanged=True))
    print('sealed feature joint complete',flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,_)=j.cohort('fixed360')
    if (done['status']!='SEALED_RISK_FEATURE_JOINT360' or protocol['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order'] or done['eos_updates']!=340):
        raise ValueError('complete original360 with 340 declared EOS updates required')
    rows={k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];times,_,_,_=j.recorded(reference,note,rank,real,stale)
        actual,clock=j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as values:
            for name in ARMS:
                rows[name].append(j.score(times,values[name],actual,protocol['threshold']))
    metrics={k:j.counts(v) for k,v in rows.items()}
    if ((metrics['static']['tp'],metrics['static']['fp'])!=(116,29)
            or (metrics['intra']['tp'],metrics['intra']['fp'])!=(118,30)):
        raise ValueError('reused same-source static/intra outcomes changed')
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b]) for a,b in
        (('intra','static'),('inter','static'),('both','inter'),('both','intra'))}
    j.r.write_new(out/'result.json',dict(status='HDF5_RISK_FEATURE_JOINT360',metrics=metrics,rows=rows,
        comparisons=comparisons,capped_comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b],capped=True)
            for a,b in (('intra','static'),('inter','static'),('both','inter'),('both','intra'))},
        independent_confirmation=False,feedback='natural EOS outcomes; not a query-budget result'))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'));parser.add_argument('out',type=Path)
    args=parser.parse_args();{'predict':predict,'score':recount}[args.phase](args.out)
