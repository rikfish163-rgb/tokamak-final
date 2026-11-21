"""Matched, deterministic source teaching with different positive-shot sampling.

Run CUDA with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before starting Python.
This process-local setting is required by the deterministic CUDA matmul path.
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

import run_joint_trajectory_20261107 as j
from risk_feature_intra_20261107 import FeatureIntra
from risk_meta_sequence_20261108 import sequence_from_note,teacher_sequence
from risk_meta_step_20261107 import MetaOrderedTrajectory
from run_joint_motion_20261107 import FIT as INITIAL
from run_risk_teacher_20261107 import teacher_pools


OLD=j.r.ROOT/'probe_outputs/risk_teacher_sequence_source_20261108_v1'
FIT=j.r.ROOT/'probe_outputs/risk_positive_pool_source_20261108_v2'
NOTE_KEYS=('shot','origins','query','label_time','group')


def positive_origins(pools):
    """One entry per eligible future-positive shot, not one per current class."""
    by_shot={}
    for group in ('0_to_1','1_to_1'):
        for i,origins in pools[group]:
            by_shot.setdefault(i,[]).append(origins)
    return [(i,np.unique(np.concatenate(by_shot[i]))) for i in sorted(by_shot)]


def replace_positive_notes(shots,pools,history,*,seed=141):
    """Keep each negative and the future-positive indicator at each step."""
    choices=positive_origins(pools)
    rng=np.random.default_rng(seed);notes=[]
    for old in history:
        if old['group'].endswith('_to_0'):
            notes.append({k:old[k] for k in NOTE_KEYS})
            continue
        i,origins=choices[int(rng.integers(len(choices)))]
        first=int(rng.choice(origins));s=shots[i]
        starts=[first+32*k for k in range(4)];q=starts[-1]+26
        if float(s['y'][q+25])!=1.:
            raise ValueError('positive pool contains a nonpositive native future target')
        notes.append(dict(shot=s['shot'],origins=starts,query=q,label_time=q+25,
                          group=f"{int(s['y'][q])}_to_1"))
    return notes


def census(notes):
    return dict(sequences=len(notes),unique_shots=len({v['shot'] for v in notes}),
        groups=dict(Counter(v['group'] for v in notes)),
        group_shots={g:len({v['shot'] for v in notes if v['group']==g})
                     for g in sorted({v['group'] for v in notes})})


def fit(out=FIT):
    started=time.monotonic();j.r.configure()
    # Repeated real mixed derivatives differed without this setting. The archived
    # fit did not set it, so both sampling arms are fitted and evaluated afresh;
    # neither arm imports the archived 118/31 outcome as its own result.
    torch.use_deterministic_algorithms(True)
    base,_=j.r.load_parent(j.r.DEFAULT_PARENT);base.cuda()
    original=deepcopy(base.state_dict())
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(INITIAL/'source_predictor.pt',map_location='cuda',weights_only=True))
    models={name:FeatureIntra(base,decoder) for name in ('recorded_sampling','pooled_sampling')}
    for model in models.values():
        model.decoder.requires_grad_(True)
    opts={k:torch.optim.AdamW(v.decoder.parameters(),lr=3e-4,weight_decay=0.) for k,v in models.items()}
    source,ids=j.load_source_train()
    previous=j.r.read(OLD/'protocol.json');old_fit=j.r.read(OLD/'fit.json')
    if previous['source_train_ids']!=ids or previous['sequence_steps']!=4:
        raise ValueError('same canonical source TRAIN and four-step parent required')
    excluded=set(previous['excluded_outer_shots'])
    shots=[s for s in source if s['shot'] not in excluded]
    by_id={s['shot']:s for s in shots}
    pools=teacher_pools(shots,steps=4)
    old=[{k:v[k] for k in NOTE_KEYS} for v in old_fit['history']]
    new=replace_positive_notes(shots,pools,old,seed=141)
    if len(old)!=512 or [v['group'].endswith('_to_1') for v in old]!=[v['group'].endswith('_to_1') for v in new]:
        raise ValueError('identical source budget and stepwise future labels required')
    if any(a!=b for a,b in zip(old,new,strict=True) if a['group'].endswith('_to_0')):
        raise ValueError('a recorded negative episode changed')
    _,cfg,_,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if warm!=20 or mapping['calibration_ids']!=cfg['order'][:20]:
        raise ValueError('original20 and numerical map required')
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'protocol.json',dict(status='POSITIVE_SHOT_POOL_CONTRAST',
        source_train_ids=ids,excluded_outer_shots=sorted(excluded),sequence_steps=4,
        source_steps=512,batch=1,lr=3e-4,positive_resampling_seed=141,initial_predictor=str(INITIAL/'source_predictor.pt'),
        previous_fit=str(OLD),parent_sha256=j.PARENT_SHA256,
        future_positive_count=sum(v['group'].endswith('_to_1') for v in old),
        negatives_unchanged=True,stepwise_future_labels_unchanged=True,
        sampling='uniform shot in union of 0_to_1 and 1_to_1, then uniform eligible origin',
        old_census=census(old),new_census=census(new),
        objective='same last-step risk credit + four-step mean signal energy',
        cold20=cfg['order'][:20],cold_steps=128,cold_batch=32,
        risk_classifier_training=False,east_label_fitting=False,source_val_or_test_selection=False,
        deterministic_algorithms=True,comparison='fresh full360 for both fitted arms; no archived outcome reuse',
        source_files={str(p):p.read_text() for p in [Path(__file__),
            j.r.ROOT/'risk_meta_sequence_20261108.py',j.r.ROOT/'risk_meta_adam_20261108.py']}))
    history=[]
    with sdpa_kernel(SDPBackend.MATH):
        for i,(a,b) in enumerate(zip(old,new,strict=True)):
            losses={}
            for name,note in (('recorded_sampling',a),('pooled_sampling',b)):
                model=models[name];model.zero_grad(set_to_none=True)
                es,qx,qa,label,_=sequence_from_note(by_id[note['shot']],note,'cuda')
                result=teacher_sequence(model,es,qx,qa,label,credit='last')
                objective=result['risk_loss']+result['signal_loss']
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite matched sampling objective')
                objective.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.,error_if_nonfinite=True)
                opts[name].step()
                losses[name]=dict(signal=float(result['signal_loss'].detach()),
                    risk=float(result['risk_loss'].detach()),gradient_norm=float(norm))
            history.append(dict(**b,recorded_note=a,losses=losses))
            if (i+1)%64==0:
                print('paired-positive-pool',i+1,losses,flush=True)
        archived=torch.load(OLD/'source_last_step.pt',map_location='cuda',weights_only=True)
        recorded=models['recorded_sampling'].decoder.state_dict()
        archived_source_exact=all(torch.equal(recorded[k],v) for k,v in archived.items())
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
            raise ValueError('qualified source risk state changed')
    for name,model in models.items():
        with (out/f'prepared_{name}.pt').open('xb') as handle:
            torch.save(model.decoder.state_dict(),handle)
    j.r.write_new(out/'fit.json',dict(status='SOURCE_TEACHER_FIT_COMPLETE',history=history,
        preparation=preparation,archived_source_exact=archived_source_exact,
        archived_outcomes_reused=False,deterministic_algorithms=True,
        source_classifier_unchanged=True,seconds=time.monotonic()-started))
    print('positive-pool fit complete',round(time.monotonic()-started,2),
          's; both sampling arms require fresh full360 evaluation',flush=True)


if __name__=='__main__':
    fit()
