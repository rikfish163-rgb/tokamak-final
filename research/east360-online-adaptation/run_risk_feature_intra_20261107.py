"""Direct observed-window risk after signal-driven internal feature updates.

No virtual-risk maximum is used in this experiment. Each temporary encoder
starts at the qualified source and is discarded after its current shot.
"""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from risk_feature_intra_20261107 import FeatureIntra,update
from run_joint_motion_20261107 import FIT


ARMS = ('static','real','wrong_time','hold')


def feature_curves(base,decoder,signal,*,warm=False,feedbacks=ARMS[1:]):
    x,a = j.r.latent.make_views(signal,base.center.device)
    with torch.no_grad():
        static=torch.cat([base(x[lo:lo+j.BATCH],a[lo:lo+j.BATCH])['logit'].sigmoid()
                          for lo in range(0,len(x),j.BATCH)]).cpu().numpy()
    traces={k:static.copy() for k in ('static',*feedbacks)}
    info=dict(releases=[],updates={k:0 for k in feedbacks},weight_norm={},gradient_max={k:0. for k in feedbacks})
    if warm or len(x)<128:
        return traces,info
    models={k:FeatureIntra(base,decoder) for k in feedbacks}
    opts={k:torch.optim.AdamW([v.weight],lr=j.INTRA_LR,weight_decay=0.) for k,v in models.items()}
    values=torch.as_tensor(signal['values'],device=x.device)
    masks=torch.as_tensor(signal['available'],device=x.device)
    lo=127
    for end in list(range(153,len(x),32))+[len(x)]:
        with torch.no_grad():
            for name,model in models.items():
                traces[name][lo:end]=model.risk(x[lo:end],a[lo:end])['logit'].sigmoid().cpu().numpy()
        if end != len(x):
            origin=end-26
            e=dict(x=x[origin:origin+1],a=a[origin:origin+1],
                truth=values[origin+1:origin+26][None],mask=masks[origin+1:origin+26][None])
            for name,model in models.items():
                stats=update(model,opts[name],e,feedback=name)
                info['updates'][name]+=1
                info['gradient_max'][name]=max(info['gradient_max'][name],stats['gradient_norm'])
            info['releases'].append(end)
        lo=end
    for name,model in models.items():
        info['weight_norm'][name]=float((model.weight-model.initial_weight).norm().detach())
    if any(not np.isfinite(v).all() for v in traces.values()):
        raise FloatingPointError('nonfinite internally adapted current risk')
    return traces,info


def predict(out):
    started=time.monotonic()
    j.r.configure()
    base,parent=j.r.load_parent(j.r.DEFAULT_PARENT)
    base.cuda()
    decoder=OrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(FIT/'prepared_predictor.pt',map_location='cuda',weights_only=True))
    reference,cfg,archive,(real,stale,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if (warm!=20 or mapping['calibration_ids']!=cfg['order'][:20]
            or parent['threshold']!=j.r.THRESHOLD):
        raise ValueError('same original20, source and threshold required')
    source_before={k:v.detach().clone() for k,v in base.state_dict().items()}
    decoder_before={k:v.detach().clone() for k,v in decoder.state_dict().items()}
    out.mkdir(parents=True,exist_ok=False)
    (out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='DIRECT_RISK_FEATURE_INTRA360',
        order=cfg['order'],arms=ARMS,threshold=j.r.THRESHOLD,warm=warm,
        parent_sha256=j.PARENT_SHA256,decoder_preparation=str(FIT),lr=j.INTRA_LR,
        prior_weight=j.PRIOR_WEIGHT,clip=1.,stride=32,first_affected=153,
        plastic_parameter='risk.encoder.layers.0.linear2.weight; 8192 scalars',
        frozen='all other risk parameters including risk head; entire trajectory decoder',
        prediction='original current-window risk head on actual observed signals; no virtual risk maximum',
        target='matured actual H25; ordered decoder objective; no risk labels or current-logit teacher',
        hold_control='repeat issuance-point observed values; common matured masks and release clock',
        temporary_reset=True,inter_updates=0,east_outcome_use=False,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in [Path(__file__),
            j.r.ROOT/'risk_feature_intra_20261107.py',j.r.ROOT/'run_joint_trajectory_20261107.py',
            j.r.ROOT/'joint_motion_20261107.py']}))
    records,witnesses=[],[]
    with patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('EAST outcomes unavailable before seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid=note['shot']
            times,baseline,_,_=j.recorded(reference,note,rank,real,stale)
            raw=j.raw_for(sid,cfg['raw_sha256'])
            np.testing.assert_array_equal(times,raw['times'])
            signal=j.r.prepare_actual(raw,rank,mapping)
            traces,info=feature_curves(base,decoder,signal,warm=rank<=warm)
            error=j.require_parity(traces['static'],baseline)
            if rank in (21,181,359):
                stop=min(len(times),max(154,len(times)//2))
                partial={k:signal[k][:stop] for k in ('values','available')}
                truncated,_=feature_curves(base,decoder,partial)
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(truncated[k],traces[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**traces)
            records.append(dict(shot=sid,rank=rank,static_error=error,adaptation=info))
            if rank%20==0:
                print('risk-feature',rank,'/360; seconds',round(time.monotonic()-started,1),flush=True)
    for model,old in [(base,source_before),(decoder,decoder_before)]:
        if any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()):
            raise ValueError('an immutable source or decoder template changed')
    j.r.write_new(out/'completed.json',dict(status='SEALED_DIRECT_RISK_FEATURE360',records=records,
        witnesses=witnesses,seconds=time.monotonic()-started,templates_unchanged=True))
    print('sealed risk-feature360',flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,warm)=j.cohort('fixed360')
    if (done['status']!='SEALED_DIRECT_RISK_FEATURE360' or protocol['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order']):
        raise ValueError('complete sealed current-risk stream required')
    rows={k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid=note['shot'];times,_,_,_=j.recorded(reference,note,rank,real,stale)
        truth,clock=j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as data:
            for name in ARMS:
                rows[name].append(j.score(times,data[name],truth,protocol['threshold']))
    metrics={k:j.counts(v) for k,v in rows.items()}
    if (metrics['static']['tp'],metrics['static']['fp'])!=(116,29):
        raise ValueError('qualified original/static360 changed')
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    j.r.write_new(out/'result.json',dict(status='HDF5_DIRECT_RISK_FEATURE360',metrics=metrics,
        paired_static={k:j.pair(rows[k],rows['static']) for k in ARMS[1:]},
        real_vs_controls={k:j.pair(rows['real'],rows[k]) for k in ('wrong_time','hold')},
        independent_intra_gain=metrics['real']['fpr']<=.20 and metrics['real']['tp']>metrics['static']['tp'],
        prefix20={k:j.counts(v[:20]) for k,v in rows.items()},
        suffix340={k:j.counts(v[20:]) for k,v in rows.items()},rows=rows,independent_confirmation=False))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    {'predict':predict,'score':recount}[args.phase](args.out)
