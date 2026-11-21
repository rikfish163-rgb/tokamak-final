"""Compare full and partially arrived feedback with a fixed source classifier."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

import run_joint_trajectory_20261107 as j
from risk_meta_step_20261107 import MetaOrderedTrajectory
from risk_partial_feedback_20261108 import partial_curves


REFERENCE=j.r.ROOT/'probe_outputs/risk_positive_pool360_20261108_v2'
FIT=j.r.ROOT/'probe_outputs/risk_positive_pool_source_20261108_v2'
ARMS=('static','full','partial_delayed','partial_early')


def predict(out,*,reference_run=REFERENCE,fit_root=FIT,device='cuda'):
    started=time.monotonic();j.r.configure();torch.use_deterministic_algorithms(True)
    base,metadata=j.r.load_parent(j.r.DEFAULT_PARENT);base.to(device)
    prior,done=(j.r.read(reference_run/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(_,_,warm)=j.cohort('fixed360')
    mapping=j.r.read(j.r.COLD/'input_map.json')
    if (done['status']!='SEALED_RISK_TEACHER360' or prior['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order'] or warm!=20
            or prior['parent_sha256']!=j.PARENT_SHA256 or prior['threshold']!=metadata['threshold']
            or metadata['threshold']!=j.r.THRESHOLD or Path(prior['fit']).resolve()!=fit_root.resolve()
            or not prior['deterministic_algorithms'] or mapping['calibration_ids']!=cfg['order'][:20]):
        raise ValueError('same sealed source, predictor, full360, cold20 and fixed threshold required')
    for name in ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py'):
        path=j.r.ROOT/name
        if prior['source_files'][str(path)]!=path.read_text():
            raise ValueError('reference full-feedback implementation changed')
    decoder=MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
    decoder.load_state_dict(torch.load(fit_root/'prepared_recorded_sampling.pt',map_location=device,weights_only=True))
    decoder.requires_grad_(False)
    originals=[deepcopy(m.state_dict()) for m in (base,decoder)]
    out.mkdir(parents=True,exist_ok=False);(out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='PARTIAL_SIGNAL_FEEDBACK360',arms=ARMS,
        order=cfg['order'],threshold=j.r.THRESHOLD,warm=20,parent_sha256=j.PARENT_SHA256,
        reference_run=str(reference_run),fit=str(fit_root),inter_updates=0,east_outcome_use=False,
        origin=127,stride=32,forecast_horizon=25,partial_horizon=16,
        first_affected=dict(full=153,partial_delayed=153,partial_early=144),
        objective='unchanged ordered H25 energy, masks restrict to actual arrived targets',
        partial_reason='two 8-point patches retain nonempty lag8 change targets',
        adaptation='temporary first FFN W only; AdamW1e-4, .01 shot-start prior, clip1',
        update_budget='same origin cadence; early feedback may mature one additional tail update',
        future_unobserved='not read; padded with center and mask false; no autoregression',
        labels='HDF5 scoring only after all curves sealed; no outcome during inference',
        source_classifier_immutable=True,temporary_reset=True,alarm_changes=False,
        deterministic_algorithms=True,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),
            j.r.ROOT/'risk_partial_feedback_20261108.py',j.r.ROOT/'risk_feature_intra_20261107.py',
            j.r.ROOT/'run_risk_feature_intra_20261107.py')}))
    records=[];witnesses=[];maximum_reference_error=0.
    with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcomes unavailable before full-stream seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid=note['shot'];raw=j.raw_for(sid,cfg['raw_sha256'])
            signal=j.r.prepare_actual(raw,rank,mapping)
            curves,info=partial_curves(base,decoder,signal,warm=rank<=warm)
            with np.load(reference_run/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as old:
                maximum_reference_error=max(maximum_reference_error,
                    j.require_parity(curves['static'],old['static']),
                    j.require_parity(curves['full'],old['recorded_sampling_real']))
            if rank in (21,181,359):
                stop=min(len(raw['times']),max(154,len(raw['times'])//2))
                short,_=partial_curves(base,decoder,{k:signal[k][:stop] for k in ('values','available')},warm=False)
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(short[k],curves[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**curves)
            records.append(dict(shot=sid,rank=rank,adaptation=info))
            if rank%20==0:
                print('partial-feedback',rank,'/',len(cfg['order']),'seconds',round(time.monotonic()-started,1),flush=True)
    for model,old in zip((base,decoder),originals,strict=True):
        if any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()):
            raise ValueError('source or fixed diagnostic predictor changed')
    j.r.write_new(out/'completed.json',dict(status='SEALED_PARTIAL_SIGNAL360',records=records,
        witnesses=witnesses,maximum_reference_error=maximum_reference_error,
        templates_unchanged=True,seconds=time.monotonic()-started))
    print('sealed partial-feedback complete',flush=True)


def recount(out):
    protocol,done=(j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,_)=j.cohort('fixed360')
    if (done['status']!='SEALED_PARTIAL_SIGNAL360' or protocol['order']!=cfg['order']
            or [v['shot'] for v in done['records']]!=cfg['order']):
        raise ValueError('complete sealed stream is required before raw-HDF5 scoring')
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
    for name,previous in (('static','static'),('full','recorded_sampling_real')):
        if j.pair(rows[name],parent['rows'][previous])!=empty:
            raise ValueError('reference source/full-feedback outcomes changed')
    metrics={k:j.counts(v) for k,v in rows.items()}
    for value in metrics.values():
        value['fpr_cap_met']={str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    pairs=(('partial_delayed','full'),('partial_early','partial_delayed'),
           ('partial_early','full'),('partial_early','static'))
    j.r.write_new(out/'result.json',dict(status='HDF5_PARTIAL_SIGNAL360',metrics=metrics,rows=rows,
        comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b]) for a,b in pairs},
        capped_comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b],capped=True) for a,b in pairs},
        prefix20={k:j.counts(v[:20]) for k,v in rows.items()},
        inter_updates=0,independent_confirmation=False))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'));parser.add_argument('out',type=Path)
    args=parser.parse_args();{'predict':predict,'score':recount}[args.phase](args.out)
