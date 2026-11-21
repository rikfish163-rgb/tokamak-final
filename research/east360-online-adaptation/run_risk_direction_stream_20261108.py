"""Cross source auxiliary teaching with runtime update geometry, EAST only."""

import argparse
from copy import deepcopy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import run_joint_trajectory_20261107 as j
from risk_direction_intra_20261108 import direction_curves
from risk_meta_step_20261107 import MetaOrderedTrajectory
from run_risk_feature_intra_20261107 import feature_curves


FIT = j.r.ROOT/'probe_outputs/risk_direction_source_20261108_v1'
REFERENCE = j.r.ROOT/'probe_outputs/risk_short_context360_20261108_v1'
ARMS = ('static','control_adam','control_direction','taught_adam','taught_direction')
ACTUAL = ARMS[2:]


def comparison_curves(base, decoders, signal, *, reference, warm=False):
    if set(decoders) != {'control','direction'}:
        raise ValueError('matched control and direction-taught auxiliary predictors required')
    curves = dict(static=np.asarray(reference['static']).copy(), control_adam=np.asarray(reference['real']).copy())
    info = {}
    for name, decoder, runner in (
            ('control_direction',decoders['control'],direction_curves),
            ('taught_adam',decoders['direction'],feature_curves),
            ('taught_direction',decoders['direction'],direction_curves)):
        kwargs = dict(feedbacks=('real',)) if name == 'taught_adam' else {}
        trace, record = runner(base,decoder,signal,warm=warm,**kwargs)
        j.require_parity(trace['static'],curves['static'])
        curves[name] = trace['real']; info[name] = record
    return curves, info


def predict(out, *, fit_root=FIT, reference_run=REFERENCE, device='cuda'):
    started = time.monotonic(); j.r.configure(); torch.use_deterministic_algorithms(True)
    base, parent = j.r.load_parent(j.r.DEFAULT_PARENT); base.to(device)
    fit, fitted = (j.r.read(fit_root/f'{k}.json') for k in ('protocol','fit'))
    prior, done = (j.r.read(reference_run/f'{k}.json') for k in ('protocol','completed'))
    _, cfg, archive, (_,_,warm) = j.cohort('fixed360')
    mapping = j.r.read(j.r.COLD/'input_map.json')
    if (fitted['status'] != 'SOURCE_DIRECTION_TEACHER_FIT_COMPLETE'
            or not all(fitted[k] for k in ('source_control_exact','prepared_control_exact','source_classifier_unchanged'))
            or fit['parent_sha256'] != prior['parent_sha256'] or fit['parent_sha256'] != j.PARENT_SHA256
            or fit['risk_classifier_training'] or fit['east_label_fitting'] or fit['jtext_online']
            or fit['source_steps'] != 512 or fit['sequence_steps'] != 4 or not fit['deterministic_algorithms']
            or done['status'] != 'SEALED_SOURCE_CONTEXT_CLOCK360' or not done['templates_unchanged']
            or prior['order'] != cfg['order'] or [r['shot'] for r in done['records']] != cfg['order']
            or prior['threshold'] != parent['threshold'] or parent['threshold'] != j.r.THRESHOLD
            or prior['inter_updates'] != 0 or not prior['deterministic_algorithms']
            or warm != 20 or prior['warm'] != warm
            or fit['cold20'] != mapping['calibration_ids'] or mapping['calibration_ids'] != cfg['order'][:20]):
        raise ValueError('completed matched fit and unchanged full360/source/map/alarm contract required')
    for name in ('run_risk_feature_intra_20261107.py','risk_feature_intra_20261107.py'):
        path = j.r.ROOT/name
        if prior['source_files'][str(path)] != path.read_text():
            raise ValueError('cached control implementation changed')
    decoders = {}
    for name in ('control','direction'):
        decoder = MetaOrderedTrajectory(base.center,base.scale,dim=base.cfg.dim,scenarios=4).to(device).eval()
        decoder.load_state_dict(torch.load(fit_root/f'prepared_{name}.pt',map_location=device,weights_only=True))
        decoders[name] = decoder.requires_grad_(False)
    old_control = torch.load(Path(prior['fit'])/'prepared_control.pt',map_location=device,weights_only=True)
    if any(not torch.equal(v,old_control[k]) for k,v in decoders['control'].state_dict().items()):
        raise ValueError('cached control auxiliary state differs from the newly fitted control')
    originals = [deepcopy(m.state_dict()) for m in (base,*decoders.values())]
    out.mkdir(parents=True,exist_ok=False); (out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='SOURCE_DIRECTION_CROSS360',arms=ARMS,order=cfg['order'],
        threshold=j.r.THRESHOLD,warm=20,parent_sha256=j.PARENT_SHA256,fit=str(fit_root),reference_run=str(reference_run),
        reused='sealed control_adam from prior control_full; exact same G and unchanged inference implementation',
        newly_inferred=ACTUAL,inter_updates=0,east_outcome_use=False,alarm_changes=False,
        clock='origin127+32k, fullH25, release=origin+26',first_affected=153,temporary_reset=True,
        update_geometry='native Adam candidate norm, accumulated first-moment direction; constant-size optimizer state',
        deterministic_algorithms=True,independent_confirmation=False,
        source_files={str(p):p.read_text() for p in (Path(__file__),j.r.ROOT/'risk_direction_intra_20261108.py',
            j.r.ROOT/'risk_meta_adam_20261108.py',j.r.ROOT/'risk_feature_intra_20261107.py',
            j.r.ROOT/'run_risk_feature_intra_20261107.py')}))
    records = []; witnesses = []
    with sdpa_kernel(SDPBackend.MATH),patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcomes unavailable until full-stream seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid = note['shot']; raw = j.raw_for(sid,cfg['raw_sha256'])
            signal = j.r.prepare_actual(raw,rank,mapping)
            with np.load(reference_run/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as data:
                reference = dict(static=data['static'].copy(),real=data['control_full'].copy())
            curves, info = comparison_curves(base,decoders,signal,reference=reference,warm=rank<=warm)
            info['control_adam'] = done['records'][rank-1]['adaptation']['control_full']
            if rank in (21,181,359):
                stop = min(len(raw['times']),max(154,len(raw['times'])//2))
                prefix, _ = comparison_curves(base,decoders,{k:signal[k][:stop] for k in ('values','available')},
                    reference={k:v[:stop] for k,v in reference.items()},warm=False)
                witnesses.append(dict(shot=sid,points=stop,max_error=max(j.require_parity(prefix[k],curves[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**curves)
            records.append(dict(shot=sid,rank=rank,adaptation=info))
            if rank%20 == 0:
                print('direction cross',rank,'/',len(cfg['order']),'seconds',round(time.monotonic()-started,1),flush=True)
    for model, old in zip((base,*decoders.values()),originals,strict=True):
        if any(not torch.equal(v,old[k]) for k,v in model.state_dict().items()):
            raise ValueError('a frozen source or auxiliary template changed')
    j.r.write_new(out/'completed.json',dict(status='SEALED_SOURCE_DIRECTION360',records=records,witnesses=witnesses,
        templates_unchanged=True,seconds=time.monotonic()-started))
    print('sealed direction cross complete',flush=True)


def recount(out):
    protocol, done = (j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    reference, cfg, archive, (real,stale,_) = j.cohort('fixed360')
    if (done['status'] != 'SEALED_SOURCE_DIRECTION360' or protocol['order'] != cfg['order']
            or [r['shot'] for r in done['records']] != cfg['order']):
        raise ValueError('all original stream members must be sealed before raw-HDF5 scoring')
    rows = {k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid = note['shot']; times,_,_,_ = j.recorded(reference,note,rank,real,stale)
        truth, clock = j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as data:
            for name in ARMS:
                rows[name].append(j.score(times,data[name],truth,protocol['threshold']))
    old = j.r.read(Path(protocol['reference_run'])/'result.json')
    empty = dict(tp_gained=[],tp_lost=[],fp_added=[],fp_removed=[])
    for name, previous in (('static','static'),('control_adam','control_full')):
        if j.pair(rows[name],old['rows'][previous]) != empty:
            raise ValueError('cached static or original intra outcomes changed')
    metrics = {k:j.counts(v) for k,v in rows.items()}
    for value in metrics.values():
        value['fpr_cap_met'] = {str(cap):value['fpr']<=cap for cap in (.10,.15,.20)}
    pairs = [(a,b) for a in ACTUAL for b in ('static','control_adam')]+[
        ('taught_direction','control_direction'),('taught_direction','taught_adam')]
    j.r.write_new(out/'result.json',dict(status='HDF5_SOURCE_DIRECTION360',metrics=metrics,rows=rows,
        comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b]) for a,b in pairs},
        capped_comparisons={a+'_vs_'+b:j.pair(rows[a],rows[b],capped=True) for a,b in pairs},
        prefix20={k:j.counts(v[:20]) for k,v in rows.items()},
        update_counts={k:sum(v['adaptation'][k]['updates']['real'] for v in done['records']) for k in ARMS[1:]},
        teaching_geometry_interaction={k:metrics['taught_direction'][k]-metrics['taught_adam'][k]
            -metrics['control_direction'][k]+metrics['control_adam'][k] for k in ('tp','fp','j')},
        inter_updates=0,independent_confirmation=False))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    args = parser.parse_args()
    {'predict':predict,'score':recount}[args.phase](args.out)
