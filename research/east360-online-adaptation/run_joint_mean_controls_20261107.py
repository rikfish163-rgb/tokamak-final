"""Missing mean-readout controls, reusing the sealed motion experiment."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from run_joint_motion_20261107 import FIT


PARENT = j.r.ROOT/'probe_outputs/joint_motion360_20261107_v1'
FEEDBACK = ('probability_only', 'wrong_time')
ARMS = tuple(f'{kind}_{arm}' for arm in FEEDBACK for kind in ('mean', 'joint'))


def predict(out):
    started = time.monotonic()
    j.r.configure()
    protocol, done = (j.r.read(PARENT/f'{k}.json') for k in ('protocol','completed'))
    reference,cfg,archive,(real,stale,warm) = j.cohort('fixed360')
    if done['status'] != 'SEALED_JOINT_TRAJECTORY360' or protocol['order'] != cfg['order']:
        raise ValueError('completed original motion360 required')
    base,parent = j.r.load_parent(j.r.DEFAULT_PARENT)
    base.cuda()
    predictor = OrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    predictor.load_state_dict(torch.load(FIT/'prepared_predictor.pt',map_location='cuda',weights_only=True))
    mapping = j.r.read(j.r.COLD/'input_map.json')
    if parent['threshold'] != protocol['threshold'] or warm != 20:
        raise ValueError('same threshold and preparation boundary required')
    frozen = {k:v.detach().clone() for k,v in base.state_dict().items()}
    out.mkdir(parents=True,exist_ok=False)
    (out/'shots').mkdir()
    j.r.write_new(out/'protocol.json',dict(status='MEAN_CONTROL_EXTENSION',parent=str(PARENT),
        order=cfg['order'],threshold=protocol['threshold'],preparation=str(FIT),arms=ARMS,
        feedback=FEEDBACK,warm=warm,source_files={str(p):p.read_text() for p in
            (Path(__file__),j.r.ROOT/'run_joint_trajectory_20261107.py')}))
    records,witnesses = [],[]
    with patch.object(j.r.io23,'read_outcome',side_effect=RuntimeError('outcomes unavailable before seal')):
        for rank,note in enumerate(archive['sealed'],1):
            sid = note['shot']
            times,_,_,_ = j.recorded(reference,note,rank,real,stale)
            raw = j.raw_for(sid,cfg['raw_sha256'])
            np.testing.assert_array_equal(times,raw['times'])
            signal = j.r.prepare_actual(raw,rank,mapping)
            context = j.prediction_context(base,signal)
            traces,info = j.adaptive_curves(base,predictor,signal,context,
                arms=FEEDBACK,warm=rank<=warm)
            with np.load(PARENT/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as old:
                errors = {arm:j.require_parity(traces[f'joint_{arm}'],old[f'joint_{arm}'])
                          for arm in FEEDBACK}
            if rank in (21,181,359):
                stop = min(len(times),max(154,len(times)//2))
                partial = {k:signal[k][:stop] for k in ('values','available')}
                ctx = j.prediction_context(base,partial)
                truncated,_ = j.adaptive_curves(base,predictor,partial,ctx,arms=FEEDBACK)
                witnesses.append(dict(shot=sid,points=stop,
                    max_error=max(j.require_parity(truncated[k],traces[k][:stop]) for k in ARMS)))
            j.r.save_arrays(out/'shots'/f'{rank:04d}_{sid}.npz',**traces)
            records.append(dict(shot=sid,rank=rank,joint_parity=errors,adaptation=info))
            if rank%20 == 0:
                print('mean controls',rank,'/360; seconds',round(time.monotonic()-started,1),flush=True)
    if any(not torch.equal(v,frozen[k]) for k,v in base.state_dict().items()):
        raise ValueError('source classifier changed during control extension')
    j.r.write_new(out/'completed.json',dict(status='SEALED_MEAN_CONTROLS360',records=records,
        witnesses=witnesses,seconds=time.monotonic()-started,source_classifier_unchanged=True))
    print('sealed mean controls360',flush=True)


def recount(out):
    protocol,done = (j.r.read(out/f'{k}.json') for k in ('protocol','completed'))
    parent = j.r.read(PARENT/'result.json')
    reference,cfg,archive,(real,stale,warm) = j.cohort('fixed360')
    if (done['status'] != 'SEALED_MEAN_CONTROLS360' or protocol['order'] != cfg['order']
            or [v['shot'] for v in done['records']] != cfg['order']):
        raise ValueError('complete sealed control extension required')
    rows = {k:[] for k in ARMS}
    for rank,note in enumerate(archive['sealed'],1):
        sid = note['shot']
        times,_,_,_ = j.recorded(reference,note,rank,real,stale)
        truth,clock = j.raw_truth(j.r.io23.path_for('EAST',sid),sid)
        np.testing.assert_array_equal(times,clock)
        with np.load(out/'shots'/f'{rank:04d}_{sid}.npz',allow_pickle=False) as data:
            for arm in ARMS:
                rows[arm].append(j.score(times,data[arm],truth,protocol['threshold']))
    for arm in FEEDBACK:
        if rows[f'joint_{arm}'] != parent['rows'][f'joint_{arm}']:
            raise ValueError('control extension no longer reproduces parent joint judgments')
    metrics = {k:j.counts(v) for k,v in rows.items()}
    paired = {k:j.pair(parent['rows']['mean_intra'],rows[k]) for k in ARMS if k.startswith('mean_')}
    j.r.write_new(out/'result.json',dict(status='HDF5_MEAN_CONTROLS360',metrics=metrics,
        actual_mean_intra_vs_control=paired,rows=rows,joint_parent_judgments_identical=True))
    print({k:(v['tp'],v['fp'],v['tp_30_to_3000']) for k,v in metrics.items()},flush=True)
    print(paired,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('predict','score'))
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    {'predict':predict,'score':recount}[args.phase](args.out)
