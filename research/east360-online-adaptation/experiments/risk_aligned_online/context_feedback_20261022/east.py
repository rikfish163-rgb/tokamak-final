"""Locked EAST TEST360 diagnostic: no target labels train or choose any arm."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..event_adapt_20261022.train import public
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..joint_feedback_20261022.diagnose_stream import compare
from ..simple_train import sha
from ..simple_online import _durable_json
from .train import model_from_warm,snapshot
from .stream import predict_pair


def recovered_trace(path,times):
    """Recover a complete sealed shot after SIGTERM, never partial predictions.

This is valid ONLY for this fixed-model, per-window/per-shot-reset experiment.
There is no persistent optimizer/model/RNG state crossing shots. The original
file is never overwritten; missing update telemetry is explicitly unavailable.
Regression tests reject truncated arrays, clock mismatches and NaNs.
"""
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!={'times','static','intra'}:
            raise ValueError('complete paired trace required')
        if not np.array_equal(data['times'],times):
            raise ValueError('recovered clock mismatch')
        result={k:data[k].copy() for k in ('static','intra')}
    for values in result.values():
        if values.shape!=times.shape: raise ValueError('complete shot predictions required')
        if not np.isfinite(values).all() or (values<0).any() or (values>1).any():
            raise ValueError('finite probabilities required')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uniform',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    if args.resume and (not args.output.is_dir() or (args.output/'completed.json').exists()):
        raise ValueError('resume only an existing incomplete run')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);begin=time.monotonic()
    receipt=json.loads((args.checkpoint.parent/'completed.json').read_text())
    if sha(args.checkpoint)!=receipt['checkpoint_sha256']: raise ValueError('checkpoint mismatch')
    ck=torch.load(args.checkpoint,map_location='cuda',weights_only=True)
    model,threshold=model_from_warm();model.load_state_dict(ck['model']);model.freeze_base()
    mode=ck['config']['mode']; use_event=mode=='event'
    if args.uniform!=(mode=='warm'): raise ValueError('warm is the uniform control only')
    source={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    contract=load_test_contract('EAST')
    if not args.resume: args.output.mkdir(parents=True)
    config=dict(format='context-feedback-EAST-v1',mode=mode,uniform=args.uniform,
        contract=contract,checkpoint_sha256=sha(args.checkpoint),threshold=threshold,
        update='same2SGD.01 on arrived128-point window; next32 samples only; reseteachwindow/shot',
        outcome_access='HDF5 IsDisrupt/DownTime only after complete paired prediction sealed; scoring only',
        target_training='none; no support110; no EAST scaler/threshold selection; no event truth',
        comparison='all three fixed final controllers plus uniform; no best-checkpoint selection on EAST',
        evidence='DEVELOPMENT-EXPOSED FIXED TEST, NOT INDEPENDENT CONFIRMATION',
        code_sha256=None)
    recovered_ids=[]
    if args.resume:
        original=json.loads((args.output/'protocol.json').read_text())
        for key in ('mode','uniform','checkpoint_sha256','threshold','contract'):
            if original[key]!=config[key]: raise ValueError('resume changed experiment contract')
        for filename,digest in original['code_sha256'].items():
            # Only the runner acquired an explicit recovery path; inference,
            # data semantics, model and loss files must be byte-identical.
            if Path(filename).name!='east.py' and sha(io23.ROOT/filename)!=digest:
                raise ValueError(f'resume changed original dependency: {filename}')
        recovered_ids=sorted(int(f.stem) for f in (args.output/'shots').glob('*.npz'))
        if recovered_ids!=contract['shot_ids'][:len(recovered_ids)]:
            raise ValueError('only a contiguous sealed prefix may be recovered')
        recovery=dict(cause='external SIGTERM exit143; Python traceback absent',
            recovered_shots=recovered_ids,original_protocol_sha256=sha(args.output/'protocol.json'),
            runner_sha256=sha(Path(__file__)),
            missing_original_telemetry='per-update gradients/weights were buffered and lost; never fabricated',
            prediction_hashes={str(s):sha(args.output/'shots'/f'{s}.npz') for s in recovered_ids})
        _durable_json(args.output/'recovery.json',recovery)
    else:
        config['code_sha256']=snapshot(args.output)
        _durable_json(args.output/'protocol.json',config)
    rows={k:[] for k in ('static','intra')}; audits=[]
    if not args.resume: (args.output/'shots').mkdir()
    for index,sid in enumerate(contract['shot_ids'],1):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        artifact=args.output/'shots'/f'{sid}.npz'
        if sid in recovered_ids:
            scores=recovered_trace(artifact,s['times'])
            info=dict(updates=None,telemetry_status='unavailable_after_SIGTERM',inter_updates=0)
        else:
            scores,info=predict_pair(model,{k:s[k] for k in ('values','available')},device='cuda',
                use_event=use_event,uniform=args.uniform)
            info['telemetry_status']='recorded'
            # Seal ALL predictions before reading this shot's outcome.
            np.savez_compressed(artifact,times=s['times'],**scores)
            _durable_json(args.output/'shots'/f'{sid}_updates.json',info)
        sealed_sha=sha(artifact)
        revealed=simple_data.attach_outcome(s,'EAST',{})
        for arm in rows:
            v=revealed['valid']
            rows[arm].append(dict(shot=sid,times=s['times'][v],scores=scores[arm][v],
                is_disrupt=revealed['is_disrupt'],down=revealed['down']))
        audits.append(dict(shot=sid,prediction_sha256=sealed_sha,**info))
        if index%30==0:
            _durable_json(args.output/'progress.json',dict(shots=index,seconds=time.monotonic()-begin))
            print(json.dumps(dict(stage='EAST_fixed_test',mode=mode,shots=index,recovered=len(recovered_ids),
                                 seconds=time.monotonic()-begin)),flush=True)
    metrics={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    _durable_json(args.output/'per_shot.json',metrics)
    _durable_json(args.output/'updates.json',audits)
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in source.items()):
        raise AssertionError('persistent model changed')
    result=dict(status='COMPLETED',metrics={k:public(v) for k,v in metrics.items()},
        paired_flips=compare(metrics['static'],metrics['intra']),
        early_alarms_above_3000ms={k:sum(p['timely'] and p['lead_ms']>3000 for p in v['predictions']) for k,v in metrics.items()},
        recorded_updates=sum(len(a['updates']) for a in audits if a['updates'] is not None),
        missing_update_telemetry_shots=[a['shot'] for a in audits if a['updates'] is None],
        source_unchanged=True,inter_updates=0,
        seconds=time.monotonic()-begin,
        goal_pass=metrics['intra']['fpr']<=.2 and metrics['intra']['J']>metrics['static']['J']
                  and metrics['intra']['tp']>=metrics['static']['tp'],
        independent_confirmation=False)
    _durable_json(args.output/'completed.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
