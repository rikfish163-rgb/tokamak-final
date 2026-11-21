"""Full source VAL diagnostic, NOT an EAST or held-out TEST result.

Current predictions are immutable. Observed-window completion updates only
the following block, from a fresh copy of the same source encoder each time.
This explicitly tests a window-local alternative to cumulative shot updates.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..event_adapt_20261022.train import load_source, public
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder, update_observed
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, select_threshold, sha
from .train import INNER_RATE, INNER_STEPS


def predict_pair(model, shot, *, device, interval=8):
    if {'y','down','valid','is_disrupt','event_target','event_intervals'} & set(shot):
        raise ValueError('outcome fields forbidden during prediction')
    if type(interval) is not int or interval < 1:
        raise ValueError('positive integer interval required')
    n = len(shot['values'])
    if not n or shot['available'].shape != shot['values'].shape:
        raise ValueError('nonempty matching signal and availability arrays required')
    source = initial_encoder(model)
    fast = source
    scores = {k:np.empty(n, 'float32') for k in ('static','intra')}
    updates = []
    for lo in range(0,n,interval):
        hi = min(lo+interval,n)
        windows = [simple_data.window(shot,t,model.cfg.window) for t in range(lo,hi)]
        x,m = [torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
        for arm, state in (('static',source),('intra',fast)):
            # Both paths use the same differentiable attention dispatch, even
            # though the prediction graph itself is discarded immediately.
            out = model.predict_fast(x,m,state)['logit']
            scores[arm][lo:hi] = out.detach().sigmoid().cpu().numpy()
            del out
        if hi < model.cfg.window or hi == n:
            continue
        fast, info = update_observed(model,x[-1:],m[-1:],
            learning_rate=INNER_RATE,steps=INNER_STEPS)
        updates.append(dict(last_observation=hi-1,first_affected_prediction=hi,
            steps=info['steps'],observations=info['observations'],delta=info['delta']))
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise FloatingPointError('nonfinite causal source score')
    return scores,dict(updates=updates,inter_updates=0,
        reset='fresh original encoder per completed observation block; no state crosses shots')


def compare(before,after):
    old = {r['shot']:r for r in before['predictions']}
    new = {r['shot']:r for r in after['predictions']}
    if set(old)!=set(new):
        raise ValueError('paired shot identities required')
    result = {k:[] for k in ('rescued','lost','false_alarm_removed','false_alarm_added')}
    for sid,a in old.items():
        b=new[sid]
        if a['is_disrupt']:
            if not a['timely'] and b['timely']: result['rescued'].append(sid)
            if a['timely'] and not b['timely']: result['lost'].append(sid)
        else:
            if a['alarm'] and not b['alarm']: result['false_alarm_removed'].append(sid)
            if not a['alarm'] and b['alarm']: result['false_alarm_added'].append(sid)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2)
    begin=time.monotonic()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ck['config']['parent_sha256']!=PARENT_SHA256: raise ValueError('source ancestry mismatch')
    base,_=load_parent(DEFAULT_PARENT)
    model=PairedFeedback(base,ck['model']['level_scale'])
    model.load_state_dict(ck['model']); model.cuda(); model.prepare_decoder()
    # Two explicitly supported saved formats: warm preparation and joint source.
    threshold=(ck['config']['threshold'] if args.checkpoint.name=='warm.pt' else ck['threshold'])
    before_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    roles=io23.source_roles()
    shots=load_source(roles['val'],io23.advanced_times())
    if len(shots)!=160 or len({s['shot'] for s in shots})!=160:
        raise ValueError('complete source VAL160 required')
    args.output.mkdir(parents=True)
    files=[Path(__file__),Path(__file__).with_name('train.py')]
    files += [Path(__file__).parents[1]/n for n in ('paired_feedback_20261022/model.py',
        'masked_state_20261022/model.py','simple_data.py','simple_train.py','simple23.py',
        'causal_input.py','io23.py','event_adapt_20261022/train.py')]
    _durable_json(args.output/'protocol.json',dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha(args.checkpoint),source_parent_sha256=PARENT_SHA256,
        selection_status='DIAGNOSTIC_LAST_CHECKPOINT; does not replace selected source',
        partition='complete J-TEXT VAL160, not TEST',ids=roles['val'],
        threshold=threshold,lead_comparison='strict >10ms',
        update='2 SGD .01 from original encoder; window128; every8 observed samples; next block only',
        threshold_diagnostic='also refit static-only VAL threshold in matched inference backend; never fit on intra',
        test_accessed=False,east_accessed=False,independent_confirmation=False,
        code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files}))
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in files: archive.write(f,str(f.relative_to(io23.ROOT)))
    rows={k:[] for k in ('static','intra')};audit=[];traces={}
    for index,s in enumerate(shots,1):
        signals={k:s[k] for k in ('values','available')}
        predictions,info=predict_pair(model,signals,device='cuda')
        audit.append(dict(shot=s['shot'],**info))
        valid=s['valid']
        for arm in rows:
            rows[arm].append(dict(shot=s['shot'],is_disrupt=s['is_disrupt'],down=s['down'],
                times=s['times'][valid],scores=predictions[arm][valid]))
            traces[f'{s["shot"]}_{arm}']=predictions[arm]
        traces[f'{s["shot"]}_times']=s['times']; traces[f'{s["shot"]}_valid']=valid
        if index%40==0:
            print(json.dumps(dict(stage='VAL_diagnostic',shots=index,seconds=time.monotonic()-begin)),flush=True)
    existing={k:metrics(v,threshold) for k,v in rows.items()}
    matched=select_threshold(rows['static'],.15)
    recalculated={k:metrics(v,matched['threshold']) for k,v in rows.items()}
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before_state.items()):
        raise AssertionError('persistent source mutated')
    np.savez_compressed(args.output/'traces.npz',**traces)
    _durable_json(args.output/'updates.json',audit)
    _durable_json(args.output/'per_shot.json',dict(existing=existing,matched=recalculated))
    result=dict(status='COMPLETED',existing_threshold={k:public(v) for k,v in existing.items()},
        existing_flips=compare(existing['static'],existing['intra']),
        matched_static_threshold={k:public(v) for k,v in recalculated.items()},
        matched_flips=compare(recalculated['static'],recalculated['intra']),
        updates=sum(len(a['updates']) for a in audit),source_unchanged=True,
        seconds=time.monotonic()-begin,test_accessed=False,east_accessed=False,
        evidence='FULL VAL160 DIAGNOSTIC ONLY; no checkpoint promotion')
    _durable_json(args.output/'completed.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
