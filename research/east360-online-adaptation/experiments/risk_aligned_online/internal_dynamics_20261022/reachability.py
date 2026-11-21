"""NONCAUSAL diagnostic: can per-shot supervised U fix static errors at all?

Uses the full target shot and revealed outcome BEFORE optimizing its own U.
Never an online performance result or a rigorous global upper bound. This
separates restricted-interface capacity from a poor unsupervised direction.
All 94 static FP and 35 non-timely disruptive shots are inspected, not a
hand-picked successful subset. Unchanged correct shots are not evaluated.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..simple_train import metrics, sha
from .stream import load_model


def windows(s, ends, model):
    examples=[simple_data.window(s, int(t), model.cfg.window) for t in ends]
    return [torch.as_tensor(np.stack(v), device='cuda') for v in zip(*examples)]


def score(model, s, fast, threshold):
    ends=np.flatnonzero(s['valid']); values=[]
    for start in range(0, len(ends), 128):
        x,m=windows(s, ends[start:start+128], model)
        values.extend(model(x,m,fast.expand(len(x),-1,-1))['logit'].detach().sigmoid().cpu().tolist())
    row=dict(shot=s['shot'], is_disrupt=s['is_disrupt'], down=s['down'], times=s['times'][ends], scores=np.asarray(values))
    result=metrics([row],threshold,30.)['predictions'][0]
    result['maximum_probability']=float(np.max(values))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radius',type=float,default=1.)
    args=parser.parse_args()
    if args.radius not in (1.,4.): parser.error('predeclared diagnostic radius: 1 or 4')
    torch.set_num_threads(2); start=time.monotonic(); rng=np.random.default_rng(431)
    runs=Path(__file__).parents[1]/'runs'
    source=runs/'internal_dynamics_M_20261022/selected.pt'
    model,ck=load_model(source,'cuda')
    frozen={k:v.clone() for k,v in model.state_dict().items()}
    static=json.loads((runs/'internal_EAST_static_20261022/completed.json').read_text())['metrics']
    bad=[r for r in static['predictions'] if (r['is_disrupt'] and not r['timely']) or (not r['is_disrupt'] and r['alarm'])]
    suffix='' if args.radius==1. else '_radius4'
    output=io23.ROOT/f'outputs/source_event_rebuild_audit_20261022/internal_oracle_reachability{suffix}.json'
    if output.exists(): raise FileExistsError(output)
    records=[]
    for number,r in enumerate(bad,1):
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal')),'EAST',{})
        valid=np.flatnonzero(s['valid'])
        if s['is_disrupt']:
            lead=np.round((s['down']-s['times'])*1000.,9)
            positive=np.flatnonzero(s['valid'] & (lead>30.) & (lead<=125.))
            negative=np.flatnonzero(s['valid'] & (lead>125.))
            if not len(positive): raise ValueError('no timely diagnostic oracle positive endpoints')
        else:
            sealed=np.load(runs/f'internal_EAST_static_20261022/test360_traces.npz')
            top=np.argsort(sealed[f'{r["shot"]}_scores'])[-min(64,len(valid)):]
            positive=[]; negative=valid
        fast=model.initial_fast(1); before=score(model,s,fast,ck['threshold'])
        for step in range(64):
            if s['is_disrupt']:
                a=rng.choice(positive,16,replace=True)
                b=rng.choice(negative if len(negative) else positive,16,replace=True)
                ends=np.r_[a,b]; target=torch.as_tensor(s['y'][ends],device='cuda')
            else:
                ends=np.r_[rng.choice(valid,16,replace=True),rng.choice(valid[top],16,replace=True)]
                target=torch.zeros(32,device='cuda')
            x,m=windows(s,ends,model)
            loss=F.binary_cross_entropy_with_logits(model(x,m,fast.expand(32,-1,-1))['logit'],target)
            grad,=torch.autograd.grad(loss,fast)
            candidate=fast-.1*grad/(grad.square().sum()+1e-16).sqrt()
            candidate=candidate*(args.radius/(candidate.square().sum()+1e-16).sqrt()).clamp(max=1.)
            fast=candidate.detach().requires_grad_(True)
        after=score(model,s,fast,ck['threshold'])
        records.append(dict(shot=r['shot'],is_disrupt=r['is_disrupt'],before=before,after=after,fast_norm=float(fast.detach().norm()),
                            corrected=bool(after['timely'] if r['is_disrupt'] else not after['alarm'])))
        if number%20==0: print(json.dumps(dict(diagnostic='NONCAUSAL_ORACLE',shots=number,corrected=sum(x['corrected'] for x in records),elapsed=time.monotonic()-start)),flush=True)
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()): raise AssertionError('persistent model changed')
    result=dict(status='COMPLETED_NONCAUSAL_DIAGNOSTIC',source_sha256=sha(source), records=records,
        n_static_errors=len(bad), corrected_FP=sum(x['corrected'] and not x['is_disrupt'] for x in records),
        corrected_timely=sum(x['corrected'] and x['is_disrupt'] for x in records),
        optimization=f'64 normalized true-label steps, step=.1, radius={args.radius}, independent U for each full shot',
        code_sha256=sha(Path(__file__)),
        limitation='future/full-shot labels used; selected static errors only; local optimizer not a guaranteed upper bound; NOT online TPR/FPR',
        elapsed_seconds=time.monotonic()-start)
    output.write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
