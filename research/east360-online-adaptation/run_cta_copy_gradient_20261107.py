"""CPU-only offline gradient compatibility audit of the fixed source copies."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.func import functional_call

import cta_copy_kernel_20261107 as kernel
import run_cta_copy_pilot_20261107 as p


FIT=p.ROOT/'probe_outputs/cta_copy_pilot_20261107_v1'
PLAN=p.ROOT/'CTA_COPY_GRADIENT_PROTOCOL_20261107.md'


def summarize(rows):
    result={}
    for group in p.GROUPS:
        selected=[q for q in rows if q['group']==group]
        by_shot=defaultdict(list)
        for q in selected:by_shot[q['shot']].append(q)
        result[group]=dict(shots=len(by_shot),windows=len(selected),
            favorable_fraction_shot_equal=float(np.mean([np.mean([q['cosine']>0 for q in v]) for v in by_shot.values()])),
            cosine_mean_shot_equal=float(np.mean([np.mean([q['cosine'] for q in v]) for v in by_shot.values()])),
            cosine_median=float(np.median([q['cosine'] for q in selected])),
            zero_gradient_windows=sum(q['zero_gradient'] for q in selected),
            unfavorable_small_step_windows=sum(q['cosine']<0 for q in selected),
            actual_full_step_harmful_windows=(sum(q['actual_risk_change']>1e-6 for q in selected)
                                             if all('actual_risk_change' in q for q in selected) else None))
    return result


def main(out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    base,checkpoint=p.r.load_parent(p.r.DEFAULT_PARENT)
    frozen=p.snapshot(base)
    models={'teacher':base,**{kind:kernel.copy_for_training(base) for kind in p.KINDS}}
    for kind in p.KINDS:
        models[kind].load_state_dict(torch.load(FIT/(kind+'.pt'),map_location='cpu',weights_only=True))
    out.mkdir()
    p.write(out/'protocol.json',dict(status='OFFLINE_GRADIENT_DIAGNOSTIC',fit=str(FIT),device='cpu',
        code={f.name:f.read_text() for f in (Path(__file__),PLAN,Path(kernel.__file__))}))
    combined={}
    for role,path,load_role in (('source_val',p.REFERENCE/'val_sealed.json','val'),('east20',p.EAST/'sealed.json','east20')):
        rows=p.read(path)['rows']
        data=p.load_windows(rows,load_role,'cpu')
        with np.load(FIT/(role+'_predictions.npz'),allow_pickle=False) as f:
            archived={k:f[k].copy() for k in f.files}
        role_result={}
        for kind,model in models.items():
            records=[]
            initial=p.snapshot(model)
            error=0.
            for i,row in enumerate(rows):
                d=p.part(data,slice(i,i+1))
                info=kernel.inner_gradient(model,d['x0'],d['a0'],d['truth'],d['mask'])
                names=tuple(info['gradients'])
                q=functional_call(model,info['parameters'],(d['xq'],d['aq']))['logit']
                signed=(1-2*row['label'])*q.sum()
                risk=torch.autograd.grad(signed,tuple(info['parameters'][n] for n in names))
                norm_signal=float(info['gradient_norm'])
                norm_risk=float(torch.sqrt(sum(v.double().square().sum() for v in risk)))
                dot=float(sum((info['gradients'][n].double()*v.double()).sum() for n,v in zip(names,risk)))
                cosine=dot/(norm_signal*norm_risk) if norm_signal and norm_risk else 0.
                original=archived['teacher' if kind=='teacher' else kind+'_static'][i]
                error=max(error,abs(float(q.detach())-float(original)))
                record=dict(shot=row['shot'],group=row['group'],cosine=cosine,signal_gradient_norm=norm_signal,
                    signed_logit_gradient_norm=norm_risk,gradient_dot=dot,zero_gradient=not(norm_signal and norm_risk))
                if kind!='teacher':
                    record['actual_risk_change']=(1-2*row['label'])*float(archived[kind+'_real'][i]-archived[kind+'_static'][i])
                records.append(record)
            if not p.unchanged(model,initial):raise ValueError('gradient audit mutated parent')
            if error>3e-5:raise ValueError('CPU parent differs from sealed GPU predictions')
            role_result[kind]=dict(summary=summarize(records),records=records,static_cpu_gpu_error=error)
            print(role,kind,role_result[kind]['summary'],flush=True)
        p.write(out/(role+'_result.json'),dict(status='OFFLINE_SIGNED_LOGIT_GRADIENTS',models=role_result))
        combined[role]={k:v['summary'] for k,v in role_result.items()}
    if not p.unchanged(base,frozen):raise ValueError('source changed')
    p.write(out/'completed.json',dict(status='COMPLETE_CPU_GRADIENT_DIAGNOSTIC',seconds=time.monotonic()-start,
        summary=combined,source_unchanged=True,trained_models=0,gpu_used=False))
    print('complete seconds',time.monotonic()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out)
