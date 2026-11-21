"""POST-HOC gradient attribution at seven disclosed outcome-selected cases.

Uses labels only for diagnosis after the complete stream is sealed. These
cases must never become a deployed shot-ID rule or a physical-cause claim.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..trajectory_memory_20261024.learning import combine_step
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ, REFERENCE, load_metric_bank, make_parent, models_for_calibration, signal, transform


def main(root):
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert READ(root/'verification.json')['completed_sha256']==sha(root/'completed.json')
    truth={r['shot']:r for r in READ(root/'outcomes.json')}
    bank=load_metric_bank(); bank['pair_gradients']=torch.load(root/'pair_gradients.pt',map_location='cuda',weights_only=True)
    parent,_=make_parent(); model=models_for_calibration(parent)['mapped_fixed']; mapping=READ(REFERENCE/'input_map.json')
    outputs=[]
    for sid in (65944,71013,73653,79712,66404,64581,69768):
        outcome=truth[sid]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            valid=np.ones(len(z['times']),bool) if not outcome['is_disrupt'] else np.round((outcome['down']-z['times'])*1000,9)>30
            ids=np.flatnonzero(valid); end=int(ids[z['off'][ids].argmax()])
            result=dict(shot=sid,is_disrupt=outcome['is_disrupt'],endpoint=end,
                lead_ms=None if not outcome['is_disrupt'] else float(round((outcome['down']-z['times'][end])*1000,9)),
                saved_probability={a:float(z[a][end]) for a in ('off','point','paired','augmented','uniform')})
        if end<128:
            raise ValueError('selected peak precedes any update; needs separate no-update explanation')
        receipt=READ(root/'receipts'/f'{sid}.json')[end//128-1]
        raw=signal(sid); mapped=transform({k:raw[k] for k in ('values','available')},mapping)
        x,m=[torch.as_tensor(a[None],device='cuda') for a in simple_data.window(mapped,end,128)]
        fast=initial_encoder(model); z=logit(model,x,m,fast)
        jac=flat(torch.autograd.grad(z.sum(),tuple(fast.values()))).detach().double()
        attribution={}
        for a in ('point','paired','augmented','uniform'):
            r=receipt[a]; ix=np.array(r['selected'],int); w=torch.tensor(r['weights'],device='cuda',dtype=torch.float32)
            g=bank['gradients'] if a=='point' else bank['pair_gradients']['augmented' if a=='augmented' else 'paired']
            mixed=w@g[ix]; norm=float(mixed.double().norm())
            contributions=-(g[ix].double()@jac)*w.double()*(r['requested_step_norm']/norm)
            state,_=combine_step(fast,g[ix],w,learning_rate=.01,length=r['requested_step_norm'])
            with torch.no_grad():
                before=logit(model,x,m,fast); after=logit(model,x,m,state)
            values=contributions.cpu().numpy(); top=np.argsort(abs(values))[::-1][:6]
            attribution[a]=dict(first_order_logit_change=float(contributions.sum()),
                actual_logit_change=float((after-before)[0]),
                first_affected_prediction=r['first_affected_prediction'],
                top_historical_contributions=[dict(shot=int(bank['shots'][ix[j]]),row=int(ix[j]),
                    label=int(bank['labels'][ix[j]]),weight=float(w[j]),linear_logit_change=float(values[j])) for j in top])
        result['attribution']=attribution;outputs.append(result)
    artifact=dict(scope='post-hoc, outcome-selected diagnosis; linearized model influence, not physical causation',
        selected_cases='all4 lostTP; removedFP66404; preservedTP64581/69768 versus augmented',
        completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),cases=outputs)
    _durable_json(root/'case_influence.json',artifact)
    for r in outputs:
        print(dict(shot=r['shot'],prob=r['saved_probability'],paired=r['attribution']['paired']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
