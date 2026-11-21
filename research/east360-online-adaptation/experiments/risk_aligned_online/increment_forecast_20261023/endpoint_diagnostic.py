"""Post-hoc fixed-endpoint sensitivity; never a deployable update selector.

The endpoint is chosen after reading an already-scored development outcome.
Its future input is used ONLY by a no-grad probe. The real update receives
the observed prefix, and every update receipt must match the sealed run.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..history_tangent_20261023.learning import logit
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import IncrementFeedback, update

READ = lambda p: json.loads(p.read_text())


def main(root):
    path = root/'endpoint63226_posthoc.json'
    if path.exists(): raise FileExistsError(path)
    p, done = READ(root/'protocol.json'), READ(root/'completed.json')
    assert READ(root/'verification.json')['status']=='VERIFIED'
    for name,digest in done['artifact_sha256'].items(): assert sha(root/name)==digest
    torch.set_num_threads(2); torch.manual_seed(131)
    sid=63226; plan=next(f for f in p['folds'] if sid in f['held']); folder=root/f'fold{plan["fold"]}'
    base,_=load_parent(DEFAULT_PARENT)
    s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
    signal=transform({k:s[k] for k in ('values','available')},READ(folder/'input_map.json'))
    with np.load(folder/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
        endpoint=int(np.flatnonzero(s['valid']&(trace['static']>=p['threshold']))[0])
        recorded={h:float(trace[h][endpoint]) for h in ('static','absolute','increment')}
    x,m=simple_data.window(signal,endpoint,128)
    x,m=[torch.as_tensor(a[None],device='cuda') for a in (x,m)]
    saved=READ(folder/'shots'/f'{sid}_updates.json'); output={}
    for h in ('absolute','increment'):
        model=IncrementFeedback(copy.deepcopy(base),h).cuda()
        model.load_state_dict(torch.load(folder/f'{h}.pt',map_location='cuda',weights_only=True)['model'])
        model.set_stage('frozen'); original=state_cpu(model); fast=initial_encoder(model)
        def probe(state):
            with torch.no_grad(): return float(logit(model,x,m,state).item())
        initial=probe(fast); before=initial; rows=[]
        for count in range(256,endpoint+1,128):
            fast,r=update(model,{k:v[:count] for k,v in signal.items()},count,fast,device='cuda')
            assert r==saved[h][len(rows)]
            after=probe(fast)
            rows.append(dict(count=count,time=float(s['times'][count-1]),logit_before=before,
                             logit_after=after,delta=after-before,signal_loss_drop=r['loss_before']-r['loss_after']))
            before=after
        final_probability=float(torch.sigmoid(torch.tensor(before,dtype=torch.float32)))
        # The probe uses batch size one, while recorded inference uses 32.
        # CUDA floating-point reduction order can differ, so report the gap.
        difference=abs(final_probability-recorded[h]); assert difference < 1e-6
        assert_unchanged(model,original)
        output[h]=dict(initial_logit=initial,final_logit=before,final_probability=final_probability,
            recorded_probability=recorded[h],batch_difference=difference,updates=len(rows),
            risk_decreasing_updates=sum(r['delta']<0 for r in rows),
            risk_increasing_updates=sum(r['delta']>0 for r in rows),
            total_delta=before-initial,largest_decrease=min(rows,key=lambda r:r['delta']),rows=rows)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='POSTHOC_DIAGNOSTIC',shot=sid,endpoint=endpoint,time=float(s['times'][endpoint]),
        lead_ms=round(float((s['down']-s['times'][endpoint])*1000),9),recorded=recorded,
        selected_using='known development outcome and static first alarm; NOT an online selection rule',
        scope='same future endpoint probed across earlier weights; labels/future never passed to update',
        completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),arms=output)
    _durable_json(path,result)
    print({**result,'arms':{h:{k:v for k,v in r.items() if k!='rows'} for h,r in output.items()}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); main(parser.parse_args().root)
