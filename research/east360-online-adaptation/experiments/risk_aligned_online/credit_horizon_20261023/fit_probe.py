"""Post-hoc training-shot probe; not a new validation or deployment result."""
import argparse
import copy
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.learning import episode_pool
from ..warmstart_alignment_20261022.alignment import transform
from .learning import episode
from .run import READ, ARMS, load_model


def probe(root):
    out=root/'fit_probe.json'
    if out.exists():raise FileExistsError(out)
    verification=READ(root/'verification.json')
    assert verification['status']=='VERIFIED' and verification['completed_sha256']==sha(root/'completed.json')
    p=READ(root/'protocol.json'); sid=63226; rows=[]
    torch.set_num_threads(2); torch.manual_seed(131)
    boundary=math.log(p['threshold']/(1-p['threshold']))
    original_signal,original_outcome=simple_data.read_shot,io23.read_outcome
    def signal_guard(device,shot,**kwargs):
        assert device=='EAST' and shot==sid
        return original_signal(device,shot,**kwargs)
    def outcome_guard(device,shot):
        assert device=='EAST' and shot==sid
        return original_outcome(device,shot)
    for plan in p['folds']:
        if sid not in plan['fit']:continue
        f=plan['fold']; folder=root/f'fold{f}'
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=outcome_guard):
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
        blocks=episode_pool([s],[sid])[0]
        assert blocks==next(r for r in READ(folder/'eligibility.json') if r['shot']==sid)
        signal=transform({k:s[k] for k in ('values','available')},READ(folder/'input_map.json'))
        warm=load_model(f)
        for arm in ('warm',)+ARMS:
            model=copy.deepcopy(warm)
            if arm!='warm':model.load_state_dict(torch.load(folder/f'{arm}.pt',map_location='cuda',weights_only=True)['model'])
            model.set_stage('frozen'); saved=state_cpu(model)
            for block in blocks['blocks']:
                r=episode(model,signal,block['count'],block['ends'],True,boundary=boundary,tail=1,device='cuda')
                windows=[simple_data.window(signal,e,128) for e in block['ends']]
                x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
                with torch.no_grad():
                    h=model.predict_fast(x,m,r['fast'],mode='risk_state')['risk_state']
                    peak=float(model.base.risk_head(h).sigmoid().max())
                rows.append(dict(fold=f,shot=sid,arm=arm,count=block['count'],ends=block['ends'],
                    risk_before=float(r['before']),risk_after=float(r['risk'].detach()),
                    forecast_loss_last_observed=float(r['forecast'].detach()),max_probability=peak,
                    block_timely_alarm=peak>=p['threshold']))
                del r
            assert_unchanged(model,saved)
        print(dict(stage='fit_probe',fold=f,rows=len(rows)),flush=True)
    report=dict(status='COMPLETED',selection='Post-hoc:63226 was the lost held warning and a training outlier; only folds where it is a preparation identity.',
        limitation='Fixed in-sample diagnostic blocks, not a new shot-level test metric; original outcomes never used in deployment updates.',
        threshold=p['threshold'],rows=rows,verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)),test360_accessed=False)
    _durable_json(out,report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);probe(parser.parse_args().root)
