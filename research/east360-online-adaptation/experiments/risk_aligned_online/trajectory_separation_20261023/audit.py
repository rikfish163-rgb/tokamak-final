"""Recount local algebra and independently rerun selected neural diagnostics."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import REFERENCE,REF_SHA,IDS,MODELS,PLAN,models_for,inspect


def main(root):
    out=root/'verification.json'
    if out.exists():raise FileExistsError(out)
    p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h
    for name,h in p['code_sha256'].items():assert sha(name)==h
    for name,h in p['checkpoints'].items():assert sha(name)==h
    assert sha(PLAN)==p['plan_sha256'] and sha(REFERENCE/'completed.json')==REF_SHA
    updates=0;amplifying=0;maxerror=0.;rows=[]
    for f in range(4):
        for sid in IDS:
            for m in MODELS:
                r=READ(root/f'fold{f}_{sid}_{m}.json');rows.append(r)
                assert r['receipts_exact']
                assert r['maximum_amplification']==max(r['steps'],key=lambda x:x['distance_ratio'])
                for s in r['steps']:
                    before=s['distance_before']**2
                    estimate=before+s['linear_distance_squared_change']+s['quadratic_distance_squared_change']
                    np.testing.assert_allclose(estimate,s['distance_after_squared'],rtol=1e-10,atol=1e-15)
                    np.testing.assert_allclose((s['distance_after_squared']/before)**.5,s['distance_ratio'],rtol=1e-12)
                    updates+=1;amplifying+=s['distance_ratio']>1
                    maxerror=max(maxerror,s['float32_distance_error'])
                actual=sum(c['linear_distance_squared_change'] for c in r['channel_components'])
                np.testing.assert_allclose(actual,r['maximum_amplification']['linear_distance_squared_change'],rtol=1e-3,atol=1e-10)
    assert updates==done['common_updates'] and amplifying==done['amplifying_updates']
    torch.set_num_threads(2);torch.manual_seed(131);replayed=[]
    for f,sid,m in ((1,77639,'prepared'),(0,77412,'prepared'),(3,75604,'prepared')):
        model=models_for(f)[m];frozen=state_cpu(model)
        def deny(*args,**kwargs):raise AssertionError('no outcome in neural diagnostic')
        with patch.object(io23,'read_outcome',side_effect=deny):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},READ(REFERENCE/f'fold{f}'/'input_map.json'))
            old=READ(root/f'fold{f}_{sid}_{m}.json')
            r=inspect(model,signal,old['critical_index'],READ(REFERENCE/f'fold{f}'/'shots'/f'{sid}_updates.json')[m])
            assert all(v==old[k] for k,v in r.items())
        assert_unchanged(model,frozen);replayed.append(dict(fold=f,shot=sid,model=m,exact=True))
    result=dict(status='VERIFIED_POSTHOC_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),records=len(rows),common_updates=updates,
        amplifying_updates=amplifying,max_float32_distance_error=maxerror,full_selected_replays=replayed,
        interpretation='Local numerical dynamics; not new performance results, independent shots or physical causes.')
    _durable_json(out,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
