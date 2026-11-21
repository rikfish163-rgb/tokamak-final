"""Independent scalar scoring; selected full CUDA prediction/receipt replay."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.audit import independently_score,aggregate
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS,predict
from .run import REFERENCE,REF_SHA,PLAN,CURVES,MODELS,models_for


def main(root):
    out=root/'verification.json'
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    for name,h in done['artifact_sha256'].items():assert sha(root/name)==h
    for name,h in p['code_sha256'].items():assert sha(io23.ROOT/name)==h
    for group in ('checkpoints','input_sha256'):
        for name,h in p[group].items():assert sha(name)==h
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    assert len(p['shot_ids'])==65 and not set(p['shot_ids'])&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2);torch.manual_seed(131)
    counts={f'{m}_{a}':0 for m in MODELS for a in ARMS};changes={};replays=[];maxnorm=0.
    for f in range(4):
        folder=root/f'fold{f}';records={a:[] for a in CURVES}
        for sid in p['shot_ids']:
            d,down=io23.read_outcome('EAST',sid)
            updates=READ(folder/'shots'/f'{sid}_updates.json')
            with np.load(folder/'shots'/f'{sid}.npz') as z,np.load(REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz') as old:
                valid=np.ones(len(z['times']),bool) if not d else (down-z['times'])*1000>=-1e-6
                for a in CURVES:records[a].append(independently_score(sid,d,down,z['times'][valid],z[a][valid],p['threshold']))
                for m in MODELS:
                    for a in ('static','intra'):np.testing.assert_array_equal(z[f'{m}_{a}'],old[f'{m}_{a}'])
                    for a in ARMS:
                        np.testing.assert_array_equal(z[f'{m}_{a}'][:256],z[f'{m}_static'][:256])
                        rs=updates[m][a];assert len(rs)==len(range(256,len(z['times']),128))
                        for i,r in enumerate(rs):
                            count=256+128*i
                            assert r['last_observation']==count-1 and r['first_affected_prediction']==count
                            assert all(e+25<=count-1 for e in r['forecast_origins'])
                            assert r['parameter_delta_norm']<=.00101
                            if a=='observation':assert r['summary_delta_norm']==0
                            counts[f'{m}_{a}']+=r['gradient_steps']
                    for a,b in zip(updates[m]['observation'],updates[m]['matched_all']):
                        assert a['requested_norm']==b['requested_norm']
                        if not b.get('zero_gradient_cannot_match',False):
                            maxnorm=max(maxnorm,abs(a['parameter_delta_norm']-b['parameter_delta_norm']))
        saved=READ(folder/'per_shot.json')
        for a in CURVES:
            assert aggregate(records[a],p['threshold'])==done['metrics'][str(f)][a]
            assert records[a]==saved[a]['predictions']
        changes[str(f)]={}
        for m in MODELS:
            for b in ('static','intra'):
                for a in ARMS:
                    pairs=list(zip(records[f'{m}_{b}'],records[f'{m}_{a}']))
                    changes[str(f)][f'{m}:{b}->{a}']=dict(
                        rescued=[x['shot'] for x,y in pairs if x['is_disrupt'] and not x['timely'] and y['timely']],
                        lost=[x['shot'] for x,y in pairs if x['is_disrupt'] and x['timely'] and not y['timely']],
                        removed_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and x['alarm'] and not y['alarm']],
                        added_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and not x['alarm'] and y['alarm']])
        models=models_for(f);frozen={m:state_cpu(v) for m,v in models.items()};mapping=READ(folder/'input_map.json')
        def deny(*args,**kwargs):raise AssertionError('no outcome during prediction replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in (54314,77639,89117):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                receipts=READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    for m,model in models.items():
                        pred,r=predict(model,signal);assert r==receipts[m]
                        for a in ARMS:np.testing.assert_array_equal(pred[a],z[f'{m}_{a}'])
                        replays.append(dict(fold=f,shot=sid,model=m,exact=True))
        for m,model in models.items():assert_unchanged(model,frozen[m])
        print(dict(fold=f,replays=len(replays),seconds=round(time.monotonic()-start,2)),flush=True)
    assert counts==done['updates'] and changes==done['changes']
    assert maxnorm<1e-6
    result=dict(status='VERIFIED_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),scalar_recounts=4*65*len(CURVES),
        full_replays=replays,updates=counts,max_matched_actual_norm_difference=maxnorm,
        source_unchanged=True,test360_accessed=False,seconds=time.monotonic()-start)
    _durable_json(out,result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
