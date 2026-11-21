"""All-shot scalar recount and predeclared three-shot/four-fold CUDA replay."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,REFERENCE,REF_SHA
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.audit import independently_score,aggregate
from ..increment_forecast_20261023.learning import predict
from ..labeled_alignment_20261023.run import historical_pool
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .run import PREPARED,PREPARED_SHA,PLAN,MODELS,CURVES,models_for


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    start=time.monotonic();p=READ(root/'protocol.json');done=READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['checkpoints'].items():assert sha(name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(REFERENCE/'completed.json')==REF_SHA and sha(PREPARED/'completed.json')==PREPARED_SHA
    historical,pool_audit=historical_pool(io23.cohort_ids('EAST'))
    used={sid for fold in p['folds'] for sid in fold['fit']+fold['held']}
    ids=sorted(r['shot'] for r in historical if r['shot'] not in used)
    assert ids==p['shot_ids'] and len(ids)==65 and len(used)==20
    assert pool_audit==READ(root/'historical_pool.json')
    assert not (set(ids)&set(io23.cohort_ids('EAST')))
    torch.set_num_threads(2);torch.manual_seed(131)
    seals={(r['fold'],r['shot']):r['sha256'] for r in READ(root/'seals.json')}
    assert len(seals)==4*65
    fixed_replay=[ids[0],ids[len(ids)//2],ids[-1]]
    update_counts={m:0 for m in MODELS};loss_descent={m:0 for m in MODELS};replayed=[];summary={};changes={}
    details={(r['fold'],r['arm'],r['shot']):r for r in READ(root/'peak_details.json')}
    for f in range(4):
        records={a:[] for a in CURVES};folder=root/f'fold{f}'
        for sid in ids:
            path=folder/'shots'/f'{sid}.npz';assert sha(path)==seals[f,sid]
            d,down=io23.read_outcome('EAST',sid);receipts=READ(folder/'shots'/f'{sid}_updates.json')
            with np.load(path) as z:
                times=z['times'];valid=np.ones(len(times),bool) if not d else ((down-times)*1000>=-1e-6)
                for a in CURVES:
                    scores=z[a]
                    r=independently_score(sid,d,down,times[valid],scores[valid],p['threshold']);records[a].append(r)
                    eligible=[i for i,t in enumerate(times) if valid[i]]
                    timely=[i for i in eligible if d and round(float((down-times[i])*1000),9)>30.]
                    labeltimely=[i for i in timely if round(float((down-times[i])*1000),9)<=125.+1e-6]
                    maxima=[max((float(scores[i]) for i in pos),default=None) for pos in (eligible,timely,labeltimely)]
                    detail=details[f,a,sid]
                    assert maxima==[detail[k] for k in ('max_valid','max_timely','max_label_timely')]
                for m in MODELS:
                    np.testing.assert_array_equal(z[f'{m}_static'][:256],z[f'{m}_intra'][:256])
                    assert len(receipts[m])==len(range(256,len(times),128))
                    for j,r in enumerate(receipts[m]):
                        count=256+j*128
                        assert r['last_observation']==count-1 and r['first_affected_prediction']==count
                        assert all(e+25<=count-1 for e in r['forecast_origins'])
                        assert r['parameter_delta_norm']<=.00101
                        update_counts[m]+=r['gradient_steps']
                        loss_descent[m]+=int(r.get('loss_after',float('inf'))<r.get('loss_before',-float('inf')))
        summary[str(f)]={a:aggregate(r,p['threshold']) for a,r in records.items()}
        assert summary[str(f)]==done['metrics'][str(f)]
        saved=READ(folder/'per_shot.json')
        for a in CURVES:assert records[a]==saved[a]['predictions']
        changes[str(f)]={}
        for name,left,right in [(m,f'{m}_static',f'{m}_intra') for m in MODELS]+[('preparation_static','warm_static','prepared_static')]:
            pairs=list(zip(records[left],records[right]))
            assert all(x['shot']==y['shot'] for x,y in pairs)
            c=dict(rescued=[x['shot'] for x,y in pairs if x['is_disrupt'] and not x['timely'] and y['timely']],
                   lost=[x['shot'] for x,y in pairs if x['is_disrupt'] and x['timely'] and not y['timely']],
                   removed_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and x['alarm'] and not y['alarm']],
                   added_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and not x['alarm'] and y['alarm']])
            changes[str(f)][name]=c
        models=models_for(f);frozen={a:state_cpu(m) for a,m in models.items()}
        mapping=READ(folder/'input_map.json');assert mapping==READ(REFERENCE/f'fold{f}'/'input_map.json')
        def deny(*args,**kwargs):raise AssertionError('outcome access during replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in fixed_replay:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    np.testing.assert_array_equal(z['times'],s['times'])
                    stored=READ(folder/'shots'/f'{sid}_updates.json')
                    for a,m in models.items():
                        pred,receipts,_=predict(m,signal)
                        assert receipts==stored[a]
                        for k,v in pred.items():np.testing.assert_array_equal(z[f'{a}_{k}'],v)
                        replayed.append(dict(fold=f,shot=sid,model=a,exact=True))
        for a,m in models.items():assert_unchanged(m,frozen[a])
        print(dict(stage='audit',fold=f,replayed=len(replayed),seconds=round(time.monotonic()-start,2)),flush=True)
    assert changes==done['changes'] and update_counts==done['updates']
    assert READ(root/'signal_access.json')==[dict(kind='signal',shot=sid) for f in range(4) for sid in ids]
    assert READ(root/'evaluation_access.json')==[dict(fold=f,shot=sid,phase='all_predictions_sealed') for f in range(4) for sid in ids]
    result=dict(status='VERIFIED_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),independent_scalar_recounts=4*65*4,
        distinct_shots=65,neural_replay=replayed,updates=update_counts,arrived_loss_decreased=loss_descent,
        seconds=time.monotonic()-start,test360_accessed=False,independent_confirmation=False,
        limitation='Same65 shots repeated under4 preparation folds, not260 independent shots; only predeclared3 identities fully re-inferred.')
    _durable_json(root/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
