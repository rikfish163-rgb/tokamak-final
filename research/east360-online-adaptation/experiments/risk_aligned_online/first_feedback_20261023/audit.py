"""Independent scalar recount plus fixed first/middle/last CUDA replay."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..external65_20261023.run import models_for, MODELS
from ..horizon_memory_20261023.audit import independently_score, aggregate
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS, predict, schedule
from .run import REFERENCE65, REFERENCE65_SHA, CURVES, PLAN


def audit(root):
    if (root/'verification.json').exists(): raise FileExistsError(root/'verification.json')
    start=time.monotonic(); p=READ(root/'protocol.json'); done=READ(root/'completed.json')
    for name,h in done['artifact_sha256'].items(): assert sha(root/name)==h,name
    for name,h in p['code_sha256'].items(): assert sha(io23.ROOT/name)==h,name
    for group in ('input_sha256','checkpoints'):
        for name,h in p[group].items(): assert sha(name)==h,name
    assert sha(PLAN)==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(REFERENCE65/'completed.json')==REFERENCE65_SHA==p['reference_sha256']
    old=READ(REFERENCE65/'protocol.json')
    for key in ('shot_ids','folds','roster','threshold','checkpoints','input_sha256'): assert p[key]==old[key],key
    ids=p['shot_ids']; replay_ids=[ids[0],ids[len(ids)//2],ids[-1]]
    assert not set(ids)&set(io23.cohort_ids('EAST'))
    assert not set(ids)&{i for fold in p['folds'] for i in fold['fit']+fold['held']}
    seals={(r['fold'],r['shot']):r['sha256'] for r in READ(root/'seals.json')}
    assert len(seals)==260
    torch.set_num_threads(2);torch.manual_seed(131)
    counts={a:0 for a in CURVES if not a.endswith('_static')}
    summary={};changes={}; replayed=[];old_differences=[];floors={};short=[];baseline=[]
    for f in range(4):
        folder=root/f'fold{f}';records={a:[] for a in CURVES}
        floors[str(f)]={m:{str(c):[] for c in (128,153,256)} for m in MODELS}
        for sid in ids:
            path=folder/'shots'/f'{sid}.npz';assert sha(path)==seals[f,sid]
            d,down=io23.read_outcome('EAST',sid)
            receipts=READ(folder/'shots'/f'{sid}_updates.json')
            with np.load(path) as z, np.load(REFERENCE65/f'fold{f}'/'shots'/f'{sid}.npz') as archived:
                times=z['times'];n=len(times)
                valid=np.ones(n,bool) if not d else ((down-times)*1000>=-1e-6)
                np.testing.assert_array_equal(times,archived['times'])
                if n<=256: short.append(dict(fold=f,shot=sid,length=n))
                for a in CURVES:
                    assert len(z[a])==n
                    records[a].append(independently_score(sid,d,down,times[valid],z[a][valid],p['threshold']))
                for m in MODELS:
                    for a,b in (('static','static'),('late4','intra')):
                        x,y=z[f'{m}_{a}'],archived[f'{m}_{b}']
                        baseline.append(dict(fold=f,shot=sid,arm=f'{m}_{a}',max_abs_error=float(np.max(np.abs(x-y))),
                            threshold_crossing_disagreements=np.flatnonzero((x>=p['threshold'])!=(y>=p['threshold'])).tolist()))
                    if not d:
                        first=next((i for i,v in enumerate(z[f'{m}_static']) if v>=p['threshold']),None)
                        for c in (128,153,256):
                            if first is not None and first<c: floors[str(f)][m][str(c)].append(sid)
                    for a in ARMS[1:]:
                        expected=schedule(n,a);rs=receipts[m][a]
                        assert len(rs)==len(expected)
                        limit=expected[0] if expected else n
                        np.testing.assert_array_equal(z[f'{m}_{a}'][:limit],z[f'{m}_static'][:limit])
                        for c,r in zip(expected,rs):
                            assert r['last_observation']==c-1 and r['first_affected_prediction']==c
                            origins=[c-1-25] if (c<384 and a!='late4') else [c-1-25*(j+1) for j in range(4)]
                            assert r['forecast_origins']==origins
                            assert all(e>=127 and e+25<=c-1 for e in origins)
                            assert r['parameter_delta_norm']<=.00101
                            counts[f'{m}_{a}']+=r['gradient_steps']
        summary[str(f)]={a:aggregate(rs,p['threshold']) for a,rs in records.items()}
        assert summary[str(f)]==done['metrics'][str(f)]
        saved=READ(folder/'per_shot.json');archived=READ(REFERENCE65/f'fold{f}'/'per_shot.json')
        for a in CURVES: assert records[a]==saved[a]['predictions'],a
        changes[str(f)]={}
        for m in MODELS:
            changes[str(f)][m]={}
            for b,a in (('late1','early1'),('late4','late1'),('static','early1')):
                pairs=list(zip(records[f'{m}_{b}'],records[f'{m}_{a}']))
                assert all(x['shot']==y['shot'] for x,y in pairs)
                changes[str(f)][m][f'{b}_to_{a}']=dict(
                    rescued=[x['shot'] for x,y in pairs if x['is_disrupt'] and not x['timely'] and y['timely']],
                    lost=[x['shot'] for x,y in pairs if x['is_disrupt'] and x['timely'] and not y['timely']],
                    removed_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and x['alarm'] and not y['alarm']],
                    added_fp=[x['shot'] for x,y in pairs if not x['is_disrupt'] and not x['alarm'] and y['alarm']])
            for a,b in (('static','static'),('late4','intra')):
                for x,y in zip(records[f'{m}_{a}'],archived[f'{m}_{b}']['predictions']):
                    if x!=y: old_differences.append(dict(fold=f,arm=f'{m}_{a}',current=x,archived=y))
        models=models_for(f);frozen={m:state_cpu(v) for m,v in models.items()}
        mapping=READ(folder/'input_map.json')
        assert mapping==READ(REFERENCE65/f'fold{f}'/'input_map.json')
        def deny(*args,**kwargs): raise AssertionError('outcome access during replay')
        with patch.object(io23,'read_outcome',side_effect=deny):
            for sid in replay_ids:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping)
                receipt=READ(folder/'shots'/f'{sid}_updates.json')
                with np.load(folder/'shots'/f'{sid}.npz') as z:
                    for m,model in models.items():
                        curves,updates=predict(model,signal)
                        assert updates==receipt[m]
                        for a,v in curves.items(): np.testing.assert_array_equal(z[f'{m}_{a}'],v)
                        replayed.append(dict(fold=f,model=m,shot=sid,all_four_arms_exact=True))
        for m,model in models.items(): assert_unchanged(model,frozen[m])
        print(dict(stage='audit',fold=f,replays=len(replayed),seconds=round(time.monotonic()-start,2)),flush=True)
    assert baseline==READ(root/'baseline_comparison.json')
    assert short==done['short_records'] and counts==done['updates'] and changes==done['changes']
    assert READ(root/'signal_access.json')==[dict(shot=sid,kind='signal') for f in range(4) for sid in ids]
    assert READ(root/'evaluation_access.json')==[dict(fold=f,shot=sid,phase='all_predictions_sealed') for f in range(4) for sid in ids]
    result=dict(status='VERIFIED_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        scalar_recounts=4*65*8,distinct_shots=65,neural_replay=replayed,updates=counts,short_records=short,
        archived_per_shot_differences=old_differences,archived_curve_max_error=max(r['max_abs_error'] for r in baseline),
        archived_threshold_crossings=sum(len(r['threshold_crossing_disagreements']) for r in baseline),
        immutable_early_fp_ids=floors,seconds=time.monotonic()-start,test360_accessed=False,
        independent_confirmation=False,limitation='Same65 identities repeated in4 folds; only3 predeclared identities fully re-inferred per model and fold.')
    _durable_json(root/'verification.json',result); print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
