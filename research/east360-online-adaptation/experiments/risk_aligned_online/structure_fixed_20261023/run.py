"""Paired structural-parameter freeze; reuse immutable static/raw controls."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..external65_20261023.run import models_for,MODELS
from ..external65_20261023.analysis import paired_changes
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS,predict

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/external65_frozen_20261023'
REF_SHA='1dbc3b674371c87fd65e12617f55d6866338dc8b1d054e3ff2880fefbc9f93c3'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/固定结构参数_65炮配对约定_20261023.md'
CURVES=tuple(f'{m}_{a}' for m in MODELS for a in ('static','intra')+ARMS)


def evaluate(root,p):
    summaries={};changes={}
    for f in range(4):
        rows={a:[] for a in CURVES}
        for sid in p['shot_ids']:
            d,down=io23.read_outcome('EAST',sid)
            with np.load(root/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                valid=io23.labels_at(z['times'],d,down,125.)[1]
                for a in CURVES:rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=d,down=down))
        results={a:metrics(rs,p['threshold'],lead=30.) for a,rs in rows.items()}
        _durable_json(root/f'fold{f}'/'per_shot.json',results)
        summaries[str(f)]={a:dict(**public(r),very_early_over3000=sum(
            x['timely'] and x['lead_ms']>3000 for x in r['predictions'])) for a,r in results.items()}
        changes[str(f)]={f'{m}:{b}->{a}':paired_changes(results[f'{m}_{b}']['predictions'],results[f'{m}_{a}']['predictions'])
            for m in MODELS for b in ('static','intra') for a in ARMS}
    return summaries,changes


def main(root):
    if root.exists():raise FileExistsError(root)
    assert sha(REFERENCE/'completed.json')==REF_SHA
    prior=READ(REFERENCE/'protocol.json');done=READ(REFERENCE/'completed.json')
    for name,h in done['artifact_sha256'].items():assert sha(REFERENCE/name)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    for group in ('checkpoints','input_sha256'):
        for name,h in prior[group].items():assert sha(name)==h
    root.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    paths=set(Path(__file__).parent.glob('*.py'));paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_structure_fixed.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for pp in sorted(paths):z.write(pp,str(pp.relative_to(io23.ROOT)))
    p=dict(reference_sha256=REF_SHA,plan_sha256=sha(PLAN),shot_ids=prior['shot_ids'],folds=prior['folds'],
        roster=prior['roster'],threshold=prior['threshold'],checkpoints=prior['checkpoints'],input_sha256=prior['input_sha256'],
        code_sha256={str(pp.relative_to(io23.ROOT)):sha(pp) for pp in paths},
        new_arms=ARMS,cached_arms=['static','intra'],test360_accessed=False,independent_confirmation=False,
        offline_steps=0,inter_steps=0,source_sha256=PARENT_SHA256)
    _durable_json(root/'protocol.json',p);counts={f'{m}_{a}':0 for m in MODELS for a in ARMS}
    def deny(*args,**kwargs):raise AssertionError('no outcome access until all predictions sealed')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for f in range(4):
            folder=root/f'fold{f}';(folder/'shots').mkdir(parents=True)
            models=models_for(f);frozen={m:state_cpu(v) for m,v in models.items()}
            mapping=READ(REFERENCE/f'fold{f}'/'input_map.json');_durable_json(folder/'input_map.json',mapping)
            for i,sid in enumerate(p['shot_ids'],1):
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal=transform({k:s[k] for k in ('values','available')},mapping);curves={};receipts={}
                oldpath=REFERENCE/f'fold{f}'/'shots'/f'{sid}.npz'
                assert sha(oldpath)==done['artifact_sha256'][str(oldpath.relative_to(REFERENCE))]
                with np.load(oldpath) as old:
                    np.testing.assert_array_equal(s['times'],old['times'])
                    for name,model in models.items():
                        scores,r=predict(model,signal);receipts[name]=r
                        for a in ('static','intra'):curves[f'{name}_{a}']=old[f'{name}_{a}'].copy()
                        for a,v in scores.items():
                            np.testing.assert_array_equal(v[:256],old[f'{name}_static'][:256])
                            curves[f'{name}_{a}']=v;counts[f'{name}_{a}']+=sum(x['gradient_steps'] for x in r[a])
                np.savez_compressed(folder/'shots'/f'{sid}.npz',times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',receipts)
                if i%10==0 or i==len(p['shot_ids']):
                    progress=dict(fold=f,shots=i,seconds=round(time.monotonic()-start,2))
                    _durable_json(root/'progress.json',progress);print(progress,flush=True)
            for name,model in models.items():assert_unchanged(model,frozen[name])
    summary,changes=evaluate(root,p)
    for path,h in p['checkpoints'].items():assert sha(path)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='COMPLETED',metrics=summary,changes=changes,updates=counts,
        seconds=time.monotonic()-start,test360_accessed=False,source_unchanged=True,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',result)
    print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
