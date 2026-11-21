"""Fold-paired attention-routing experiment; no new historical preparation."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..horizon_memory_20261023.learning import FoldAccess
from ..increment_forecast_20261023.learning import IncrementFeedback
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS,predict,masks

READ=lambda p:json.loads(p.read_text())
REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/increment_forecast_cv20_20261023'
REF_SHA='3f5e0d0003104b1c54e413badfa5d85147a59b377b566d1d8a532bf20ff23e61'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/注意力时序关联适配_事前约定_20261023.md'


def evaluate(root,plans,threshold):
    rows={a:[] for a in ('static','full',*ARMS)};access=[]
    for plan in plans:
        for sid in plan['held']:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            access.append(dict(shot=sid,phase='after_all_seals'))
            with np.load(root/f'fold{plan["fold"]}'/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
                np.testing.assert_array_equal(trace['times'],s['times']);v=s['valid']
                for a in rows:rows[a].append(dict(shot=sid,times=s['times'][v],scores=trace[a][v],is_disrupt=s['is_disrupt'],down=s['down']))
    outcomes={a:metrics(r,threshold,lead=30.) for a,r in rows.items()}
    _durable_json(root/'per_shot.json',outcomes);_durable_json(root/'evaluation_access.json',access)
    return {a:dict(**public(r),very_early_over3000=sum(s['timely'] and s['lead_ms']>3000 for s in r['predictions'])) for a,r in outcomes.items()}


def main(root):
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    assert sha(REFERENCE/'completed.json')==REF_SHA
    prior=READ(REFERENCE/'completed.json');p=READ(REFERENCE/'protocol.json')
    for name,digest in prior['artifact_sha256'].items():assert sha(REFERENCE/name)==digest,name
    assert READ(REFERENCE/'verification.json')['status']=='VERIFIED'
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    base,info=load_parent(DEFAULT_PARENT);origin=state_cpu(base)
    root.mkdir(parents=True);paths=set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_attention_scope.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):archive.write(path,str(path.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),reference_sha256=REF_SHA,source_sha256=PARENT_SHA256,
        folds=p['folds'],threshold=info['threshold'],plan_path=str(PLAN),plan_sha256=sha(PLAN),
        code_sha256={str(path.relative_to(io23.ROOT)):sha(path) for path in paths},
        arms=ARMS,test360_accessed=False,independent_confirmation=False,seed=131,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    _durable_json(root/'protocol.json',protocol);steps=0;seals=[]
    read_signal=simple_data.read_shot;source_ids=io23.source_roles()['train']
    for plan in p['folds']:
        f=plan['fold'];out=root/f'fold{f}';ref=REFERENCE/f'fold{f}'
        out.mkdir();(out/'shots').mkdir()
        model=IncrementFeedback(copy.deepcopy(base),'increment').cuda()
        checkpoint=ref/'increment.pt';model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['model'])
        model.set_stage('frozen');frozen=state_cpu(model)
        mapping=READ(ref/'input_map.json');_durable_json(out/'input_map.json',mapping)
        q,v=masks(initial_encoder(model),model.cfg.dim)
        _durable_json(out/'parent_receipt.json',dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),
            qk_coordinates=sum(int(t.sum()) for t in q.values()),value_coordinates=sum(int(t.sum()) for t in v.values()),
            full_coordinates=sum(t.numel() for t in q.values())))
        access=FoldAccess(plan,source_ids);access.phase='predict'
        def signal_guard(device,sid,**kwargs):
            access.check('signal',device,sid);return read_signal(device,sid,**kwargs)
        def no_outcome(*args,**kwargs):raise AssertionError('no outcome before all predictions are sealed')
        with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=no_outcome):
            for sid in plan['held']:
                s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                scores,updates=predict(model,transform({k:s[k] for k in ('values','available')},mapping))
                with np.load(ref/'shots'/f'{sid}.npz',allow_pickle=False) as old:
                    np.testing.assert_array_equal(scores['static'],old['static']);scores['full']=old['increment'].copy()
                path=out/'shots'/f'{sid}.npz';up=out/'shots'/f'{sid}_updates.json'
                np.savez_compressed(path,times=s['times'],**scores);_durable_json(up,updates)
                seals.append(dict(fold=f,shot=sid,prediction_sha256=sha(path),updates_sha256=sha(up)))
                steps+=sum(r['gradient_steps'] for rows in updates.values() for r in rows)
        assert_unchanged(model,frozen);_durable_json(out/'access.json',access.log)
        print(dict(stage='fold_complete',fold=f,shots=len(seals),seconds=time.monotonic()-start),flush=True)
    _durable_json(root/'prediction_receipts.json',seals)
    result=evaluate(root,p['folds'],info['threshold'])
    assert result['static']==prior['metrics']['static'] and result['full']==prior['metrics']['increment']
    assert_unchanged(base,origin);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    done=dict(status='COMPLETED',metrics=result,seconds=time.monotonic()-start,neural_steps=steps,
        source_unchanged=True,test360_accessed=False,independent_confirmation=False,
        artifact_sha256={str(path.relative_to(root)):sha(path) for path in root.rglob('*') if path.is_file()})
    _durable_json(root/'completed.json',done);print({k:v for k,v in done.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
