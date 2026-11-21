"""Fixed-model, same-step-budget first-feedback timing experiment on65 shots."""
import argparse
from pathlib import Path
import sys
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ, REFERENCE
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..external65_20261023.run import models_for, MODELS
from ..external65_20261023.analysis import paired_changes
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha, metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .learning import ARMS, predict

REFERENCE65 = io23.ROOT/'experiments/risk_aligned_online/runs/external65_frozen_20261023'
REFERENCE65_SHA = '1dbc3b674371c87fd65e12617f55d6866338dc8b1d054e3ff2880fefbc9f93c3'
PLAN = io23.ROOT/'docs/research/source_event_rebuild_20261022/首次反馈提前_配对约定_20261023.md'
CURVES = tuple(f'{m}_{a}' for m in MODELS for a in ARMS)


def evaluate(root, protocol):
    summaries = {}; changes = {}; accesses = []
    for f in range(4):
        rows = {a:[] for a in CURVES}
        for sid in protocol['shot_ids']:
            d,down = io23.read_outcome('EAST',sid)
            accesses.append(dict(fold=f,shot=sid,phase='all_predictions_sealed'))
            with np.load(root/f'fold{f}'/'shots'/f'{sid}.npz') as z:
                valid = io23.labels_at(z['times'],d,down,125.)[1]
                for a in CURVES:
                    rows[a].append(dict(shot=sid,times=z['times'][valid],scores=z[a][valid],is_disrupt=d,down=down))
        results = {a:metrics(r,protocol['threshold'],lead=30.) for a,r in rows.items()}
        _durable_json(root/f'fold{f}'/'per_shot.json',results)
        summaries[str(f)] = {a:dict(**public(r),very_early_over3000=sum(
            x['timely'] and x['lead_ms']>3000 for x in r['predictions'])) for a,r in results.items()}
        changes[str(f)] = {m:{f'{b}_to_{a}':paired_changes(
            results[f'{m}_{b}']['predictions'],results[f'{m}_{a}']['predictions'])
            for b,a in (('late1','early1'),('late4','late1'),('static','early1'))} for m in MODELS}
    _durable_json(root/'evaluation_access.json',accesses)
    return summaries,changes


def main(root):
    if root.exists(): raise FileExistsError(root)
    if not torch.cuda.is_available(): raise RuntimeError('rl_env CUDA required')
    assert sha(REFERENCE65/'completed.json') == REFERENCE65_SHA
    prior = READ(REFERENCE65/'protocol.json'); old = READ(REFERENCE65/'completed.json')
    assert READ(REFERENCE65/'verification.json')['status'] == 'VERIFIED_DIAGNOSTIC'
    for name,h in old['artifact_sha256'].items(): assert sha(REFERENCE65/name)==h,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    for group in ('checkpoints','input_sha256'):
        for path,h in prior[group].items(): assert sha(path)==h,path
    ids = prior['shot_ids']; assert len(ids)==65 and not set(ids)&set(io23.cohort_ids('EAST'))
    torch.set_num_threads(2); torch.manual_seed(131); torch.cuda.reset_peak_memory_stats()
    start = time.monotonic(); root.mkdir(parents=True)
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.add(io23.ROOT/'experiments/risk_aligned_online/tests/test_first_feedback.py')
    for name,module in list(sys.modules.items()):
        if name.startswith('experiments.risk_aligned_online.') and getattr(module,'__file__',None):
            paths.add(Path(module.__file__))
    with zipfile.ZipFile(root/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths): z.write(path,str(path.relative_to(io23.ROOT)))
    p = dict(shot_ids=ids,folds=prior['folds'],roster=prior['roster'],threshold=prior['threshold'],
        source_sha256=PARENT_SHA256,checkpoints=prior['checkpoints'],input_sha256=prior['input_sha256'],
        reference_sha256=REFERENCE65_SHA,plan_path=str(PLAN),plan_sha256=sha(PLAN),
        code_sha256={str(v.relative_to(io23.ROOT)):sha(v) for v in paths},
        first_counts=dict(late4=256,late1=256,early1=153),subsequent='384,512,...; interval128',
        learning_rate=.001,gradient_clip=1.,offline_steps=0,inter_steps=0,
        independent_confirmation=False,test360_accessed=False,retrospective_selection=True,
        prediction_batch_boundaries='union of32-grid and all arms update counts',
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    _durable_json(root/'protocol.json',p)
    counts = {a:0 for a in CURVES if not a.endswith('_static')}
    seals = []; access = []; baseline = []; short = []; read_signal = simple_data.read_shot
    def signal_guard(device,sid,**kwargs):
        if device!='EAST' or sid not in ids: raise AssertionError('signal outside declared65')
        access.append(dict(shot=sid,kind='signal'))
        return read_signal(device,sid,**kwargs)
    def deny(*args,**kwargs): raise AssertionError('outcome access before prediction seal')
    with patch.object(simple_data,'read_shot',side_effect=signal_guard),patch.object(io23,'read_outcome',side_effect=deny):
        for f in range(4):
            folder = root/f'fold{f}'; (folder/'shots').mkdir(parents=True)
            models = models_for(f); frozen = {m:state_cpu(v) for m,v in models.items()}
            mapping = READ(REFERENCE/f'fold{f}'/'input_map.json')
            assert mapping == READ(REFERENCE65/f'fold{f}'/'input_map.json')
            _durable_json(folder/'input_map.json',mapping)
            for number,sid in enumerate(ids,1):
                s = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
                signal = transform({k:s[k] for k in ('values','available')},mapping)
                curves = {}; receipts = {}
                if len(s['times'])<=256: short.append(dict(fold=f,shot=sid,length=len(s['times'])))
                for name,model in models.items():
                    pred,receipt = predict(model,signal)
                    curves.update({f'{name}_{a}':v for a,v in pred.items()}); receipts[name]=receipt
                    for a,r in receipt.items(): counts[f'{name}_{a}']+=sum(v['gradient_steps'] for v in r)
                path = folder/'shots'/f'{sid}.npz'
                np.savez_compressed(path,times=s['times'],**curves)
                _durable_json(folder/'shots'/f'{sid}_updates.json',receipts)
                seals.append(dict(fold=f,shot=sid,sha256=sha(path)))
                with np.load(REFERENCE65/f'fold{f}'/'shots'/f'{sid}.npz') as archived:
                    np.testing.assert_array_equal(archived['times'],s['times'])
                    for name in MODELS:
                        for a,b in (('static','static'),('late4','intra')):
                            x,y=curves[f'{name}_{a}'],archived[f'{name}_{b}']
                            baseline.append(dict(fold=f,shot=sid,arm=f'{name}_{a}',
                                max_abs_error=float(np.max(np.abs(x-y))),
                                threshold_crossing_disagreements=np.flatnonzero((x>=p['threshold'])!=(y>=p['threshold'])).tolist()))
                if number%5==0 or number==len(ids):
                    progress=dict(fold=f,shots=number,seconds=round(time.monotonic()-start,2))
                    _durable_json(root/'progress.json',progress); print(progress,flush=True)
            for name,model in models.items(): assert_unchanged(model,frozen[name])
            del models,model,frozen
    _durable_json(root/'seals.json',seals); _durable_json(root/'signal_access.json',access)
    _durable_json(root/'baseline_comparison.json',baseline)
    summary,changes=evaluate(root,p)
    for path,h in p['checkpoints'].items(): assert sha(path)==h
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='COMPLETED',metrics=summary,changes=changes,updates=counts,
        short_records=short,seconds=time.monotonic()-start,peak_cuda_mb=torch.cuda.max_memory_allocated()/2**20,
        source_unchanged=True,test360_accessed=False,independent_confirmation=False,
        offline_steps=0,inter_steps=0,distinct_shots=65,folds=4,
        artifact_sha256={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()})
    _durable_json(root/'completed.json',result)
    print({k:v for k,v in result.items() if k!='artifact_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
