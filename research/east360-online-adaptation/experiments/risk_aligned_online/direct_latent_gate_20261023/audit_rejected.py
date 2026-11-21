"""Verify the stage-one rejection; not an audit of an executed online stream."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_dependence_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import fit_predictor,origins
from .run import extract


def audit(root):
    read=lambda p:json.loads(p.read_text())
    p=read(root/'protocol.json');done=read(root/'completed.json');ref=Path(p['preparation_reference'])
    assert done['status']=='COMPLETED' and not done['forecast_gate']['passes']
    assert done['risk_gate'] is None and done['actual_probe_updates']==0 and not done['recommend_full_stream']
    assert not done['test_accessed'] and read(root/'sealed_probes.json')==read(root/'scored_probes.json')==[]
    assert sha(root/'sealed_probes.json')==done['sealed_probes_sha256']
    assert sha(root/'predictor.pt')==done['predictor_sha256']
    assert sha(Path(p['plan_path']))==p['plan_sha256'] and sha(DEFAULT_PARENT)==PARENT_SHA256
    for n,h in p['code_sha256'].items():assert sha(io23.ROOT/n)==h,n
    for n,h in p['reference_hashes'].items():assert sha(ref/n)==h,n
    fit,held=set(p['fit']),set(p['held'])
    assert len(fit)==15 and len(held)==5 and not fit&held and fit|held==set(p['selected'])
    assert not (fit|held)&set(read(ref/'protocol.json')['contract']['shot_ids'])
    cache={};seals=read(root/'cache_receipts.json');assert [r['shot'] for r in seals]==p['selected']
    for r in seals:
        path=root/'cache'/f'{r["shot"]}.npz';assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:cache[r['shot']]={k:z[k] for k in z.files}
    args=[np.concatenate([cache[s][k] for s in p['fit']]) for k in ('current','previous','future')]
    rebuilt=fit_predictor(*args);stored=torch.load(root/'predictor.pt',map_location='cpu',weights_only=True)['predictor']
    for k,v in rebuilt.state_dict().items():torch.testing.assert_close(v,stored[k],rtol=0,atol=0)
    mean=stored['mean'].double().numpy();scale=stored['scale'].double().numpy();coef=stored['coefficient'].double().numpy()
    errors={};max_mse_difference=0.
    for sid,c in cache.items():
        h,prev,t=[c[k].astype(float) for k in ('current','previous','future')]
        x=np.column_stack(((np.concatenate((h,h-prev),axis=1)-mean)/scale,np.ones(len(h))))
        pred=h+x@coef
        a=float(((h-t)**2).mean());b=float(((pred-t)**2).mean())
        reported=done['forecast_gate']['per_shot'][str(sid)]
        np.testing.assert_allclose([a,b],[reported['persistence_mse'],reported['learned_mse']],rtol=2e-5,atol=2e-7)
        max_mse_difference=max(max_mse_difference,abs(a-reported['persistence_mse']),abs(b-reported['learned_mse']))
        errors[sid]=dict(n=len(h),persistence_mse=a,learned_mse=b)
    a=float(np.mean([errors[s]['persistence_mse'] for s in p['held']]))
    b=float(np.mean([errors[s]['learned_mse'] for s in p['held']]))
    assert b>=a and a>0
    torch.set_num_threads(2);torch.manual_seed(131);model,_=load_model(ref);before=state_cpu(model)
    mapping=read(root/'input_map.json');replays=[]
    for sid in (p['fit'][0],p['held'][0],p['held'][-1]):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},mapping)
        actual,excluded=extract(model,signal)
        for k,v in actual.items():np.testing.assert_array_equal(v,cache[sid][k])
        assert excluded==read(root/'excluded_origins.json')[str(sid)]
        assert set(actual['ends'])<=set(origins(len(s['values'])))
        replays.append(dict(shot=sid,examples=len(actual['ends']),states='previous/current/future',all_exact=True))
    assert_unchanged(model,before)
    result=dict(status='VERIFIED_STAGE_ONE_REJECTION',protocol_sha256=sha(root/'protocol.json'),
        completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        refit_predictor_bitwise_exact=True,numpy_mse_max_abs_difference=max_mse_difference,
        held_mse_ratio=b/a,fit_examples=sum(errors[s]['n'] for s in fit),held_examples=sum(errors[s]['n'] for s in held),
        sampled_cuda_cache_replays=replays,source_unchanged=True,online_updates_executed=0,
        independent_confirmation=False,physical_causality_identified=False)
    _durable_json(root/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
