"""TRAIN-only two-stage screen; never opens TEST signals or outcomes."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..temporal_dependence_20261023.run import load_model,REFERENCE
from ..warmstart_alignment_20261022.alignment import transform
from .learning import DirectLatent,fit_predictor,origins,encode_at,arrived_step,risk_summary

HELD=(62557,62583,62630,62945,70868)
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/直接未来表示反馈_小样本筛查约定_20261023.md'


def extract(model,signal):
    ends=origins(len(signal['values']))
    kept=[e for e in ends if (signal['available'][e+1:e+26] & np.isfinite(signal['values'][e+1:e+26])).any()]
    if not kept:raise ValueError('no observed future-state examples in historical shot')
    result={k:[] for k in ('previous','current','future')}
    for lo in range(0,len(kept),32):
        for name,delta in (('previous',-25),('current',0),('future',25)):
            h=encode_at(model,signal,[e+delta for e in kept[lo:lo+32]])
            result[name].append(h.detach().cpu().numpy())
    return dict(ends=np.array(kept),**{k:np.concatenate(v) for k,v in result.items()}),[e for e in ends if e not in kept]


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    read=lambda p:json.loads(p.read_text())
    mapping=read(REFERENCE/'input_map.json');selected=sorted(mapping['calibration_ids'])
    fit=[sid for sid in selected if sid not in HELD]
    assert len(selected)==20 and len(fit)==15 and set(HELD)<=set(selected)
    refprotocol=read(REFERENCE/'protocol.json')
    assert not set(selected)&set(refprotocol['contract']['shot_ids'])
    assert sha(REFERENCE/'completed.json')==read(REFERENCE/'verification.json')['completed_sha256']
    model,threshold=load_model();frozen=state_cpu(model)
    output.mkdir(parents=True);(output/'cache').mkdir()
    # Freeze dependency files by carrying the checked reference's full source
    # snapshot manifest, plus this explicitly separate small screening module.
    files=[io23.ROOT/n for n in refprotocol['code_sha256']]
    files += list(Path(__file__).parent.glob('*.py'))+[io23.ROOT/'experiments/risk_aligned_online/tests/test_direct_latent_gate.py',
        io23.ROOT/'experiments/risk_aligned_online/temporal_dependence_20261023/run.py']
    hashes={str(f.relative_to(io23.ROOT)):sha(f) for f in files}
    for n,h in refprotocol['code_sha256'].items():assert hashes[n]==h,n
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in hashes:z.write(io23.ROOT/n,n)
    protocol=dict(scope='HISTORICAL TRAIN SCREEN, NOT TEST360 PERFORMANCE',selected=selected,fit=fit,held=list(HELD),
        preparation_reference=str(REFERENCE),reference_hashes={n:sha(REFERENCE/n) for n in (
            'input_only.pt','input_map.json','protocol.json','completed.json','verification.json')},
        threshold=threshold,code_sha256=hashes,plan_path=str(PLAN),plan_sha256=sha(PLAN),
        horizon=25,later_query=50,window=128,origins_per_shot_max=128,ridge_penalty=.001,
        risk_state_dim=64,fast_parameters=model.encoder_names(),learning_rate=.001,gradient_clip=1.,
        forecast_gate='equal-shot held5 MSE strictly lower than persistence',
        direction_gate='positive and negative BCE nonworsening, balanced BCE strictly lower',
        source_risk_unchanged=True,test_signals_accessed=False,test_labels_accessed=False,
        held_caveat='held only from predictor fitting; used in prior mapping/method development',
        event_labels_used=False,physical_truth_target=False)
    _durable_json(output/'protocol.json',protocol);_durable_json(output/'input_map.json',mapping)
    signals={};cache={};receipts=[];excluded={}
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no outcomes before all probe probabilities sealed')):
        for sid in selected:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signals[sid]=transform({k:s[k] for k in ('values','available')},mapping)
            cache[sid],excluded[sid]=extract(model,signals[sid])
            path=output/'cache'/f'{sid}.npz';np.savez_compressed(path,**cache[sid])
            receipts.append(dict(shot=sid,sha256=sha(path)))
        _durable_json(output/'cache_receipts.json',receipts);_durable_json(output/'excluded_origins.json',excluded)
        args=[np.concatenate([cache[sid][name] for sid in fit]) for name in ('current','previous','future')]
        predictor=fit_predictor(*args).cuda();frozen_predictor=state_cpu(predictor)
        torch.save(dict(predictor=frozen_predictor),output/'predictor.pt')
        errors={}
        for sid,c in cache.items():
            h,p,t=[torch.as_tensor(c[k],device='cuda') for k in ('current','previous','future')]
            predicted=predictor(h,p)
            errors[sid]=dict(n=len(h),persistence_mse=float((h-t).square().mean()),
                learned_mse=float((predicted-t).square().mean()),
                target_innovation_energy=float((t-h).square().mean()))
        persistence=float(np.mean([errors[s]['persistence_mse'] for s in HELD]))
        learned=float(np.mean([errors[s]['learned_mse'] for s in HELD]))
        gate=learned<persistence
        forecast=dict(per_shot=errors,held_equal_shot_persistence=persistence,held_equal_shot_learned=learned,
                      relative_improvement=1-learned/persistence if persistence>0 else None,passes=gate)
        _durable_json(output/'forecast_gate.json',forecast)
        print(json.dumps(dict(stage='forecast_gate',passed=gate,persistence=persistence,learned=learned)),flush=True)
        records=[]
        if gate:
            identity=DirectLatent(np.zeros(128),np.ones(128),np.zeros((129,64))).cuda()
            for sid in HELD:
                for e in cache[sid]['ends'].tolist():
                    q=e+50;signal=signals[sid]
                    h=encode_at(model,signal,[q]);logits=dict(static=float(model.base.risk_head(h)[0,0].detach()))
                    updates={}
                    for arm,head in (('learned',predictor),('persistence',identity)):
                        fast,r=arrived_step(model,head,signal,e)
                        h=encode_at(model,signal,[q],fast=fast)
                        logits[arm]=float(model.base.risk_head(h)[0,0].detach());updates[arm]=r
                    records.append(dict(shot=sid,origin=e,query_end=q,logits=logits,updates=updates))
                print(json.dumps(dict(stage='one_step_probes',shot=sid,probes=len(records))),flush=True)
        _durable_json(output/'sealed_probes.json',records)
    scored=[];outcomes={}
    if gate:
        for sid in HELD:
            s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
            outcomes[sid]=s
        for r in records:
            s=outcomes[r['shot']];q=r['query_end']
            lead=round(float((s['down']-s['times'][q])*1000),9) if s['is_disrupt'] else None
            valid=bool(s['valid'][q] and (lead is None or lead>30))
            scored.append(dict(r,valid=valid,label=int(s['y'][q]),lead_ms=lead))
    risk=risk_summary([r for r in scored if r['valid']]) if gate else None
    _durable_json(output/'scored_probes.json',scored)
    assert_unchanged(model,frozen);assert_unchanged(predictor,frozen_predictor)
    complete=dict(status='COMPLETED',seconds=time.monotonic()-start,source_risk_unchanged=True,
        forecast_gate=forecast,risk_gate=risk,recommend_full_stream=bool(gate and risk['passes_direction_gate']),
        actual_probe_updates=sum(u['gradient_steps'] for r in records for u in r['updates'].values()),
        test_accessed=False,predictor_sha256=sha(output/'predictor.pt'),sealed_probes_sha256=sha(output/'sealed_probes.json'))
    _durable_json(output/'completed.json',complete);print(json.dumps(complete),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
