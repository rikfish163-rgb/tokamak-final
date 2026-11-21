"""Frozen-classifier target-supervised feedback experiment; local CUDA only."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import snapshot,public
from .learning import episode_pool,block_episode
from .stream import predict_five

REFERENCE=Path(__file__).parents[1]/'runs/labeled_alignment20_test360_20261023'
ARMS=('warm','forecast','point_meta','maximum_meta')
STEPS=64
HOLDOUT=(62557,62583,62630,62945,70868)


def prepare(warm,shots,output,threshold):
    fit=[s for s in shots if s['shot'] not in HOLDOUT]
    pools=episode_pool(fit,[s['shot'] for s in fit])
    groups={d:[r for r in pools if r['is_disrupt']==d and r['blocks']] for d in (False,True)}
    assert len(fit)==15 and len(groups[True])==3 and len(groups[False])==12
    _durable_json(output/'eligibility.json',pools)
    byid={s['shot']:s for s in fit}
    models={k:copy.deepcopy(warm) for k in ARMS};before=state_cpu(warm)
    opts={}
    for arm in ARMS[1:]:
        models[arm].base.forecast_head.requires_grad_(True)
        opts[arm]=torch.optim.Adam(models[arm].base.forecast_head.parameters(),lr=1e-4)
    rng=np.random.default_rng(131);history=[];started=time.monotonic()
    boundary=math.log(threshold/(1-threshold))
    with (output/'training_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            episodes=[]
            for d in (False,True):
                r=groups[d][int(rng.integers(len(groups[d])))];b=r['blocks'][int(rng.integers(len(r['blocks'])))]
                episodes.append(dict(shot=r['shot'],is_disrupt=d,**b))
            manifest.write(json.dumps(dict(step=step,episodes=episodes))+'\n');manifest.flush()
            row=dict(step=step)
            for arm in ARMS[1:]:
                model=models[arm];opt=opts[arm];opt.zero_grad(set_to_none=True);rs=[]
                for e in episodes:
                    s=byid[e['shot']];signal={k:s[k] for k in ('values','available')}
                    r=block_episode(model,signal,e['count'],e['ends'],e['is_disrupt'],boundary=boundary,
                        objective='maximum' if arm=='maximum_meta' else 'point',device='cuda',create_graph=True)
                    delta=r['risk_after']-r['risk_before']
                    objective=r['forecast']+(delta/(.001*len(r['differentiated_counts'])) if arm!='forecast' else 0.)
                    if not torch.isfinite(objective):raise FloatingPointError('nonfinite meta objective')
                    (objective/2).backward()
                    rs.append(dict(**e,objective=float(objective.detach()),risk_before=float(r['risk_before']),
                        risk_after=float(r['risk_after'].detach()),forecast=float(r['forecast'].detach()),
                        update_counts=r['all_update_counts'],differentiated_counts=r['differentiated_counts']))
                    del r,delta,objective
                assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('base.forecast_head.'))
                norm=torch.nn.utils.clip_grad_norm_(model.base.forecast_head.parameters(),1.,error_if_nonfinite=True)
                opt.step();assert_unchanged(model,before,('base.forecast_head.weight','base.forecast_head.bias'))
                row[arm]=dict(records=rs,head_gradient_norm=float(norm))
            history.append(row)
            if (step+1)%8==0:
                _durable_json(output/'training.json',history)
                print(json.dumps(dict(stage='target_feedback_training',step=step+1,seconds=time.monotonic()-started)),flush=True)
    receipts={}
    for arm,model in models.items():
        model.set_stage('frozen')
        assert_unchanged(model,before,('base.forecast_head.weight','base.forecast_head.bias') if arm!='warm' else ())
        torch.save(dict(model=state_cpu(model),config=model.cfg.to_dict()),output/f'{arm}.pt')
        receipts[arm]=dict(sha256=sha(output/f'{arm}.pt'),classifier_bitwise_unchanged=True)
    _durable_json(output/'preparation_receipts.json',receipts)
    return models


def scored_row(shot,score):
    valid=shot['valid']
    return dict(shot=shot['shot'],times=shot['times'][valid],scores=score[valid],
                is_disrupt=shot['is_disrupt'],down=shot['down'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    if root.exists():raise FileExistsError(root)
    if not torch.cuda.is_available():raise RuntimeError('rl_env local CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    old=json.loads((REFERENCE/'protocol.json').read_text());selection=json.loads((REFERENCE/'selection.json').read_text())['selected']
    verified=json.loads((REFERENCE/'verification.json').read_text());assert verified['status']=='VERIFIED'
    assert sha(REFERENCE/'completed.json')==verified['completed_sha256']
    assert set(HOLDOUT)<set(old['selected_ids'])
    base,parent_info=load_parent(DEFAULT_PARENT);threshold=float(parent_info['threshold'])
    warm=PairedFeedback(base,np.ones(23)).cuda()
    ck=REFERENCE/'input_only.pt';receipt=json.loads((REFERENCE/'preparation_receipts.json').read_text())['input_only']
    assert sha(ck)==receipt['checkpoint_sha256']
    warm.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model']);warm.set_stage('frozen')
    mapping=json.loads((REFERENCE/'input_map.json').read_text());contract=load_test_contract('EAST')
    assert not set(old['selected_ids'])&set(contract['shot_ids'])
    root.mkdir(parents=True);(root/'shots').mkdir();(root/'holdout').mkdir()
    hashes=snapshot(root)
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/n for n in (
        'matched_forecast_meta_20261022/learning.py','deepcoral_prefix_20261022/learning.py',
        'deepcoral_prefix_20261022/run.py','tests/test_target_feedback_meta.py')]
    with zipfile.ZipFile(root/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as archive:
        for f in files:
            name=str(f.relative_to(io23.ROOT));archive.write(f,name);hashes[name]=sha(f)
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/EAST标签教炮内反馈_事前约定_20261023.md'
    p=dict(source_sha256=PARENT_SHA256,warm_checkpoint=str(ck),warm_sha256=sha(ck),
        reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        code_sha256=hashes,plan_path=str(plan),plan_sha256=sha(plan),contract=contract,
        threshold=threshold,arms=ARMS,steps=STEPS,learning_rate=1e-4,inner_learning_rate=1e-3,
        selected_ids=old['selected_ids'],fit_ids=[r['shot'] for r in selection if r['shot'] not in HOLDOUT],
        holdout_ids=HOLDOUT,head_warmup_and_input_mapping_used_all20=True,
        threshold_selection=False,checkpoint_selection=False,independent_confirmation=False,
        causal_scope='retrospective historical preparation; per-shot temporary causal encoder adaptation',
        persistent_optimized_parameters=['base.forecast_head.weight','base.forecast_head.bias'],
        outer_objectives=['forecast only','point future margin loss','next-block maximum margin loss'],
        not_whole_shot_maximum=True,no_test_outcomes_before_all_seals=True)
    _durable_json(root/'protocol.json',p);_durable_json(root/'input_map.json',mapping)
    allowed=set(old['selected_ids']);access=[];read_outcome=io23.read_outcome
    def guarded(device,sid):
        if device!='EAST' or sid not in allowed:raise AssertionError('unauthorized label access before sealing')
        access.append(int(sid));return read_outcome(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guarded):
        shots=[]
        for sid in old['selected_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            s=simple_data.attach_outcome(s,'EAST',{})
            shots.append(dict(s,**transform({k:s[k] for k in ('values','available')},mapping)))
        models=prepare(warm,shots,root,threshold);frozen={k:state_cpu(m) for k,m in models.items()}
        held={}
        for s in shots:
            if s['shot'] not in HOLDOUT:continue
            preds,updates=predict_five(models,{k:s[k] for k in ('values','available')})
            np.savez_compressed(root/'holdout'/f"{s['shot']}.npz",times=s['times'],**preds)
            _durable_json(root/'holdout'/f"{s['shot']}_updates.json",updates)
            for name,v in preds.items():held.setdefault(name,[]).append(scored_row(s,v))
        _durable_json(root/'holdout_metrics.json',{k:metrics(v,threshold,lead=30.) for k,v in held.items()})
        seals=[]
        for index,sid in enumerate(contract['shot_ids']):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            preds,updates=predict_five(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(REFERENCE/'shots'/f'{sid}.npz',allow_pickle=False) as ref:
                np.testing.assert_array_equal(preds['static'],ref['input_only_static'])
                np.testing.assert_array_equal(preds['warm_intra'],ref['input_only_intra'])
            path=root/'shots'/f'{sid}.npz';u=root/'shots'/f'{sid}_updates.json'
            np.savez_compressed(path,times=s['times'],**preds);_durable_json(u,updates)
            seals.append(dict(shot=sid,prediction_sha256=sha(path),updates_sha256=sha(u)))
            if (index+1)%40==0:
                progress=dict(stage='stream',shots=index+1,seconds=time.monotonic()-started)
                _durable_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        _durable_json(root/'prediction_receipts.json',seals);_durable_json(root/'pre_eval_label_access.json',access)
    rows={}
    for sid in contract['shot_ids']:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],s['times'])
            for name in ['static']+[f'{k}_intra' for k in ARMS]:rows.setdefault(name,[]).append(scored_row(s,trace[name]))
    results={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    for k,m in models.items():assert_unchanged(m,frozen[k])
    assert sha(DEFAULT_PARENT)==PARENT_SHA256 and sha(ck)==p['warm_sha256']
    _durable_json(root/'per_shot.json',results)
    summary={k:dict(**public(v),very_early_over3000=sum(r['timely'] and r['lead_ms']>3000 for r in v['predictions'])) for k,v in results.items()}
    _durable_json(root/'completed.json',dict(status='COMPLETED',metrics=summary,seconds=time.monotonic()-started,
        unchanged_static=True,unchanged_warm_intra=True,source_unchanged=True,temporary_resets=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
