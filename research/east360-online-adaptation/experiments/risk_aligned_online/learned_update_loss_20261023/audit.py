"""Independent BCE recount; all held probes and all twelve fits replayed on CUDA."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import UpdateLoss,context_features
from .run import ALL,LEARNED,PLAN,tensor_window,logit,schedule,train,evaluate


def bce(z,y):return float(np.logaddexp(0.,float(z))-float(y)*float(z))


def main(root):
    read=lambda p:json.loads(p.read_text())
    protocol,done=read(root/'protocol.json'),read(root/'completed.json')
    assert sha(root/'protocol.json')==done['protocol_sha256']
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in protocol['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(PLAN)==protocol['plan_sha256']
    ref=Path(protocol['reference'])
    for name,digest in protocol['reference_hashes'].items():assert sha(ref/name)==digest
    assert sha(ref/'conditional_moments.pt')==protocol['parent_sha256']
    assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']
    samples=read(root/'sampling.json');plans=read(root/'folds.json')
    held,fit=read(root/'held_probes.json'),read(root/'fit_probes.json')
    assert len(samples)==640 and len(held)==3200 and len(fit)==5760
    assert len({(r['shot'],r['query']) for r in samples})==640
    all_ids=set(protocol['selected_ids']);held_sets=[]
    for plan in plans:
        assert len(plan['fit'])==15 and len(plan['held'])==5
        assert set(plan['fit']).isdisjoint(plan['held']) and set(plan['fit'])|set(plan['held'])==all_ids
        assert plan['schedule']==schedule(samples,plan['fit'],plan['fold'])
        held_sets.extend(plan['held'])
    assert sorted(held_sets)==sorted(all_ids)
    for r in held+fit:
        plan=plans[r['fold']]
        assert r['shot'] in plan[r['role']] and r['query']==r['count']+24
        np.testing.assert_allclose((r['query_time']-r['observed_time'])*1000,25.,atol=1e-8)
        for when in ('before','after'):
            np.testing.assert_allclose(r[when],bce(r['logit_'+when],r['label']),rtol=1e-4,atol=7e-7)
        u=r['update'];norm=u['parameter_delta_norm'] if r['arm']=='forecast' else u['delta_norm']
        assert norm<=.00101 and u['gradient_steps']==1
    recounted={}
    for arm in ALL:
        groups={};good=0
        for group in ('safe','disruptive_timely','disruptive_early'):
            part=[r for r in held if r['arm']==arm and r['group']==group]
            per_shot=[]
            for sid in sorted({r['shot'] for r in part}):
                rs=[r for r in part if r['shot']==sid]
                rec={when:float(np.mean([bce(r['logit_'+when],r['label']) for r in rs])) for when in ('before','after')}
                per_shot.append(rec)
                if group=='disruptive_timely':good+=rec['after']<rec['before']
            groups[group]={when:float(np.mean([r[when] for r in per_shot])) for when in ('before','after')}
            for when in ('before','after'):
                np.testing.assert_allclose(groups[group][when],done['summary'][arm]['groups'][group][when],rtol=1e-4,atol=7e-7)
        main=[groups[g] for g in ('safe','disruptive_timely')]
        passed=bool(good>=3 and all(r['after']<=r['before'] for r in main) and sum(r['after'] for r in main)<sum(r['before'] for r in main))
        assert passed==done['summary'][arm]['passes_screen']
        recounted[arm]=dict(groups=groups,passes_screen=passed,improved_disruptive_shots=int(good))
    torch.set_num_threads(2);model,_=load_model();frozen=state_cpu(model)
    torch.manual_seed(131);rule0=UpdateLoss(model.cfg.dim).cuda()
    items=[];signals={};mapping=read(ref/'input_map.json');replayed=0;refit=0
    history=read(root/'training.json');original_read=simple_data.read_shot
    def allowed(device,sid,**kwargs):
        if device!='EAST' or sid not in all_ids:raise AssertionError('test signal forbidden')
        return original_read(device,sid,**kwargs)
    with patch.object(io23,'read_outcome',side_effect=AssertionError('audit uses sealed historical labels only')),patch.object(simple_data,'read_shot',side_effect=allowed):
        for sid in protocol['selected_ids']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping);signals[sid]=signal
            for r in [r for r in samples if r['shot']==sid]:
                x,m=tensor_window(signal,r['count']-1);qx,qm=tensor_window(signal,r['query'])
                item=dict(r,x=x,m=m,qx=qx,qm=qm,context=context_features(x,m,model.base.center,model.base.scale))
                assert float(logit(model,item,initial_encoder(model)).detach())==r['logit_before']
                assert s['times'][r['query']]==r['query_time'] and s['times'][r['count']-1]==r['observed_time']
                items.append(item)
        for plan in plans:
            f=plan['fold'];part=[r for r in items if r['shot'] in plan['held']]
            for arm in ALL:
                if arm in LEARNED:
                    rule=copy.deepcopy(rule0)
                    trace=train(model,rule,items,plan['schedule'],arm,plan['fit'])
                    assert trace==history[f'{f}_{arm}']
                    saved=torch.load(root/f'fold{f}_{arm}.pt',map_location='cuda',weights_only=True)
                    assert saved['arm']==arm and saved['fold']==f
                    for k,v in rule.state_dict().items():torch.testing.assert_close(v,saved['rule'][k],rtol=0,atol=0)
                    refit+=1
                else:rule=None
                actual=evaluate(model,rule,part,arm,f,'held',signals)
                expected=[r for r in held if r['fold']==f and r['arm']==arm]
                assert actual==expected
                replayed+=len(actual)
            print(dict(fold=f,held_replayed=replayed,loss_heads_refitted=refit),flush=True)
    assert_unchanged(model,frozen)
    assert len(read(root/'label_access.json'))==20 and {r['shot'] for r in read(root/'label_access.json')}==all_ids
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        recounted_rows=len(held)+len(fit),cuda_replayed_held_rows=replayed,exact_loss_head_refits=refit,
        recounted_summary=recounted,parent_unchanged=True,no_test_access=True,
        extra_meta_tests_sha256=sha(io23.ROOT/'experiments/risk_aligned_online/tests/test_learned_update_loss_meta.py'))
    _durable_json(root/'verification.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
