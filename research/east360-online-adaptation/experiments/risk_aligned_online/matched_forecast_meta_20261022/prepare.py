"""Train only the original direct-forecast head, with matched causal burn-in."""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..paired_feedback_20261022.model import PairedFeedback,initial_encoder
from ..redesign_20261004.prepare_prior import load_parent,load_source_train,DEFAULT_PARENT,PARENT_SHA256,assert_classifier_unchanged,classifier_state
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.run import snapshot
from ..deepcoral_prefix_20261022.learning import adapt_arrived
from .learning import episode,episode_pools,draw_episode,observed_step,INNER_LR

ARMS=('forecast','meta')
STEPS=256


def summarize(records):
    def part(rs):
        return dict(n=len(rs),risk_before=float(np.mean([r['risk_before'] for r in rs])),
            risk_after=float(np.mean([r['risk_after'] for r in rs])),
            delta=float(np.mean([r['risk_after']-r['risk_before'] for r in rs])),
            helpful=sum(r['risk_after']<r['risk_before'] for r in rs),
            forecast=float(np.mean([r['forecast'] for r in rs])))
    return dict(all=part(records),positive=part([r for r in records if r['label']==1]),
                negative=part([r for r in records if r['label']==0]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    base,parent=load_parent(DEFAULT_PARENT);frozen=classifier_state(base)
    template=PairedFeedback(base,np.ones(23)).cuda();template.set_stage('frozen')
    models={k:copy.deepcopy(template) for k in ARMS}
    for m in models.values():m.base.forecast_head.requires_grad_(True)
    optimizers={k:torch.optim.Adam(m.base.forecast_head.parameters(),lr=1e-4) for k,m in models.items()}
    args.output.mkdir(parents=True)
    hashes=snapshot(args.output)
    dependencies=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parents[1]/'deepcoral_prefix_20261022/learning.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        for f in dependencies:z.write(f,str(f.relative_to(io23.ROOT)));hashes[str(f.relative_to(io23.ROOT))]=sha(f)
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/反馈目标匹配部署_事前约定_20261022.md'
    source,ids=load_source_train()
    train=[s for s in source if s['shot']%5!=0];report=[s for s in source if s['shot']%5==0]
    pools,ledger=episode_pools(train,[s['shot'] for s in train])
    rpools,rledger=episode_pools(report,[s['shot'] for s in report])
    protocol=dict(source_sha256=PARENT_SHA256,source_checkpoint=str(DEFAULT_PARENT),threshold=float(parent['threshold']),
        code_sha256=hashes,plan_path=str(plan),plan_sha256=sha(plan),steps=STEPS,batch=2,seed=131,
        learning_rate=1e-4,inner_learning_rate=INNER_LR,tbptt_last_updates=2,
        arms=ARMS,source_train_ids=ids,head_fit_ids=[s['shot'] for s in train],
        head_report_ids=[s['shot'] for s in report],report_is_independent=False,
        source_classifier_trained_on_report_ids=True,target_data_accessed=False,
        validation_test_outcomes_accessed=False,
        objectives=dict(forecast='D',meta='D+(R_after-stopgrad(R_static))/(K*inner_lr)'),
        no_checkpoint_selection=True,only_forecast_head_persistently_trainable=True)
    _durable_json(args.output/'protocol.json',protocol)
    _durable_json(args.output/'eligibility.json',dict(fit=ledger,report=rledger))
    rng=np.random.default_rng(131);history=[];operator_checked=False
    with (args.output/'training_manifest.jsonl').open('x') as manifest:
        for step in range(STEPS):
            examples=[draw_episode(train,pools,rng,label) for label in (0,1)]
            manifest.write(json.dumps(dict(step=step,episodes=[r for _,r in examples]))+'\n');manifest.flush()
            row=dict(step=step+1)
            for arm,model in models.items():
                opt=optimizers[arm];opt.zero_grad(set_to_none=True);records=[]
                for shot,receipt in examples:
                    signal={k:shot[k] for k in ('values','available')}
                    if not operator_checked:
                        fast=initial_encoder(model)
                        a,_=observed_step(model,signal,256,fast,device='cuda',create_graph=False)
                        b,_=adapt_arrived(model,signal,256,fast,device='cuda')
                        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
                        operator_checked=True
                    r=episode(model,signal,receipt['query'],float(receipt['label']),device='cuda',create_graph=True)
                    delta=r['risk_after']-r['risk_before']
                    objective=r['forecast']+(delta/(INNER_LR*len(r['differentiated_counts'])) if arm=='meta' else 0.)
                    if not torch.isfinite(objective):raise FloatingPointError('nonfinite auxiliary meta objective')
                    (objective/len(examples)).backward()
                    records.append(dict(**receipt,is_disrupt=shot['is_disrupt'],
                        objective=float(objective.detach()),risk_before=float(r['risk_before']),
                        risk_after=float(r['risk_after'].detach()),forecast=float(r['forecast'].detach()),
                        burn_in_updates=len(r['all_update_counts'])-len(r['differentiated_counts'])))
                    del r,objective,delta
                params=list(model.base.forecast_head.parameters())
                if any(p.grad is None for p in params):raise AssertionError('head gradient missing')
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('base.forecast_head.'))
                row[arm]=dict(records=records,head_gradient_norm=float(norm))
            history.append(row)
            if (step+1)%16==0:
                _durable_json(args.output/'training.json',history)
                print(json.dumps(dict(stage='train',step=step+1,seconds=time.monotonic()-start,
                    loss={k:float(np.mean([v['objective'] for v in row[k]['records']])) for k in ARMS})),flush=True)
    for arm,m in models.items():
        assert_classifier_unchanged(m.base,frozen)
        torch.save(dict(forecast_head={k:v.detach().cpu() for k,v in m.base.forecast_head.state_dict().items()},
                        parent_sha256=PARENT_SHA256),args.output/f'{arm}.pt')
        m.set_stage('frozen')
    # Fixed report episodes; both heads are FINAL-step models. No epoch choice.
    rrng=np.random.default_rng(7131)
    selected=[draw_episode(report,rpools,rrng,label) for _ in range(64) for label in (0,1)]
    reported={k:[] for k in ARMS};reported['original']=[]
    for i,(shot,receipt) in enumerate(selected):
        signal={k:shot[k] for k in ('values','available')}
        for arm,m in dict(original=template,**models).items():
            r=episode(m,signal,receipt['query'],float(receipt['label']),device='cuda',create_graph=False)
            reported[arm].append(dict(**receipt,is_disrupt=shot['is_disrupt'],
                risk_before=float(r['risk_before']),risk_after=float(r['risk_after'].detach()),
                forecast=float(r['forecast'].detach())))
        if (i+1)%32==0:print(json.dumps(dict(stage='source_report',episodes=i+1)),flush=True)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'source_report.json',dict(scope='source TRAIN held out only from new head fitting; not independent',
        records=reported,summary={k:summarize(v) for k,v in reported.items()}))
    _durable_json(args.output/'prepare_completed.json',dict(status='COMPLETED',seconds=time.monotonic()-start,
        static_classifier_bitwise_unchanged=True,cuda_step_matches_deployment=True,
        training_manifest_sha256=sha(args.output/'training_manifest.jsonl'),
        head_sha256={k:sha(args.output/f'{k}.pt') for k in ARMS}))
    print(json.dumps(dict(prepared=True,summary={k:summarize(v) for k,v in reported.items()})),flush=True)


if __name__=='__main__':main()
