"""Four-fold loss-head screen on the same twenty historical preparation shots."""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..arrived_teacher_20261023.learning import sample_queries,step as forecast_step
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import REFERENCE,READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import UpdateLoss,context_features,adapt_window,folds,summarize

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/可学习炮内损失_事前约定_20261023.md'
LEARNED=('learned','no_context','shuffled')
ALL=LEARNED+('entropy','forecast')


def tensor_window(signal,end,device='cuda'):
    return [torch.as_tensor(v[None],device=device) for v in simple_data.window(signal,end,128)]


def logit(model,item,fast):
    h=model.predict_fast(item['qx'],item['qm'],fast,mode='risk_state')['risk_state']
    return model.base.risk_head(h).reshape(())


def prepare(model,ids,mapping):
    records=[];items=[];signals={}
    for sid in ids:
        s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
        signal=transform({k:s[k] for k in ('values','available')},mapping);signals[sid]=signal
        records.append(dict(shot=sid,is_disrupt=bool(s['is_disrupt'])))
        for row in sample_queries(s):
            x,m=tensor_window(signal,row['count']-1);qx,qm=tensor_window(signal,row['query'])
            item=dict(row,shot=sid,label=int(s['y'][row['query']]),x=x,m=m,qx=qx,qm=qm,
                      context=context_features(x,m,model.base.center,model.base.scale),
                      query_time=float(s['times'][row['query']]),observed_time=float(s['times'][row['count']-1]))
            z=logit(model,item,initial_encoder(model)).detach()
            item.update(logit_before=float(z),before=float(F.binary_cross_entropy_with_logits(z,z.new_tensor(float(item['label'])))))
            items.append(item)
    return records,items,signals


def schedule(items,fit,fold):
    rng=np.random.default_rng(131+fold)
    eligible=[(j,r) for j,r in enumerate(items) if r['shot'] in fit and r['group']!='disruptive_early']
    pools={g:{sid:[j for j,r in eligible if r['shot']==sid and r['group']==g]
              for sid in sorted({r['shot'] for _,r in eligible if r['group']==g})}
           for g in ('safe','disruptive_timely')}
    swaps=np.array([False]*96+[True]*96);rng.shuffle(swaps)
    result=[]
    for index in range(192):
        pair=[]
        for g in ('safe','disruptive_timely'):
            sid=int(rng.choice(list(pools[g])));pair.append(int(rng.choice(pools[g][sid])))
        result.append(dict(indices=pair,swap=bool(swaps[index])))
    return result


def train(model,rule,items,steps,arm,fit):
    opt=torch.optim.Adam(rule.parameters(),lr=.003);history=[]
    for index,entry in enumerate(steps):
        opt.zero_grad(set_to_none=True);losses=[]
        for j in entry['indices']:
            item=items[j]
            if item['shot'] not in fit:raise AssertionError('held identity in loss-head training')
            fast,receipt=adapt_window(model,rule,item['x'],item['m'],item['context'],create_graph=True,
                                      mode='no_context' if arm=='no_context' else 'observed')
            z=logit(model,item,fast)
            y=1-item['label'] if arm=='shuffled' and entry['swap'] else item['label']
            losses.append(F.binary_cross_entropy_with_logits(z,z.new_tensor(float(y))))
        outer=torch.stack(losses).mean();outer.backward()
        grads=[p.grad for p in rule.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('invalid meta-gradient')
        norm=float(torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads])))
        history.append(dict(step=index+1,outer=float(outer.detach()),meta_gradient_norm=norm,
                            inner_delta_last=receipt['delta_norm']))
        opt.step()
    return history


def evaluate(model,rule,items,arm,fold,role,signals=None):
    rows=[]
    for item in items:
        if arm=='forecast':
            fast,receipt,_=forecast_step(model,signals[item['shot']],item['count'],'forecast')
        else:
            fast,receipt=adapt_window(model,rule,item['x'],item['m'],item['context'],
                mode='no_context' if arm=='no_context' else 'entropy' if arm=='entropy' else 'observed')
        z=logit(model,item,fast).detach()
        row={k:v for k,v in item.items() if not isinstance(v,torch.Tensor)}
        row.update(arm=arm,fold=fold,role=role,logit_after=float(z),
                   after=float(F.binary_cross_entropy_with_logits(z,z.new_tensor(float(item['label'])))),update=receipt)
        rows.append(row)
    return rows


def main(output):
    if output.exists():raise FileExistsError(output)
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    rp,rv=READ(REFERENCE/'protocol.json'),READ(REFERENCE/'verification.json')
    assert rv['status']=='VERIFIED' and sha(REFERENCE/'completed.json')==rv['completed_sha256']
    for f,digest in rp['code_sha256'].items():assert sha(io23.ROOT/f)==digest,f
    ids=rp['selected_ids'];assert len(ids)==20 and not set(ids)&set(rp['contract']['shot_ids'])
    parent=REFERENCE/'conditional_moments.pt'
    assert sha(parent)==READ(REFERENCE/'preparation_receipts.json')['conditional_moments']['checkpoint_sha256']
    model,threshold=load_model();frozen=state_cpu(model);mapping=READ(REFERENCE/'input_map.json')
    torch.manual_seed(131);rule0=UpdateLoss(model.cfg.dim).cuda()
    output.mkdir(parents=True)
    paths={io23.ROOT/f for f in rp['code_sha256']};paths.update(Path(__file__).parent.glob('*.py'))
    for f in ('arrived_teacher_20261023/learning.py','task_geometry_20261023/run.py',
              'task_geometry_20261023/learning.py','tests/test_learned_update_loss.py'):
        paths.add(io23.ROOT/'experiments/risk_aligned_online'/f)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(io23.ROOT)))
    protocol=dict(reference=str(REFERENCE),parent_sha256=sha(parent),source_sha256=PARENT_SHA256,
        source_checkpoint=str(DEFAULT_PARENT),selected_ids=ids,threshold=threshold,
        code_sha256={str(p.relative_to(io23.ROOT)):sha(p) for p in paths},plan_sha256=sha(PLAN),
        reference_hashes={n:sha(REFERENCE/n) for n in ('protocol.json','completed.json','verification.json','input_map.json')},
        parameter_scope=model.encoder_names(),loss_parameters=sum(p.numel() for p in rule0.parameters()),
        steps=192,outer_lr=.003,inner_lr=.001,clip_norm=1.,seed=131,
        scope='historical one-step cross-fitted LOSS HEAD only; parent and mapping already fitted all20',
        no_test_access=True,no_alarm_change=True,no_inter=True,no_new_label_identities=True,
        outcome='EAST fixed125ms; positive screen lead>30ms; earlier negatives diagnostic only',
        chronology='historical preparation, NOT first20 chronological shots',
        torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name())
    _durable_json(output/'protocol.json',protocol)
    access=[];original_outcome=io23.read_outcome;original_signal=simple_data.read_shot
    def allowed_outcome(device,sid):
        if device!='EAST' or sid not in ids:raise AssertionError('TEST/out-of-budget outcome forbidden')
        access.append(dict(device=device,shot=sid));return original_outcome(device,sid)
    def allowed_signal(device,sid,**kwargs):
        if device!='EAST' or sid not in ids:raise AssertionError('TEST/out-of-budget signal forbidden')
        return original_signal(device,sid,**kwargs)
    with patch.object(io23,'read_outcome',side_effect=allowed_outcome),patch.object(simple_data,'read_shot',side_effect=allowed_signal):
        records,items,signals=prepare(model,ids,mapping)
        splits=folds(records);plans=[dict(r,schedule=schedule(items,r['fit'],r['fold'])) for r in splits]
        _durable_json(output/'sampling.json',[{k:v for k,v in r.items() if not isinstance(v,torch.Tensor)} for r in items])
        _durable_json(output/'folds.json',plans);_durable_json(output/'label_access.json',access)
        rows=[];fit_rows=[];training={}
        for split in plans:
            f=split['fold'];held=[r for r in items if r['shot'] in split['held']]
            fit=[r for r in items if r['shot'] in split['fit']]
            for arm in LEARNED:
                rule=copy.deepcopy(rule0)
                history=train(model,rule,items,split['schedule'],arm,split['fit'])
                training[f'{f}_{arm}']=history
                torch.save(dict(rule={k:v.detach().cpu() for k,v in rule.state_dict().items()},fold=f,arm=arm),output/f'fold{f}_{arm}.pt')
                rows.extend(evaluate(model,rule,held,arm,f,'held'))
                fit_rows.extend(evaluate(model,rule,fit,arm,f,'fit'))
                progress=dict(fold=f,arm=arm,seconds=time.monotonic()-started,held_rows=len(rows))
                _durable_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            for arm in ('entropy','forecast'):
                rows.extend(evaluate(model,None,held,arm,f,'held',signals))
        assert_unchanged(model,frozen)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256 and sha(parent)==protocol['parent_sha256']
    _durable_json(output/'held_probes.json',rows);_durable_json(output/'fit_probes.json',fit_rows)
    _durable_json(output/'training.json',training)
    summary={a:summarize([r for r in rows if r['arm']==a]) for a in ALL}
    fit_summary={f'{f}_{a}':summarize([r for r in fit_rows if r['fold']==f and r['arm']==a])
                 for f in range(4) for a in LEARNED}
    done=dict(status='COMPLETED',seconds=time.monotonic()-started,summary=summary,fit_summary=fit_summary,
        recommend_full_stream=summary['learned']['passes_screen'],parent_unchanged=True,source_unchanged=True,
        test_accessed=False,training_neural_steps=4*3*192*2,held_neural_steps=len(rows),
        fit_diagnostic_neural_steps=len(fit_rows),protocol_sha256=sha(output/'protocol.json'),
        artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.suffix in ('.json','.pt','.zip')})
    _durable_json(output/'completed.json',done)
    print(json.dumps(dict(status='COMPLETED',recommend_full_stream=done['recommend_full_stream'],
        summary={a:{k:v for k,v in r.items() if k!='per_shot'} for a,r in summary.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
