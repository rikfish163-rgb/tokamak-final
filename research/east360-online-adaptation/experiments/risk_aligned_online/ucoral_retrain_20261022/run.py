"""Paired source-retrained z-score / U-CORAL reference, not neural online gain."""
import argparse
import copy
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract,validate_test_ids
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import load_shots,pools,select_threshold,validation_rank,score,metrics,sha
from ..warmstart_alignment_20261022.run import predict_many,public
from .alignment import fit_alignment,validate_prefix,KEYS
from .model import CoordinateClassifier

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/warmstart_alignment_test360_v2_20261022'
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/源到目标CORAL重训_事前约定_20261022.md'
ARMS=('zscore','ucoral')


def phase_metrics(rows,parent_threshold,selected_threshold):
    """Do not retroactively apply a post-prefix threshold to first20 predictions."""
    predictions=[]
    for i,r in enumerate(rows):
        threshold=parent_threshold if i<20 else selected_threshold
        p=metrics([r],threshold,lead=30.)['predictions'][0]
        predictions.append(dict(p,threshold=float(threshold)))
    nd=sum(p['is_disrupt'] for p in predictions);ns=len(predictions)-nd
    tp=sum(p['timely'] for p in predictions);fp=sum(p['alarm'] and not p['is_disrupt'] for p in predictions)
    leads=[p['lead_ms'] for p in predictions if p['timely']]
    tpr=tp/nd if nd else None;fpr=fp/ns if ns else None
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,tpr=tpr,fpr=fpr,J=tpr-fpr if nd and ns else None,
                parent_threshold=parent_threshold,selected_threshold=selected_threshold,
                median_lead_ms=float(np.median(leads)) if leads else None,
                very_early_over3000ms=sum(v>3000 for v in leads),predictions=predictions)


def training_batch(shots,groups,rng,cfg):
    manifest=[];examples=[]
    for _ in range(64):
        label=int(rng.integers(2));group=groups[label];i,ends=group[int(rng.integers(len(group)))]
        end=int(rng.choice(ends));s=shots[i];x,m=simple_data.window(s,end,cfg.window)
        examples.append((x,m,float(label)));manifest.append(dict(shot=s['shot'],end=end,label=label))
    tensors=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples)]
    return tensors,manifest


@torch.no_grad()
def predict_target(model,shot):
    model.eval();windows=[]
    for key in ('values','available'):
        p=np.pad(shot[key],((model.cfg.window-1,0),(0,0)))
        windows.append(np.lib.stride_tricks.sliding_window_view(p,model.cfg.window,axis=0).swapaxes(1,2))
    out=[]
    for lo in range(0,len(shot['values']),32):
        x,m=[torch.as_tensor(np.array(w[lo:lo+32],copy=True),device='cuda') for w in windows]
        p=model(x,m,domain='target')['logit'].sigmoid().cpu().numpy()
        if not np.isfinite(p).all():raise FloatingPointError('nonfinite target probabilities')
        out.append(p)
    return np.concatenate(out)


def snapshot(output):
    root=Path(__file__).parents[1]
    files=list(Path(__file__).parent.glob('*.py'))+[root/n for n in (
        'simple23.py','simple_train.py','simple_data.py','io23.py','causal_input.py','fixed_test_protocol.py',
        'simple_online.py','redesign_20261004/prepare_prior.py','paired_feedback_20261022/model.py',
        'masked_state_20261022/model.py','warmstart_alignment_20261022/run.py','warmstart_alignment_20261022/alignment.py')]
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(io23.ROOT)))
    return {str(p.relative_to(io23.ROOT)):sha(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output
    if out.exists():raise FileExistsError(out)
    if not torch.cuda.is_available():raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);torch.backends.mha.set_fastpath_enabled(False)
    started=time.monotonic();base,parent=load_parent(DEFAULT_PARENT);threshold=float(parent['threshold'])
    contract=load_test_contract('EAST');order=contract['shot_ids'];out.mkdir(parents=True);(out/'shots').mkdir()
    protocol=dict(format='source-target-ucoral-retrain-v1',contract=contract,arms=ARMS,
        source_sha256=PARENT_SHA256,parent_threshold=threshold,plan=str(PLAN),plan_sha256=sha(PLAN),
        code_sha256=snapshot(out),reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        source_train_ids=simple_data.source_roles()['train'],target_stat_ids=order[:20],seed=131,
        epochs=4,steps_per_epoch=256,batch_size=64,lr=.0003,weight_decay=.01,
        optimizer='AdamW',gradient_clip=1.,frozen=['forecast_head','fast_origin'],
        inference_backend='math SDPA; MHA fastpath disabled for identical train/eval dispatch',
        target_outcomes_for_fit=0,source_validation_fpr_strict_cap=.1,
        activation='after 20 ended-shot prediction seals; no retroactive warmup rescoring',
        independent_confirmation=False,intra_shot_updates=0,inter_shot_updates=0)
    _durable_json(out/'protocol.json',protocol)
    # Recompute and verify first20 ORIGINAL raw predictions before target fit.
    frozen=PairedFeedback(copy.deepcopy(base),np.ones(23)).cuda();frozen.set_stage('frozen')
    refs={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    prefix=[];seals=[]
    for sid in order[:20]:
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        p=predict_many(frozen,dict(raw={k:s[k] for k in ('values','available')}),device='cuda')['raw']
        path=REFERENCE/'shots'/f'{sid}.npz'
        if sha(path)!=refs[sid]['prediction_sha256']:raise AssertionError('reference prediction hash changed')
        with np.load(path,allow_pickle=False) as ref:
            np.testing.assert_array_equal(ref['times'],s['times'])
            np.testing.assert_allclose(p,ref['static'],atol=2e-6,rtol=0)
            np.testing.assert_array_equal(p>=threshold,ref['static']>=threshold)
        file=out/'shots'/f'{sid}.npz';np.savez_compressed(file,times=s['times'],**{a:p for a in ARMS})
        seals.append(dict(shot=sid,prediction_sha256=sha(file)));prefix.append({k:s[k] for k in KEYS})
    del frozen
    validate_prefix(prefix,order,{str(r['shot']):r['prediction_sha256'] for r in seals})
    _durable_json(out/'activation_receipt.json',dict(after_seals=seals.copy(),target_outcomes_accessed=False))
    train,train_ids=load_source_train()
    # Source values/masks are unaltered. No source holdout used to fit moments.
    source=[{k:s[k] for k in KEYS} for s in train];fit=fit_alignment(source,prefix)
    fit['channels']=io23.RAW23;fit['source_ids']=train_ids;fit['target_ids']=order[:20]
    _durable_json(out/'alignment.json',fit);del source,prefix
    advanced=simple_data.advanced_times();roles=simple_data.source_roles();val=load_shots(roles['val'],advanced)
    models={a:CoordinateClassifier(copy.deepcopy(base),fit,coral=a=='ucoral').cuda() for a in ARMS}
    params={a:[p for p in m.parameters() if p.requires_grad] for a,m in models.items()}
    opts={a:torch.optim.AdamW(params[a],lr=.0003,weight_decay=.01) for a in ARMS}
    best={a:None for a in ARMS};history=[];selected={};groups=pools(train);rng=np.random.default_rng(131)
    def evaluate(epoch,losses):
        for a,m in models.items():
            measured=select_threshold(score(m,val,'cuda'),.1);rank=validation_rank(measured)
            rec=dict(epoch=epoch,arm=a,loss=float(np.mean(losses[a])) if losses[a] else None,validation=public(measured))
            history.append(rec);print(json.dumps(rec),flush=True)
            path=out/f'{a}_epoch{epoch}.pt'
            torch.save(dict(state={k:v.detach().cpu() for k,v in m.state_dict().items()},epoch=epoch,
                            threshold=measured['threshold'],validation=public(measured),parent_sha256=PARENT_SHA256),path)
            if best[a] is None or rank<best[a]:
                best[a]=rank;selected[a]=dict(path=str(path.resolve()),sha256=sha(path),epoch=epoch,threshold=measured['threshold'])
        _durable_json(out/'history.json',history)
    evaluate(0,{a:[] for a in ARMS})
    train_started=time.monotonic()
    with (out/'training_manifest.jsonl').open('x') as manifest:
        for epoch in range(1,5):
            losses={a:[] for a in ARMS}
            for step in range(256):
                (x,m,y),examples=training_batch(train,groups,rng,base.cfg)
                manifest.write(json.dumps(dict(epoch=epoch,step=step,examples=examples))+'\n')
                for a,model in models.items():
                    model.train();opts[a].zero_grad(set_to_none=True)
                    loss=F.binary_cross_entropy_with_logits(model(x,m)['logit'],y.float())
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params[a],1.,error_if_nonfinite=True);opts[a].step()
                    losses[a].append(float(loss.detach()))
            manifest.flush();evaluate(epoch,losses)
    _durable_json(out/'selection.json',dict(models=selected,source_validation_only=True,
        training_manifest_sha256=sha(out/'training_manifest.jsonl'),training_and_validation_seconds=time.monotonic()-train_started))
    # Source TEST after selection, never used for tuning this comparison.
    source_test=load_shots(roles['test'],advanced);source_result={}
    for a,model in models.items():
        payload=torch.load(selected[a]['path'],map_location='cpu',weights_only=True);model.load_state_dict(payload['state'])
        model.eval();model.requires_grad_(False)
        result=metrics(score(model,source_test,'cuda'),selected[a]['threshold'],lead=10.)
        source_result[a]=dict(result,qualified=result['tpr']>=.9 and result['fpr']<.2)
    _durable_json(out/'source_test220.json',source_result)
    for i,sid in enumerate(order[20:],20):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        predictions={a:predict_target(model,s) for a,model in models.items()}
        file=out/'shots'/f'{sid}.npz';np.savez_compressed(file,times=s['times'],**predictions)
        seals.append(dict(shot=sid,prediction_sha256=sha(file)))
        if (i+1)%40==0:
            progress=dict(shots=i+1,seconds=time.monotonic()-started)
            _durable_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    _durable_json(out/'prediction_receipts.json',seals)
    rows={a:[] for a in ARMS}
    # FIRST target outcome access: every prediction above already sealed.
    for sid in order:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(out/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for a in ARMS:
                v=s['valid'];rows[a].append(dict(shot=sid,times=s['times'][v],scores=trace[a][v],
                                                is_disrupt=s['is_disrupt'],down=s['down']))
    full={};result={}
    for a,rs in rows.items():
        validate_test_ids('EAST',[r['shot'] for r in rs]);assert sum(r['is_disrupt'] for r in rs)==180
        full[a]=phase_metrics(rs,threshold,selected[a]['threshold'])
        result[a]=dict(full360=public(full[a]),suffix340=public(metrics(rs[20:],selected[a]['threshold'],lead=30.)),
                       full360_common_parent_threshold=public(metrics(rs,threshold,lead=30.)),
                       source_test220=public(source_result[a]))
        for n,p in models[a].base.named_parameters():
            if n=='fast_origin' or n.startswith('forecast_head.'):
                torch.testing.assert_close(p.cpu(),dict(base.named_parameters())[n],rtol=0,atol=0)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('qualified parent changed')
    _durable_json(out/'per_shot.json',full)
    _durable_json(out/'completed.json',dict(status='COMPLETED',metrics=result,selected=selected,
        seconds=time.monotonic()-started,source_archive_unchanged=True,independent_confirmation=False,
        intra_shot_gain_tested=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
