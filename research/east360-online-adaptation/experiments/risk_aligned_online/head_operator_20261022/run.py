"""Forecast-head x temporary-update-operator controlled diagnostic.

Within each operator, only the auxiliary head changes. Across operators both
the parameter coordinates AND step rule differ; do not call that difference
a parameter-count-only ablation. No new training or output risk correction.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..fast_update import normalized_step
from ..simple_online import _durable_json
from ..simple_train import sha,metrics
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..deepcoral_prefix_20261022.run import predict_pairs
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import predict_many,public,snapshot

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/deepcoral_prefix_test360_v2_20261022'
ARMS=('static','encoder_original','encoder_calibrated','token_original','token_calibrated')


def token_update(model,signal,count,fast,initial,*,device):
    if model.cfg.fast_site!='tokens' or model.cfg.fast_rule!='normalized':
        raise ValueError('qualified token-coordinate normalized update required')
    x,m,target,tm,origins=forecast_examples(signal,count,model.cfg,device)
    current=fast.detach().requires_grad_(True)
    loss=model.forecast_loss(model(x,m,current.expand(len(x),-1))['forecast'],target,tm)
    observations=int(tm.sum())
    receipt=dict(last_observation=count-1,first_affected_prediction=count,
                 forecast_origins=origins,observed_targets=observations,gradient_steps=0)
    if not observations:
        receipt.update(distance_from_origin=float((fast-initial).norm()))
        return fast,receipt
    gradient,=torch.autograd.grad(loss,current)
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite token feedback')
    updated=normalized_step(current.detach(),initial,gradient,step=.05,radius=1.).detach()
    # Matching the original step coordinates/radius does not establish risk
    # improvement or complete historical-replay equivalence: this experiment
    # deliberately retains the common H25x4 / 128-point feedback schedule.
    with torch.enable_grad():
        after=model.forecast_loss(model(x,m,updated.clone().requires_grad_(True).expand(len(x),-1))['forecast'],target,tm)
    receipt.update(gradient_steps=1,gradient_norm=float(gradient.norm()),
        loss_before=float(loss.detach()),loss_after=float(after.detach()),
        distance_from_origin=float((updated-initial).norm()),step_norm=float((updated-fast).norm()))
    return updated,receipt


def token_predictions(models,signal,*,device='cuda'):
    if set(signal)!={'values','available'} or not models:
        raise ValueError('signal-only input and explicit model arms required')
    n=len(signal['values']); scores={k:[] for k in models};receipts={k:[] for k in models}
    initial={k:m.initial_fast(1).detach() for k,m in models.items()}
    fast={k:v.clone() for k,v in initial.items()}
    for lo in range(0,n,128):
        hi=min(n,lo+128)
        for start in range(lo,hi,32):
            stop=min(hi,start+32)
            windows=[simple_data.window(signal,e,128) for e in range(start,stop)]
            x,m=[torch.as_tensor(np.stack(v),device=device) for v in zip(*windows)]
            for name,model in models.items():
                # Same gradient-enabled math attention path and endpoint batch32
                # as the saved frozen baseline. No future points in a window.
                state=fast[name].detach().requires_grad_(True).expand(len(x),-1)
                p=model(x,m,state)['logit'].sigmoid().detach().cpu().numpy()
                scores[name].append(p)
        if hi<153 or hi==n: continue
        for name,model in models.items():
            fast[name],r=token_update(model,signal,hi,fast[name],initial[name],device=device)
            receipts[name].append(r)
    result={k:np.concatenate(v) for k,v in scores.items()}
    if not all(np.isfinite(v).all() for v in result.values()): raise FloatingPointError('nonfinite token predictions')
    return result,receipts


def load_calibrated():
    prep=json.loads((REFERENCE/'preparation_receipts.json').read_text())
    path=REFERENCE/'density.pt'; assert sha(path)==prep['density']['checkpoint_sha256']
    base,_=load_parent(DEFAULT_PARENT); model=PairedFeedback(base,np.ones(23))
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    original,_=load_parent(DEFAULT_PARENT)
    for key,value in model.base.state_dict().items():
        if not key.startswith('forecast_head.'):
            torch.testing.assert_close(value,original.state_dict()[key],rtol=0,atol=0)
    model.cuda();model.set_stage('frozen')
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(2);torch.manual_seed(131);start=time.monotonic()
    base,parent=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda();model.set_stage('frozen')
    original={k:v.detach().cpu().clone() for k,v in model.base.state_dict().items()}
    threshold=float(parent['threshold']);contract=load_test_contract('EAST');order=contract['shot_ids']
    old_protocol=json.loads((REFERENCE/'protocol.json').read_text())
    assert threshold==old_protocol['threshold'] and order==old_protocol['contract']['shot_ids']
    mapping=json.loads((REFERENCE/'density_map.json').read_text())
    assert mapping['calibration_ids']==order[:20]
    old_seals={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    args.output.mkdir(parents=True);(args.output/'shots').mkdir()
    hashes=snapshot(args.output)
    dependencies=[Path(__file__),Path(__file__).parents[1]/'fast_update.py',
        Path(__file__).parents[1]/'deepcoral_prefix_20261022/learning.py',
        Path(__file__).parents[1]/'deepcoral_prefix_20261022/run.py']
    with zipfile.ZipFile(args.output/'code_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:
        for f in dependencies:
            z.write(f,str(f.relative_to(io23.ROOT)));hashes[str(f.relative_to(io23.ROOT))]=sha(f)
    plan=io23.ROOT/'docs/research/source_event_rebuild_20261022/预测头与更新位置_事前约定_20261022.md'
    _durable_json(args.output/'protocol.json',dict(contract=contract,threshold=threshold,arms=ARMS,
        source_checkpoint=str(DEFAULT_PARENT),source_sha256=PARENT_SHA256,
        reference=str(REFERENCE),reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        code_sha256=hashes,plan_path=str(plan),plan_sha256=sha(plan),
        density_mapping_sha256=sha(REFERENCE/'density_map.json'),
        warmup20='common original predictions; mapping and calibrated head activated only afterward',
        encoder_operator='embedding+position+CLS+firstblock; SGD1e-3; gradient norm cap1',
        token_operator='128 temporary patch-token gain/bias; normalized step0.05; origin radius1',
        common_feedback='H25x4 every128points; already-arrived signals; temporary states reset eachshot',
        reused_arm='encoder_calibrated predictions are immutable prior-run traces, not a new independent replicate',
        no_new_training=True,no_target_labels_in_updates=True,independent_confirmation=False))
    calibrated=None;frozen_calibrated=None;seals=[]
    for index,sid in enumerate(order):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal={k:s[k] for k in ('values','available')}
        refpath=REFERENCE/'shots'/f'{sid}.npz';assert sha(refpath)==old_seals[sid]['prediction_sha256']
        with np.load(refpath,allow_pickle=False) as ref:
            if index<20:
                raw=predict_many(model,dict(raw=signal),device='cuda')['raw']
                predictions={k:raw.copy() for k in ARMS};updates={}
            else:
                aligned=transform(signal,mapping)
                encoder,enc_updates=predict_pairs({'original':model},aligned)
                token,tok_updates=token_predictions({'original':model.base,'calibrated':calibrated.base},aligned)
                predictions=dict(static=encoder['original_static'],encoder_original=encoder['original_intra'],
                    encoder_calibrated=ref['density_intra'].copy(),token_original=token['original'],
                    token_calibrated=token['calibrated'])
                updates=dict(encoder_original=enc_updates['original'],token_original=tok_updates['original'],
                             token_calibrated=tok_updates['calibrated'])
                for k in ('token_original','token_calibrated'):
                    np.testing.assert_allclose(predictions[k][:256],predictions['static'][:256],rtol=0,atol=2e-6)
            np.testing.assert_array_equal(predictions['static'],ref['density_static'])
        path=args.output/'shots'/f'{sid}.npz'
        np.savez_compressed(path,times=s['times'],**predictions)
        _durable_json(args.output/'shots'/f'{sid}_updates.json',updates)
        seals.append(dict(shot=sid,prediction_sha256=sha(path)))
        if index==19:
            calibrated=load_calibrated()
            frozen_calibrated={k:v.detach().cpu().clone() for k,v in calibrated.base.state_dict().items()}
            _durable_json(args.output/'activation_receipt.json',dict(after_sealed_ids=order[:20],
                preceding_seals=seals.copy(),mapping_sha256=sha(REFERENCE/'density_map.json'),
                calibrated_checkpoint_sha256=sha(REFERENCE/'density.pt')))
        if (index+1)%40==0:
            progress=dict(shots=index+1,seconds=time.monotonic()-start)
            _durable_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    _durable_json(args.output/'prediction_receipts.json',seals)
    rows={k:[] for k in ARMS}
    for sid in order:
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(args.output/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm in ARMS:
                v=s['valid'];rows[arm].append(dict(shot=sid,times=s['times'][v],scores=trace[arm][v],
                    is_disrupt=s['is_disrupt'],down=s['down']))
    full={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
    result={k:dict(full360=public(full[k]),common_suffix340=public(metrics(v[20:],threshold,lead=30.))) for k,v in rows.items()}
    for m,expected in ((model,original),(calibrated,frozen_calibrated)):
        for k,v in m.base.state_dict().items(): torch.testing.assert_close(v.cpu(),expected[k],rtol=0,atol=0)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'per_shot.json',full)
    _durable_json(args.output/'completed.json',dict(status='COMPLETED',metrics=result,
        persistent_models_bitwise_unchanged=True,seconds=time.monotonic()-start))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
