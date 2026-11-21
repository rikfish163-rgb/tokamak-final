"""Source whole-shot qualification of an EAST update rule, not a JTEXT lane.

No base retraining, no new threshold, no target inputs. Full source signals are
predicted and sealed before outcome/valid-endpoint filtering for source metrics.
This distinguishes point-label loss from safe-shot false alarms.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from . import simple_data
from .cross_channel_prefix import CrossChannelModel, predict_shot
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .simple_online import _durable_json,_hash
from .simple_train import metrics

HEADS=Path(__file__).parent/'runs/cross_channel_source_20261021'
ARMS=('static','signal_embedding','risk_embedding','signal_ffn','risk_ffn')


def evaluate_shot(model,sid,folder,advanced,*,intra=True,predictor=predict_shot,data_api=simple_data):
    shot=data_api.read_shot('J-TEXT',sid)
    if shot['device']!='J-TEXT' or int(shot['shot'])!=int(sid):
        raise ValueError('source identity mismatch')
    scores,diagnostic=predictor(model,shot,intra=intra,device='cuda')
    scores=np.asarray(scores)
    if (scores.shape!=(len(shot['times']),) or not np.isfinite(scores).all()
            or (scores<0).any() or (scores>1).any()):
        raise ValueError('finite full-record probabilities required before outcome access')
    path=Path(folder)/f'{sid}.npz'
    with path.open('xb') as h:
        np.savez_compressed(h,times=shot['times'],scores=scores)
    digest=_hash(path)
    labelled=data_api.attach_outcome(shot,'J-TEXT',advanced)
    valid=np.asarray(labelled['valid'],bool)
    if valid.shape!=scores.shape or not valid.any():raise ValueError('source scoring endpoints required')
    row=dict(shot=int(sid),times=shot['times'][valid],scores=scores[valid],
             is_disrupt=bool(labelled['is_disrupt']),down=labelled['down'])
    return row,dict(diagnostic,full_prediction_count=len(scores),scored_count=int(valid.sum()),
                    sealed_sha256=digest,outcomes_opened_only_after_full_seal=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heads',type=Path,default=HEADS)
    p.add_argument('--arms',choices=ARMS,nargs='+',default=list(ARMS))
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if args.arms[0]!='static' or len(set(args.arms))!=len(args.arms):
        p.error('static first and distinct declared arms required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(4);torch.manual_seed(131)
    args.output.mkdir(parents=True)
    roles=simple_data.source_roles();ids=list(map(int,roles['val']))
    if len(ids)!=160 or len(set(ids))!=160 or set(ids)&(set(roles['train'])|set(roles['test'])):
        raise ValueError('canonical disjoint source VAL160 required')
    advanced=simple_data.advanced_times()
    advanced={sid:advanced[sid] for sid in ids if sid in advanced}
    base,parent=load_parent(DEFAULT_PARENT);threshold=float(parent['threshold'])
    reference={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_shot_screen.py','cross_channel_prefix.py','simple_train.py',
            'simple_data.py','io23.py','simple_online.py','simple23.py','fast_update.py',
            'redesign_20261004/intra.py','redesign_20261004/prepare_prior.py'):
            z.write(root/name,name)
    _durable_json(args.output/'protocol.json',dict(
        scope='SOURCE VAL160 OFFLINE QUALIFICATION OF EAST RULE, NOT JTEXT ONLINE BRANCH',
        parent_sha256=PARENT_SHA256,threshold=threshold,ids=ids,arms=args.arms,
        source_base_changed=False,east_data_used=False,threshold_reselection=False,
        source_timing='strict lead>10ms',temporary_reset_every_shot=True,
        early_errors_counted=True,labels_and_valid_filter_after_prediction_seal=True,
        update=dict(first_update=7,stride=32,step=.05,radius=1.),
        source_preparation=str(args.heads.resolve()),
        head_hashes={p.name:_hash(p) for p in args.heads.glob('*_head.pt')},
        source_preparation_protocol_sha256=_hash(args.heads/'protocol.json'),
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    results={};started=time.monotonic()
    for arm in args.arms:
        folder=args.output/arm;folder.mkdir()
        base,_=load_parent(DEFAULT_PARENT)
        site='ffn' if arm.endswith('ffn') else 'embedding'
        model=CrossChannelModel(base,site=site).cuda().eval().requires_grad_(False)
        branch=arm if arm.startswith('risk') else 'signal'
        head=torch.load(args.heads/f'{branch}_head.pt',map_location='cpu',weights_only=True)
        if head['parent_sha256']!=PARENT_SHA256 or head['branch']!=branch:
            raise ValueError('auxiliary head provenance mismatch')
        model.decoder.load_state_dict(head['decoder'])
        rows=[];accepted=attempted=0
        for index,sid in enumerate(ids):
            row,diag=evaluate_shot(model,sid,folder,advanced,intra=arm!='static')
            rows.append(row);accepted+=diag['accepted_updates'];attempted+=diag['attempted_updates']
            _durable_json(folder/f'{sid}_update.json',diag)
            if (index+1)%40==0:print(json.dumps(dict(arm=arm,completed=index+1)),flush=True)
        result=metrics(rows,threshold,lead=10.)
        if arm=='static' and (result['tp'],result['fp'])!=(73,6):
            raise AssertionError('qualified source VAL160 baseline does not reproduce73TP/6FP')
        # Independently reconstruct metric rows from durable, complete traces.
        recount=[]
        for row in rows:
            shot=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',row['shot']),'J-TEXT',advanced)
            with np.load(folder/f"{row['shot']}.npz",allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['times'],shot['times'])
                recount.append(dict(row,scores=saved['scores'][shot['valid']],times=saved['times'][shot['valid']]))
        assert metrics(recount,threshold,lead=10.)==result
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),reference[k],rtol=0,atol=0)
        _durable_json(folder/'metrics.json',result)
        results[arm]={k:v for k,v in result.items() if k!='predictions'}
        results[arm].update(accepted_updates=accepted,attempted_updates=attempted)
        _durable_json(args.output/'progress.json',results)
        print(json.dumps(dict(arm=arm,metrics=results[arm])),flush=True)
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'completed.json',dict(results=results,
        elapsed_seconds=time.monotonic()-started,base_bitwise_unchanged=True,
        independent_sealed_recount=True,east_data_used=False))


if __name__=='__main__':main()
