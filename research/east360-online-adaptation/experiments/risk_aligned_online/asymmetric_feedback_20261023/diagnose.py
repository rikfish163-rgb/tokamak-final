"""Preparation-gradient conflict on existing TRAIN draws; no held labels."""
import argparse
import copy
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..credit_horizon_20261023.run import READ,load_model
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..east_joint_feedback_20261023.learning import enable,episode
from ..east_joint_source_replay_20261023.learning import replay_loss
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_source_train
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import gradient_relation

REFERENCE=io23.ROOT/'experiments/risk_aligned_online/runs/east_joint_source_replay_cv20_20261023'
REF_SHA='6878a0c1941e018ed824b8fbbb8a2366b862d8dc449354e6149a83e568ac0101'


def gradient(loss,model):
    pars=tuple(p for k,p in model.named_parameters() if k in model.encoder_names())
    return torch.cat([v.detach().flatten().double() for v in torch.autograd.grad(loss,pars,retain_graph=True)])


def main(root):
    if root.exists():raise FileExistsError(root)
    assert sha(REFERENCE/'completed.json')==REF_SHA and READ(REFERENCE/'verification.json')['status']=='VERIFIED'
    root.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(REFERENCE/'protocol.json');source_ids=io23.source_roles()['train'];read=io23.read_outcome;access=[]
    def source_guard(device,sid):
        assert device=='J-TEXT' and sid in source_ids
        return read(device,sid)
    with patch.object(io23,'read_outcome',side_effect=source_guard):source,_=load_source_train()
    source={s['shot']:s for s in source};rows=[]
    for plan in p['folds']:
        f=plan['fold'];folder=REFERENCE/f'fold{f}';mapping=READ(folder/'input_map.json')
        def target_guard(device,sid):
            assert device=='EAST' and sid in plan['fit'];access.append(dict(fold=f,shot=sid));return read(device,sid)
        with patch.object(io23,'read_outcome',side_effect=target_guard):
            shots={}
            for sid in plan['fit']:
                s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
                shots[sid]=dict(s,**transform({k:s[k] for k in ('values','available')},mapping))
        warm=load_model(f);trained=copy.deepcopy(warm)
        trained.load_state_dict(torch.load(folder/'meta.pt',map_location='cuda',weights_only=True)['model'])
        for stage,m in (('initial',warm),('after_meta',trained)):
            enable(m);before=state_cpu(m)
            for entry in READ(folder/'training.json')[:8]:
                gradients={k:torch.zeros(sum(dict(m.named_parameters())[n].numel() for n in m.encoder_names()),device='cuda',dtype=torch.float64)
                           for k in ('before','after','forecast')}
                for e in entry['meta']['episodes']:
                    s=shots[e['shot']]
                    r=episode(m,{k:s[k] for k in ('values','available')},e['count'],e['ends'],e['disruptive'],
                        boundary=math.log(p['threshold']/(1-p['threshold'])),track=True,device='cuda')
                    for k in gradients:gradients[k]+=gradient(r[k],m)/2
                    del r
                examples=[]
                for e in entry['source_examples']:
                    s=source[e['shot']];assert s['valid'][e['end']] and int(s['y'][e['end']]>.5)==e['label']
                    x,mask=simple_data.window(s,e['end'],128);examples.append((x,mask,np.float32(e['label'])))
                sx,sm,sy=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*examples)]
                gs=gradient(replay_loss(m,sx,sm,sy),m);gf=.2*gradients['forecast']
                gr=.5*(gradients['before']+gradients['after'])
                rows.append(dict(fold=f,stage=stage,step=entry['step'],
                    target=gradient_relation(gr,gf),source=gradient_relation(gs,gf),
                    supervised_total=gradient_relation(gr+gs,gf)))
            assert_unchanged(m,before)
            print(dict(fold=f,stage=stage,rows=len(rows)),flush=True)
    summary={}
    for stage in ('initial','after_meta'):
        summary[stage]={}
        for name in ('target','source','supervised_total'):
            group=[r[name] for r in rows if r['stage']==stage]
            summary[stage][name]=dict(n=len(group),negative_dot=sum(r['dot']<0 for r in group),
                median_cosine=float(np.median([r['cosine'] for r in group if r['cosine'] is not None])),
                median_aux_to_risk_norm=float(np.median([r['norm_b']/r['norm_a'] for r in group if r['norm_a']>0])))
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='COMPLETED',reference_sha256=REF_SHA,source_unchanged=True,test360_accessed=False,
        target_outcome_access=access,rows=rows,summary=summary,code_sha256=sha(Path(__file__)),
        seconds=time.monotonic()-start,interpretation='Local Euclidean gradient relation on 8 TRAIN draws per fold, not an Adam causal attribution or a held-shot performance result.')
    _durable_json(root/'diagnostic.json',result);print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
