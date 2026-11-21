"""Does local fitting help beyond a fixed attenuation or reversed pairing?

Designed after viewing the first six-arm development result. Fits a scalar on
the SAME20 historical preparation shots only; never selects it on TEST.
"""
import argparse
import copy
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..deepcoral_prefix_20261022.run import assert_unchanged,state_cpu
from ..fixed_test_protocol import score_fixed_test
from ..simple_online import _durable_json
from ..simple_train import sha
from ..target_feedback_meta_20261023.audit import recount
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from .model import LocalTransition
from .run import READ,load_model

PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261022/局部动力学快参数_衰减对照约定_20261023.md'


def fit_scale(before,after):
    u,v=np.asarray(before,dtype=float),np.asarray(after,dtype=float)
    if u.ndim!=2 or u.shape!=v.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('matching finite correction vectors required')
    denominator=np.square(u).sum()
    if denominator<=0:raise ValueError('nonzero reference correction required')
    return float(np.clip((u*v).sum()/denominator,0.,1.))


def predict(models,signal):
    if set(signal)!={'values','available'}:raise ValueError('signals only')
    pieces={k:[] for k in models}
    for lo in range(0,len(signal['values']),32):
        batch=[simple_data.window(signal,t,128) for t in range(lo,min(lo+32,len(signal['values'])))]
        x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*batch)]
        for name,(model,mode) in models.items():
            pieces[name].append(model(x,m,mode=mode)['logit'].sigmoid().detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in pieces.items()}


def main(root):
    output=root/'mechanism_controls'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);torch.manual_seed(131);started=time.monotonic()
    p,first,v=[READ(root/f) for f in ('protocol.json','completed.json','verification.json')]
    assert v['status']=='VERIFIED' and v['completed_sha256']==sha(root/'completed.json')
    ck=root/'static_trained.pt';assert sha(ck)==first['checkpoint_sha256']['static_trained']
    parent,threshold=load_model();model=LocalTransition(copy.deepcopy(parent.base)).cuda()
    model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True)['model'])
    model.eval().requires_grad_(False);initial=state_cpu(model)
    mapping=READ(Path(p['reference'])/'input_map.json');ids=p['selected_ids'];order=p['contract']['shot_ids']
    assert not set(ids)&set(order)
    output.mkdir();(output/'shots').mkdir()
    _durable_json(output/'protocol.json',dict(base_protocol_sha256=sha(root/'protocol.json'),
        base_completed_sha256=sha(root/'completed.json'),checkpoint_sha256=sha(ck),
        plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),selected_ids=ids,
        source_code_sha256=p['code_sha256'],threshold=threshold,contract=p['contract'],
        test_previously_exposed=True,independent_confirmation=False,extra_label_identities=0,
        fixed_scale='least-squares internal correction fit to prior on640 historical endpoints; clip0..1',
        new_arms=['static_shuffled','static_scaled']))
    access=[];original_read=io23.read_outcome
    def guard(device,sid):
        if device!='EAST' or sid not in ids:raise AssertionError('unauthorized outcome before seals')
        access.append(sid);return original_read(device,sid)
    with patch.object(io23,'read_outcome',side_effect=guard):
        before=[];after=[];selection=[]
        for sid in ids:
            s=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            valid=np.flatnonzero(s['valid'] & (np.arange(len(s['times']))>=127))
            if not len(valid):raise ValueError('historical shot lacks one full window')
            ends=valid[np.rint(np.linspace(0,len(valid)-1,32)).astype(int)]
            selection.append(dict(shot=sid,endpoints=ends.tolist()))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            batch=[simple_data.window(signal,int(t),128) for t in ends]
            x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*batch)]
            before.append(model(x,m,mode='prior')['correction'].detach().cpu().numpy())
            after.append(model(x,m,mode='fit')['correction'].detach().cpu().numpy())
        u,vectors=np.concatenate(before),np.concatenate(after);alpha=fit_scale(u,vectors)
        np.savez_compressed(output/'scale_fit.npz',prior=u,fitted=vectors)
        _durable_json(output/'scale_fit.json',dict(alpha=alpha,selection=selection,
            vector_sha256=sha(output/'scale_fit.npz'),uncorrected_mse=float(np.square(u-vectors).mean()),
            scaled_mse=float(np.square(alpha*u-vectors).mean())))
        scaled=copy.deepcopy(model)
        with torch.no_grad():scaled.readout.weight.mul_(alpha)
        models={'static_shuffled':(model,'shuffled'),'static_scaled':(scaled,'prior')}
        states={k:state_cpu(m) for k,(m,_) in models.items()};seals=[]
        for index,sid in enumerate(order):
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            pred=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
            with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(s['times'],z['times'])
                for value in pred.values():np.testing.assert_array_equal(value[:15],z['parent'][:15])
            path=output/'shots'/f'{sid}.npz';np.savez_compressed(path,times=s['times'],**pred)
            seals.append(dict(shot=sid,sha256=sha(path)))
            if (index+1)%60==0:print(dict(stage='controls',shots=index+1,seconds=time.monotonic()-started),flush=True)
        _durable_json(output/'prediction_receipts.json',seals);_durable_json(output/'label_access.json',access)
    assert access==ids
    rows={k:[] for k in models};independent={k:[] for k in models}
    for seal in seals:
        sid=seal['shot'];path=output/'shots'/f'{sid}.npz';assert sha(path)==seal['sha256']
        d,down=io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as z:
            valid=(down-z['times'])*1000>=-1e-6 if d else np.ones(len(z['times']),dtype=bool)
            for name in models:
                rows[name].append(dict(shot=sid,times=z['times'][valid],scores=z[name][valid],is_disrupt=d,down=down))
                independent[name].append(recount(sid,z['times'],z[name],d,down,threshold))
    scored={k:score_fixed_test('EAST',r,threshold) for k,r in rows.items()}
    for k,r in scored.items():assert r['predictions']==independent[k]
    replays=[]
    for sid in (order[0],order[len(order)//2],order[-1]):
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        prediction=predict(models,transform({k:s[k] for k in ('values','available')},mapping))
        with np.load(output/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            for k,value in prediction.items():np.testing.assert_array_equal(value,z[k])
        replays.append(dict(shot=sid,curves=2,max_abs_difference=0.))
    assert_unchanged(model,initial)
    for k,(m,_) in models.items():assert_unchanged(m,states[k])
    previous=READ(root/'per_shot.json');_durable_json(output/'per_shot.json',scored)
    result=dict(status='COMPLETED_AND_VERIFIED',seconds=time.monotonic()-started,alpha=alpha,
        protocol_sha256=sha(output/'protocol.json'),scale_sha256=sha(output/'scale_fit.json'),
        original_static_checkpoint_unchanged=sha(ck)==first['checkpoint_sha256']['static_trained'],
        independent_confirmation=False,traces_independently_recounted=720,sampled_replays=replays,
        metrics={k:dict(**public(r),very_early_over3000=sum(v['timely'] and v['lead_ms']>3000 for v in r['predictions'])) for k,r in scored.items()},
        paired_vs_true_fit={k:paired_flips(previous['static_fit']['predictions'],r['predictions']) for k,r in scored.items()},
        paired_vs_prior={k:paired_flips(previous['static_prior']['predictions'],r['predictions']) for k,r in scored.items()})
    _durable_json(output/'completed.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
