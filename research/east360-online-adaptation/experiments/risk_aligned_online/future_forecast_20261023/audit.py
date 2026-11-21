"""Torch double-precision recount and three complete forecast replays."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..forecast_token_scope_20261023.learning import ARMS, PatchForecast
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .diagnostic import replay
from .run import REFERENCE

READ = lambda p: json.loads(p.read_text())


def audit(root):
    if (root/'verification.json').exists():raise FileExistsError(root/'verification.json')
    p,done=READ(root/'protocol.json'),READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    assert sha(Path(p['plan_path']))==p['plan_sha256']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256==p['source_sha256']
    assert sha(REFERENCE/'completed.json')==p['reference_sha256']
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT);parent=PairedFeedback(base,np.ones(23)).cuda();parent.set_stage('frozen')
    original=state_cpu(parent);rows=READ(root/'windows.json')
    lookup={(r['shot'],r['origin'],r['arm']):r for r in rows}
    seals={r['shot']:r for r in READ(root/'prediction_receipts.json')}
    checked=0;replays=[];channel_rows=[];max_error=0.;removed=0;empty=0
    for plan in p['folds']:
        f=plan['fold'];folder=root/f'fold{f}';ref=REFERENCE/f'fold{f}';mapping=READ(ref/'input_map.json')
        models={}
        for head in ('full','patch'):
            m=copy.deepcopy(parent)
            if head=='patch':m.base.forecast_head=PatchForecast(m.base.forecast_head,m.cfg.dim)
            m.load_state_dict(torch.load(ref/f'{head}.pt',map_location='cuda',weights_only=True)['model'])
            m.set_stage('frozen');models[head]=m
        for sid in plan['held']:
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            path=folder/f'{sid}.npz';assert sha(path)==seals[sid]['sha256']
            if sid in (63226,64481,70868):
                def forbidden(*args,**kwargs):raise AssertionError('outcome is not a forecast input')
                with patch.object(io23,'read_outcome',side_effect=forbidden):
                    packets,updates=replay(models,signal)
                assert updates==READ(ref/'shots'/f'{sid}_updates.json')
                with np.load(path,allow_pickle=False) as cached:
                    for name,a in packets.items():np.testing.assert_array_equal(a,cached[name])
                replays.append(dict(shot=sid,origins=len(packets['origins']),exact=True))
            with np.load(path,allow_pickle=False) as pack:
                np.testing.assert_array_equal(pack['center'],parent.base.center.cpu().numpy())
                np.testing.assert_array_equal(pack['scale'],parent.base.scale.cpu().numpy())
                for j,count in enumerate(pack['counts']):
                    assert pack['origins'][j]==count-1 and count+25<=len(signal['values'])
                    np.testing.assert_array_equal(pack['last_value'][j],signal['values'][count-1])
                    np.testing.assert_array_equal(pack['last_available'][j],signal['available'][count-1])
                    raw=signal['values'][count:count+25];valid=signal['available'][count:count+25]&np.isfinite(raw)
                    last=signal['values'][count-1];lastvalid=signal['available'][count-1]&np.isfinite(last)
                    mask=torch.as_tensor(valid&lastvalid[None])
                    center=torch.as_tensor(pack['center']).double();scale=torch.as_tensor(pack['scale']).double()
                    truth=torch.asinh((torch.as_tensor(np.where(valid,raw,pack['center'])).double()-center)/scale)
                    persistence=torch.asinh((torch.as_tensor(np.where(lastvalid,last,pack['center'])).double()-center)/scale)[None].expand_as(truth)
                    removed+=int(valid.sum())-int(mask.sum());empty+=int(not mask.any())
                    for a in ARMS:
                        saved=lookup[sid,int(count)-1,a]
                        predictions={k:torch.as_tensor(pack[f'{k}_{a}'][j]).double() for k in ('before','after')}
                        predictions['static']=torch.as_tensor(pack[f'static_{a.split("_")[0]}'][j]).double()
                        predictions['persistence']=persistence
                        channel={}
                        for name,pred in predictions.items():
                            loss=F.smooth_l1_loss(pred,truth,reduction='none')
                            mean=float(loss[mask].mean()) if mask.any() else None
                            if mean is None:assert saved[name] is None
                            else:
                                err=abs(mean-saved[name]);max_error=max(max_error,err)
                                assert err<2e-12
                            channel[name]=[float(loss[:,c][mask[:,c]].mean()) if mask[:,c].any() else None for c in range(23)]
                            if name!='persistence':
                                for x,y in zip(channel[name],saved['per_channel'][name]):
                                    assert x is None and y is None or x is not None and y is not None and abs(x-y)<2e-12
                        channel_rows.append(dict(shot=sid,arm=a,origin=int(count)-1,**channel));checked+=1
    assert checked==len(rows)==done['complete_future_origins']*4
    # An independent grouping calculation verifies that long shots do not
    # receive more weight simply because they contain more update windows.
    summary=READ(root/'summary.json');channel_summary={}
    for group,condition in (
        ('all',lambda r:True),('valid',lambda r:r['future_all_valid']),
        ('safe',lambda r:r['future_all_valid'] and not r['is_disrupt']),
        ('disruptive',lambda r:r['future_all_valid'] and r['is_disrupt']),
        ('positive',lambda r:r['future_all_valid'] and r['future_any_positive'])):
        for a in ARMS:
            selected=[r for r in rows if r['arm']==a and condition(r) and r['observed']]
            ids=sorted({r['shot'] for r in selected})
            if not ids:assert a not in summary[group];continue
            means={k:float(np.mean([np.mean([r[k] for r in selected if r['shot']==sid]) for sid in ids]))
                   for k in ('before','after','static','persistence')}
            for k,v in means.items():assert abs(v-summary[group][a]['mean_losses'][k])<2e-12
            assert len(ids)==summary[group][a]['n_shots'] and len(selected)==summary[group][a]['n_windows']
        if group!='all':continue
        for a in ARMS:
            selected=[r for r in channel_rows if r['arm']==a];ids=sorted({r['shot'] for r in selected})
            channel_summary[a]=[]
            for c,name in enumerate(io23.RAW23):
                means={}
                for method in ('before','after','static','persistence'):
                    per_shot=[[r[method][c] for r in selected if r['shot']==sid and r[method][c] is not None] for sid in ids]
                    seen=[float(np.mean(v)) for v in per_shot if v]
                    means[method]=float(np.mean(seen)) if seen else None
                channel_summary[a].append(dict(channel=name,**means))
    assert_unchanged(parent,original);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(root/'channel_diagnostic.json',channel_summary)
    report=dict(status='VERIFIED',completed_sha256=sha(root/'completed.json'),audit_sha256=sha(Path(__file__)),
        independent_window_arm_recounts=checked,max_loss_abs_difference=max_error,forecast_replays=replays,
        common_mask_excluded_targets=removed,empty_future_targets=empty,
        channels_sha256=sha(root/'channel_diagnostic.json'),equal_shot_aggregation_verified=True,
        source_unchanged=True,scope='posthoc CV20; three same-implementation full forecast replays; not new warning results')
    _durable_json(root/'verification.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);audit(parser.parse_args().root)
