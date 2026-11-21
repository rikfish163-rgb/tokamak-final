"""Recount saved channel directions without loading or fitting any model."""
import argparse
from pathlib import Path

import numpy as np

from ..context_memory_20261028.checkpoint import read
from ..simple_online import _durable_json
from ..simple_train import sha


def run(root):
    if (root/'summary.json').exists():raise FileExistsError(root/'summary.json')
    d=read(root,'complete.json');p=read(root,'protocol.json');data=read(root,'records.json');rows=data['records']
    if (not d['complete'] or d['records_sha256']!=sha(root/'records.json')
            or d['protocol_sha256']!=sha(root/'protocol.json') or len(rows)+len(data['excluded'])!=340):
        raise ValueError('incomplete diagnostic records')
    cos=np.array([r['channel_cosine'] for r in rows]);dot=np.array([r['channel_dot'] for r in rows])
    norm=np.array([r['channel_gradient_norm'] for r in rows]);loss=np.array([r['channel_loss'] for r in rows])
    chosen=np.array([r['oracle_channel'] for r in rows])
    if any(a.shape!=(len(rows),23) or not np.isfinite(a).all() for a in (cos,dot,norm,loss)):
        raise ValueError('finite full channel ledger required')
    expected=np.where(cos.max(1)>1e-6,cos.argmax(1),-1)
    if not np.array_equal(chosen,expected):raise ValueError('channel selection mismatch')
    for j in (2,10):
        if any(np.any(a[:,j]!=0) for a in (cos,dot,norm,loss)):raise ValueError('unobserved/excluded channel contributed')
    if np.any((cos>1e-6)&(dot<=0)) or np.any((cos< -1e-6)&(dot>=0)):
        raise ValueError('gradient sign mismatch')
    flags=np.array([r['flag'] for r in rows]);miss=np.array([r['flag'] and not r['cold_timely'] for r in rows])
    fp=np.array([not r['flag'] and r['cold_alarm'] for r in rows])
    channels=[]
    for c,name in enumerate(p['channels']):
        channels.append(dict(name=name,observed_points=int((norm[:,c]>0).sum()),
            oracle_selections=int((chosen==c).sum()),
            positive_direction_disruptive=int((cos[flags,c]>1e-6).sum()),positive_direction_safe=int((cos[~flags,c]>1e-6).sum()),
            positive_direction_missed=int((cos[miss,c]>1e-6).sum()),positive_direction_fp=int((cos[fp,c]>1e-6).sum()),
            mean_gradient_norm_share=float((norm[:,c]/norm.sum(1)).mean()),mean_loss_share=float((loss[:,c]/loss.sum(1)).mean())))
    groups={}
    for name,take in dict(all=np.ones(len(rows),bool),disruptive_positive=flags,safe=~flags,cold_missed=miss,cold_fp=fp).items():
        local=[r for r,t in zip(rows,take) if t]
        counts=dict(n=len(local),all_channels_risk_improved=sum(r['risk_loss_change']['all']<0 for r in local),
            oracle_channel_risk_improved=sum(r['risk_loss_change']['oracle_channel']<0 for r in local),
            any_positive_channel=int((cos[take].max(1)>1e-6).sum()))
        if any(d['groups'][name][k]!=v for k,v in counts.items()):raise ValueError('summary counts differ')
        groups[name]=counts
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),groups=groups,
        channels=channels,excluded=data['excluded'],scope='recount and algebra only; not independent gradient recomputation',
        statistical_scope='one fixed development cohort; no independent sampling/significance claim',
        summary_code_sha256=sha(Path(__file__)))
    _durable_json(root/'summary.json',result);print(dict(groups=groups,records=len(rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
