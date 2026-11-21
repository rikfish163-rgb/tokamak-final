"""Posthoc trajectory-specificity checks, never imported by online learning."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .model import tokens,masked,latent_loss,risk_kl,readout
from .train import source_windows,CACHE


def run(root):
    checked=read(root,'audit.json');p=read(root,'protocol.json')
    if not checked['complete'] or checked['complete_sha256']!=sha(root/'complete.json'):
        raise ValueError('audited completed stream required')
    records=[]
    for i,sid in enumerate(p['order'][20:],21):
        flag,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{i:04d}_{sid}.npz') as z:
            valid=(down-z['times'])*1000>=-1e-6 if flag else np.ones(len(z['times']),bool)
            a=z['task'][valid].astype('float64')-z['cold'][valid]
            b=z['task_reverse'][valid].astype('float64')-z['cold'][valid]
            den=np.linalg.norm(a)*np.linalg.norm(b)
            records.append(dict(shot=sid,flag=flag,cosine=float(a@b/den) if den>0 else None,
                mean_delta=float(a.mean()),max_abs_delta=float(np.abs(a).max())))
    groups={}
    for name,sub in (('all340',records),('D',[r for r in records if r['flag']]),('S',[r for r in records if not r['flag']])):
        groups[name]=dict(n=len(sub),increased_mean_risk=sum(r['mean_delta']>0 for r in sub),
            mean_risk_delta=float(np.mean([r['mean_delta'] for r in sub])),
            median_task_reverse_cosine=float(np.median([r['cosine'] for r in sub if r['cosine'] is not None])),
            median_shot_max_change=float(np.median([r['max_abs_delta'] for r in sub])))
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    base,_=load_parent(DEFAULT_PARENT);base.cuda()
    training=Path(p['source_root']);sp=read(training,'protocol.json')
    inv={r['shot']:r for r in read(CACHE,'inventory.json')}
    x,a,flags,windows=source_windows(sp['val_ids'],inv,None)
    if windows!=read(training,'val_windows.json'):raise ValueError('source validation windows differ')
    errors=[];kls=[]
    with torch.no_grad():
        for lo in range(0,len(x),32):
            xx,aa=x[lo:lo+32].cuda(),a[lo:lo+32].cuda()
            h,v=tokens(base,xx,aa,base.center.new_zeros(128));z=readout(base,h,v,None)
            mx,ma=masked(xx,aa);mh,mv=tokens(base,mx,ma,base.center.new_zeros(128))
            # A no-training baseline: replace each masked signal token by the
            # mean of observed signal tokens in this SAME arrived window.
            # Keep masked-view CLS and observed tokens. No shot outcome,
            # future window or source-fitted parameter enters this baseline.
            count=mv[:,:-1].sum(1,keepdim=True)
            mean=(mh[:,:-1]*mv[:,:-1,None]).sum(1)/count.clamp_min(1)
            filled=torch.where(mv[...,None],mh,mean[:,None])
            target=h.repeat_interleave(2,0);valid=v.repeat_interleave(2,0)
            errors.extend(latent_loss(filled,target,valid).reshape(-1,2).mean(-1).cpu().tolist())
            kls.extend(risk_kl(readout(base,filled,valid,None),z.repeat_interleave(2)).reshape(-1,2).mean(-1).cpu().tolist())
    mean_fill={name:dict(mse=float(np.asarray(errors)[mask].mean()),kl=float(np.asarray(kls)[mask].mean()))
               for name,mask in (('all',np.ones(len(flags),bool)),('D',flags),('S',~flags))}
    result=dict(scope='POSTHOC_SPECIFICITY_NOT_A_NEW_ONLINE_METHOD',stream_complete_sha256=sha(root/'complete.json'),
        preterminal_valid=True,groups=groups,records=records,source_val_mean_fill=mean_fill,
        source_val_ids=sp['val_ids'],source_validation_windows_sha256=sha(training/'val_windows.json'),
        code_sha256=sha(Path(__file__)),online_policy_changed=False)
    if (root/'diagnosis.json').exists():raise FileExistsError(root/'diagnosis.json')
    _durable_json(root/'diagnosis.json',result);print(groups,flush=True);print(mean_fill,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
