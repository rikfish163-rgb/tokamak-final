"""Recount source strata and replay all saved finite-step validation values."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..attention_basis_20261029.learning import MatrixAttention
from ..context_memory_20261028.checkpoint import read
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import GROUPS,assemble,paired_risk
from .run import load_split


def run(root):
    if (root/'audit.json').exists():raise FileExistsError(root/'audit.json')
    p=read(root,'protocol.json');done=read(root,'complete.json');v=read(root,'validation.json');training=read(root,'training.json')
    for name,key in (('protocol','protocol'),('validation','validation'),('training','training'),('input_hashes','input_hashes')):
        if sha(root/f'{name}.json')!=done[f'{key}_sha256']:raise ValueError('artifact changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('run implementation changed')
    if (sha(DEFAULT_PARENT)!=PARENT_SHA256 or p['source_sha256']!=PARENT_SHA256
            or not done['complete'] or done['target_or_test_signals_read']!=0
            or sha(io23.ROOT/'QK_MATCHED_HEAD_PLAN.md')!=p['plan_sha256']):raise ValueError('source/plan changed')
    roles=simple_data.source_roles()
    if p['train_ids']!=roles['train'] or p['val_ids']!=roles['val']:raise ValueError('role mismatch')
    if len(training['history'])!=512 or len(training['receipts'])!=6144:raise ValueError('budget mismatch')
    for r in training['receipts']:
        if r['shot'] not in roles['train'] or r['last_target']!=r['count']-1 or not r['count']<=r['query']<r['count']+128:
            raise ValueError('training episode outside role/clock')
    for h in training['history']:
        for arm in ('forecast','matched'):
            if not all(np.isfinite(x) for x in h[arm].values()) or h[arm]['head_gradient_norm']<=0:
                raise ValueError('nonfinite/non-updating training arm')
    hashes=read(root,'input_hashes.json')
    if set(hashes)!=set(map(str,roles['train']+roles['val'])):raise ValueError('input manifest mismatch')
    for sid,h in hashes.items():
        if sha(io23.path_for('J-TEXT',int(sid)))!=h:raise ValueError('raw source changed')
    advanced=simple_data.advanced_times();shots=load_split(roles['val'],{i:advanced[i] for i in roles['val'] if i in advanced})
    indices={s['shot']:i for i,s in enumerate(shots)};expected=[];coverage=[]
    for group in GROUPS:
        for s in shots:
            q=np.arange(len(s['times']));valid=s['valid']&(q>=256)
            if group=='safe':keep=valid&not_disruptive(s)
            elif group=='early':keep=valid&(s['y']<=.5)&s['is_disrupt']
            else:
                keep=valid&(s['y']>.5)&s['is_disrupt']
                if s['is_disrupt']:keep&=np.round((s['down']-s['times'])*1000,9)>10
            eligible=q[keep]
            if len(eligible):
                for end in eligible[np.linspace(0,len(eligible)-1,min(4,len(eligible)),dtype=int)]:
                    expected.append((s['shot'],int(end),group))
    if expected!=[(r['shot'],r['query'],r['group']) for r in v['per_endpoint']['matched_qk']]:raise ValueError('VAL sampling changed')
    for s in shots:
        timely=(s['y']>.5)&s['valid']
        if s['is_disrupt']:timely&=np.round((s['down']-s['times'])*1000,9)>10
        coverage.append(dict(shot=s['shot'],flag=s['is_disrupt'],valid_points=int(s['valid'].sum()),
            timely_positive_points=int(timely.sum()),timely_positive_after256=int(np.sum(timely&(np.arange(len(timely))>=256)))))
    summary={};latency={}
    for arm,rows in v['per_endpoint'].items():
        summary[arm]={};latency[arm]={}
        for g in GROUPS:
            grouped={}
            for r in rows:
                if r['group']==g:grouped.setdefault(r['shot'],[]).append(r)
            differences=[np.mean([r['after']-r['before'] for r in rs]) for rs in grouped.values()]
            measure=float(np.mean(differences));summary[arm][g]=measure
            if abs(measure-v['summary'][arm][g]['mean_risk_change'])>1e-12:raise ValueError('shot-weighted summary differs')
            latency[arm][g]=[]
            for lower in (0,32,64,96):
                chosen=[r for r in rows if r['group']==g and lower<=((r['query']-256)%128)<lower+32]
                latency[arm][g].append(dict(query_lag_range=[lower,lower+31],endpoints=len(chosen),
                    mean_change=float(np.mean([r['after']-r['before'] for r in chosen])) if chosen else None))
        summary[arm]['macro_change']=float(np.mean([summary[arm][g] for g in GROUPS]))
    m=summary['matched_qk']
    accepted=all(m[g]<=0 for g in ('safe','danger')) and all(m['macro_change']<summary[k]['macro_change']-1e-6
        for k in ('original_qk','forecast_qk','matched_reverse','matched_hold'))
    if accepted!=v['necessary_source_screen'] or accepted!=done['necessary_source_screen']:raise ValueError('source screen differs')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    nets={}
    for name in ('original','forecast','matched'):
        base,_=load_parent(DEFAULT_PARENT);base.cuda();nets[name]=MatrixAttention(base)
        if name!='original':
            if sha(root/f'{name}_head.pt')!=done['heads'][name]:raise ValueError('head changed')
            base.forecast_head.load_state_dict(torch.load(root/f'{name}_head.pt',map_location='cuda',weights_only=True))
    arms=dict(original_tokens=('original','tokens','true'),original_qk=('original','qk','true'),
        forecast_qk=('forecast','qk','true'),matched_qk=('matched','qk','true'),
        matched_reverse=('matched','qk','reverse'),matched_hold=('matched','qk','hold'))
    max_error=0.;queries=[(indices[sid],q) for sid,q,_ in expected]
    for lo in range(0,len(queries),12):
        e=assemble(shots,queries[lo:lo+12],'cuda')
        if e['receipts']!=v['receipts'][lo:lo+len(e['y'])]:raise ValueError('feedback target receipts changed')
        for arm,(name,site,mode) in arms.items():
            actual=paired_risk(nets[name],e,site=site,mode=mode)
            for key,t in actual.items():
                if key=='alignment':continue
                saved=np.array([r[key] for r in v['per_endpoint'][arm][lo:lo+len(e['y'])]])
                error=float(np.max(np.abs(t.detach().cpu().numpy()-saved)));max_error=max(max_error,error)
                if error>2e-6:raise ValueError('finite-step validation replay changed')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),source_sha256=PARENT_SHA256,
        all_validation_replayed=True,endpoint_count=len(expected),arm_count=len(arms),max_replay_error=max_error,
        summary=summary,latency_by_endpoint=latency,coverage=coverage,source_screen=accepted,
        code_sha256=sha(Path(__file__)),independent_confirmation=False)
    _durable_json(root/'audit.json',result)
    print(dict(endpoints=len(expected),max_replay_error=max_error,screen=accepted,
        disruptive_val=sum(s['is_disrupt'] for s in shots),
        disruptive_without_timing_opportunity=sum(r['flag'] and r['timely_positive_after256']==0 for r in coverage),
        matched_latency=latency['matched_qk']),flush=True)


def not_disruptive(s):
    return not s['is_disrupt']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
