"""Re-evaluate saved capacity witnesses; never turns them into online results."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .probe import logits


def run(root):
    p=read(root,'protocol.json'); done=read(root,'complete.json'); data=read(root,'records.json')
    if not done['complete'] or done['scope']!='POSTHOC_LABEL_INFORMED_POINT_DIAGNOSTIC':raise ValueError('diagnostic only')
    for name in ('protocol','records','witnesses'):
        suffix='npz' if name=='witnesses' else 'json'
        if sha(root/f'{name}.{suffix}')!=done[f'{name}_sha256']:raise ValueError('artifact changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('implementation changed')
    previous=Path(p['previous']); reference=Path(p['reference'])
    for name in ('complete','audit','per_shot'):
        if sha(previous/f'{name}.json')!=p[f'previous_{name}_sha256']:raise ValueError('reference changed')
    for path,h in p['reference_hashes'].items():
        if sha(reference/path)!=h:raise ValueError('cold parent changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();threshold=parent['threshold']
    slow=torch.load(reference/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    mapping=read(reference,'input_map.json');hashes=read(reference,'input_hashes.json')
    rows=data['records'];covered=data['coverage'];xs=[];masks=[];early=[];post_fp=[];post_missed=[]
    if [r['shot'] for r in covered]!=p['order'][20:] or len({(r['shot'],r['end']) for r in rows})!=len(rows):
        raise ValueError('cohort coverage or duplicate point')
    for index,cover in enumerate(covered,20):
        sid=cover['shot'];rr=[r for r in rows if r['shot']==sid]
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        flag,down=io23.read_outcome('EAST',sid)
        curve=previous/'shots'/f'{index+1:04d}_{sid}.npz'
        seal=read(previous/'shots',f'{index+1:04d}_{sid}_seal.json')
        if sha(curve)!=seal['sha256']:raise ValueError('curve changed')
        with np.load(curve) as z:prob=z['cold'];times=z['times']
        if not np.array_equal(times,raw['times']):raise ValueError('times mismatch')
        lead=np.round((down-times)*1000,9) if flag else None
        valid=lead>=-1e-6 if flag else np.ones(len(times),bool)
        alarm=bool(((prob>=threshold)&valid).any())
        timely=bool(flag and ((prob>=threshold)&(lead>30)).any())
        prefix_alarm=bool(not flag and (prob[:128]>=threshold).any())
        if alarm!=cover['cold_alarm'] or timely!=cover['cold_timely'] or flag!=cover['flag'] or prefix_alarm!=cover['early_alarm']:
            raise ValueError('whole-shot baseline mismatch')
        if prefix_alarm:early.append(sid)
        if flag and not timely:post_missed.append(sid)
        if not flag and alarm:post_fp.append(sid)
        eligible=np.arange(len(times))>=128
        selection=[('timely_peak',eligible&(lead>30)),('positive_window_peak',eligible&(lead>30)&(lead<=125))] if flag else [('safe_peak',eligible)]
        expected={}
        for name,ok in selection:
            ix=np.flatnonzero(ok)
            if len(ix):expected.setdefault(int(ix[prob[ix].argmax()]),[]).append(name)
        if [(r['end'],r['reasons']) for r in rr]!=sorted(expected.items()) or cover['points']!=len(rr):
            raise ValueError('endpoint selection mismatch')
        for r in rr:
            if r['flag']!=flag or r['sealed_probability']!=float(prob[r['end']]) or r['reference_curve_sha256']!=seal['sha256']:
                raise ValueError('point provenance mismatch')
        if rr:
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            x,a=batch(sig,[r['end'] for r in rr],'cpu');xs.append(x);masks.append(a)
    x=torch.cat(xs);a=torch.cat(masks);signs=np.array([1 if r['flag'] else -1 for r in rows]);summary={}
    with np.load(root/'witnesses.npz') as w:
        for radius in p['radii']:
            key=f'radius{radius}';state=w[key+'_state'];history=w[key+'_logits'];norms=w[key+'_norms'];steps=w[key+'_best_step']
            if (state.shape!=(len(rows),128) or history.shape!=(65,len(rows)) or not np.isfinite(history).all()
                    or not np.isfinite(state).all() or not np.isfinite(norms).all()
                    or (norms>radius+2e-6).any() or (np.linalg.norm(state,axis=1)>radius+2e-6).any()
                    or not np.array_equal(steps,np.argmax(history*signs[None],axis=0))):
                raise ValueError('invalid optimization witness')
            probabilities=[];max_error=0.
            for lo in range(0,len(rows),32):
                with torch.no_grad():
                    best=logits(base,x[lo:lo+32].cuda(),a[lo:lo+32].cuda(),torch.from_numpy(state[lo:lo+32]).cuda(),slow).cpu().numpy()
                    zero=logits(base,x[lo:lo+32].cuda(),a[lo:lo+32].cuda(),torch.zeros(len(best),128,device='cuda'),slow).cpu().numpy()
                if not np.allclose(zero,history[0,lo:lo+len(best)],atol=3e-6,rtol=0):raise ValueError('zero state mismatch')
                recorded=np.array([r[key]['best_logit'] for r in rows[lo:lo+len(best)]])
                max_error=max(max_error,float(np.max(np.abs(best-recorded))))
                if not np.allclose(best,history[steps[lo:lo+len(best)],np.arange(lo,lo+len(best))],atol=3e-6,rtol=0):raise ValueError('best trace mismatch')
                # Match the live float32 sigmoid, not a different NumPy exp implementation.
                probabilities.extend(torch.from_numpy(best).sigmoid().tolist())
            reached=sorted({r['shot'] for r,pb in zip(rows,probabilities) if r['shot'] in post_missed and pb>=threshold})
            safe_below=[r['shot'] for r,pb in zip(rows,probabilities) if r['shot'] in post_fp and pb<threshold]
            if max_error>3e-6 or reached!=done['groups'][key]['missed_shots_with_timely_point_crossing'] or safe_below!=done['groups'][key]['safe_fp_peak_below_threshold']:
                raise ValueError('replayed diagnosis mismatch')
            summary[key]=dict(missed_point_crossings=len(reached),fp_peak_reductions=len(safe_below),max_replay_logit_error=max_error)
    if early!=done['early_fp_shots'] or len(rows)!=done['points']:raise ValueError('summary mismatch')
    result=dict(complete=True,complete_sha256=sha(root/'complete.json'),shots=len(covered),points=len(rows),
        postcold_missed=len(post_missed),postcold_fp=len(post_fp),early_fp=early,groups=summary,
        audit_code_sha256=sha(Path(__file__)),online_success=False)
    _durable_json(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
