"""Matched-point oracle diagnosis of a later SIGNAL-token adaptation site."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..latent_completion_20261028.model import tokens,readout
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .probe import optimize,summarize


def adapt_signals(h,state):
    gain,bias=state.chunk(2,-1)
    signal=h[:,:-1]*(1+.1*gain[:,None].tanh())+.1*bias[:,None].tanh()
    return torch.cat((signal,h[:,-1:]),1)


def late_logits(base,x,available,state,slow):
    h,v=tokens(base,x,available,torch.zeros_like(state))
    return readout(base,adapt_signals(h,state),v,slow)


def run(root):
    if (root/'late_site.json').exists():raise FileExistsError(root/'late_site.json')
    p=read(root,'protocol.json');a=read(root,'audit.json');d=read(root,'records.json')
    if not a['complete'] or a['complete_sha256']!=sha(root/'complete.json'):raise ValueError('audited parent point experiment required')
    done=read(root,'complete.json')
    if sha(root/'records.json')!=done['records_sha256']:raise ValueError('point records changed')
    for path,h in p['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('parent implementation changed')
    ref=Path(p['reference']);mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json')
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold model changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();original=classifier_state(base)
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda();slow_before=slow.clone()
    rows=d['records'];xs=[];masks=[];start=time.monotonic()
    code_hash=sha(Path(__file__));plan_hash=sha(io23.ROOT/'COLD_SITE_PLAN.md')
    for sid in p['order'][20:]:
        selected=[r['end'] for r in rows if r['shot']==sid]
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('input changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        if selected:
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            x,m=batch(sig,selected,'cpu');xs.append(x);masks.append(m)
    x=torch.cat(xs);m=torch.cat(masks);witness={};max_error=0.
    for radius in p['radii']:
        key=f'radius{radius}';states=[];histories=[];norms=[];steps=[]
        for lo in range(0,len(x),32):
            xx=x[lo:lo+32].cuda();mm=m[lo:lo+32].cuda();rr=rows[lo:lo+32]
            signs=xx.new_tensor([1 if r['flag'] else -1 for r in rr])
            result=optimize(lambda s:late_logits(base,xx,mm,s,slow),xx.new_zeros(len(xx),128),signs,
                radius=radius,steps=p['steps'],step=p['step'])
            with torch.no_grad():best=late_logits(base,xx,mm,result['state'],slow)
            for i,r in enumerate(rr):
                z=float(result['logits'][0,i]);prob=float(result['logits'][0,i].sigmoid())
                if abs(z-r['initial_logit'])>3e-6 or (prob>=parent['threshold'])!=(r['sealed_probability']>=parent['threshold']):
                    raise ValueError('late zero state changed parent')
                k=int(result['best_step'][i]);max_error=max(max_error,abs(float(best[i])-float(result['logits'][k,i])))
                r[key]=dict(best_logit=float(best[i]),best_probability=float(best[i].sigmoid()),
                    best_step=k,state_norm=float(result['state'][i].norm()),initial_gradient_norm=float(result['gradient_norm'][i]))
            states.append(result['state'].cpu());histories.append(result['logits'].cpu())
            norms.append(result['norms'].cpu());steps.append(result['best_step'].cpu())
        witness[key+'_state']=torch.cat(states).numpy();witness[key+'_logits']=torch.cat(histories,1).numpy()
        witness[key+'_norms']=torch.cat(norms,1).numpy();witness[key+'_best_step']=torch.cat(steps).numpy()
    if max_error>3e-6 or not torch.equal(slow,slow_before):raise ValueError('witness or frozen model mismatch')
    assert_classifier_unchanged(base,original)
    if sha(Path(__file__))!=code_hash or sha(io23.ROOT/'COLD_SITE_PLAN.md')!=plan_hash:raise ValueError('code or plan changed')
    with (root/'late_witnesses.npz').open('xb') as f:np.savez_compressed(f,**witness)
    result=dict(complete=True,scope='POSTHOC_LABEL_INFORMED_SITE_COMPARISON_NOT_ONLINE',
        reference_complete_sha256=sha(root/'complete.json'),reference_audit_sha256=sha(root/'audit.json'),
        code_sha256=code_hash,plan_sha256=plan_hash,witnesses_sha256=sha(root/'late_witnesses.npz'),
        records=rows,groups={f'radius{r}':summarize(rows,parent['threshold'],r) for r in p['radii']},
        max_witness_replay_error=max_error,frozen_model_unchanged=True,seconds=time.monotonic()-start,
        warning='same parameter norm is NOT same functional perturbation; no whole-shot online gain or physical-law claim')
    _durable_json(root/'late_site.json',result);print(result['groups'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
