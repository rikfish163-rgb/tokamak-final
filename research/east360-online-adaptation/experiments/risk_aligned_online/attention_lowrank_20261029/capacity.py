"""Fixed398-point label-informed diagnosis, strictly separate from online use."""
import argparse
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..cold_capacity_20261029.probe import optimize,summarize
from ..context_memory_20261028.checkpoint import read
from ..delayed_recurrent_memory_20261025.learning import batch
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT,classifier_state,assert_classifier_unchanged
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .model import AttentionAdapter


def run(output,previous):
    if output.exists() or output.absolute().parent!=io23.ROOT/'probe_outputs':raise ValueError('new worktree output required')
    p=read(previous,'protocol.json');audit=read(previous,'audit.json');done=read(previous,'complete.json')
    if not audit['complete'] or audit['complete_sha256']!=sha(previous/'complete.json') or done['records_sha256']!=sha(previous/'records.json'):
        raise ValueError('audited previous point experiment required')
    ref=Path(p['reference'])
    for path,h in p['reference_hashes'].items():
        if sha(ref/path)!=h:raise ValueError('cold reference changed')
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base,parent=load_parent(DEFAULT_PARENT);base.cuda();net=AttentionAdapter(base);original=classifier_state(base)
    frozen={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    slow=torch.load(ref/'models/cold.pt',map_location='cpu',weights_only=True)['slow'].cuda()
    mapping=read(ref,'input_map.json');hashes=read(ref,'input_hashes.json')
    data=read(previous,'records.json');rows=data['records'];xs=[];masks=[]
    start=time.monotonic();output.mkdir()
    files={Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('experiments.risk_aligned_online.') and getattr(m,'__file__',None)}
    protocol=dict(scope='POSTHOC_LABEL_INFORMED_ATTENTION_CAPACITY_NOT_ONLINE',previous=str(previous),
        previous_complete_sha256=sha(previous/'complete.json'),previous_records_sha256=sha(previous/'records.json'),
        previous_audit_sha256=sha(previous/'audit.json'),reference=str(ref),reference_hashes=p['reference_hashes'],
        order=p['order'],threshold=parent['threshold'],modes=['qk','v'],radii=[1,4],steps=64,step=.1,dimensions=512,
        plan_sha256=sha(io23.ROOT/'ATTENTION_LOWRANK_PLAN.md'),code_sha256={str(f.relative_to(io23.ROOT)):sha(f) for f in files},
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    _durable_json(output/'protocol.json',protocol)
    with zipfile.ZipFile(output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):z.write(f,str(f.relative_to(io23.ROOT)))
        z.write(io23.ROOT/'ATTENTION_LOWRANK_PLAN.md','ATTENTION_LOWRANK_PLAN.md')
    for sid in p['order'][20:]:
        if sha(io23.path_for('EAST',sid))!=hashes[str(sid)]:raise ValueError('raw changed')
        raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        ends=[r['end'] for r in rows if r['shot']==sid]
        if ends:
            sig=transform({k:raw[k] for k in ('values','available')},mapping)
            x,a=batch(sig,ends,'cpu');xs.append(x);masks.append(a)
    x=torch.cat(xs);a=torch.cat(masks);witnesses={};groups={}
    for mode in protocol['modes']:
        target=[{k:v for k,v in r.items() if not k.startswith('radius')} for r in rows]
        for radius in protocol['radii']:
            states=[];hist=[];norms=[];steps=[];error=0.
            for lo in range(0,len(x),32):
                xx=x[lo:lo+32].cuda();aa=a[lo:lo+32].cuda();rr=target[lo:lo+32]
                signs=xx.new_tensor([1 if r['flag'] else -1 for r in rr])
                result=optimize(lambda s:net(xx,aa,s,slow,mode,forecast=False)['logit'],xx.new_zeros(len(xx),512),signs,
                    radius=radius,steps=64,step=.1)
                with torch.no_grad():best=net(xx,aa,result['state'],slow,mode,forecast=False)['logit']
                ds=net.deltas(result['state'],mode)
                delta_norm=sum(v.square().sum((1,2)) for v in ds.values()).sqrt()
                for i,r in enumerate(rr):
                    z=float(result['logits'][0,i]);p0=float(result['logits'][0,i].sigmoid())
                    if abs(p0-r['sealed_probability'])>2e-6 or (p0>=parent['threshold'])!=(r['sealed_probability']>=parent['threshold']):
                        raise ValueError('zero attention state differs from sealed cold curve')
                    k=int(result['best_step'][i]);error=max(error,abs(float(best[i])-float(result['logits'][k,i])))
                    r['initial_logit']=z
                    r[f'radius{radius}']=dict(best_logit=float(best[i]),best_probability=float(best[i].sigmoid()),
                        best_step=k,state_norm=float(result['state'][i].norm()),matrix_delta_frobenius=float(delta_norm[i]))
                states.append(result['state'].cpu());hist.append(result['logits'].cpu())
                norms.append(result['norms'].cpu());steps.append(result['best_step'].cpu())
            if error>3e-6:raise ValueError('best witness does not replay')
            key=f'{mode}_radius{radius}'
            witnesses[key+'_state']=torch.cat(states).numpy();witnesses[key+'_logits']=torch.cat(hist,1).numpy()
            witnesses[key+'_norms']=torch.cat(norms,1).numpy();witnesses[key+'_steps']=torch.cat(steps).numpy()
            groups[key]=dict(summarize(target,parent['threshold'],radius),max_replay_error=error)
            print(dict(mode=mode,radius=radius,seconds=time.monotonic()-start,groups=groups[key]),flush=True)
        _durable_json(output/f'{mode}_records.json',target)
    with (output/'witnesses.npz').open('xb') as f:np.savez_compressed(f,**witnesses)
    assert_classifier_unchanged(base,original)
    if not all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in net.state_dict().items()):raise ValueError('persistent parameters changed')
    for path,h in protocol['code_sha256'].items():
        if sha(io23.ROOT/path)!=h:raise ValueError('code changed during run')
    result=dict(complete=True,scope=protocol['scope'],groups=groups,points=len(rows),shots=len(data['coverage']),
        seconds=time.monotonic()-start,scientific_goal_achieved=False,
        artifacts={n:sha(output/n) for n in ('protocol.json','qk_records.json','v_records.json','witnesses.npz')})
    _durable_json(output/'complete.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();run(a.output,a.previous)
