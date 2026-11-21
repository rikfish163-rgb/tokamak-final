"""Post-result mechanism diagnostic, no model update or performance selection."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from ..matured_negative_20261023.learning import tensors
from .run import (io23,READ,sha,make_parent,models_for_calibration,REFERENCE,
    read_signal,transform,state_cpu,assert_unchanged,_durable_json)
from .model import CurvatureAdapter


def decompose(layer,h,valid):
    _,k,v=layer.coordinates(h);observed=valid.clone();observed[:,-1]=False
    n=observed.sum(-1).clamp_min(1);mask=observed[:,:,None]
    km=(k*mask).sum(1)/n[:,None];vm=(v*mask).sum(1)/n[:,None]
    kc=(k-km[:,None])*mask;vc=(v-vm[:,None])*mask
    error=(k@layer.origin-v)*mask
    full=k.transpose(1,2)@error/n[:,None,None]
    level=km[:,:,None]*(km@layer.origin-vm)[:,None,:]
    varying=kc.transpose(1,2)@(kc@layer.origin-vc)/n[:,None,None]
    torch.testing.assert_close(full,level+varying,atol=3e-6,rtol=3e-5)
    # Zero-energy rows carry no measured variation: keep an explicit count,
    # rather than treating an undefined ratio as positive mechanistic evidence.
    ke=(k.square()*mask).sum((1,2));ve=(v.square()*mask).sum((1,2));gn=full.square().sum((1,2)).sqrt()
    eligible=(ke>0)&(ve>0)&(gn>0)&(n>=2)
    return dict(eligible=eligible,
        key_variation_fraction=kc.square().sum((1,2))/torch.where(ke>0,ke,1),
        value_variation_fraction=vc.square().sum((1,2))/torch.where(ve>0,ve,1),
        varying_gradient_ratio=varying.square().sum((1,2)).sqrt()/torch.where(gn>0,gn,1),
        level_gradient_ratio=level.square().sum((1,2)).sqrt()/torch.where(gn>0,gn,1))


def main(root):
    torch.set_num_threads(2);torch.manual_seed(131)
    p=READ(root/'protocol.json');verification=READ(root/'verification.json')
    assert verification['completed_sha256']==sha(root/'completed.json')
    source,_=make_parent();parent=models_for_calibration(source)['mapped_fixed']
    model=CurvatureAdapter(copy.deepcopy(parent.base)).cuda().eval()
    state=torch.load(root/'layers.pt',map_location='cpu',weights_only=True)
    model.layer.load_state_dict(state['adaptive']);before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');samples=[];data={};eligible=[]
    def deny(*args,**kwargs):raise AssertionError('no outcomes in latent mechanism probe')
    with patch.object(io23,'read_outcome',side_effect=deny),torch.no_grad():
        for sid in p['shot_ids']:
            s=read_signal(sid);signal=transform({k:s[k] for k in ('values','available')},mapping)
            ends=np.arange(127,len(s['values']),128)
            for lo in range(0,len(ends),32):
                batch=ends[lo:lo+32];x,m=tensors(signal,batch,'cuda');h,valid=model.tokens(x,m)
                rows=decompose(model.layer,h,valid)
                eligible.append(rows.pop('eligible').cpu().numpy())
                for k,v in rows.items():data.setdefault(k,[]).append(v.cpu().numpy())
                samples.extend(dict(shot=sid,end=int(e)) for e in batch)
    mask=np.concatenate(eligible);arrays={k:np.concatenate(v) for k,v in data.items()}
    path=root/'latent_probe.npz';np.savez_compressed(path,eligible=mask,**arrays)
    assert_unchanged(model,before)
    summary=dict(scope='POST_RESULT_DIAGNOSTIC; every128th fully observed endpoint; no labels, no fitting',
        completed_sha256=sha(root/'completed.json'),probe_sha256=sha(Path(__file__)),data_sha256=sha(path),
        sampled_endpoints=len(samples),eligible=int(mask.sum()),sampling=samples,
        quantiles={k:{str(q):float(np.quantile(v[mask],q)) for q in (.1,.5,.9,.99)} for k,v in arrays.items()})
    _durable_json(root/'latent_probe.json',summary)
    print({k:v for k,v in summary.items() if k!='sampling'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
