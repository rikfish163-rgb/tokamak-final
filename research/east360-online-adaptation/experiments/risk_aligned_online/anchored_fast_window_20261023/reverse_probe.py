"""Post-hoc magnitude-matched reversal; no fitting and no new shot metrics."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .run import (READ,sha,io23,make_parent,models_for_calibration,REFERENCE,read_signal,
    transform,state_cpu,assert_unchanged,DEFAULT_PARENT,PARENT_SHA256,_durable_json)
from .model import AnchoredAdapter,centered_step
from ..matured_negative_20261023.learning import tensors


def match_write_norm(candidate,reference):
    a=candidate.square().sum((1,2)).sqrt(); b=reference.square().sum((1,2)).sqrt()
    if torch.any((a==0)&(b>0)):
        raise ValueError('a zero candidate has no direction to match a nonzero reference')
    denominator=torch.where(a>0,a,torch.ones_like(a))
    return candidate*(b/denominator)[:,None,None]


@torch.no_grad()
def main(root):
    target=root/'reverse_probe.json'
    if target.exists() or (root/'reverse_probe.npz').exists(): raise FileExistsError(target)
    p=READ(root/'protocol.json'); verified=READ(root/'verification.json')
    assert verified['completed_sha256']==sha(root/'completed.json')
    torch.set_num_threads(2);torch.manual_seed(131);base,_=make_parent();parent=models_for_calibration(base)['mapped_fixed']
    net=AnchoredAdapter(copy.deepcopy(parent.base),rule='normalized').cuda().eval()
    net.layer.load_state_dict(torch.load(root/'layers.pt',map_location='cpu',weights_only=True)['normalized'])
    snapshot=state_cpu(net); mapping=READ(REFERENCE/'input_map.json'); collected=[]
    def deny(*args,**kwargs): raise AssertionError('signal-only probe may not read outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny):
        for index,sid in enumerate(p['shot_ids']):
            s=read_signal(sid); signal=transform({k:s[k] for k in ('values','available')},mapping)
            ends=np.arange(127,len(s['times']),128)
            for lo in range(0,len(ends),32):
                use=ends[lo:lo+32]; x,m=tensors(signal,use,'cuda');h,valid=net.tokens(x,m)
                q,k,v=net.layer.coordinates(h); observed=valid.clone();observed[:,-1]=False
                a,ai=centered_step(k,v,observed,net.layer.origin,net.layer.eta,rule='normalized',reverse=False)
                r,ri=centered_step(k,v,observed,net.layer.origin,net.layer.eta,rule='normalized',reverse=True)
                matched=match_write_norm(r,a)
                def score(delta):
                    hh=h+net.layer.output(q@delta)*valid[...,None]
                    return net.from_tokens(hh,valid,mode='bypass')[0].sigmoid()
                actual=score(a); exact=net.from_tokens(h,valid,mode='on')[0].sigmoid()
                torch.testing.assert_close(actual,exact,atol=0,rtol=0)
                an=a.square().sum((1,2)).sqrt(); rn=r.square().sum((1,2)).sqrt()
                torch.testing.assert_close(matched.square().sum((1,2)).sqrt(),an,atol=1e-6,rtol=1e-5)
                parent_score=net.from_tokens(h,valid,mode='off')[0].sigmoid()
                values=dict(shot=np.full(len(use),sid),endpoint=use,actual_norm=an.cpu().numpy(),
                    reverse_norm=rn.cpu().numpy(),parent=parent_score.cpu().numpy(),actual=actual.cpu().numpy(),
                    reverse=score(r).cpu().numpy(),matched_reverse=score(matched).cpu().numpy(),
                    cosine=(a*r).sum((1,2)).div((an*rn).clamp_min(torch.finfo(an.dtype).tiny)).cpu().numpy())
                collected.append(values)
            if (index+1)%90==0: print(dict(stage='reverse_probe',shots=index+1),flush=True)
    z={k:np.concatenate([r[k] for r in collected]) for k in collected[0]}
    assert all(np.isfinite(v).all() for v in z.values())
    use=z['actual_norm']>0; ratios=z['reverse_norm'][use]/z['actual_norm'][use]
    by_shot=[float(np.median(z['reverse_norm'][(z['shot']==sid)&use]/z['actual_norm'][(z['shot']==sid)&use]))
             for sid in np.unique(z['shot']) if np.any((z['shot']==sid)&use)]
    summary=dict(scope='POST_HOC_DIAGNOSTIC_ONLY; every128th full past window; no outcomes, no fitting, not TPR/FPR',
        windows=len(z['shot']),shots=len(np.unique(z['shot'])),zero_reference_writes=int((~use).sum()),
        reversal_write_norm_ratio_quantiles=np.quantile(ratios,[.1,.5,.9]).tolist(),
        median_shot_median_write_ratio=float(np.median(by_shot)),median_cosine=float(np.median(z['cosine'][use])),
        score_summary={k:dict(median=float(np.median(z[k])),fraction_sampled_points_over_threshold=float(np.mean(z[k]>=p['threshold'])))
            for k in ('parent','actual','reverse','matched_reverse')},
        median_abs_change_after_matching=float(np.median(np.abs(z['matched_reverse']-z['actual']))),
        completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),source_unchanged=True)
    assert_unchanged(net,snapshot); assert sha(DEFAULT_PARENT)==PARENT_SHA256
    np.savez_compressed(root/'reverse_probe.npz',**z)
    summary['arrays_sha256']=sha(root/'reverse_probe.npz'); _durable_json(target,summary); print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
