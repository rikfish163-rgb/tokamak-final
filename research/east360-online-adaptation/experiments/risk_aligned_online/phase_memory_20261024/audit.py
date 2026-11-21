"""Independent AUC/phase arithmetic; exact replay from archived input windows."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import READ,WITHIN,MODES,METHODS,load_metric_bank,representations,evaluate,DEFAULT_PARENT,PARENT_SHA256


def main(root):
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    for f,h in done['artifact_sha256'].items():
        assert sha(root/f)==h,f
    for f,h in p['input_sha256'].items():
        assert sha(Path(f))==h,f
    details=READ(root/'folds.json');results=READ(root/'results.json')
    with np.load(root/'embeddings.npz') as z:
        data={k:z[k].copy() for k in z.files}
    shots=data['shots'];labels=data['labels'];ends=data['early_ends']
    assert len(set(zip(shots[labels==1].tolist(),ends[labels==1].tolist())))==done['unique_early_windows']==18
    checks=0;phase_checks=0;diagnostic={};geometry={}
    for mode in MODES:
        e=data[mode+'_early'];l=data[mode+'_late'];qe=data[mode+'_query_early'];ql=data[mode+'_query_late']
        vectors=np.stack([(l-e)[shots==sid].mean(0) for sid in np.unique(shots[labels==1])])
        lengths=np.linalg.norm(vectors,axis=1);v=vectors/lengths[:,None]
        geometry[mode]=dict(shots=np.unique(shots[labels==1]).tolist(),mean_change_norm=lengths.tolist(),cosine=(v@v.T).tolist())
        phase_by_shot={}
        for method in METHODS:
            assert len(details[mode][method])==64
            for r in details[mode][method]:
                dsid,ssid=r['held_shots'];keep=~np.isin(shots,[dsid,ssid]);pos=keep&(labels==1)
                assert r['history_shots']==sorted(map(int,np.unique(shots[keep])))
                assert len(r['history_shots'])==18 and not set(r['held_shots'])&set(r['history_shots'])
                li=np.array(r['late_rows']);ei=np.array(r['early_unique_rows']);si=np.array(r['safe_rows'])
                assert (shots[li]==dsid).all() and (shots[ei]==dsid).all() and (shots[si]==ssid).all()
                assert len(set(ends[ei]))==len(ei)
                a,b,c=[np.array(r[k]) for k in ('late_scores','early_scores','safe_scores')]
                for negative,key in ((b,'phase_auc'),(c,'late_safe_auc')):
                    yy=np.r_[np.ones(len(a)),np.zeros(len(negative))]
                    # Pair-count AUC and sklearn's trapezoidal ROC integrate
                    # in different orders. All3840 actual comparisons differ
                    # by <=1.11e-16; allow one float64 epsilon, not metric drift.
                    np.testing.assert_allclose(roc_auc_score(yy,np.r_[a,negative]),r[key],
                                               rtol=0,atol=np.finfo('float64').eps)
                    checks+=1
                diff=a-b[np.array(r['pair_to_unique'])]
                assert float(np.mean((diff>0)+.5*(diff==0)))==r['paired_order'];checks+=1
                if method=='phase':
                    q=np.concatenate([ql[li],qe[ei],ql[si]])
                    separation=((e[pos]-l[pos])**2).sum(1)
                    tau=float(np.median(separation[separation>1e-12]));assert tau==r['temperature']
                    indexes=np.flatnonzero(pos);de=((q[:,None]-e[pos][None])**2).sum(-1)
                    dl=((q[:,None]-l[pos][None])**2).sum(-1)
                    chosen=np.stack([np.flatnonzero(shots[pos]==sid)[np.minimum(de[:,shots[pos]==sid],dl[:,shots[pos]==sid]).argmin(1)]
                                     for sid in np.unique(shots[pos])],1)
                    np.testing.assert_array_equal(indexes[chosen],r['selected_original_rows'])
                    early_d=np.take_along_axis(de,chosen,1);late_d=np.take_along_axis(dl,chosen,1)
                    affinity=np.exp(-np.logaddexp(0.,-(early_d-late_d)/tau))
                    logits=-np.minimum(early_d,late_d)/tau
                    w=np.exp(logits-logits.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
                    np.testing.assert_allclose(w,r['weights'],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose((w*affinity).sum(1),np.r_[a,b,c],atol=1e-12,rtol=1e-12)
                    phase_checks+=1
                    # phase uses no safe memory, so these values must be the
                    # same across all16 safety partners; record each D once.
                    item=dict(late_mean=float(a.mean()),early_mean=float(b.mean()),
                        history_shots=np.unique(shots[pos]).tolist(),
                        mean_vote_late=affinity[:len(a)].mean(0).tolist(),
                        mean_vote_early=affinity[len(a):len(a)+len(b)].mean(0).tolist(),
                        mean_weight_late=w[:len(a)].mean(0).tolist(),mean_weight_early=w[len(a):len(a)+len(b)].mean(0).tolist())
                    if str(dsid) in phase_by_shot:
                        assert phase_by_shot[str(dsid)]==item
                    else:
                        phase_by_shot[str(dsid)]=item
        diagnostic[mode]=phase_by_shot
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    def deny(*args,**kwargs):
        raise AssertionError('no fresh shot or outcome during screen replay')
    with patch.object(io23,'read_outcome',side_effect=deny),patch.object(io23,'read_signals',side_effect=deny):
        bank=load_metric_bank()
        with np.load(WITHIN/'pairs.npz') as z:
            views=representations(bank,z['early_x'],z['early_mask'])
        for mode,view in views.items():
            for key,array in view.items():
                np.testing.assert_array_equal(array,data[f'{mode}_{key}'])
            metrics,rows=evaluate(view,shots,labels,ends)
            assert metrics==results[mode] and rows==details[mode]
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    verification=dict(completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),
        checked_auc_and_order=checks,independent_phase_recomputations=phase_checks,exact_representation_and_fold_replays=len(MODES),
        historical_unique_early=18,heldout_identity_leak=False,source_risk_unchanged=True,
        scope='retrieval diagnostic only; fixed earlier input map includes all20 historical shots')
    _durable_json(root/'verification.json',verification)
    _durable_json(root/'geometry_diagnostic.json',dict(scope='post-hoc representation geometry, not physical-cause identification',
        phase_votes=diagnostic,mean_stage_change=geometry))
    print(verification,flush=True)
    print(dict(risk_phase_votes=diagnostic['risk'],risk_direction_geometry=geometry['risk']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
