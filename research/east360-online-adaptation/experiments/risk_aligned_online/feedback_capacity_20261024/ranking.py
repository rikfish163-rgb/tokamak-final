"""Post-hoc endpoint ranking and additive attribution; no model selection."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from .run import READ,sha,io23,_durable_json,REPRESENTATIONS,feedback


def weighted_auc(y,score,weight):
    y,s,w=[np.asarray(a) for a in (y,score,weight)]
    if (y.ndim!=1 or s.shape!=y.shape or w.shape!=y.shape or set(np.unique(y))!={0,1}
            or not np.isfinite(s).all() or not np.isfinite(w).all() or (w<=0).any()):
        raise ValueError('finite scores, positive weights, and two classes required')
    order=np.argsort(s,kind='stable');s=s[order];y=y[order];w=w[order]
    starts=np.r_[0,1+np.flatnonzero(np.diff(s))]
    pos=np.add.reduceat(w*(y==1),starts);neg=np.add.reduceat(w*(y==0),starts)
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))


def main(root):
    path=root/'ranking_diagnostic.json'
    if path.exists():raise FileExistsError(path)
    completed=sha(root/'completed.json')
    assert READ(root/'verification.json')['completed_sha256']==completed
    done=READ(root/'completed.json')
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    p=READ(root/'protocol.json');access=READ(root/'evaluation_access.json');folds=[];attributions=[]
    for fold in p['folds']:
        f=fold['fold'];folder=root/f'fold{f}';ys=[];weights=[];values={}
        for r in [a for a in access if a['fold']==f and a['role']=='held']:
            sid=r['shot'];dis=r['disrupted'];down=r['down']
            with np.load(folder/f'held_{sid}_predictions.npz') as z:
                use=(np.round((down-z['times'])*1000,9)<=125)&(np.round((down-z['arrived_times'])*1000,9)>30) if dis else np.ones(len(z['times']),bool)
                n=int(use.sum());ys.extend([int(dis)]*n);weights.extend([1./n]*n)
                for arm in z.files:
                    if arm not in ('times','arrived_times'):values.setdefault(arm,[]).extend(z[arm][use].tolist())
                increments={name:float(np.mean(z[name+'_real'][use]-z[name+'_prior'][use])) for name in REPRESENTATIONS}
            if dis:
                with np.load(folder/f'held_{sid}_inputs.npz') as z:a={k:z[k].copy() for k in z.files}
                for name in REPRESENTATIONS:
                    with np.load(folder/f'{name}_model.npz') as z:model={k:z[k].copy() for k in z.files}
                    # Exact additive arithmetic in this linear probe, not a physical intervention.
                    contribution=(feedback(a['delta'],a['mask'],model)*model['real']).reshape(-1,5,23)[use].sum(1).mean(0)
                    np.testing.assert_allclose(contribution.sum(),increments[name],atol=1e-12,rtol=1e-12)
                    attributions.append(dict(fold=f,shot=sid,representation=name,mean_increment=increments[name],
                        channel_mean_contribution=dict(zip(io23.RAW23,map(float,contribution)))))
        auc={}
        for arm,z in values.items():
            auc[arm]=weighted_auc(ys,z,weights)
            np.testing.assert_allclose(auc[arm],roc_auc_score(ys,z,sample_weight=weights),atol=1e-12,rtol=1e-12)
        folds.append(dict(fold=f,disruption=next(a['shot'] for a in access if a['fold']==f and a['role']=='held' and a['disrupted']),auc=auc))
    result=dict(scope='POST_HOC endpoint ranking, shot-equal within class; NOT shot-level AUC or TPR/FPR',
        completed_sha256=completed,code_sha256=sha(Path(__file__)),folds=folds,
        mean_fold_auc={arm:float(np.mean([r['auc'][arm] for r in folds])) for arm in folds[0]['auc']},
        attributions=attributions,attribution_scope='Additive decomposition of a diagnostic linear score; correlated signals, no causal/physics claim')
    _durable_json(path,result)
    print(dict(fold_auc=folds,mean_fold_auc=result['mean_fold_auc']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
