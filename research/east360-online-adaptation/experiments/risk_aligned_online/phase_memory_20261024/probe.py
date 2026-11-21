"""A frozen-representation diagnostic, NEVER appended to risk predictions."""
import argparse
from pathlib import Path
import time

import numpy as np

from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import auc,summarize
from .run import READ,MODES,io23

METHODS=('ordered','unpaired','shuffled_order')
PLAN=io23.ROOT/'docs/research/source_event_rebuild_20261023/阶段信息线性探针_诊断延伸约定_20261024.md'


def fit_direction(early,late,shots,method):
    if (early.shape!=late.shape or early.ndim!=2 or shots.shape!=(len(early),)
            or not np.isfinite(early).all() or not np.isfinite(late).all() or method not in METHODS):
        raise ValueError('finite matching historical phase arrays required')
    groups=[np.flatnonzero(shots==sid) for sid in np.unique(shots)]
    if len(groups)!=3 or len({len(g) for g in groups})!=1 or len(groups[0])%2:
        raise ValueError('three historical shots with equal even pair counts required')
    permutation=np.arange(len(shots));targets=np.ones(len(shots))
    if method=='unpaired':
        for group,other in zip(groups,groups[1:]+groups[:1]):
            permutation[group]=other
    elif method=='shuffled_order':
        rng=np.random.default_rng(131)
        for group in groups:
            targets[rng.permutation(group)[:len(group)//2]]=-1
    delta=late-early[permutation]
    ridge=float(np.mean((delta*delta).sum(1)))
    if not np.isfinite(ridge) or ridge<=0:
        raise ValueError('nonzero historical phase movement required')
    a=delta/np.sqrt(len(delta));b=targets/np.sqrt(len(delta))
    coef=a.T@np.linalg.solve(a@a.T+ridge*np.eye(len(a)),b)
    residual=a.T@(a@coef-b)+ridge*coef
    relative=float(np.linalg.norm(residual)/np.linalg.norm(a.T@b))
    if relative>1e-10:
        raise ArithmeticError('probe solve violates normal equation')
    return coef,dict(ridge=ridge,relative_normal_residual=relative,
        early_indexes=permutation.tolist(),targets=targets.tolist(),coefficient_norm=float(np.linalg.norm(coef)),
        training_mse=float(np.mean((delta@coef-targets)**2)))


def main(screen):
    root=screen/'linear_probe'
    if root.exists():
        raise FileExistsError(root)
    assert READ(screen/'verification.json')['completed_sha256']==sha(screen/'completed.json')
    with np.load(screen/'embeddings.npz') as z:
        data={k:z[k].copy() for k in z.files}
    shots=data['shots'];labels=data['labels'];folds=READ(screen/'folds.json')['risk']['phase']
    start=time.monotonic();root.mkdir()
    _durable_json(root/'protocol.json',dict(scope='POST_HOC_LINEAR_DIAGNOSTIC; not online learning, not final risk score',
        source_screen_sha256=sha(screen/'completed.json'),embedding_sha256=sha(screen/'embeddings.npz'),
        plan_sha256=sha(PLAN),code_sha256=sha(Path(__file__)),methods=METHODS,modes=MODES,
        ridge='mean squared historical displacement norm; no hyperparameter search',
        train_shots_per_fold=3,train_pairs_per_fold=48,stream360_read=False,source_risk_updated=False,timeout_seconds=600))
    results={};details={};fits={};coefs={}
    for mode in MODES:
        if time.monotonic()-start>600:
            raise TimeoutError('declared10minutes exceeded')
        e=data[mode+'_early'];l=data[mode+'_late'];qe=data[mode+'_query_early'];ql=data[mode+'_query_late']
        rows={method:[] for method in METHODS}
        for dsid in np.unique(shots[labels==1]):
            keep=(labels==1)&(shots!=dsid)
            for method in METHODS:
                w,r=fit_direction(e[keep],l[keep],shots[keep],method)
                key=f'{mode}_{dsid}_{method}';coefs[key]=w
                fits[key]=dict(r,held_disruptive=int(dsid),training_shots=np.unique(shots[keep]).tolist(),
                               training_rows=np.flatnonzero(keep).tolist())
                for f in folds:
                    if f['held_shots'][0]!=dsid:
                        continue
                    li=np.array(f['late_rows']);ei=np.array(f['early_unique_rows']);si=np.array(f['safe_rows'])
                    a=ql[li]@w;b=qe[ei]@w;c=ql[si]@w
                    diff=a-b[np.array(f['pair_to_unique'])]
                    rows[method].append(dict(held_shots=f['held_shots'],late_rows=li.tolist(),early_unique_rows=ei.tolist(),safe_rows=si.tolist(),
                        pair_to_unique=f['pair_to_unique'],late_scores=a.tolist(),early_scores=b.tolist(),safe_scores=c.tolist(),
                        phase_auc=auc(a,b),late_safe_auc=auc(a,c),paired_order=float(np.mean((diff>0)+.5*(diff==0)))))
        results[mode]={k:summarize(v) for k,v in rows.items()};details[mode]=rows
        print(dict(mode=mode,metrics={k:{m:v[m] for m in ('phase_auc','paired_order','late_safe_auc')} for k,v in results[mode].items()}),flush=True)
    np.savez_compressed(root/'coefficients.npz',**coefs)
    _durable_json(root/'fits.json',fits);_durable_json(root/'results.json',results);_durable_json(root/'folds.json',details)
    done=dict(status='completed',fits=len(fits),seconds=time.monotonic()-start,
        eligible_for_neural_trial=False,candidate_promoted=False,scientific_goal_achieved=False,
        reason='diagnostic only; original distance-phase gate remains failed',
        artifact_sha256={f.name:sha(f) for f in root.iterdir() if f.is_file()})
    _durable_json(root/'completed.json',done);print(done,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--screen',type=Path,required=True)
    main(parser.parse_args().screen)
