"""POSTHOC score-order diagnostic, explicitly not target-threshold selection.

True outcomes from the evaluated65 are used. Oracle thresholds therefore MUST
NOT be deployed or reported as held-out calibrated performance. This quantifies
what a single threshold could do on the existing traces, without new inference.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..credit_horizon_20261023.run import READ
from ..simple_online import _durable_json
from ..simple_train import sha


def diagnostic(rows,caps=(.1,.15,.2)):
    labels=np.asarray([r['is_disrupt'] for r in rows],bool)
    values=np.asarray([r['max_timely'] if r['is_disrupt'] else r['max_valid'] for r in rows],float)
    if (len({r['shot'] for r in rows})!=len(rows) or not labels.any() or labels.all()
            or not np.isfinite(values).all() or (values<0).any() or (values>1).any()
            or any(not 0<=cap<1 for cap in caps)):
        raise ValueError('unique shots, both classes, finite probabilities and caps in[0,1) required')
    pos,neg=values[labels],values[~labels]
    auc=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())
    assert auc==roc_auc_score(labels,values) or abs(auc-roc_auc_score(labels,values))<1e-14
    candidates=np.r_[np.nextafter(1.,2.),np.unique(values)]
    result=[]
    for cap in caps:
        feasible=[]
        for threshold in candidates:
            tp,fp=int((pos>=threshold).sum()),int((neg>=threshold).sum())
            if fp/len(neg)<=cap:feasible.append((-tp,fp,-float(threshold),float(threshold)))
        _,_,_,threshold=min(feasible)
        tp,fp=int((pos>=threshold).sum()),int((neg>=threshold).sum())
        result.append(dict(cap=float(cap),threshold=threshold,tp=tp,fp=fp,
            tpr=tp/len(pos),fpr=fp/len(neg)))
    return dict(auc=auc,positive_shots=len(pos),negative_shots=len(neg),oracle=result,
        deployment_authorized=False,meaning='posthoc threshold ceiling on these same labelled traces only')


def main(root):
    path=root/'posthoc_ranking.json'
    if path.exists():raise FileExistsError(path)
    done=READ(root/'completed.json');detail_path=root/'peak_details.json'
    assert sha(detail_path)==done['artifact_sha256']['peak_details.json']
    details=READ(detail_path);result={}
    for f in range(4):
        result[str(f)]={}
        for arm in ('warm_static','warm_intra','prepared_static','prepared_intra'):
            rows=[r for r in details if r['fold']==f and r['arm']==arm]
            assert len(rows)==65
            result[str(f)][arm]=diagnostic(rows)
    payload=dict(status='POSTHOC_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        code_sha256=sha(Path(__file__)),uses_evaluation_outcomes=True,changes_to_predictions=False,
        prespecified_in_main_plan=False,results=result)
    _durable_json(path,payload)
    print(payload,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
