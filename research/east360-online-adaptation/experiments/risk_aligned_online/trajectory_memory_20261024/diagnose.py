"""Post-hoc ranking, routing and history-gradient analysis; never fits a model."""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ..simple_online import _durable_json
from .run import READ
from .learning import ARMS


def describe(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p10=float(np.quantile(x,.1)) if len(x) else None,p90=float(np.quantile(x,.9)) if len(x) else None)


def timely_update(times,count,down):
    """Post-hoc timing diagnostic; never called by runtime adaptation."""
    return bool(30<round((down-times[count-1])*1000,9)<=125
                and round((down-times[count])*1000,9)>30)


def main(root):
    torch.set_num_threads(2)
    protocol=READ(root/'protocol.json');truth=READ(root/'outcomes.json');records=READ(root/'per_shot.json')
    bank_info=READ(root/'bank.json')
    with np.load(root/'bank.npz') as z:labels=z['labels'];losses=z['losses']
    g=torch.load(root/'gradients.pt',map_location='cpu',weights_only=True)['gradients'].double()
    gram=(g@g.T).numpy();parent_gradient={}
    for label in (0,1):
        idx=np.flatnonzero(labels==label)
        parent_gradient[str(label)]=dict(loss=describe(losses[idx]),gradient_norm=describe(g[idx].norm(dim=1).numpy()))
    peaks={a:[] for a in ('off',)+ARMS};weights={a:[] for a in ARMS};weight_groups=[]
    useful_update=[];routing_l1={a:[] for a in ARMS[1:]};selected_overlap={a:[] for a in ARMS[1:]}
    for outcome in truth:
        sid=outcome['shot'];flag=outcome['is_disrupt'];down=outcome['down']
        with np.load(root/'shots'/f'{sid}.npz') as z:
            times=z['times'];valid=np.ones(len(times),bool) if not flag else np.round((down-times)*1000,9)>30
            if not valid.any():raise ValueError('no timely observation; no silent shot omission')
            for a in peaks:peaks[a].append(float(z[a][valid].max()))
            receipts=READ(root/'receipts'/f'{sid}.json');has_useful=False
            for r in receipts:
                count=r['trajectory']['first_affected_prediction'];t=times[count-1]
                group='safe' if not flag else 'timely_positive' if 30<round((down-t)*1000,9)<=125 else 'other_disruptive'
                if flag:has_useful = has_useful or timely_update(times,count,down)
                weight_groups.append(group)
                for a in ARMS:
                    w=np.zeros(len(labels));w[r[a]['selected']]=r[a]['weights'];weights[a].append(w)
                for a in ARMS[1:]:
                    routing_l1[a].append(float(np.abs(weights['trajectory'][-1]-weights[a][-1]).sum()))
                    selected_overlap[a].append(len(set(r[a]['selected'])&set(r['trajectory']['selected']))/20)
            if flag:useful_update.append(dict(shot=sid,has_update_in_30_125=has_useful))
    yy=np.array([r['is_disrupt'] for r in truth]);ranking={}
    for a,values in peaks.items():
        values=np.asarray(values,dtype='float32');safe=np.sort(values[~yy]);operating={}
        for cap in (.1,.15,.2):
            allowed=int(round(cap*len(safe)))
            threshold=float(np.nextafter(safe[len(safe)-allowed-1],np.float32(np.inf)))
            operating[str(cap)]=dict(threshold=threshold,tp=int(np.sum(values[yy]>=threshold)),
                                    fp=int(np.sum(values[~yy]>=threshold)))
        ranking[a]=dict(timely_peak_auc=float(roc_auc_score(yy,values)),
                         diagnostic_oracle_thresholds=operating)
    weights={a:np.stack(v) for a,v in weights.items()};groups=np.asarray(weight_groups);cosines={}
    def dot(a,b):return np.einsum('bi,ij,bj->b',a,gram,b,optimize=True)
    realnorm=np.sqrt(np.maximum(dot(weights['trajectory'],weights['trajectory']),0))
    for a in ARMS[1:]:
        norm=np.sqrt(np.maximum(dot(weights[a],weights[a]),0));valid=realnorm*norm>1e-14
        cosines[a]=describe(dot(weights['trajectory'],weights[a])[valid]/(realnorm[valid]*norm[valid]))
    masses={a:{group:describe((w@labels)[groups==group]) for group in np.unique(groups)} for a,w in weights.items()}
    flips=READ(root/'flips.json');flip_counts={}
    for a,rows in flips.items():
        flip_counts[a]={name:sum(r['before']['is_disrupt']==d and r['beneficial']==benefit for r in rows)
                        for name,d,benefit in [('rescued',True,True),('lost',True,False),('removed_fp',False,True),('added_fp',False,False)]}
    out=dict(scope='post-hoc DEVELOPMENT_ONLY; oracle thresholds use test outcomes, NOT deployable scores or capacity bounds',
        ranking=ranking,flips=flip_counts,historical_loss_and_gradient=parent_gradient,
        routing_weight_l1={a:describe(v) for a,v in routing_l1.items()},
        selected_overlap={a:describe(v) for a,v in selected_overlap.items()},gradient_cosine=cosines,
        positive_history_mass=masses,useful_update_count=sum(r['has_update_in_30_125'] for r in useful_update),
        useful_update_per_disruptive_shot=useful_update)
    _durable_json(root/'mechanism_diagnostic.json',out)
    print({k:v for k,v in out.items() if k!='useful_update_per_disruptive_shot'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
