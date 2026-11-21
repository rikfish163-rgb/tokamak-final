"""Same-history paired screen, not a deployed classifier or online endpoint."""
import numpy as np

from .learning import neighbour_score


def pair_out_auc(query, memory, shots, y, *, k=3):
    """Hold one positive and one negative SHOT out together.

    Both query shots use exactly the same historical bank. Compare scores only
    within that fold, then weight shot pairs equally. y is evaluation truth;
    neighbour_score receives historical labels only. This bank intentionally
    has one risk class per shot; mixed-class banks need another evaluation.
    """
    if (query.shape != memory.shape or len(query) != len(shots)
            or y.shape != shots.shape or set(np.unique(y)) != {0, 1}):
        raise ValueError('matched queries and bank with both risk classes required')
    classes={c:np.unique(shots[y == c]) for c in (0, 1)}
    if set(classes[0]) & set(classes[1]):
        raise ValueError('pair-out shot labels must be unambiguous')
    if any(len(s) <= k for s in classes.values()):
        raise ValueError('not enough distinct history shots after pair exclusion')
    rows=[]
    for positive in classes[1]:
        for negative in classes[0]:
            held=(positive,negative);keep=~np.isin(shots,held)
            query_rows=[np.flatnonzero(shots == sid) for sid in held]
            ix=np.concatenate(query_rows)
            score=neighbour_score(query[ix],memory[keep],shots[keep],y[keep],
                                  exclude=shots[ix],k=k)
            p,n=np.split(score,[len(query_rows[0])])
            diff=p[:,None]-n[None,:]
            auc=float(np.mean((diff>0)+.5*(diff==0)))
            rows.append(dict(held_shots=list(map(int,held)),
                history_shots=sorted(map(int,np.unique(shots[keep]))),
                positive_scores=p.tolist(),negative_scores=n.tolist(),auc=auc))
    return float(np.mean([r['auc'] for r in rows])),rows


def qualifies(aggregate, raw_east_auc, risk_east_seeds):
    """Necessary screen only; this cannot certify actual intra-shot benefit."""
    return bool(
        len(risk_east_seeds) == 3 and all(v > .5 for v in risk_east_seeds)
        and aggregate['risk']['val_conditional_auc'] > .5
        and aggregate['risk']['east_auc'] > max(.5,raw_east_auc)
        and all(aggregate['risk'][metric] > aggregate[m][metric]
                for metric in ('val_conditional_auc','east_auc')
                for m in ('mean','shuffle','family','reverse')))
