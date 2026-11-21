"""Source-storage QA convention, not a calibrated physical safety factor.

The snapshot obeys qa_proxy = C * Bt / Ip_A in source TRAIN1353.
EAST contains multiple C conventions, including an outcome-associated split.
This diagnostic reconstructs only this derived slot from the current Bt/Ip;
it does not match a target population or use its outcomes. The geometry and
units of the original proxy remain uncertified.
"""
import numpy as np

from .io23 import RAW23

BT,IP,QA=(RAW23.index(n) for n in ('Bt','ip','qa_proxy'))


def _signals(record):
    x=np.asarray(record['values']);a=np.asarray(record['available'])
    if (x.ndim!=2 or x.shape[1]!=23 or not len(x) or x.dtype!=np.float32
            or a.shape!=x.shape or a.dtype!=bool):
        raise ValueError('nonempty float32 flat23 and boolean availability required')
    return x,a


def fit_source_qa_convention(shots):
    """Equally weighted per-shot coefficient; source signals only.

    A 1% relative-spread gate tests the *numerical constant* hypothesis, not
    physical accuracy. It is far above the observed snapshot rounding/noise
    and is not tuned on risk labels. Non-constant conventions are rejected.
    """
    rows=list(shots);ids=[r['shot'] for r in rows]
    if not rows or len(set(ids))!=len(ids):
        raise ValueError('nonempty distinct source shots required')
    coefficients=[];spreads=[]
    for r in rows:
        if r.get('device')!='J-TEXT' or any(k in r for k in (
                'y','down','is_disrupt','horizon','_sealed_scores')):
            raise ValueError('J-TEXT source signals only, no outcome fields')
        x,a=_signals(r)
        valid=a[:,BT]&a[:,IP]&a[:,QA]&np.isfinite(x[:,BT])&np.isfinite(x[:,IP])&np.isfinite(x[:,QA])&(x[:,BT]!=0)
        if not valid.any():raise ValueError('source shot has no identifiable QA convention')
        v=x[valid].astype(np.float64)
        ratio=v[:,QA]*v[:,IP]/v[:,BT]
        lo,mid,hi=np.quantile(ratio,[.05,.5,.95])
        if not np.isfinite([lo,mid,hi]).all() or mid<=0:
            raise ValueError('positive finite source convention required')
        spread=(hi-lo)/mid
        if spread>.01:raise ValueError('source QA is not a constant Bt/Ip convention')
        coefficients.append(mid);spreads.append(spread)
    coefficient=float(np.median(coefficients))
    if np.max(abs(np.asarray(coefficients)-coefficient))/coefficient>.01:
        raise ValueError('source shots do not share a single QA convention')
    return dict(coefficient=coefficient,source_ids=ids,
                formula='qa_source_coordinate = coefficient * Bt / Ip_A',
                source_coefficient_range=[float(min(coefficients)),float(max(coefficients))],
                max_within_shot_relative_spread=float(max(spreads)),
                numerical_spread_gate=.01,target_information_used=0,
                physical_q_calibration=False)


def rebuild_qa(record,coefficient):
    """Pointwise reconstruction: no stored QA, future samples or outcomes.

    This explicitly replaces QA availability by derivability from Bt and Ip.
    A zero denominator is undefined, so that time point is unavailable; no
    interpolation or future-derived fill is permitted.
    """
    if not np.isfinite(coefficient) or coefficient<=0:
        raise ValueError('positive finite source convention required')
    x,a=_signals(record);x=x.copy();a=a.copy()
    valid=a[:,BT]&a[:,IP]&np.isfinite(x[:,BT])&np.isfinite(x[:,IP])&(x[:,IP]!=0)
    x[:,QA]=0.
    value=coefficient*x[valid,BT].astype(np.float64)/x[valid,IP].astype(np.float64)
    if not np.isfinite(value).all() or (abs(value)>np.finfo(np.float32).max).any():
        raise FloatingPointError('QA reconstruction is not representable')
    x[valid,QA]=value.astype(np.float32);a[:,QA]=valid
    return dict(record,values=x,available=a,qa_rebuilt=True)
