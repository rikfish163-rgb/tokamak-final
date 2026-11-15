"""Score each mature relation against its saved forecast, then learn from it."""

import numpy as np

from . import commissioning_lag_data_20261110 as lag_data
from .commissioning_calibration_20261108 import prefix_mapping
from .commissioning_lag_source_20261110 import shot_coordinates
from .commissioning_transfer_20261108 import STARTUP, features
from .commissioning_transition_east_20261110 import fit, scores
from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE
from .hdf5_score_recheck_20261103 import score_curve
from .mask_repair_20261104.east_evaluation import prepare_actual


MODES = ('fixed', 'real', 'shifted', 'lag')
FAMILY_PAIRS = ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3))
FIRST_EFFECT = FIRST + STRIDE + HORIZON + 1


def _family_matrix():
    groups = lag_data.GROUPS
    index = [FAMILY_PAIRS.index(tuple(sorted((g,h))))
             for i,g in enumerate(groups) for h in groups[i+1:] if g != h]
    index *= len(lag_data.LAGS)
    raw = np.eye(len(FAMILY_PAIRS),dtype=np.float32)[index]
    balanced = raw * np.sqrt(np.float32(len(index))/raw.sum(0))
    return balanced


def prequential_views(length, q, aligned, shifted, eligible, prior):
    """Publish [pre-update match, signed deviation, same-sign persistence].

    A q+1..q+25 block supplies a unit relation target and releases at q+26.
    STRIDE > HORIZON, so the state saved at q stays unchanged until this
    feedback arrives. A missing block publishes no feature and breaks the
    consecutive-deviation term, while the direction and its reference stay put.
    """
    pairs = _family_matrix()
    floor = np.linalg.norm(prior)
    if prior.shape != (len(pairs),) or not floor > 0.:
        raise ValueError('nonzero source direction in cross-edge order required')
    if len(q) and (q[0] != FIRST or np.any(np.diff(q) != STRIDE)):
        raise ValueError('expected scheduled nonoverlapping forecast origins')
    k = len(FAMILY_PAIRS)
    v = {a:np.zeros((length,3*k),np.float32) for a in MODES}
    state = np.broadcast_to(prior,(len(MODES),len(prior))).copy()
    earlier = np.zeros((len(MODES),k),np.float32)
    previous_deviation = np.zeros_like(earlier)
    last = None
    contiguous = False
    wins = np.zeros(len(MODES),int)
    loss = np.zeros(len(MODES),np.float64)
    observed = 0
    for i,origin in enumerate(q):
        if origin+HORIZON >= length:
            break
        # Freeze the forecast before reading this block's target.
        forecast = state / np.maximum(np.linalg.norm(state,axis=1,keepdims=True),floor)
        if not eligible[i]:
            contiguous = False
            continue
        target = aligned[i]
        compatibility = forecast*target[None,:]
        match = compatibility@pairs
        margin = compatibility.sum(1)
        wins += margin>0
        loss += np.logaddexp(0.,-margin)
        difference = match-earlier if last is not None else np.zeros_like(match)
        persistence = (np.where(difference*previous_deviation>0.,
                                np.sign(difference)*np.minimum(abs(difference),abs(previous_deviation)),0.)
                       if contiguous else np.zeros_like(match))
        rows = np.concatenate((match,difference,persistence),axis=1)
        first = int(origin+HORIZON+1)
        stop = min(length,first+STRIDE)
        for a,row in zip(MODES,rows,strict=True):
            v[a][first:stop] = row
        earlier = match if last is None else .85*earlier+.15*match
        previous_deviation = difference
        contiguous = True
        # These new directions can only affect the next forecast.
        state[1:] = .85*state[1:] + .15*np.stack((target,shifted[i],target if last is None else last))
        last = target
        observed += 1
    return v,dict(observed_blocks=observed,updates=dict.fromkeys(MODES[1:],observed),
                  true_order_loss_sum=dict(zip(MODES,loss.tolist(),strict=True)),
                  true_order_wins=dict(zip(MODES,wins.tolist(),strict=True)))


def encode(base,net,signal,center,scale):
    raw,_ = features(base,signal,'cuda')
    z,m,*_ = shot_coordinates(signal,center,scale)
    q = np.arange(FIRST,len(z)-HORIZON,STRIDE)
    r,s,ok = lag_data.feedback(z,m,q+HORIZON)
    views,audit = prequential_views(len(z),q,r,s,ok,net.source_mean.numpy())
    return raw,views,audit


def cold_probe(base,net,raws,outcomes,source_stats,center,scale,original):
    """Four folds; held shots are excluded from both the map and risk fit."""
    positives = [s for s in raws if outcomes[s][0]]
    safe = [s for s in raws if not outcomes[s][0]]
    if (len(positives),len(safe)) != (8,12):
        raise ValueError('expected 8 disruptive and 12 safe ended cold shots')
    folds = []
    totals = {k:[0,0] for k in MODES}
    pair_wins = dict.fromkeys(MODES,0)
    pair_count = 0
    for f in range(4):
        held = positives[f::4]+safe[f::4]
        trained = [s for s in raws if s not in held]
        mapping = prefix_mapping(source_stats,[raws[s] for s in trained])
        coef,_ = fit(base,net,raws,outcomes,trained,mapping,center,scale,original,encode_fn=encode)
        result = []
        for sid in held:
            raw,v,_ = encode(base,net,prepare_actual(raws[sid],STARTUP+1,mapping),center,scale)
            p = scores(raw,v,coef,MODES)
            t = raws[sid]['times']
            d,down = outcomes[sid]
            window = np.round((down-t)*1000.,9)>30 if d else np.ones(len(t),bool)
            result.append(dict(shot=sid,disruptive=d,
                               peaks={k:float(y[window].max()) if window.any() else None for k,y in p.items()},
                               alarms={k:score_curve(t,y,d,down,original) for k,y in p.items()}))
            for k,a in result[-1]['alarms'].items():
                totals[k][0] += d and a['timely']
                totals[k][1] += not d and a['alarm']
        for d in result[:2]:
            for s in result[2:]:
                pair_count += 1
                for k in MODES:
                    v = d['peaks'][k]
                    pair_wins[k] += v is not None and v>s['peaks'][k]
        folds.append(dict(held=held,train=trained,rows=result))
    return dict(scope='ENDED_COLD20_FOUR_SHOT_FOLDS_NOT_FULL_STREAM',
                threshold=original,folds=folds,timely_and_false_alarm=totals,
                matched_fold_peak_pairs=pair_count,peak_pair_wins=pair_wins)
