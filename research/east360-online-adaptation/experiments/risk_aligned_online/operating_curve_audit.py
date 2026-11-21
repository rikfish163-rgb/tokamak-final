"""Descriptive, posthoc matched-FPR audit of immutable operational curves.

Every threshold search here uses the whole completed development stream's
outcomes. It is NOT a causal threshold policy or an independent test result.
No emitted prediction, model, normalization or alarm setting is changed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import simple_data
from .density_mapping_audit import RAW, MAPPED
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash

ROOT=Path(__file__).parent


def operational_maxima(rows):
    positive=[];negative=[];ids=set();no_d=[];no_s=[]
    for row in rows:
        sid=row['shot'];label=row['is_disrupt']
        times=np.asarray(row['times']);scores=np.asarray(row['scores']);valid=np.asarray(row['valid'])
        if (sid in ids or label not in (True,False) or times.ndim!=1 or scores.shape!=times.shape
                or valid.shape!=times.shape or valid.dtype!=bool or not np.isfinite(times).all()
                or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1))
                or np.any(np.diff(times)<0)):
            raise ValueError('distinct shots and matching finite time/probability/boolean-mask arrays required')
        ids.add(sid);take=valid.copy()
        if label:
            down=row['down']
            if down is None or not np.isfinite(down):raise ValueError('finite disruption time required')
            take &= np.round((float(down)-times)*1000.,9)>30.
        maximum=float(np.max(scores[take])) if take.any() else -math.inf
        (positive if label else negative).append(maximum)
        if not take.any():(no_d if label else no_s).append(sid)
    if not positive or not negative:raise ValueError('both classes required for matched TPR/FPR evidence')
    return dict(positive=np.asarray(positive,dtype=np.float64),negative=np.asarray(negative,dtype=np.float64),
                disruptions_without_eligible_point=no_d,safe_shots_without_eligible_point=no_s)


def operating_curve(rows,threshold,*,caps=(.10,.15,.20)):
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('finite original probability threshold required')
    if not caps or any(not np.isfinite(c) or not 0<=c<=1 for c in caps):
        raise ValueError('nonempty caps within [0,1] required')
    maxima=operational_maxima(rows);d=maxima['positive'];s=maxima['negative']

    def point(t):
        # Use float64 maxima/comparison. NumPy's weak Python-scalar promotion
        # against float32 scores could otherwise erase a nextafter tie break.
        tp=int(np.sum(d>=t)) if t is not None else 0
        fp=int(np.sum(s>=t)) if t is not None else 0
        return dict(threshold=t,decision='threshold' if t is not None else 'always_no_alarm',
                    tp=tp,fp=fp,tpr=tp/len(d),fpr=fp/len(s),J=tp/len(d)-fp/len(s))

    observed=np.unique(np.concatenate((d[np.isfinite(d)],s[np.isfinite(s)])))
    candidates={0.,1.,float(threshold),*map(float,observed)}
    candidates.update(float(np.nextafter(value,math.inf)) for value in observed if value<1.)
    curve=[point(t) for t in sorted(candidates)]
    # Saturated safe scores equal to 1 cannot be excluded by any legal
    # probability threshold. Label the no-alarm action explicitly; never
    # manufacture a deployable threshold >1 or drop such shots.
    curve.append(point(None))
    best={}
    for cap in caps:
        allowed=math.floor(cap*len(s)+1e-12)
        eligible=[p for p in curve if p['fp']<=allowed]
        selected=max(eligible,key=lambda p:(p['tp'],-p['fp'],p['threshold'] is not None,
                                            p['threshold'] if p['threshold'] is not None else -1.))
        best[f'{cap:.2f}']=dict(selected,cap=cap,allowed_fp=allowed)
    return dict(n_disrupt=len(d),n_safe=len(s),strict_minimum_lead_ms=30.,
                disruptions_without_eligible_point=maxima['disruptions_without_eligible_point'],
                safe_shots_without_eligible_point=maxima['safe_shots_without_eligible_point'],
                fixed_source_threshold=point(float(threshold)),posthoc_best_at_cap=best,
                uses_all_stream_labels_posthoc=True,deployable_threshold_selection=False,
                independent_validation=False,curve=curve)


def load_sealed(folder,order,signals,threshold):
    records=[];hashes=[]
    for index,(item,shot) in enumerate(zip(order,signals),1):
        sid=item['shot'];seal=json.loads((folder/f'{index:04d}_{sid}_seal.json').read_text())
        audit=json.loads((folder/f'{index:04d}_{sid}_audit.json').read_text())
        digest=_hash(folder/seal['npz'])
        if (seal['shot']!=sid or seal['threshold']!=threshold or digest!=seal['sha256']
                or audit['seal_sha256']!=digest):raise AssertionError('sealed identity or threshold changed')
        with np.load(folder/seal['npz'],allow_pickle=False) as trace:
            if int(trace['shot'])!=sid or not np.array_equal(trace['times'],shot['times']):
                raise AssertionError('saved trace and actual signal times differ')
            records.append(dict(shot=sid,is_disrupt=shot['is_disrupt'],down=shot['down'],
                                valid=shot['valid'],times=trace['times'].copy(),scores=trace['scores'].copy()))
        hashes.append(dict(shot=sid,npz_sha256=digest))
    return records,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new output required; existing evidence is immutable')
    verification=json.loads((args.root/'full_verification.json').read_text())
    if verification['verification']!='PASS' or not verification['all_arms_complete']:
        raise AssertionError('fully completed and verified arms required')
    order=json.loads((args.root/'intra_control/order.json').read_text())
    if json.loads((MAPPED.parent/'order.json').read_text())!=order:
        raise AssertionError('different input-control sample/order')
    protocol=json.loads((args.root/'intra_control/protocol.json').read_text())
    threshold=protocol['threshold']
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('qualified source changed')
    signals=[simple_data.attach_outcome(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'),
                                         'EAST',{}) for r in order]
    folders=dict(raw_static=RAW,mapped_static=MAPPED,intra_control=args.root/'intra_control/stream',
                 inter_only=args.root/'inter_only_v2/stream',both=args.root/'both_v2/stream')
    results={};seals={}
    for arm,folder in folders.items():
        records,seals[arm]=load_sealed(folder,order,signals,threshold)
        results[arm]={period:operating_curve(records[start:stop],threshold)
                     for period,start,stop in (('all',0,len(order)),('first30',0,30),('after30',30,len(order)))}
        if arm in verification['results']:
            actual=results[arm]['all']['fixed_source_threshold'];saved=verification['results'][arm]['metrics']['all']
            if any(actual[k]!=saved[k] for k in ('tp','fp','tpr','fpr','J')):
                raise AssertionError('operational maxima do not reproduce sealed first-alarm metrics')
    result=dict(scope='POSTHOC DEVELOPMENT MATCHED-FPR DIAGNOSTIC, NOT DEPLOYMENT OR CONFIRMATION',
        source_sha256=PARENT_SHA256,code_sha256=_hash(Path(__file__)),order_sha256=_hash(args.root/'intra_control/order.json'),
        verification_sha256=_hash(args.root/'full_verification.json'),sealed_curves_checked=len(order)*len(folders),
        future_oracle_thresholds_used_by_learner=False,threshold_search_uses_full_period_outcomes=True,
        original_predictions_and_model_unchanged=True,results=results,seals=seals)
    _durable_json(args.output,result)
    print(json.dumps(dict(sealed_curves_checked=result['sealed_curves_checked'],all={arm:dict(
        fixed=row['all']['fixed_source_threshold'],caps=row['all']['posthoc_best_at_cap'])
        for arm,row in results.items()})),flush=True)


if __name__=='__main__':main()
