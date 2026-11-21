"""Independent CPU rescoring and explicitly paired diagnostic summaries.

This is posthoc analysis, not an online policy or an estimate of unseen-shot
performance. Each candidate difference is computed at the SAME saved state.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha


def validate_sample_metadata(row,times,prediction,disruptive,down,valid,threshold):
    count=row['count'];stop=min(count+128,len(times));eligible=valid[count:stop].copy()
    if disruptive:eligible &= np.round((down-times[count:stop])*1000,9)>30
    assert row['query_stop_exclusive']==stop
    assert row['eligible_query_points']==int(eligible.sum())
    assert row['is_disrupt']==disruptive
    assert row['alarm_already_emitted']==bool(np.any(prediction[:count][valid[:count]]>=threshold))
    assert row['status']==('measured' if eligible.any() else 'no_future_timely_or_valid_point')


def validate_bins(row,times,prediction,valid,disruptive,down):
    expected={}
    if disruptive:
        lead=np.round((down-times)*1000,9)
        for lo,hi in ((0,10),(10,25),(25,30),(30,50),(50,125),(125,500),(500,float('inf'))):
            selected=valid & (lead>=0 if lo==0 else lead>lo) & (lead<=hi)
            expected[f'{lo}_{hi}']=dict(observed=int(selected.sum()),
                peak=float(prediction[selected].max()) if selected.any() else None)
    assert row['lead_bin_peaks']==expected


def validate_source(source):
    """Re-read TRAIN outcomes/timestamps and reconstruct labels arithmetically."""
    train=io23.source_roles()['train'];advanced=io23.advanced_times();expected=[]
    assert source['n_train']==len(train) and len(train)==len(set(train))
    assert sha(Path(source['advanced_path']))==source['advanced_sha256']
    assert Path(source['advanced_path']).resolve()==io23.JTEXT_ADVANCED_TIME_PATH.resolve()
    from ..redesign_20261004.prepare_prior import DEFAULT_PARENT
    assert sha(DEFAULT_PARENT.parent/'protocol.json')==source['parent_protocol_sha256']
    for shot in train:
        disrupted,down=io23.read_outcome('J-TEXT',shot)
        if not disrupted:continue
        raw=io23.read_signals('J-TEXT',shot);a=advanced.get(shot)
        h=max(25.,a) if a is not None else 25.
        unrounded=(down-raw['times'])*1000;lead=np.round(unrounded,9)
        valid=unrounded>=-1e-6;positive=valid & (unrounded<=h+1e-6)
        expected.append(dict(shot=shot,horizon_ms=h,raw_advanced_ms=a,missing=a is None,
            observed=int(valid.sum()),positive=int(positive.sum()),
            positive_gt10=int((positive & (lead>10)).sum()),positive_gt30=int((positive & (lead>30)).sum()),
            eligible_gt30=int((valid & (lead>30)).sum())))
    assert source['rows']==expected and source['n_disrupt']==len(expected)
    assert source['horizon_gt30']==sum(r['horizon_ms']>30 for r in expected)
    assert source['missing_advanced']==sum(r['missing'] for r in expected)
    assert source['disruptions_with_positive_gt30']==sum(r['positive_gt30']>0 for r in expected)
    for key in ('positive','positive_gt10','positive_gt30'):
        assert source[key]==sum(r[key] for r in expected)


def paired_summary(rows):
    differences=[];by_shot=defaultdict(list)
    for row in rows:
        c=row['same_state_candidates']
        d=c['lag_bias']['predicted_risk_loss_change']-c['recent']['predicted_risk_loss_change']
        if not np.isfinite(d):raise ValueError('finite paired effects required')
        differences.append(d);by_shot[row['shot']].append(d)
    return dict(sampled_updates=len(rows),distinct_shots=len(by_shot),
        lag_lower_predicted_loss=sum(d < -1e-7 for d in differences),
        lag_higher_predicted_loss=sum(d > 1e-7 for d in differences),
        negligible_difference=sum(abs(d)<=1e-7 for d in differences),
        sample_mean_difference=float(np.mean(differences)) if rows else None,
        equal_shot_mean_difference=float(np.mean([np.mean(v) for v in by_shot.values()])) if rows else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output-name',default='verification_v2.json')
    args=p.parse_args();root=args.diagnostic
    if Path(args.output_name).name!=args.output_name:raise ValueError('output basename required')
    output=root/args.output_name
    if output.exists():raise FileExistsError(output)
    load=lambda path:json.loads(path.read_text())
    protocol=load(root/'protocol.json');completed=load(root/'completed.json')
    assert completed['status']=='COMPLETED'
    assert sha(root.parents[1]/'temporal_feedback_diagnostic_20261023/run.py')==protocol['code_sha256']
    parent=Path(protocol['run']);original=load(parent/'protocol.json')
    assert sha(parent/'verification.json')==protocol['verification_sha256']
    timing=load(root/'timing.json');cases=load(root/'cases.json')
    by_shot=defaultdict(list)
    for row in cases:by_shot[row['shot']].append(row)
    receipts=load(parent/'prediction_receipts.json');expected_pairs=set();update_count=0
    labels={};rescored={arm:Counter() for arm in original['arms']}
    categories=('late_0_10','late_10_25','late_25_30','timely_30_125',
                'timely_125_3000','early_over3000')
    for position,receipt in enumerate(receipts):
        shot=receipt['shot'];path=parent/'shots'/f'{shot}.npz'
        upath=parent/'shots'/f'{shot}_updates.json'
        assert sha(path)==receipt['prediction_sha256'] and sha(upath)==receipt['updates_sha256']
        disruptive,down=io23.read_outcome('EAST',shot);labels[shot]=disruptive
        with np.load(path,allow_pickle=False) as z:
            times=z['times'];valid=(down-times)*1000>=-1e-6 if disruptive else np.ones(len(times),bool)
            for arm in original['arms']:
                hit=np.flatnonzero(valid & (z[arm]>=original['threshold']))
                first=int(hit[0]) if len(hit) else None
                lead=round(float(down-times[first])*1000,9) if disruptive and first is not None else None
                category=('safe_fp' if len(hit) else 'safe_tn') if not disruptive else (
                    'never' if not len(hit) else categories[np.searchsorted([10,25,30,125,3000],lead,side='left')])
                r=timing[arm][position]
                assert (r['shot'],r['category'],r['alarm_index'],r['lead_ms'])==(shot,category,first,lead)
                validate_bins(r,times,z[arm],valid,disruptive,down)
                rescored[arm][category]+=1
            for row in by_shot[shot]:
                validate_sample_metadata(row,times,z[row['arm']],disruptive,down,valid,original['threshold'])
            if position>=20:
                counts=list(range(256,len(times),128));updates=load(upath)
                for arm in ('recent','lag_bias'):
                    assert len(updates[arm])==len(counts)
                    # The completed experiment has no empty-supervision updates.
                    # Fail explicitly if that contract changes; do not silently omit them.
                    assert all(r['gradient_steps']==1 for r in updates[arm])
                    update_count+=len(counts)
                    if counts:
                        expected_pairs.update((shot,arm,c) for c in {counts[0],counts[len(counts)//2],counts[-1]})
    assert {k:dict(v) for k,v in rescored.items()}==completed['timing_counts']
    actual=[(r['shot'],r['arm'],r['count']) for r in cases]
    assert len(actual)==len(set(actual)) and set(actual)==expected_pairs
    assert update_count==completed['verified_update_receipts']
    assert len(cases)==completed['sampled_updates']
    measured=[r for r in cases if r['status']=='measured']
    assert len(measured)==completed['measured']
    for row in measured:
        for values in row['same_state_candidates'].values():
            np.testing.assert_allclose(sum(c['predicted_risk_loss_change'] for c in values['channels']),
                values['predicted_risk_loss_change'],rtol=5e-4,atol=2e-7)
    groups={}
    for arm in ('recent','lag_bias'):
        for phase in ('safe','disruptive'):
            rows=[r for r in measured if r['arm']==arm and not r['alarm_already_emitted']
                  and labels[r['shot']]==(phase=='disruptive')]
            groups[f'{arm}/decision_open/{phase}']=paired_summary(rows)
    validate_source(load(root/'source_timing.json'))
    result=dict(status='VERIFIED',audit_code_sha256=sha(Path(__file__)),
        sealed_diagnostic_files={n:sha(root/n) for n in ('protocol.json','completed.json','timing.json','cases.json','source_timing.json')},
        rescored_traces=len(receipts)*len(original['arms']),verified_update_receipts=update_count,
        sampled_updates=len(cases),measured_updates=len(measured),
        sample_status_counts=dict(Counter(r['status'] for r in cases)),
        source_labels_recomputed_from_hdf5=True,sample_eligibility_recomputed=True,lead_bin_peaks_recomputed=True,
        independent_timing_counts={k:dict(v) for k,v in rescored.items()},same_state_paired_comparisons=groups,
        limitations=[
            'First/middle/last is outcome-independent sampling, not random or representative sampling.',
            'Original phase groups use before-update peak location; blocks may span both sides of 125 ms.',
            'Candidate differences are first-order predictions, not both finite counterfactual effects.',
            'Cross-arm groups have different conditioning sets; only within-row candidates are paired.',
            'All update receipts replayed; only measured sampled probability blocks replayed, not every full trace.',
            'Future outcomes used for diagnosis only; not a deployable direction selector or independent confirmation.'])
    _durable_json(output,result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
