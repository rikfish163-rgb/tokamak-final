"""Read-only retrospective error attribution on the locked EAST test set.

Labels are used only after completed prediction seals. This audit does not
change any threshold, prediction, model or online update policy. Elapsed time
means time since the first recorded sample, NOT plasma ignition time.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import simple_data
from .fixed_test_protocol import load_test_contract, score_fixed_test
from .operating_curve_audit import load_sealed, operating_curve
from .simple_online import _durable_json, _hash
from .simple_train import metrics


def diagnose(rows, threshold):
    scored=[]; details=[]
    for row in rows:
        times=np.asarray(row['times']); scores=np.asarray(row['scores'])
        valid=np.asarray(row['valid']).copy()
        if (times.ndim!=1 or len(times)==0 or scores.shape!=times.shape
                or valid.shape!=times.shape or valid.dtype!=bool
                or not np.isfinite(times).all() or (np.diff(times)<=0).any()
                or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any()):
            raise ValueError('ordered finite time/probability arrays with a boolean validity mask required')
        if row['is_disrupt']:
            if row['down'] is None or not np.isfinite(row['down']):
                raise ValueError('finite disruption time required for retrospective scoring')
            valid &= times<=row['down']
        if not valid.any():raise ValueError('no eligible observed prediction endpoint')
        filtered=dict(row,times=times[valid],scores=scores[valid])
        decision=metrics([filtered],threshold,lead=30.)['predictions'][0]
        category=('TP' if decision['timely'] else 'TARDY' if decision['alarm'] else 'MISS') \
            if row['is_disrupt'] else ('FP' if decision['alarm'] else 'TN')
        alarms=np.flatnonzero(valid & (scores>=threshold))
        scored.append(filtered)
        details.append(dict(**decision,category=category,
            first_alarm_elapsed_ms=float(round((times[alarms[0]]-times[0])*1000.,9)) if len(alarms) else None,
            recorded_duration_ms=float(round((times[-1]-times[0])*1000.,9)),
            fraction_eligible_endpoints_above_threshold=float(np.mean(scores[valid]>=threshold))))
    ids=[r['shot'] for r in details]
    if len(ids)!=len(set(ids)):raise ValueError('distinct shot identities required')
    fp=[r for r in details if r['category']=='FP']
    values=[r['fraction_eligible_endpoints_above_threshold'] for r in fp]
    totals=metrics(scored,threshold,lead=30.)
    return dict(metrics={k:v for k,v in totals.items() if k!='predictions'},
        counts={k:sum(r['category']==k for r in details) for k in ('TP','TARDY','MISS','FP','TN')},
        fp_before_elapsed_ms={str(t):sum(r['first_alarm_elapsed_ms']<t for r in fp)
                              for t in (2,8,32,64,128,250,500,1000)},
        fp_endpoint_fraction_quantiles=dict(zip(('min','q25','median','q75','max'),
            map(float,np.quantile(values,[0,.25,.5,.75,1])))) if values else None,
        early_over_3000=sum(r['timely'] and r['lead_ms']>3000 for r in details),
        elapsed_time_reference='first recorded sample; not physical ignition',shots=details)


def paired_changes(before,after):
    old={r['shot']:r for r in before['shots']}; new={r['shot']:r for r in after['shots']}
    if (set(old)!=set(new) or len(old)!=len(before['shots']) or len(new)!=len(after['shots'])
            or any(old[s]['is_disrupt']!=new[s]['is_disrupt'] for s in old)):
        raise ValueError('paired shot identities and outcomes must be identical')
    changed=[]
    for sid in old:
        a,b=old[sid],new[sid]
        if a['category']!=b['category']:
            changed.append(dict(shot=sid,before=a['category'],after=b['category'],
                before_lead_ms=a['lead_ms'],after_lead_ms=b['lead_ms']))
    count=lambda a,b:sum(r['before'] in a and r['after'] in b for r in changed)
    return dict(rescued_fp=count({'FP'},{'TN'}),new_fp=count({'TN'},{'FP'}),
        rescued_not_timely=count({'TARDY','MISS'},{'TP'}),
        new_not_timely=count({'TP'},{'TARDY','MISS'}),
        net_tp_change=after['metrics']['tp']-before['metrics']['tp'],
        net_fp_change=after['metrics']['fp']-before['metrics']['fp'],changed_shots=changed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--parent-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new output file required; do not replace evidence')
    candidate=json.loads((args.candidate/'completed.json').read_text())
    if candidate['status']!='COMPLETED':raise ValueError('completed candidate arms required')
    parent=json.loads((args.parent_run/'completed.json').read_text())
    if not all(parent[k] for k in ('fixed_test_sets','parent_unchanged','sealed_recount')):
        raise ValueError('fully verified fixed-test parent run required')
    protocol=json.loads((args.candidate/'protocol.json').read_text())
    contract=load_test_contract('EAST');order=[dict(shot=s) for s in contract['shot_ids']]
    if json.loads((args.candidate/'order.json').read_text())!=order:
        raise ValueError('candidate did not use the locked test order')
    threshold=protocol['threshold']
    signals=[simple_data.attach_outcome(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'),
                                       'EAST',{}) for r in order]
    folders=dict(original_static=args.parent_run/'east_test360/stream',
        prepared_static=args.candidate/'prepared_static/stream',
        prepared_intra=args.candidate/'prepared_intra/stream')
    summaries={};hashes={};operating_diagnostics={}
    for name,folder in folders.items():
        rows,hashes[name]=load_sealed(folder,order,signals,threshold)
        # Count the same full fixed test separately before timing analysis.
        filtered=[]
        for row in rows:
            valid=row['valid'].copy()
            if row['is_disrupt']:valid &= row['times']<=row['down']
            filtered.append(dict(row,times=row['times'][valid],scores=row['scores'][valid]))
        exact=score_fixed_test('EAST',filtered,threshold)
        summaries[name]=diagnose(rows,threshold)
        if summaries[name]['metrics']!={k:v for k,v in exact.items() if k!='predictions'}:
            raise AssertionError('diagnostic and fixed-test recount differ')
        curve=operating_curve(rows,threshold)
        operating_diagnostics[name]={k:v for k,v in curve.items() if k!='curve'}
    pairs={name:paired_changes(summaries[a],summaries[b]) for name,a,b in (
        ('source_preparation','original_static','prepared_static'),
        ('intra_effect','prepared_static','prepared_intra'),
        ('total_vs_original','original_static','prepared_intra'))}
    result=dict(scope='RETROSPECTIVE FIXED-TEST ERROR AUDIT; DEVELOPMENT-EXPOSED',
        experiment_complete=True,test=contract,threshold=threshold,threshold_changed=False,
        parent_completion_sha256=_hash(args.parent_run/'completed.json'),
        candidate_completion_sha256=_hash(args.candidate/'completed.json'),
        source_file_sha256=_hash(Path(__file__)),sealed_predictions_checked=360*len(folders),
        summaries=summaries,paired=pairs,seals=hashes,
        posthoc_operating_diagnostics=operating_diagnostics,
        operating_diagnostics_warning='uses all test outcomes after the run; not deployed thresholds or confirmatory results',
        interpretation='descriptive associations; neither mechanism identification nor independent confirmation')
    _durable_json(args.output,result)
    print(json.dumps(dict(results={name:{k:v for k,v in s.items() if k!='shots'}
                                   for name,s in summaries.items()},
        paired={name:{k:v for k,v in p.items() if k!='changed_shots'} for name,p in pairs.items()})),flush=True)


if __name__=='__main__':main()
