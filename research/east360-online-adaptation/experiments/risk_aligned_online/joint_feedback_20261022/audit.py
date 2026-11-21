"""Independently recompute paired outcomes and explain label/metric mismatch."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .. import io23
from ..event_adapt_20261022.train import load_source, public
from ..fixed_test_protocol import score_fixed_test
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import metrics, sha
from .diagnose_stream import compare


RUNS=Path(__file__).parents[1]/'runs'
ARMS=('joint','meta','joint_weighted','meta_weighted')


def snapshot_check(path, config):
    with zipfile.ZipFile(path/'code_snapshot.zip') as archive:
        for name,expected in config['code_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                raise AssertionError(f'archived code digest mismatch {name}')


def first_alarm_context(shot, probability, threshold):
    use=np.flatnonzero(shot['valid'] & (probability>=threshold))
    if not len(use): return None
    k=int(use[0])
    return dict(index=k,time_seconds=float(shot['times'][k]),
        lead_ms=float(round((shot['down']-shot['times'][k])*1000,9)) if shot['is_disrupt'] else None,
        source_point_label=int(shot['y'][k]>.5),
        horizon_ms=float(shot['horizon']),probability=float(probability[k]),
        before_first_adaptation=k<128)


def main():
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/joint_feedback_summary.json'
    if output.exists(): raise FileExistsError(output)
    if sha(DEFAULT_PARENT)!=PARENT_SHA256: raise AssertionError('original source modified')
    warm=torch.load(RUNS/'paired_feedback_warm_20261022/warm.pt',map_location='cpu',weights_only=True)
    reference=np.load(RUNS/'paired_feedback_warm_20261022/test220_traces.npz')
    roles=io23.source_roles(); training=[]; sequences=set()
    for arm in ARMS:
        path=RUNS/f'joint_feedback_{arm}_20261022'
        done=json.loads((path/'completed.json').read_text())
        config=json.loads((path/'config.json').read_text())
        snapshot_check(path,config)
        inputs=json.loads((path/'training_inputs.json').read_text())
        sequences.add(inputs['sequence_sha256'])
        ck=torch.load(path/'source.pt',map_location='cpu',weights_only=True)
        if done['selected_epoch']!=0 or ck['epoch']!=0:
            raise AssertionError('audit expectation must be reviewed for nonzero selection')
        if any(not torch.equal(v,ck['model'][k]) for k,v in warm['model'].items()):
            raise AssertionError('selected initialization differs')
        if done['checkpoint_sha256']!=sha(path/'source.pt'): raise AssertionError('checkpoint hash changed')
        traces=np.load(path/'test220_traces.npz'); rows=[]
        difference=max(float(np.max(np.abs(traces[k]-reference[k]))) for k in reference.files)
        for sid in roles['test']:
            disruptive,down=io23.read_outcome('J-TEXT',sid)
            rows.append(dict(shot=sid,is_disrupt=disruptive,down=down,
                times=traces[f'{sid}_times'],scores=traces[f'{sid}_scores']))
        measured=score_fixed_test('J-TEXT',rows,ck['threshold'])
        for key in ('tp','fp','tpr','fpr','J'):
            if measured[key]!=done['metrics'][key]: raise AssertionError(f'TEST mismatch {arm}/{key}')
        at_old=score_fixed_test('J-TEXT',rows,warm['config']['threshold'])
        if (at_old['tp'],at_old['fp'])!=(100,13): raise AssertionError('old-threshold reproduction failed')
        history=json.loads((path/'history.json').read_text())
        training.append(dict(arm=arm,selected_epoch=0,inputs=inputs,
            test220=public(measured),test220_at_original_threshold=public(at_old),
            trace_max_difference_from_original=difference,last_epoch=history[-1],
            source_model_equal_warm=True,probe_gate=done['probe_gate']))
    if len(sequences)!=1: raise AssertionError('four training sequences do not match')
    shots=load_source(roles['val'],io23.advanced_times()); stream=[]
    for arm in ('warm',*ARMS):
        path=RUNS/f'joint_feedback_val_{arm}_20261022'
        protocol=json.loads((path/'protocol.json').read_text()); snapshot_check(path,protocol)
        done=json.loads((path/'completed.json').read_text()); data=np.load(path/'traces.npz')
        if sorted(protocol['ids'])!=sorted(roles['val']): raise AssertionError('VAL membership mismatch')
        threshold=done['matched_static_threshold']['static']['threshold']
        rows={k:[] for k in ('static','intra')}; contexts=[]
        for s in shots:
            sid=s['shot']; valid=s['valid']
            if not np.array_equal(valid,data[f'{sid}_valid']): raise AssertionError('validity mismatch')
            if not np.array_equal(s['times'],data[f'{sid}_times']): raise AssertionError('time grid mismatch')
            r=dict(shot=sid,is_disrupt=bool(s['is_disrupt']),horizon_ms=s['horizon'])
            r['has_mutable_timely_endpoint']=bool(np.any(valid & (np.arange(len(valid))>=128) &
                (np.round((s['down']-s['times'])*1000,9)>10))) if s['is_disrupt'] else True
            for mode in rows:
                scores=data[f'{sid}_{mode}']
                rows[mode].append(dict(shot=sid,is_disrupt=s['is_disrupt'],down=s['down'],
                                      times=s['times'][valid],scores=scores[valid]))
                r[mode]=first_alarm_context(s,scores,threshold)
            contexts.append(r)
        calculated={k:metrics(v,threshold) for k,v in rows.items()}
        for k,value in calculated.items():
            if public(value)!=done['matched_static_threshold'][k]: raise AssertionError('VAL metrics mismatch')
        flips=compare(calculated['static'],calculated['intra'])
        if flips!=done['matched_flips']: raise AssertionError('paired flips mismatch')
        flags={r['shot']:r for r in calculated['static']['predictions']}
        timely=[r for r in contexts if flags[r['shot']]['timely']]
        outside=[r['shot'] for r in timely if r['static']['source_point_label']==0]
        static_fp=[r for r in contexts if not r['is_disrupt'] and r['static'] is not None]
        mutable_misses=[r['shot'] for r in contexts if r['is_disrupt'] and not flags[r['shot']]['timely']
                        and r['has_mutable_timely_endpoint']]
        stream.append(dict(arm=arm,matched_metrics={k:public(v) for k,v in calculated.items()},
            existing_threshold_metrics=done['existing_threshold'],flips=flips,
            static_timely_alarms=len(timely),timely_alarm_with_zero_point_label=len(outside),
            zero_label_timely_shots=outside,
            static_false_alarms_before_first_update=[r['shot'] for r in static_fp if r['static']['before_first_adaptation']],
            missed_disruptions_with_mutable_timely_endpoints=mutable_misses,
            contexts=contexts,updates=done['updates']))
    result=dict(status='VERIFIED',training=training,full_val_diagnostics=stream,
        source_original_sha256=PARENT_SHA256,matched_training_sequences=True,
        windows_per_arm=8192,full_val_shots_per_model=160,
        claims='single-seed development experiments; no new EAST result, no physical mechanism claim',
        qualification='original qualified source retained; selected epochs all zero')
    _durable_json(output,result)
    print(json.dumps(dict(path=str(output),status='VERIFIED',training_sequences=len(sequences),
        diagnostics=[{k:r[k] for k in ('arm','flips','static_timely_alarms',
            'timely_alarm_with_zero_point_label','static_false_alarms_before_first_update',
            'missed_disruptions_with_mutable_timely_endpoints')} for r in stream]),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
