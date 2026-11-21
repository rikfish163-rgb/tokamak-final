"""Recompute every sealed EAST result; no model or policy modification."""
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..joint_feedback_20261022.diagnose_stream import compare
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_train import sha
from ..simple_online import _durable_json


def main():
    root=io23.ROOT/'experiments/risk_aligned_online/runs'
    if sha(DEFAULT_PARENT)!=PARENT_SHA256: raise AssertionError('qualified parent changed')
    modes=('uniform','point','shot','event')
    contract=load_test_contract('EAST');records={};static_reference={};sampling=[]
    for mode in modes:
        run=root/f'context_feedback_east_{mode}_20261022'
        done=json.loads((run/'completed.json').read_text())
        protocol=json.loads((run/'protocol.json').read_text())
        recovery=json.loads((run/'recovery.json').read_text()) if (run/'recovery.json').exists() else None
        threshold=protocol['threshold']; updates=json.loads((run/'updates.json').read_text())
        if [r['shot'] for r in updates]!=contract['shot_ids']: raise AssertionError('stream identities changed')
        rows={k:[] for k in ('static','intra')};before_update_fp=[];stats=[]
        weights=[];deltas=[]
        for info in updates:
            sid=info['shot'];file=run/'shots'/f'{sid}.npz'
            if sha(file)!=info['prediction_sha256']: raise AssertionError('sealed prediction changed')
            with np.load(file) as data:
                times=data['times']; d,down=io23.read_outcome('EAST',sid)
                _,valid=io23.labels_at(times,d,down,125.)
                if mode=='uniform': static_reference[sid]=data['static'].copy()
                elif not np.array_equal(data['static'],static_reference[sid]):
                    raise AssertionError('nonidentical bare-source comparison')
                for arm in rows:
                    rows[arm].append(dict(shot=sid,times=times[valid],scores=data[arm][valid],is_disrupt=d,down=down))
                if not np.array_equal(data['static'][:128],data['intra'][:128]):
                    raise AssertionError('update affected pre-update predictions')
                indices=np.flatnonzero((data['static']>=threshold)&valid)
                alarm_index=int(indices[0]) if len(indices) else None
                if not d and alarm_index is not None and alarm_index<128:
                    before_update_fp.append(sid)
                stats.append(dict(shot=sid,is_disrupt=d,record_ms=float((times[-1]-times[0])*1000),
                    first_static_alarm_index=alarm_index,
                    first_static_alarm_ms_after_record_start=float((times[alarm_index]-times[0])*1000) if alarm_index is not None else None))
            if info['updates'] is None:
                if recovery is None or sid not in recovery['recovered_shots']:
                    raise AssertionError('unexplained missing update telemetry')
                if recovery['prediction_hashes'][str(sid)]!=info['prediction_sha256']:
                    raise AssertionError('recovered immutable trace changed')
                continue
            for r in info['updates']:
                if r['first_affected_prediction']!=r['last_observation']+1 or r['last_observation']<127:
                    raise AssertionError('causal update timing changed')
                weights.append(r['weights']);deltas.append(r['delta'])
        recomputed={k:score_fixed_test('EAST',v,threshold) for k,v in rows.items()}
        for arm in rows:
            for key,value in done['metrics'][arm].items():
                if recomputed[arm][key]!=value: raise AssertionError(f'{mode} {arm} {key} mismatch')
        flips=compare(recomputed['static'],recomputed['intra'])
        if flips!=done['paired_flips']: raise AssertionError('flip ledger mismatch')
        w=np.asarray(weights)
        records[mode]=dict(metrics=done['metrics'],paired_flips=flips,goal_pass=done['goal_pass'],
            static_false_alarms_before_first_possible_update=before_update_fp,
            irrevocable_fpr_floor_from_burnin=len(before_update_fp)/180.,
            recorded_updates=len(weights),
            missing_update_telemetry_shots=done['missing_update_telemetry_shots'],
            weight_summary_scope='resumed tail only; original prefix telemetry lost at SIGTERM',
            delta_norm_quantiles=np.quantile(deltas,[0,.25,.5,.75,1.]).tolist(),
            channel_weights=dict(zip(io23.RAW23,w.mean(0).tolist())),
            minimum_channel_weight=float(w.min()),maximum_channel_weight=float(w.max()),
            seconds=done['seconds'],per_shot_timing=stats)
        if mode!='uniform':
            preparation=root/f'context_feedback_{mode}_20261022'
            sample=json.loads((preparation/'sampling.json').read_text());sampling.append(sample['sha256'])
            checkpoint=torch.load(preparation/'last.pt',weights_only=True,map_location='cpu')
            warm=torch.load(root/'context_feedback_warm_20261022/warm.pt',weights_only=True,map_location='cpu')
            noncontroller=[k for k in warm['model'] if not k.startswith('controller.')]
            if any(not torch.equal(warm['model'][k],checkpoint['model'][k]) for k in noncontroller):
                raise AssertionError('preparation changed more than controller')
    if len(set(sampling))!=1: raise AssertionError('source training shots not matched')
    output=io23.ROOT/'outputs/source_event_rebuild_audit_20261022/context_feedback_summary.json'
    if output.exists(): raise FileExistsError(output)
    result=dict(status='VERIFIED',source_parent_sha256=PARENT_SHA256,
        arms=records,matched_source_sampling_sha256=sampling[0],static_traces_bitwise_identical=True,
        evidence='single-seed fixed TEST360, development exposed; not independent confirmation',
        verification='complete sealed-trace rescoring and hashes; not an independent inference rerun')
    _durable_json(output,result)
    for mode,r in records.items():
        print(json.dumps(dict(mode=mode,metrics=r['metrics'],paired_flips=r['paired_flips'],
            before_first_update_fp=r['static_false_alarms_before_first_possible_update'],
            weight_range=[r['minimum_channel_weight'],r['maximum_channel_weight']]),ensure_ascii=False))


if __name__=='__main__':main()
