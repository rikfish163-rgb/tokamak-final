"""Verify all seals/decisions and replay fixed full-shot samples on CUDA."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..fixed_test_protocol import load_test_contract, score_fixed_test
from ..paired_feedback_20261022.model import PairedFeedback
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT, PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from ..warmstart_alignment_20261022.run import public
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..operating_curve_audit import operating_curve
from .run import ARMS, REFERENCE, MAP_PATH, predict_arms


def independent_counts(records, threshold):
    tp=fp=nd=ns=early=0
    for r in records:
        crossing = np.flatnonzero(np.asarray(r['scores']) >= threshold)
        if r['is_disrupt']:
            nd += 1
            if len(crossing):
                lead = round(float(r['down']-r['times'][crossing[0]])*1000,9)
                tp += int(lead > 30.); early += int(lead > 3000.)
        else:
            ns += 1; fp += int(len(crossing)>0)
    return dict(tp=tp,fp=fp,n_disrupt=nd,n_safe=ns,very_early_over3000ms=early)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    done=json.loads((root/'completed.json').read_text()); assert done['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text())
    full=json.loads((root/'per_shot.json').read_text())
    order=load_test_contract('EAST')['shot_ids']; assert protocol['contract']['shot_ids']==order
    assert protocol['arms']==list(ARMS)
    for path,h in protocol['code_sha256'].items(): assert sha(io23.ROOT/path)==h
    assert sha(protocol['plan_path'])==protocol['plan_sha256']
    assert sha(REFERENCE/'protocol.json')==protocol['reference_protocol_sha256']
    assert sha(REFERENCE/'prediction_receipts.json')==protocol['reference_receipts_sha256']
    assert sha(MAP_PATH)==protocol['density_mapping_sha256']
    mapping=json.loads(MAP_PATH.read_text()); assert mapping['calibration_ids']==order[:20]
    seals=json.loads((root/'prediction_receipts.json').read_text())
    old_seals={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    assert [r['shot'] for r in seals]==order
    activation=json.loads((root/'activation_receipt.json').read_text())
    assert activation['after_sealed_ids']==order[:20] and activation['preceding_seals']==seals[:20]
    assert activation['mapping_sha256']==sha(MAP_PATH)
    rows={k:[] for k in ARMS}; all_updates={k:[] for k in ('recent','lag_bias')}
    for index,(sid,seal) in enumerate(zip(order,seals)):
        path=root/'shots'/f'{sid}.npz'; upath=root/'shots'/f'{sid}_updates.json'
        ref=REFERENCE/'shots'/f'{sid}.npz'
        assert sha(path)==seal['prediction_sha256'] and sha(upath)==seal['updates_sha256']
        assert sha(ref)==old_seals[sid]['prediction_sha256']
        updates=json.loads(upath.read_text())
        s=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as t, np.load(ref,allow_pickle=False) as old:
            np.testing.assert_array_equal(t['times'],s['times'])
            np.testing.assert_array_equal(t['static'],old['static'])
            np.testing.assert_array_equal(t['original4'],old['encoder_original'])
            for arm in ARMS:
                assert t[arm].shape==s['times'].shape and np.isfinite(t[arm]).all()
                assert np.all((t[arm]>=0)&(t[arm]<=1))
                np.testing.assert_array_equal(t[arm][:256],t['static'][:256])
                if index<20: np.testing.assert_array_equal(t[arm],t['static'])
                v=s['valid']
                rows[arm].append(dict(shot=sid,times=s['times'][v],scores=t[arm][v].copy(),
                                     is_disrupt=s['is_disrupt'],down=s['down']))
        if index<20:
            assert not updates
            continue
        expected=list(range(256,len(s['values']),128))
        assert set(updates)==set(all_updates)
        for arm,records in updates.items():
            assert [r['first_affected_prediction'] for r in records]==expected
            for r in records:
                c=r['first_affected_prediction']
                assert r['last_observation']==c-1
                assert r['calibration_origins']==[c-101,c-76]
                assert r['gradient_origins']==[c-51,c-26]
                assert r['calibration_last_target']==c-51 and r['first_gradient_target']==c-50
                assert r['gradient_steps'] in (0,1)
                if r['gradient_steps']:
                    assert r['observed_gradient_targets']>0 and 0<=r['parameter_delta_norm']<=.00101
                    assert all(np.isfinite(v) for v in r.values() if isinstance(v,float))
                else:
                    assert r['observed_gradient_targets']==0 and r['reason']=='no_shared_observed_targets'
                all_updates[arm].append(r)
        for a,b in zip(updates['recent'],updates['lag_bias']):
            for field in ('calibration_origins','gradient_origins','observed_old_targets',
                          'observed_gradient_targets','eligible_channels','first_affected_prediction'):
                assert a[field]==b[field]
    summaries={}; counts={}; curves={}
    for arm,rs in rows.items():
        assert score_fixed_test('EAST',rs,protocol['threshold'])==full[arm]
        assert public(full[arm])==done['metrics'][arm]['full360']
        counts[arm]=independent_counts(rs,protocol['threshold'])
        for key in ('tp','fp','n_disrupt','n_safe'): assert counts[arm][key]==full[arm][key]
        curves[arm]=operating_curve([dict(r,valid=np.ones(len(r['times']),bool)) for r in rs],
                                    protocol['threshold'])
        for key in ('tp','fp','tpr','fpr','J'):
            assert curves[arm]['fixed_source_threshold'][key]==full[arm][key]
    for arm,records in all_updates.items():
        actual=[r for r in records if r['gradient_steps']]
        summaries[arm]=dict(scheduled_updates=len(records),gradient_steps=len(actual),
            loss_decreased=sum(r['loss_after']<r['loss_before'] for r in actual),
            gradient_norm_max=max(r['gradient_norm'] for r in actual),
            step_norm_max=max(r['parameter_delta_norm'] for r in actual),
            median_raw_loss=float(np.median([r['recent_raw_loss'] for r in actual])),
            median_corrected_loss=float(np.median([r['recent_corrected_loss'] for r in actual])),
            mean_raw_energy=float(np.mean([r['recent_raw_mean_square'] for r in actual])),
            mean_remaining_energy=float(np.mean([r['recent_corrected_mean_square'] for r in actual])))
    torch.set_num_threads(2); torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT); model=PairedFeedback(base,np.ones(23)).cuda(); model.set_stage('frozen')
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    replay=[]
    for index in (20,179,359):
        sid=order[index]; s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},mapping)
        fresh,up=predict_arms(model,signal)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as t:
            for arm,value in fresh.items(): np.testing.assert_array_equal(value,t[arm])
        assert up==json.loads((root/'shots'/f'{sid}_updates.json').read_text())
        replay.append(sid)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    comparisons={f'static_to_{k}':paired_flips(full['static']['predictions'],full[k]['predictions']) for k in ARMS[1:]}
    comparisons['recent_to_lag_bias']=paired_flips(full['recent']['predictions'],full['lag_bias']['predictions'])
    result=dict(status='VERIFIED',rescored_traces=len(order)*len(ARMS),
        replayed_shots=replay,replayed_full_traces=len(replay)*3,max_replay_error=0.,
        source_archive_unchanged=True,all_predictions_and_updates_hash_verified=True,
        first20_cost_included=True,all_update_time_boundaries_valid=True,
        independent_counts=counts,update_summary=summaries,paired_comparisons=comparisons,
        audit_code_sha256=sha(Path(__file__)))
    _durable_json(root/'verification.json',result)
    _durable_json(root/'operating_diagnostic.json',dict(
        scope='POSTHOC: all360 outcomes used to sweep thresholds, not deployable or independent',
        thresholds_written_back=False,results=curves))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
