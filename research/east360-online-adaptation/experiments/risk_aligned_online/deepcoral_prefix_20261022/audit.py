"""Post-run verification and explicitly posthoc operating diagnostics."""
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
from ..operating_curve_audit import operating_curve, operational_maxima
from ..warmstart_alignment_20261022.alignment import transform
from .run import ARMS, predict_pairs


def paired_flips(before,after):
    if [r['shot'] for r in before]!=[r['shot'] for r in after]: raise ValueError('unpaired shots')
    result=dict(rescued_timely=[],lost_timely=[],removed_fp=[],added_fp=[])
    for a,b in zip(before,after):
        if a['is_disrupt']:
            if not a['timely'] and b['timely']: result['rescued_timely'].append(a['shot'])
            if a['timely'] and not b['timely']: result['lost_timely'].append(a['shot'])
        else:
            if a['alarm'] and not b['alarm']: result['removed_fp'].append(a['shot'])
            if not a['alarm'] and b['alarm']: result['added_fp'].append(a['shot'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    completed=json.loads((root/'completed.json').read_text()); assert completed['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text())
    for path,digest in protocol['code_sha256'].items(): assert sha(io23.ROOT/path)==digest
    assert sha(protocol['plan_path'])==protocol['plan_sha256']
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    receipts=json.loads((root/'prediction_receipts.json').read_text())
    order=load_test_contract('EAST')['shot_ids']; assert [r['shot'] for r in receipts]==order
    mapping=json.loads((root/'density_map.json').read_text()); assert mapping['calibration_ids']==order[:20]
    full=json.loads((root/'per_shot.json').read_text())
    rows={k:[] for k in full}
    update_summary={k:dict(updates=0,loss_decreased=0,step_norm_max=0.,gradient_norm_max=0.) for k in ARMS}
    for index,(sid,seal) in enumerate(zip(order,receipts)):
        path=root/'shots'/f'{sid}.npz'; updates_path=root/'shots'/f'{sid}_updates.json'
        assert sha(path)==seal['prediction_sha256'] and sha(updates_path)==seal['updates_sha256']
        shot=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['times'],shot['times'])
            for name in rows:
                if index<20: np.testing.assert_array_equal(trace[name],trace['density_static'])
                else: np.testing.assert_array_equal(trace[name][:256],trace[name.removesuffix('_intra')+'_static'][:256]
                                                   if name.endswith('_intra') else trace[name][:256])
                rows[name].append(dict(shot=sid,times=shot['times'],scores=trace[name].copy(),valid=shot['valid'],
                                       is_disrupt=shot['is_disrupt'],down=shot['down']))
        for arm,updates in json.loads(updates_path.read_text()).items():
            if index<20: assert not updates
            for u in updates:
                count=u['first_affected_prediction']; assert u['last_observation']==count-1
                assert count%128==0 and 256<=count<len(shot['values'])
                assert all(count-128<=e and e+25<=count-1 for e in u['forecast_origins'])
                if u['gradient_steps']:
                    stat=update_summary[arm]; stat['updates']+=1
                    stat['loss_decreased']+=u['loss_after']<u['loss_before']
                    stat['step_norm_max']=max(stat['step_norm_max'],u['parameter_delta_norm'])
                    stat['gradient_norm_max']=max(stat['gradient_norm_max'],u['gradient_norm'])
                    assert u['parameter_delta_norm']<=.00101
    operating={}
    for name,records in rows.items():
        cropped=[dict(r,times=r['times'][r['valid']],scores=r['scores'][r['valid']]) for r in records]
        assert score_fixed_test('EAST',cropped,protocol['threshold'])==full[name]
        operating[name]={}
        for period,rs in (('full360',records),('common_suffix340',records[20:])):
            curve=operating_curve(rs,protocol['threshold']); maxima=operational_maxima(rs)
            d,s=maxima['positive'][:,None],maxima['negative'][None,:]
            auc=float(((d>s).astype(float)+.5*(d==s)).mean())
            for key in ('tp','fp','tpr','fpr','J'):
                assert curve['fixed_source_threshold'][key]==completed['metrics'][name][period][key]
            operating[name][period]=dict(timely_peak_ranking_auc=auc,**curve)
    # Verify actual changes in copied checkpoints, not just requires_grad flags.
    torch.set_num_threads(2); models={}
    parent,_=load_parent(DEFAULT_PARENT); source=parent.state_dict()
    checkpoint_receipts=json.loads((root/'preparation_receipts.json').read_text())
    for arm in ARMS:
        path=root/f'{arm}.pt'; assert sha(path)==checkpoint_receipts[arm]['checkpoint_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        base,_=load_parent(DEFAULT_PARENT)
        model=PairedFeedback(base,np.ones(23)); model.load_state_dict(payload['model'],strict=True)
        allowed={k.removeprefix('base.') for k in model.encoder_names()} if arm!='density' else set()
        allowed|={'forecast_head.weight','forecast_head.bias'}
        for key,value in model.base.state_dict().items():
            if key not in allowed: torch.testing.assert_close(value,source[key],rtol=0,atol=0)
        model.cuda(); model.set_stage('frozen'); models[arm]=model
    replayed=[]
    for index in (20,179,359):
        sid=order[index]
        shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:shot[k] for k in ('values','available')},mapping)
        predictions,_=predict_pairs(models,signal)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm,p in predictions.items(): np.testing.assert_array_equal(p,trace[arm])
        replayed.append(sid)
    comparisons={f'{k}_static_to_intra':paired_flips(full[f'{k}_static']['predictions'],full[f'{k}_intra']['predictions'])
                 for k in ARMS}
    comparisons['replay_static_to_coral_static']=paired_flips(full['replay_static']['predictions'],full['coral_static']['predictions'])
    report=dict(status='VERIFIED',rescored_traces=len(order)*len(rows),replayed_full_shots=replayed,
        replayed_full_traces=len(replayed)*len(rows),max_replay_difference=0.,
        all_update_time_boundaries_valid=True,source_archive_unchanged=True,
        update_summary=update_summary,paired_comparisons=comparisons,
        very_early_over3000ms={k:sum(p['timely'] and p['lead_ms']>3000 for p in r['predictions']) for k,r in full.items()})
    _durable_json(root/'verification.json',report)
    _durable_json(root/'operating_diagnostic.json',dict(
        scope='POSTHOC: all evaluation-period labels used for threshold sweep; NOT deployed or independent',
        thresholds_written_back=False,results=operating))
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
