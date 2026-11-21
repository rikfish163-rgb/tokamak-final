"""Post-completion rescore, causal receipt checks, and actual CUDA replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..simple_train import sha
from ..simple_online import _durable_json
from ..fixed_test_protocol import load_test_contract,score_fixed_test
from ..deepcoral_prefix_20261022.run import predict_pairs
from ..deepcoral_prefix_20261022.audit import paired_flips
from ..head_operator_20261022.audit import validate_update
from ..warmstart_alignment_20261022.alignment import transform
from ..operating_curve_audit import operating_curve,operational_maxima
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from .run import ARMS,REFERENCE,MAP_PATH,load_models
from .learning import episode


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    done=json.loads((root/'completed.json').read_text());assert done['status']=='COMPLETED'
    protocol=json.loads((root/'protocol.json').read_text());order=load_test_contract('EAST')['shot_ids']
    for path,h in protocol['code_sha256'].items():assert sha(io23.ROOT/path)==h
    prepared=Path(protocol['prepared'])
    assert sha(prepared/'protocol.json')==protocol['preparation_protocol_sha256']
    assert sha(prepared/'prepare_completed.json')==protocol['preparation_completion_sha256']
    assert sha(MAP_PATH)==protocol['mapping_sha256']
    assert sha(REFERENCE/'protocol.json')==protocol['reference_protocol_sha256']
    prep=json.loads((prepared/'protocol.json').read_text())
    prep_done=json.loads((prepared/'prepare_completed.json').read_text())
    assert sha(prepared/'training_manifest.jsonl')==prep_done['training_manifest_sha256']
    assert not set(prep['head_fit_ids'])&set(prep['head_report_ids'])
    assert set(prep['source_train_ids'])==set(io23.source_roles()['train'])
    assert set(prep['head_fit_ids'])|set(prep['head_report_ids'])==set(prep['source_train_ids'])
    manifest=[json.loads(line) for line in (prepared/'training_manifest.jsonl').read_text().splitlines()]
    assert len(manifest)==256
    for index,row in enumerate(manifest):
        assert row['step']==index and [r['label'] for r in row['episodes']]==[0,1]
        assert all(r['shot'] in prep['head_fit_ids'] and r['query']>=256 for r in row['episodes'])
    seals=json.loads((root/'prediction_receipts.json').read_text())
    assert [r['shot'] for r in seals]==order
    recovered=None
    if 'resumed_from' in done:
        recovery_path=root/f"resume_{done['resumed_from']}.json"
        assert sha(recovery_path)==done['resume_receipt_sha256']
        recovered=json.loads(recovery_path.read_text())
        assert recovered['status']=='VERIFIED_PREFIX' and recovered['target_outcomes_accessed'] is False
        assert recovered['hashes_captured_at_recovery'] is True
        assert recovered['original_protocol_sha256']==sha(root/'protocol.json')
        assert recovered['resume_code_sha256']==sha(Path(__file__).with_name('resume.py'))
        assert recovered['recovered_shots']==done['resumed_from']
        assert recovered['recovered_file_hashes']==seals[:done['resumed_from']]
        assert recovered['replayed_last_shot']==order[done['resumed_from']-1]
    activation=json.loads((root/'activation_receipt.json').read_text())
    assert activation['after_seals']==seals[:20] and activation['mapping_sha256']==sha(MAP_PATH)
    refs={r['shot']:r for r in json.loads((REFERENCE/'prediction_receipts.json').read_text())}
    mapping=json.loads(MAP_PATH.read_text());assert mapping['calibration_ids']==order[:20]
    full=json.loads((root/'per_shot.json').read_text());rows={k:[] for k in ARMS}
    stats={k:dict(updates=0,loss_decreased=0,max_step=0.) for k in ('forecast','meta')}
    for index,(sid,seal) in enumerate(zip(order,seals)):
        path=root/'shots'/f'{sid}.npz';upath=root/'shots'/f'{sid}_updates.json'
        refpath=REFERENCE/'shots'/f'{sid}.npz'
        assert sha(path)==seal['prediction_sha256'] and sha(upath)==seal['updates_sha256']
        assert sha(refpath)==refs[sid]['prediction_sha256']
        shot=simple_data.attach_outcome(simple_data.read_shot('EAST',sid,rate_contract='nominal'),'EAST',{})
        with np.load(path,allow_pickle=False) as trace,np.load(refpath,allow_pickle=False) as old:
            np.testing.assert_array_equal(trace['times'],shot['times'])
            np.testing.assert_array_equal(trace['static'],old['static'])
            np.testing.assert_array_equal(trace['original'],old['encoder_original'])
            for arm in ARMS:
                if index<20:np.testing.assert_array_equal(trace[arm],trace['static'])
                else:np.testing.assert_array_equal(trace[arm][:256],trace['static'][:256])
                rows[arm].append(dict(shot=sid,times=shot['times'],scores=trace[arm].copy(),valid=shot['valid'],
                    is_disrupt=shot['is_disrupt'],down=shot['down']))
        updates=json.loads(upath.read_text())
        if index<20:assert not updates
        else:
            assert set(updates)=={'forecast','meta'}
            for arm,us in updates.items():
                assert [u['first_affected_prediction'] for u in us]==list(range(256,len(shot['values']),128))
                for u in us:
                    validate_update(arm,u,len(shot['values']))
                    if u['gradient_steps']:
                        stats[arm]['updates']+=1
                        stats[arm]['loss_decreased']+=u['loss_after']<u['loss_before']
                        stats[arm]['max_step']=max(stats[arm]['max_step'],u['parameter_delta_norm'])
    operating={}
    for arm,rs in rows.items():
        cropped=[dict(r,times=r['times'][r['valid']],scores=r['scores'][r['valid']]) for r in rs]
        assert score_fixed_test('EAST',cropped,protocol['threshold'])==full[arm]
        operating[arm]={}
        for period,records in (('full360',rs),('common_suffix340',rs[20:])):
            curve=operating_curve(records,protocol['threshold']);peaks=operational_maxima(records)
            d,s=peaks['positive'][:,None],peaks['negative'][None,:]
            for k in ('tp','fp','tpr','fpr','J'):assert curve['fixed_source_threshold'][k]==done['metrics'][arm][period][k]
            operating[arm][period]=dict(timely_peak_ranking_auc=float(((d>s)+.5*(d==s)).mean()),**curve)
    torch.set_num_threads(2);models,_=load_models(prepared)
    replayed=[]
    for index in (20,179,359):
        sid=order[index];shot=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:shot[k] for k in ('values','available')},mapping)
        fresh,updates=predict_pairs(models,signal)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
            for arm in models:
                np.testing.assert_array_equal(fresh[f'{arm}_static'],trace['static'])
                np.testing.assert_array_equal(fresh[f'{arm}_intra'],trace[arm])
        assert updates==json.loads((root/'shots'/f'{sid}_updates.json').read_text())
        replayed.append(sid)
    # Recompute two complete source burn-ins for both FINAL heads. These are
    # source TRAIN reporting cases, not target labels or epoch selectors.
    report=json.loads((prepared/'source_report.json').read_text())
    advanced=simple_data.advanced_times();source_replayed=[]
    for index in (0,1):
        ref=report['records']['meta'][index];sid=ref['shot']
        assert sid in prep['head_report_ids']
        shot=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
        assert shot['valid'][ref['query']] and shot['y'][ref['query']]==ref['label']
        signal={k:shot[k] for k in ('values','available')}
        for arm,m in models.items():
            r=episode(m,signal,ref['query'],float(ref['label']),device='cuda',create_graph=False)
            for key in ('risk_before','risk_after','forecast'):
                assert float(r[key].detach())==report['records'][arm][index][key]
        source_replayed.append(dict(shot=sid,query=ref['query']))
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    comparisons={f'static_to_{a}':paired_flips(full['static']['predictions'],full[a]['predictions']) for a in ARMS[1:]}
    comparisons['forecast_to_meta']=paired_flips(full['forecast']['predictions'],full['meta']['predictions'])
    receipt=dict(status='VERIFIED',rescored_traces=len(order)*len(ARMS),replayed_full_shots=replayed,
        replayed_unique_full_traces=3*len(replayed),max_replay_difference=0.,
        recovery_checked=recovered is not None,
        recovered_prefix_count=0 if recovered is None else recovered['recovered_shots'],
        source_final_head_episodes_replayed=source_replayed,source_archive_unchanged=True,
        target_update_boundaries_valid=True,training_manifest_checked=True,update_summary=stats,
        paired_comparisons=comparisons,
        very_early_over3000ms={k:sum(p['timely'] and p['lead_ms']>3000 for p in f['predictions']) for k,f in full.items()})
    _durable_json(root/'verification.json',receipt)
    _durable_json(root/'operating_diagnostic.json',dict(scope='POSTHOC all-stream-label threshold sweep, not deployed',
        thresholds_written_back=False,results=operating))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
