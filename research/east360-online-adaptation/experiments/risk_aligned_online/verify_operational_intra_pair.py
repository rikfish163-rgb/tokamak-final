"""Verify the locked 240-shot fast/no-fast pair and original prefix reproduction."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import simple_data
from .density_mapping_audit import RAW, MAPPED
from .ended_context_experiment import REFERENCE
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash
from .verify_operational_memory import flips

ROOT=Path(__file__).parent


def verify_prediction_audit(audit,*,index,intra):
    assert audit['intra']==intra and audit['fast_discarded']
    assert audit['model_parameter_change_norm']==audit['model_buffer_change_norm']==0
    feedback=audit['persistent']
    assert isinstance(feedback,dict),'same explicit postseal auxiliary feedback required'
    head=feedback['head'];memory=feedback['memory']
    assert head['effective_from_index']==index+1
    assert head['actual_steps'] in ((0,4) if index<30 else (40,) if index==30 else (0,))
    assert memory['actual_steps']==0 and not memory['accepted']


def verify(root):
    order=json.loads((root/'intra_control/order.json').read_text())
    assert len(order)==240 and order==json.loads((REFERENCE/'order.json').read_text())
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    protocols={arm:json.loads((root/arm/'protocol.json').read_text()) for arm in ('mapped_static','intra_control')}
    threshold=protocols['intra_control']['threshold']
    snapshots={};states={};heads={};results={};curves={};seals={}
    signals=[simple_data.attach_outcome(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'),
                                       'EAST',{}) for r in order]
    for arm in protocols:
        folder=root/arm;p=protocols[arm];intra=arm=='intra_control'
        completed=json.loads((folder/'completed.json').read_text())
        assert json.loads((folder/'order.json').read_text())==order
        assert p['arm']==arm and p['selected_shots']==len(order) and p['parent_sha256']==PARENT_SHA256
        assert p['threshold']==threshold and p['order_sha256']==_hash(REFERENCE/'order.json')
        assert p['intra']==intra and not p['persistent_enabled'] and not p['alarm_tuning']
        assert p['auxiliary_postseal_feedback_in_all_arms'] and p['auxiliary_head_frozen_after30']
        assert p['code_snapshot_sha256']==_hash(folder/'code_snapshot.zip')
        with zipfile.ZipFile(folder/'code_snapshot.zip') as z:
            snapshots[arm]={name:z.read(name) for name in z.namelist()}
        records,seals[arm]=load_sealed(folder/'stream',order,signals,threshold)
        curves[arm]=records
        results[arm]={period:operating_curve(records[start:stop],threshold)
            for period,start,stop in (('all',0,len(order)),('first30',0,30),('after30',30,len(order)),
                                       ('first120',0,120),('last120',120,len(order)))}
        comparison=json.loads((folder/'comparison.json').read_text())
        cumulative_steps=0
        for index,item in enumerate(order,1):
            audit=json.loads((folder/'stream'/f'{index:04d}_{item["shot"]}_audit.json').read_text())
            verify_prediction_audit(audit,index=index,intra=intra)
            cumulative_steps+=audit['persistent']['head']['actual_steps']
        state=torch.load(folder/'final_state.pt',map_location='cpu',weights_only=True)
        assert state['ended_ids']==[r['shot'] for r in order] and state['parent_sha256']==PARENT_SHA256
        assert not state['memory'].any() and state['head_optimizer_steps']==40 and cumulative_steps==152
        assert completed['source_file_and_tensors_unchanged'] and completed['observed_shots']==240
        assert completed['head_optimizer_steps']==40 and completed['memory_updates']==completed['memory_proposals']==0
        for period in comparison:
            fixed=results[arm][period]['fixed_source_threshold']
            assert all(fixed[k]==comparison[period][k] for k in ('tp','fp','tpr','fpr','J'))
        heads[arm]=state['decoder'];states[arm]=comparison
    assert snapshots['mapped_static']==snapshots['intra_control'],'same entry and all model/training code required'
    assert protocols['mapped_static']['input_mapping']==protocols['intra_control']['input_mapping']
    for name,v in heads['mapped_static'].items():
        torch.testing.assert_close(v,heads['intra_control'][name],rtol=0,atol=0)
    archived=dict(mapped_static=MAPPED,intra_control=ROOT/'runs/operational_memory120_20261010/intra_control/stream')
    max_error={}
    for arm,folder in archived.items():
        old,_=load_sealed(folder,order[:120],signals[:120],threshold)
        error=0.
        for original,new in zip(old,curves[arm][:120]):
            assert np.array_equal(original['times'],new['times'])
            error=max(error,float(np.max(np.abs(original['scores']-new['scores']))))
        assert error==0.,'240-shot extension must reproduce the exact original 120-shot candidate'
        max_error[arm]=error
    raw,seals['raw_static']=load_sealed(RAW,order,signals,threshold)
    results['raw_static']={period:operating_curve(raw[start:stop],threshold)
        for period,start,stop in (('all',0,len(order)),('first30',0,30),('after30',30,len(order)),
                                   ('first120',0,120),('last120',120,len(order)))}
    return dict(verification='PASS',scope='DEVELOPMENT locked240; no independent future confirmation',
        source_sha256=PARENT_SHA256,code_sha256=_hash(Path(__file__)),source_and_alarm_unchanged=True,
        primary_sealed_curves_checked=720,archived_prefix_curves_checked=240,
        actual_seal_hash_checks=960,auxiliary_heads_bitexact=True,head_cumulative_steps=152,
        head_final_adam_steps=40,persistent_ffn_updates=0,prefix_curve_max_error=max_error,
        order_sha256=_hash(REFERENCE/'order.json'),results=results,seals=seals,
        versus_mapped_static=flips(states['mapped_static']['all']['predictions'],states['intra_control']['all']['predictions']),
        threshold_sweep_is_posthoc_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new verification output required')
    result=verify(args.root);_durable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','seals')}),flush=True)
    print(json.dumps(dict(all={arm:dict(fixed=row['all']['fixed_source_threshold'],
        caps=row['all']['posthoc_best_at_cap']) for arm,row in result['results'].items()})),flush=True)


if __name__=='__main__':main()
