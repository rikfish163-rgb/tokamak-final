"""Shared sealed-curve verification for explicitly separated pair hypotheses.

MNN target-only: all static curves must reproduce the old trajectory exactly.
Density mean/std: first30 static AND intra curves must reproduce the old ones,
but later curves intentionally have different numerical inputs. Source weights
and alarm remain frozen. Source-token site: static matches all old curves, but
the changed intra path is not required to match old FFN updates. Reuse is only
for verification, never for pooling different experimental contributions.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import io23,simple_data
from .causal_input import apply_input_contract
from .efd_normalization import load_source_statistics
from .ended_context_experiment import REFERENCE
from .operating_curve_audit import load_sealed, operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash
from .verify_ended_decoder import classify
from .verify_operational_intra_pair import verify_prediction_audit
from .verify_operational_memory import flips

ROOT=Path(__file__).parent
OLD=ROOT/'runs/operational_intra_pair240_v2_20261010'
PERIODS=(('all',0,240),('first30',0,30),('after30',30,240),
         ('first120',0,120),('last120',120,240))


def verify_protocol_pair(protocols):
    for arm,p in protocols.items():
        if (p['arm']!=arm or p['signal_contract']!='without_mnn'
                or p['input_feature_count']!=23 or p['auxiliary_excluded_features']!=['mode_number_n']
                or p['mode_number_n_is_assumed_categorical'] or p['persistent_enabled']
                or p['intra']!=(arm=='intra_control') or p['alarm_tuning']
                or not p['auxiliary_postseal_feedback_in_all_arms']
                or not p['auxiliary_head_frozen_after30']):
            raise ValueError('explicit same-input MNN auxiliary-only ablation pair required')
    for key in ('parent_sha256','threshold','order_sha256','input_mapping','seed',
                'first30_steps','bootstrap_steps','intra_step','intra_radius','stride','horizon'):
        if protocols['mapped_static'][key]!=protocols['intra_control'][key]:
            raise ValueError(f'unmatched pair field: {key}')


def verify_efd_protocol_pair(protocols):
    for arm,p in protocols.items():
        if (p['arm']!=arm or p['signal_contract']!='all23' or p['auxiliary_excluded_features']
                or p['input_feature_count']!=23 or p['normalization']!='efd_density_zscore'
                or p['normalization_channels']!=['ne_nG'] or not p['source_moments_file_sha256']
                or p['persistent_enabled'] or p['alarm_tuning'] or p['intra']!=(arm=='intra_control')
                or not p['auxiliary_postseal_feedback_in_all_arms']
                or not p['auxiliary_head_frozen_after30']):
            raise ValueError('explicit density-only mean/std pair with original signal targets required')
    for key in ('parent_sha256','threshold','order_sha256','input_mapping','seed','source_moments_file_sha256',
                'first30_steps','bootstrap_steps','intra_step','intra_radius','stride','horizon'):
        if protocols['mapped_static'][key]!=protocols['intra_control'][key]:
            raise ValueError(f'unmatched pair field: {key}')


def verify_token_protocol_pair(protocols):
    for arm,p in protocols.items():
        if (p['arm']!=arm or p['temporary_site']!='source_tokens' or p['temporary_count']!=128
                or not p['temporary_is_offset_from_parent_origin'] or p['persistent_site']!='not_present'
                or p['signal_contract']!='all23' or p['auxiliary_excluded_features']
                or p['normalization']!='density_robust' or p['source_moments_file_sha256'] is not None
                or p['normalization_channels']!=['ne_nG'] or p['input_feature_count']!=23
                or p['persistent_enabled'] or p['alarm_tuning'] or p['intra']!=(arm=='intra_control')
                or not p['auxiliary_postseal_feedback_in_all_arms']
                or not p['auxiliary_head_frozen_after30']):
            raise ValueError('explicit original source-token offset pair with unchanged input/loss required')
    for key in ('parent_sha256','threshold','order_sha256','input_mapping','seed',
                'first30_steps','bootstrap_steps','intra_step','intra_radius','stride','horizon'):
        if protocols['mapped_static'][key]!=protocols['intra_control'][key]:
            raise ValueError(f'unmatched pair field: {key}')


def curve_error(left,right):
    if len(left)!=len(right):raise ValueError('equal-length comparison scopes required')
    error=0.
    for old,new in zip(left,right):
        assert np.array_equal(old['times'],new['times'])
        error=max(error,float(np.max(np.abs(old['scores']-new['scores']))))
    return error


def verify_density_statistics(root,signals,protocols):
    prefix=[apply_input_contract(s) for s in signals[:30]]
    samples=[np.concatenate([s['values'][:,c][s['available'][:,c]&np.isfinite(s['values'][:,c])]
                             for s in prefix]).astype(np.float64) for c in range(23)]
    for arm,p in protocols.items():
        path=root/arm/'source_moments.json'
        assert _hash(path)==p['source_moments_file_sha256']
        source,_=load_source_statistics(path,expected_parent_sha256=PARENT_SHA256,
                                        train_ids=simple_data.source_roles()['train'])
        target=json.loads((root/arm/'normalization.json').read_text())
        assert target['calibration_ids']==[s['shot'] for s in prefix]
        assert target['statistics_mode']=='fixed' and not target['future_signals_used']
        for c,v in enumerate(samples):
            assert target['counts'][c]==len(v)
            if not len(v):
                assert target['mean'][c]==target['std'][c]==0.
            else:
                np.testing.assert_allclose(target['mean'][c],v.mean(),rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(target['std'][c],v.std(ddof=0),rtol=1e-12,atol=1e-12)
    c=io23.RAW23.index('ne_nG')
    return dict(target_record_ids=[s['shot'] for s in prefix],all_target_observations=int(target['counts'][c]),
        source_mean=float(source['mean'][c]),source_std=float(source['std'][c]),
        target_mean=float(target['mean'][c]),target_std=float(target['std'][c]),
        target_statistics_recomputed_independently=True,source_train_only=True,physical_unit_certified=False)


def verify_robust_density_statistics(root,signals):
    prefix=[apply_input_contract(s) for s in signals[:30]];c=io23.RAW23.index('ne_nG')
    values=[]
    for s in prefix:
        v=s['values'][:,c][s['available'][:,c]&np.isfinite(s['values'][:,c])]
        if len(v):values.append(v[::max(1,len(v)//128)])
    if not values:raise ValueError('no density observations for declared prefix')
    v=np.concatenate(values);q25,median,q75=np.quantile(v,[.25,.5,.75])
    for arm in ('mapped_static','intra_control'):
        r=json.loads((root/arm/'normalization.json').read_text())
        assert r['calibration_ids']==[s['shot'] for s in prefix] and r['usable'][c]
        assert r['counts'][c]==len(v)
        assert r['center'][c]==float(np.float32(median)) and r['scale'][c]==float(np.float32(q75-q25))
    return dict(contract='same locked robust density mapping',calibration_ids=r['calibration_ids'],
        sampled_count=len(v),center=r['center'][c],scale=r['scale'][c],physical_unit_certified=False)


def sealed_decisions(records,threshold):
    # classify already returns is_disrupt; add only the shot identifier.
    return [dict(shot=r['shot'],**classify(r['times'],r['scores'],r['valid'],threshold=threshold,
                                         disruptive=r['is_disrupt'],down=r['down'])) for r in records]


def verify(root,*,hypothesis='mnn_target'):
    if hypothesis not in ('mnn_target','density_mean_std','source_token_site'):raise ValueError('explicit pair hypothesis required')
    density=hypothesis=='density_mean_std';token=hypothesis=='source_token_site'
    protocols={a:json.loads((root/a/'protocol.json').read_text()) for a in ('mapped_static','intra_control')}
    (verify_token_protocol_pair if token else verify_efd_protocol_pair if density else verify_protocol_pair)(protocols)
    order=json.loads((REFERENCE/'order.json').read_text())
    assert len(order)==240 and _hash(DEFAULT_PARENT)==PARENT_SHA256
    threshold=protocols['intra_control']['threshold']
    signals=[simple_data.attach_outcome(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'),
                                        'EAST',{}) for r in order]
    curves={};metrics={};snapshots={};heads={};checks=0;predictions={}
    for name,folder in [('mapped_static',root/'mapped_static'),('intra_control',root/'intra_control'),
                        ('old_mapped_static',OLD/'mapped_static'),('old_intra_control',OLD/'intra_control')]:
        assert json.loads((folder/'order.json').read_text())==order
        p=json.loads((folder/'protocol.json').read_text())
        assert p['parent_sha256']==PARENT_SHA256 and p['threshold']==threshold
        assert p['order_sha256']==_hash(REFERENCE/'order.json') and not p['alarm_tuning']
        records,seal_info=load_sealed(folder/'stream',order,signals,threshold)
        checks+=len(records);curves[name]=records
        metrics[name]={period:operating_curve(records[start:stop],threshold)
                       for period,start,stop in PERIODS}
        stored=json.loads((folder/'comparison.json').read_text())
        predictions[name]=sealed_decisions(records,threshold)
        assert len(stored['all']['predictions'])==len(predictions[name])
        for actual,saved in zip(predictions[name],stored['all']['predictions']):
            assert all(actual[k]==saved[k] for k in ('shot','is_disrupt','alarm','timely','lead_ms'))
        for period in stored:
            fixed=metrics[name][period]['fixed_source_threshold']
            assert all(fixed[k]==stored[period][k] for k in ('tp','fp','tpr','fpr','J'))
        if name.startswith('old_'):continue
        assert p['code_snapshot_sha256']==_hash(folder/'code_snapshot.zip')
        with zipfile.ZipFile(folder/'code_snapshot.zip') as z:
            snapshots[name]={n:z.read(n) for n in z.namelist()}
        completed=json.loads((folder/'completed.json').read_text())
        state=torch.load(folder/'final_state.pt',map_location='cpu',weights_only=True)
        cumulative=0
        for index,item in enumerate(order,1):
            audit=json.loads((folder/'stream'/f'{index:04d}_{item["shot"]}_audit.json').read_text())
            verify_prediction_audit(audit,index=index,intra=name=='intra_control')
            cumulative+=audit['persistent']['head']['actual_steps']
        assert cumulative==152 and state['head_optimizer_steps']==40
        assert state['ended_ids']==[r['shot'] for r in order]
        if token:assert state['memory'] is None,'token-only pilot must not manufacture FFN memory'
        else:assert not state['memory'].any()
        assert completed['source_file_and_tensors_unchanged'] and completed['observed_shots']==240
        assert completed['memory_updates']==completed['memory_proposals']==0
        heads[name]=state['decoder']
    assert snapshots['mapped_static']==snapshots['intra_control']
    for key,v in heads['mapped_static'].items():
        torch.testing.assert_close(v,heads['intra_control'][key],rtol=0,atol=0)
    comparison_count=30 if density else 240
    static_error=curve_error(curves['old_mapped_static'][:comparison_count],curves['mapped_static'][:comparison_count])
    assert static_error==0.,'input/classification changed outside the declared comparison scope'
    intra_prefix_error=curve_error(curves['old_intra_control'][:30],curves['intra_control'][:30]) if density else None
    if density:assert intra_prefix_error==0.,'density transition altered the pre-transition learning trajectory'
    normalization_evidence=(verify_robust_density_statistics(root,signals) if token else
                            verify_density_statistics(root,signals,protocols) if density else None)
    return dict(verification='PASS',scope='DEVELOPMENT ablation, not independent future validation',
        hypothesis=hypothesis,code_sha256=_hash(Path(__file__)),
        parent_sha256=PARENT_SHA256,order_sha256=_hash(REFERENCE/'order.json'),
        actual_seal_hash_checks=checks,signal_contract=protocols['intra_control']['signal_contract'],input_feature_count=23,
        temporary_site='source_tokens' if token else 'final_ffn',
        categorical_claim=False,static_curve_max_error=static_error,
        static_reference_compared_shots=comparison_count,intra_first30_reference_error=intra_prefix_error,
        normalization_evidence=normalization_evidence,
        auxiliary_heads_bitexact=True,cumulative_head_steps=152,final_adam_steps=40,
        persistent_ffn_updates=0,threshold_unchanged=True,results=metrics,
        versus_same_head_static=flips(predictions['mapped_static'],predictions['intra_control']),
        versus_original_intra=flips(predictions['old_intra_control'],predictions['intra_control']),
        threshold_sweep_is_posthoc_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hypothesis',choices=('mnn_target','density_mean_std','source_token_site'),default='mnn_target')
    args=parser.parse_args()
    if args.output.exists():parser.error('new verification file required')
    result=verify(args.root,hypothesis=args.hypothesis);_durable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
    print(json.dumps({a:r['all']['fixed_source_threshold'] for a,r in result['results'].items()}),flush=True)


if __name__=='__main__':main()
