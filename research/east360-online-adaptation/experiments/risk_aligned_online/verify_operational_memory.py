"""Posthoc audit of sealed curves and the actually applied full-replay updates."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .ended_context_experiment import REFERENCE, side_metrics
from .operational_memory import _digest
from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256
from .simple_online import _durable_json, _hash

ROOT=Path(__file__).parent
ARM_FOLDERS={'intra_control':'intra_control','inter_only':'inter_only_v2','both':'both_v2'}


def verify_receipt(row, *, index, ended_ids):
    if index<=30:
        assert not row['accepted'] and row['actual_steps']==0
        return
    assert row['effective_from_index']==index+1 and row['full_guard_replay']
    assert row['history_ids']==ended_ids[max(0,index-32):index]
    if not row['actual_steps']:
        assert not row['accepted']
        return
    assert row['actual_steps']==1
    guard=row['guard'];source=row['source_guard']
    assert guard['checked_shots']==len(row['history_ids'])
    assert [s['shot'] for s in guard['before']]==row['history_ids']
    assert [s['shot'] for s in guard['after']]==row['history_ids']
    lost_tp=[a['shot'] for a,b in zip(guard['before'],guard['after']) if a['timely'] and not b['timely']]
    lost_tn=[a['shot'] for a,b in zip(guard['before'],guard['after'])
             if not a['label'] and not a['false_alarm'] and b['false_alarm']]
    assert guard['lost_tp']==lost_tp and guard['lost_tn']==lost_tn
    assert guard['passed']==(not(lost_tp or lost_tn))
    assert source['checked_points']==128
    assert source['passed']==(source['protected_tp']+source['protected_tn']>0
                             and source['lost_tp']==source['lost_tn']==0)
    assert row['accepted']==bool(guard['passed'] and source['passed']
                               and row['task_gain']>row['task_tolerance'])
    assert math.isclose(row['task_gain'],row['task_before']-row['task_after'],abs_tol=1e-12)
    assert sorted(r['label'] for r in row['proposal']['task_records'])==[0,1]
    assert row['pair_ids'][-1] in row['history_ids']
    assert ended_ids[index-1] in row['pair_ids']
    baseline={s['shot']:s for s in guard['before']}
    for record in row['proposal']['task_records']:
        reference=baseline[record['shot']]
        assert record['query']==reference['query'] and record['label']==reference['label']
        assert abs(record['probability']-reference['probability'])<2e-6
    if row['accepted']:
        assert row['memory_change_norm']>0 and row['memory_before_sha256']!=row['memory_after_sha256']
    else:
        assert row['memory_change_norm']==0 and row['memory_before_sha256']==row['memory_after_sha256']


def load_curve(folder,index,sid):
    stream=folder/'stream'
    seal=json.loads((stream/f'{index:04d}_{sid}_seal.json').read_text())
    audit=json.loads((stream/f'{index:04d}_{sid}_audit.json').read_text())
    assert _hash(stream/seal['npz'])==seal['sha256']==audit['seal_sha256']
    with np.load(stream/seal['npz'],allow_pickle=False) as trace:
        return trace['scores'].copy(),trace['times'].copy(),audit


def flips(reference,candidate):
    assert [r['shot'] for r in reference]==[r['shot'] for r in candidate]
    out={k:[] for k in ('rescued_disruptive','lost_disruptive','removed_false_alarm','new_false_alarm')}
    for a,b in zip(reference,candidate):
        if a['is_disrupt']:
            if not a['timely'] and b['timely']:out['rescued_disruptive'].append(a['shot'])
            if a['timely'] and not b['timely']:out['lost_disruptive'].append(a['shot'])
        else:
            if a['alarm'] and not b['alarm']:out['removed_false_alarm'].append(a['shot'])
            if not a['alarm'] and b['alarm']:out['new_false_alarm'].append(a['shot'])
    return out


def verify(root, *, completed_only=False):
    folders={arm:root/name for arm,name in ARM_FOLDERS.items()}
    missing=[a for a,f in folders.items() if not (f/'completed.json').exists()]
    if missing and not completed_only:
        raise ValueError(f'full comparison is not complete: {missing}')
    if 'intra_control' in missing:
        raise ValueError('completed control is required even for explicit partial verification')
    folders={a:f for a,f in folders.items() if a not in missing}
    order=json.loads((folders['intra_control']/'order.json').read_text())
    ids=[r['shot'] for r in order]
    assert order==json.loads((REFERENCE/'order.json').read_text())[:len(order)]
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    protocols={a:json.loads((f/'protocol.json').read_text()) for a,f in folders.items()}
    snapshots={}
    for arm,folder in folders.items():
        p=protocols[arm]
        assert (folder/'completed.json').exists() and json.loads((folder/'order.json').read_text())==order
        assert p['arm']==arm and p['parent_sha256']==PARENT_SHA256
        assert p['order_sha256']==_hash(REFERENCE/'order.json')
        assert p['code_snapshot_sha256']==_hash(folder/'code_snapshot.zip')
        with zipfile.ZipFile(folder/'code_snapshot.zip') as archive:
            snapshots[arm]={name:archive.read(name) for name in archive.namelist()}
    control=snapshots['intra_control']
    for arm in folders.keys()-{'intra_control'}:
        assert set(control)==set(snapshots[arm])
        for name,body in control.items():
            if name=='operational_memory.py':
                # The control never instantiates this learner. Its archived
                # version predates ONLY the tested JSON float serialization fix.
                old=b'tolerance=8*np.finfo(np.float32).eps*max(abs(before_loss),abs(after_loss),1e-12)'
                new=b'tolerance=float(8*np.finfo(np.float32).eps*max(abs(before_loss),abs(after_loss),1e-12))'
                body=body.replace(old,new)
            assert body==snapshots[arm][name],(arm,name)
    threshold=protocols['intra_control']['threshold']
    assert all(p['threshold']==threshold and p['input_mapping']==protocols['intra_control']['input_mapping']
               for p in protocols.values())
    results={};curves={};heads={}
    for arm,folder in folders.items():
        periods={p:side_metrics(folder/'stream',order,threshold,a,b)
                 for p,a,b in (('all',0,len(ids)),('first30',0,30),('after30',30,len(ids)))}
        saved=json.loads((folder/'comparison.json').read_text())
        assert saved==periods
        for period in periods.values():
            rows=period['predictions']
            assert period['tp']==sum(r['timely'] for r in rows)
            assert period['fp']==sum(not r['is_disrupt'] and r['alarm'] for r in rows)
            assert math.isclose(period['tpr'],period['tp']/period['n_disrupt'])
            assert math.isclose(period['fpr'],period['fp']/period['n_safe'])
        curves[arm]={};memory_rows=[];head_steps=0
        for index,sid in enumerate(ids,1):
            scores,times,audit=load_curve(folder,index,sid)
            assert audit['fast_discarded'] and audit['model_parameter_change_norm']==0
            assert audit['model_buffer_change_norm']==0
            curves[arm][sid]=(scores,times)
            row=audit['persistent']['memory']
            head_steps+=audit['persistent']['head']['actual_steps']
            if arm!='intra_control':verify_receipt(row,index=index,ended_ids=ids)
            else:assert row['actual_steps']==0 and not row['accepted']
            memory_rows.append(row)
        state=torch.load(folder/'final_state.pt',map_location='cpu',weights_only=True)
        completed=json.loads((folder/'completed.json').read_text())
        assert state['ended_ids']==ids and state['parent_sha256']==PARENT_SHA256
        assert completed['source_file_and_tensors_unchanged'] and completed['head_optimizer_steps']==40
        assert state['head_optimizer_steps']==40 and head_steps==152
        assert torch.isfinite(state['memory']).all() and state['memory'].double().norm()<=1.00001
        if arm=='intra_control':assert not state['memory'].any()
        else:assert _digest(state['memory'])==memory_rows[-1]['memory_after_sha256']
        heads[arm]=state['decoder']
        accepted=sum(r['accepted'] for r in memory_rows)
        assert completed['memory_updates']==accepted
        assert completed['memory_proposals']==sum(r['actual_steps'] for r in memory_rows)
        results[arm]=dict(metrics={p:{k:v for k,v in m.items() if k!='predictions'} for p,m in periods.items()},
            memory_updates=accepted,memory_norm=float(state['memory'].double().norm()),
            decisions=dict(Counter(r['reason'] for r in memory_rows)),
            gradient_probability_max_error=max((abs(r['probability']-next(
                s['probability'] for s in row['guard']['before'] if s['shot']==r['shot']))
                for row in memory_rows if row['actual_steps'] for r in row['proposal']['task_records']),default=0),
            predictions=periods['all']['predictions'])
    for arm in folders.keys()-{'intra_control'}:
        for name,tensor in heads['intra_control'].items():
            torch.testing.assert_close(heads[arm][name],tensor,rtol=0,atol=0)
    if 'both' in curves:
        for sid in ids[:30]:
            for a,b in zip(curves['intra_control'][sid],curves['both'][sid]):assert np.array_equal(a,b)
    archived=ROOT/'runs/peak_bag120_20261010/single_early_fixed'
    for index,sid in enumerate(ids,1):
        scores,times,_=load_curve(archived,index,sid)
        assert np.array_equal(scores,curves['intra_control'][sid][0])
        assert np.array_equal(times,curves['intra_control'][sid][1])
    differences={a:flips(results['intra_control']['predictions'],results[a]['predictions'])
                 for a in folders.keys()-{'intra_control'}}
    static_folder=ROOT/'runs/ended_decoder120_20261010/static'
    static_periods={p:side_metrics(static_folder,order,threshold,a,b)
                    for p,a,b in (('all',0,len(ids)),('first30',0,30),('after30',30,len(ids)))}
    versus_static={a:flips(static_periods['all']['predictions'],results[a]['predictions'])
                   for a in folders}
    return dict(verification='PASS_COMPLETED_ARMS_ONLY' if missing else 'PASS',
        missing_complete_arms=missing,all_arms_complete=not missing,
        scope='development pilot, no independent future validation',
        source_sha256=PARENT_SHA256,sealed_curves_checked=len(folders)*len(ids),
        source_and_alarm_unchanged=True,all_heads_bitexact=True,archived_control_curves_bitexact=True,
        control_unused_module_only_json_float_fix=True,results=results,versus_intra_control=differences,
        static_reference_metrics={p:{k:v for k,v in m.items() if k!='predictions'}
                                  for p,m in static_periods.items()},versus_mapped_static=versus_static)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--completed-only',action='store_true',
        help='explicitly verify completed arms only; report every missing arm and do not certify the full comparison')
    args=parser.parse_args()
    if args.output.exists():parser.error('new verification output required')
    result=verify(args.root,completed_only=args.completed_only);_durable_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
