"""Fixed-parent, fixed-alarm TEST360: feedback views x encoding-step bound.

This is a development experiment. The cap is source-only; the selected method
and hypothesis have been informed by earlier EAST development failures.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from . import simple_data
from .cross_channel_stable import HEAD,FourViewModel,predict_shot
from .ended_context_experiment import side_metrics
from .fixed_test_protocol import load_test_contract,score_fixed_test
from .operating_curve_audit import load_sealed,operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json,_hash
from .verify_event_projection import recount_prediction

ARMS=('static','rotate_unbounded','joint_unbounded','rotate_limited','joint_limited')


def locked_test_order():
    # Classification fields in the manifest must never reach the predictor.
    return [dict(shot=sid) for sid in load_test_contract('EAST')['shot_ids']]


def read_cap(path):
    data=json.loads(Path(path).read_text())
    if (data.get('source_outcomes_read') is not False or data.get('east_data_used') is not False
            or data.get('parent_sha256')!=PARENT_SHA256 or data.get('head_sha256')!=_hash(HEAD)):
        raise ValueError('source-only calibration with same parent and head required')
    cap=float(data['encoding_rms_cap'])
    if not np.isfinite(cap) or cap<=0:raise ValueError('positive finite source cap required')
    return cap,data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    cap,calibration=read_cap(args.calibration)
    torch.set_num_threads(4);torch.manual_seed(131);args.output.mkdir(parents=True)
    order=locked_test_order()
    _durable_json(args.output/'order.json',order)
    base,parent=load_parent(DEFAULT_PARENT);threshold=float(parent['threshold'])
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_factorial.py','fixed_test_protocol.py','cross_channel_stable.py','cross_channel_prefix.py',
            'simple23.py','simple_data.py','io23.py','causal_input.py','fast_update.py',
            'simple_online.py','simple_train.py','ended_context_experiment.py','operating_curve_audit.py',
            'verify_event_projection.py','redesign_20261004/intra.py','redesign_20261004/run.py',
            'redesign_20261004/prepare_prior.py'):
            z.write(root/name,name)
    protocol=dict(scope='FIXED TEST360 FACTORIAL; DEVELOPMENT-EXPOSED, NOT INDEPENDENT CONFIRMATION',arms=ARMS,
        test_contract=load_test_contract('EAST'),
        parent_sha256=PARENT_SHA256,head_sha256=_hash(HEAD),threshold=threshold,
        calibration=calibration,calibration_sha256=_hash(args.calibration),
        order_sha256=_hash(args.output/'order.json'),first_update_index=7,interval_points=32,
        feedback=['rotate one of four','mean of four'],encoding_step_cap=cap,
        computed_views_per_update=4,step=.05,radius=1.,
        source_scaler_only=True,target_support=False,target_statistics_from_other_shots=False,
        target_labels_used_for_update=False,temporary_parameters_reset_every_shot=True,
        all_startup_errors_counted=True,alarms='first crossing of original source threshold',
        primary_goal='TPR strict lead>30ms improves AND FPR<=.20 AND J improves',
        limitation='encoding bound is per-step on currently observed window, not future risk guarantee',
        method_design_uses_prior_target_development=True,real_time_latency_validated=False,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    provider=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
        attach_outcome=simple_data.attach_outcome,window=simple_data.window)
    results={};curves={};signals=None;static_decisions=None;started=time.monotonic()
    for arm in ARMS:
        folder=args.output/arm;folder.mkdir()
        base,_=load_parent(DEFAULT_PARENT)
        model=FourViewModel(base,site='embedding').cuda().eval().requires_grad_(False)
        head=torch.load(HEAD,map_location='cpu',weights_only=True)
        if head['parent_sha256']!=PARENT_SHA256 or head['branch']!='signal':
            raise ValueError('source auxiliary head mismatch')
        model.decoder.load_state_dict(head['decoder'])
        feedback='joint' if arm.startswith('joint') else 'rotate'
        bound=cap if arm.endswith('_limited') else None
        result=run_arm(model,order=order,arm='static' if arm=='static' else 'intra',
            threshold=threshold,output_dir=folder/'stream',stride=32,data_api=provider,
            predictor=partial(predict_shot,feedback=feedback,cap=bound))
        metric=side_metrics(folder/'stream',order,threshold,0,len(order))
        if any(metric[k]!=result['metrics'][k] for k in ('tp','fp','tpr','fpr','J')):
            raise AssertionError('side recount disagrees with emitted predictions')
        if signals is None:
            signals=[simple_data.attach_outcome(provider.read_shot('EAST',r['shot']),'EAST',{}) for r in order]
        sealed,hashes=load_sealed(folder/'stream',order,signals,threshold)
        scored=[]
        for row in sealed:
            valid=row['valid'].copy()
            if row['is_disrupt']:valid &= row['times']<=row['down']
            scored.append(dict(row,times=row['times'][valid],scores=row['scores'][valid]))
        fixed=score_fixed_test('EAST',scored,threshold)
        if fixed!=result['metrics']:
            raise AssertionError('fixed test contract differs from replay metrics')
        decisions=[recount_prediction(r,threshold) for r in sealed]
        if (sum(r['timely'] for r in decisions)!=metric['tp'] or
                sum(r['alarm'] and not r['is_disrupt'] for r in decisions)!=metric['fp']):
            raise AssertionError('independent sealed first-alarm recount disagrees')
        results[arm]={k:v for k,v in metric.items() if k!='predictions'}
        results[arm]['accepted_updates']=result['accepted_intra_updates']
        curves[arm]=operating_curve(sealed,threshold)
        if arm=='static':
            static_decisions=decisions
        else:
            baseline=results['static']
            results[arm]['meets_main_goal']=bool(metric['tpr']>baseline['tpr'] and
                metric['fpr']<=.20 and metric['J']>baseline['J'])
            results[arm]['flips_vs_static']=dict(
                disrupt_saved=[n['shot'] for b,n in zip(static_decisions,decisions) if n['is_disrupt'] and n['timely'] and not b['timely']],
                disrupt_lost=[n['shot'] for b,n in zip(static_decisions,decisions) if n['is_disrupt'] and b['timely'] and not n['timely']],
                false_alarms_removed=[n['shot'] for b,n in zip(static_decisions,decisions) if not n['is_disrupt'] and b['alarm'] and not n['alarm']],
                false_alarms_added=[n['shot'] for b,n in zip(static_decisions,decisions) if not n['is_disrupt'] and n['alarm'] and not b['alarm']])
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),initial[k],rtol=0,atol=0)
        assert not model.memory.any()
        _durable_json(folder/'verification.json',dict(decisions=decisions,seals=hashes,
            parent_bitwise_unchanged=True,reset_temporary_state=True,independent_sealed_recount=True))
        _durable_json(args.output/'progress.json',results)
        print(json.dumps(dict(completed=arm,tp=metric['tp'],fp=metric['fp'],J=metric['J'])),flush=True)
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'operating_curves.json',curves)
    _durable_json(args.output/'completed.json',dict(results=results,
        independent_sealed_recount=True,elapsed_seconds=time.monotonic()-started,
        no_inter_shot_state=True,no_target_labels_for_updates=True))


if __name__=='__main__':main()
