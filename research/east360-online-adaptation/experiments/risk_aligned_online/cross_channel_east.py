"""Pure-intra EAST test: source assets + only the present shot's prefix.

No first30 alignment statistics, support110, target-label updates, or alarm
retuning. Source checks and target diagnostics are recorded separately.
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
from .cross_channel_prefix import CrossChannelModel,predict_shot
from .ended_context_experiment import side_metrics
from .operating_curve_audit import load_sealed,operating_curve
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json,_hash
from .startup_sensitivity import REFERENCE
from .verify_event_projection import recount_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heads',type=Path,required=True)
    p.add_argument('--source-screen',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    screen=json.loads(args.source_screen.read_text())
    if screen.get('east_data_used') is not False:raise ValueError('source-only preparation required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(131)
    order=json.loads((REFERENCE/'order.json').read_text())
    if len(order)!=240 or len({int(r['shot']) for r in order})!=240:
        raise ValueError('same fixed240 development roster required')
    _durable_json(args.output/'order.json',order)
    base,parent=load_parent(DEFAULT_PARENT);threshold=float(parent['threshold'])
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    root=Path(__file__).parent
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in ('cross_channel_east.py','cross_channel_prefix.py','cross_channel_operational.py',
            'simple_data.py','simple23.py','io23.py','causal_input.py','fast_update.py',
            'simple_online.py','simple_train.py','ended_context_experiment.py','operating_curve_audit.py',
            'verify_event_projection.py','redesign_20261004/intra.py','redesign_20261004/run.py'):
            z.write(root/name,name)
    protocol=dict(scope='PURE INTRA EAST DEVELOPMENT REPLAY, NOT CONFIRMATORY VALIDATION',
        parent_sha256=PARENT_SHA256,threshold=threshold,source_screen=screen,
        source_screen_sha256=_hash(args.source_screen),order_sha256=_hash(REFERENCE/'order.json'),
        head_sha256={p.name:_hash(p) for p in args.heads.glob('*_head.pt')},
        arms=['static','signal_embedding','risk_embedding'],all_startup_errors_counted=True,
        source_scaler_only=True,target_support=False,target_statistics_from_other_shots=False,
        target_labels_used_for_update=False,first_update_index=7,interval_points=32,
        step=.05,radius=1.,temporary_parameters_reset_every_shot=True,
        preserved_alarm='first crossing of original source threshold',
        success='TPR strict lead>30ms improves AND FPR<=.20 AND J improves',
        real_time_latency_validated=False,
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'))
    _durable_json(args.output/'protocol.json',protocol)
    provider=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
                            attach_outcome=simple_data.attach_outcome,window=simple_data.window)
    started=time.monotonic();results={};curves={};baseline=None;signals=None
    for arm in protocol['arms']:
        folder=args.output/arm;folder.mkdir()
        base,_=load_parent(DEFAULT_PARENT)
        model=CrossChannelModel(base,site='embedding').cuda().eval().requires_grad_(False)
        branch='risk_embedding' if arm=='risk_embedding' else 'signal'
        checkpoint=torch.load(args.heads/f'{branch}_head.pt',map_location='cpu',weights_only=True)
        if checkpoint['parent_sha256']!=PARENT_SHA256 or checkpoint['branch']!=branch:
            raise ValueError('source auxiliary provenance mismatch')
        model.decoder.load_state_dict(checkpoint['decoder'])
        run=run_arm(model,order=order,arm='static' if arm=='static' else 'intra',
            threshold=threshold,output_dir=folder/'stream',stride=32,
            predictor=predict_shot,data_api=provider)
        m=side_metrics(folder/'stream',order,threshold,0,len(order))
        if any(m[k]!=run['metrics'][k] for k in ('tp','fp','tpr','fpr','J')):
            raise AssertionError('replay and side recount disagree')
        if signals is None:
            signals=[simple_data.attach_outcome(provider.read_shot('EAST',r['shot']),'EAST',{}) for r in order]
        sealed,hashes=load_sealed(folder/'stream',order,signals,threshold)
        decisions=[recount_prediction(r,threshold) for r in sealed]
        if sum(r['timely'] for r in decisions)!=m['tp'] or sum(
                r['alarm'] and not r['is_disrupt'] for r in decisions)!=m['fp']:
            raise AssertionError('independent first-alarm recount disagreement')
        curves[arm]=operating_curve(sealed,threshold)
        results[arm]={k:v for k,v in m.items() if k!='predictions'}
        results[arm]['accepted_updates']=run['accepted_intra_updates']
        if arm=='static':
            baseline=results[arm]
            old,_=load_sealed(REFERENCE/'static/stream',order,signals,threshold)
            error=max(float(np.max(np.abs(a['scores']-b['scores']))) for a,b in zip(sealed,old))
            if error>2e-6 or (m['tp'],m['fp'])!=(82,82):raise AssertionError('original static not reproduced')
            results[arm]['old_static_max_error']=error
        else:
            results[arm]['meets_main_goal']=bool(m['tpr']>baseline['tpr'] and m['fpr']<=.20 and m['J']>baseline['J'])
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v.cpu(),initial[k],rtol=0,atol=0)
        assert not model.memory.any()
        _durable_json(folder/'verification.json',dict(decisions=decisions,seals=hashes,
            parent_bitwise_unchanged=True,reset_temporary_state=True,independent_sealed_recount=True))
        _durable_json(args.output/'progress.json',results)
        print(json.dumps(dict(completed=arm,metrics=results[arm])),flush=True)
    assert _hash(DEFAULT_PARENT)==PARENT_SHA256
    _durable_json(args.output/'operating_curves.json',curves)
    _durable_json(args.output/'completed.json',dict(results=results,
        independent_sealed_recount=True,elapsed_seconds=time.monotonic()-started,
        no_inter_shot_state=True,no_target_labels_for_updates=True))


if __name__=='__main__':main()
