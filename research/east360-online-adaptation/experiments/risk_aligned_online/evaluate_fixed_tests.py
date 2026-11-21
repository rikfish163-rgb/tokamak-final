"""Recount the unchanged qualified parent on TEST220>10ms and TEST360>30ms.

No target training, adaptation, support statistics, or threshold selection.
All probability traces are saved before accessing the corresponding outcome.
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
from .cross_channel_shot_screen import evaluate_shot
from .ended_context_experiment import side_metrics
from .fixed_test_protocol import load_test_contract,score_fixed_test
from .operating_curve_audit import load_sealed
from .redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256,load_parent
from .redesign_20261004.run import run_arm
from .simple_online import _durable_json,_hash
from .source_gate import qualify_source
from .verify_event_projection import recount_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('new output directory required')
    if not torch.cuda.is_available():raise RuntimeError('rl_env CUDA required')
    torch.set_num_threads(4);torch.manual_seed(131)
    gate=qualify_source(DEFAULT_PARENT)
    if not gate['qualified']:raise ValueError('qualified original source required')
    contracts={device:load_test_contract(device) for device in ('J-TEXT','EAST')}
    base,state=load_parent(DEFAULT_PARENT);threshold=float(state['threshold'])
    original={k:v.detach().clone() for k,v in base.state_dict().items()}
    args.output.mkdir(parents=True)
    root=Path(__file__).parent
    files=('evaluate_fixed_tests.py','fixed_test_protocol.py','cross_channel_prefix.py',
        'cross_channel_shot_screen.py','simple_data.py','io23.py','simple23.py','simple_train.py',
        'simple_online.py','causal_input.py','ended_context_experiment.py','operating_curve_audit.py',
        'verify_event_projection.py','source_gate.py','fast_update.py','persistent_ffn.py',
        'masked_prefix_train.py','redesign_20261004/run.py','redesign_20261004/intra.py',
        'redesign_20261004/prepare_prior.py')
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(root/name,name)
    _durable_json(args.output/'protocol.json',dict(
        contracts=contracts,parent_sha256=PARENT_SHA256,threshold=threshold,
        threshold_origin='existing source VAL160-selected parent checkpoint',
        source_gate=gate,target_support_used=False,target_updates=False,
        target_scaler_fit=False,threshold_reselection=False,source_retrained=False,
        first_alarm='first original score crossing; no alarm retuning',
        labels_accessed='after complete per-shot prediction seal, scoring only',
        order='ascending shot ID; no adaptive state between shots',
        rate_contract=dict(JTEXT='strict1kHz',EAST='explicit nominal1kHz; retain actual times'),
        code_snapshot_sha256=_hash(args.output/'code_snapshot.zip')))
    results={};started=time.monotonic()
    for device,contract in contracts.items():
        folder=args.output/('jtext_test220' if device=='J-TEXT' else 'east_test360')
        folder.mkdir()
        ids=contract['shot_ids'];order=[dict(shot=sid) for sid in ids]
        _durable_json(folder/'order.json',order)
        base,_=load_parent(DEFAULT_PARENT)
        model=CrossChannelModel(base,site='embedding').cuda().eval().requires_grad_(False)
        # The random auxiliary decoder is not called when intra=False.
        # Risk predictions are exactly the fixed parent with zero fast delta.
        if device=='J-TEXT':
            advanced=simple_data.advanced_times()
            advanced={sid:advanced[sid] for sid in ids if sid in advanced}
            rows=[];seals=[]
            for index,sid in enumerate(ids,1):
                row,diag=evaluate_shot(model,sid,folder,advanced,intra=False)
                if diag['accepted_updates'] or diag['attempted_updates']:
                    raise AssertionError('static evaluation must not update')
                rows.append(row);seals.append(dict(shot=sid,sha256=diag['sealed_sha256']))
                _durable_json(folder/f'{sid}_audit.json',diag)
                if index%40==0:print(json.dumps(dict(device=device,completed=index)),flush=True)
            result=score_fixed_test(device,rows,threshold)
            recount=[]
            for row,receipt in zip(rows,seals):
                sid=row['shot'];path=folder/f'{sid}.npz'
                if _hash(path)!=receipt['sha256']:raise AssertionError('source seal changed')
                shot=simple_data.attach_outcome(simple_data.read_shot(device,sid),device,advanced)
                with np.load(path,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['times'],shot['times'])
                    recount.append(dict(shot=sid,is_disrupt=shot['is_disrupt'],down=shot['down'],
                        times=saved['times'][shot['valid']],scores=saved['scores'][shot['valid']]))
            if score_fixed_test(device,recount,threshold)!=result:
                raise AssertionError('source sealed recount differs')
        else:
            provider=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
                attach_outcome=simple_data.attach_outcome,window=simple_data.window)
            emitted=run_arm(model,order=order,arm='static',threshold=threshold,
                output_dir=folder/'stream',stride=32,data_api=provider,predictor=predict_shot)
            if emitted['accepted_intra_updates'] or emitted['accepted_inter_updates']:
                raise AssertionError('static target evaluation must not update')
            signals=[simple_data.attach_outcome(provider.read_shot(device,sid),device,{}) for sid in ids]
            sealed,seals=load_sealed(folder/'stream',order,signals,threshold)
            rows=[]
            for row in sealed:
                valid=row['valid'].copy()
                if row['is_disrupt']:valid &= row['times']<=row['down']
                rows.append(dict(row,times=row['times'][valid],scores=row['scores'][valid]))
            result=score_fixed_test(device,rows,threshold)
            if result!=emitted['metrics']:raise AssertionError('target sealed recount differs')
            side=side_metrics(folder/'stream',order,threshold,0,len(ids))
            if any(result[k]!=side[k] for k in result):raise AssertionError('target side recount differs')
            decisions=[recount_prediction(r,threshold) for r in sealed]
            if decisions!=result['predictions']:raise AssertionError('independent first-alarm recount differs')
        for k,v in model.base.state_dict().items():
            torch.testing.assert_close(v.cpu(),original[k],rtol=0,atol=0)
        if model.memory.any():raise AssertionError('cross-shot memory prohibited')
        result['early_over_3000']=sum((p['lead_ms'] or 0)>3000. for p in result['predictions'] if p['timely'])
        _durable_json(folder/'metrics.json',result)
        _durable_json(folder/'verification.json',dict(contract=contract,seals=seals,
            parent_bitwise_unchanged=True,sealed_recount=True,no_updates=True))
        results[device]={k:v for k,v in result.items() if k!='predictions'}
        _durable_json(args.output/'progress.json',results)
        print(json.dumps(dict(device=device,metrics=results[device])),flush=True)
    if _hash(DEFAULT_PARENT)!=PARENT_SHA256:raise AssertionError('parent file changed')
    _durable_json(args.output/'completed.json',dict(results=results,
        elapsed_seconds=time.monotonic()-started,fixed_test_sets=True,
        parent_unchanged=True,sealed_recount=True,independent_confirmation=False))


if __name__=='__main__':main()
