"""Retrospective capacity/gradient diagnostic; NEVER an online result.

Cover every error in the fixed 120-shot static receipt. True outcomes select
the diagnostic endpoints and oracle direction. No parameter is retained,
no threshold is chosen, and no signal is used before its scheduled arrival.
The oracle is a bounded visited search, not a certified capacity upper bound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import io23, simple_data
from .causal_input import apply_input_contract
from .ended_density_replay import EndedDensityReplay
from .fast_update import normalized_step
from .routed_ffn import load_router
from .routed_ffn_weight import RoutedWeightFfnModel
from .ffn_intra_20261005.model import FfnModel
from .redesign_20261004.source_replay import (
    DEFAULT_PARENT,PARENT_SHA256,load_teacher,load_source_bank)
from .simple_online import _durable_json,_hash


def diagnostic_endpoint(times,scores,disruptive,down,threshold):
    times,scores=np.asarray(times),np.asarray(scores)
    if (times.ndim!=1 or times.shape!=scores.shape or not len(times)
            or not np.isfinite(times).all() or (np.diff(times)<=0).any()
            or not np.isfinite(scores).all()):
        raise ValueError('finite aligned chronological diagnostic curve required')
    if disruptive:
        if down is None or not np.isfinite(down): raise ValueError('known diagnostic DownTime required')
        eligible=np.flatnonzero(np.round((down-times)*1000.,9)>30.)
        endpoint=int(eligible[-1]) if len(eligible) else -1
    else:
        hit=np.flatnonzero(scores>=threshold)
        endpoint=int(hit[0]) if len(hit) else -1
    if endpoint<26: return None
    origin=((endpoint-26)//100)*100
    return dict(endpoint=endpoint,origin=origin,maturity=origin+25,effective=origin+26)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('new output required')
    root=Path(__file__).parent
    previous=root/'runs/routed_ffn_local120_20261009'
    protocol=json.loads((previous/'protocol.json').read_text())
    reported=json.loads((previous/'comparison.json').read_text())['static']['metrics']['predictions']
    if protocol['source_sha256']!=PARENT_SHA256 or len(reported)!=120:
        raise ValueError('fixed source and full diagnostic cohort required')
    bank=load_source_bank(root/'runs/redesign_source_bank_20261004.pt',expected_parent_sha256=PARENT_SHA256)
    router=load_router(previous/'source_router.pt',bank_sha256=bank['bank_sha256'])
    torch.set_num_threads(4);torch.manual_seed(131)
    base=load_teacher(DEFAULT_PARENT).to('cuda')
    model=RoutedWeightFfnModel(base,router['router']).to('cuda').eval()
    snapshot={k:v.detach().clone() for k,v in model.state_dict().items()}
    x,a=bank['x'].to('cuda'),bank['available'].to('cuda')
    reference=FfnModel(base)(x,a,base.center.new_zeros(16576,requires_grad=True))
    zero=model(x,a)
    for key in reference:
        if not torch.equal(reference[key],zero[key]): raise ValueError('CUDA zero parent equality failed')
    statistics=json.loads((previous/'static/alignment.json').read_text())
    provider=EndedDensityReplay(base.center.cpu().numpy(),base.scale.cpu().numpy(),
        output_dir=previous/'static',order=[r['shot'] for r in reported])
    records=[]
    for index,row in enumerate(reported):
        if row['timely'] or (not row['is_disrupt'] and not row['alarm']): continue
        sid=row['shot']
        receipt=json.loads((previous/'static'/f'{index+1:04d}_{sid}_seal.json').read_text())
        path=previous/'static'/receipt['npz']
        if _hash(path)!=receipt['sha256']: raise ValueError('changed diagnostic seal')
        with np.load(path,allow_pickle=False) as stored:
            times,scores=stored['times'].copy(),stored['scores'].copy()
        disruptive,down=io23.read_outcome('EAST',sid)
        if bool(disruptive)!=row['is_disrupt']: raise ValueError('changed HDF5 diagnostic outcome')
        point=diagnostic_endpoint(times,scores,disruptive,down,protocol['threshold'])
        record=dict(shot=sid,is_disrupt=bool(disruptive),category='timely_miss' if disruptive else 'safe_false_alarm')
        if point is None:
            records.append(dict(record,eligible=False,reason='no matured H25 before chosen endpoint'))
            continue
        provider.statistics=({k:np.asarray(v) for k,v in statistics.items()} if index>=30 else None)
        shot=provider.reencode(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')))
        if not np.array_equal(times,shot['times']): raise ValueError('diagnostic signal grid changed')
        def window(end):
            values,available=simple_data.window(shot,end,128)
            return torch.as_tensor(values,device='cuda')[None],torch.as_tensor(available,device='cuda')[None]
        xo,ao=window(point['origin']);xe,ae=window(point['endpoint'])
        truth=torch.as_tensor(shot['values'][point['origin']+1:point['maturity']+1],device='cuda')[None]
        seen=torch.as_tensor(shot['available'][point['origin']+1:point['maturity']+1],device='cuda')[None]
        fast=model.initial_fast(1).requires_grad_()
        signal_loss=model.forecast_loss(model(xo,ao,fast)['forecast'],truth,seen)
        gradient,=torch.autograd.grad(signal_loss,fast)
        risk_fast=model.initial_fast(1).requires_grad_()
        output=model(xe,ae,risk_fast)['logit']
        target=output.new_full(output.shape,float(disruptive))
        risk_gradient,=torch.autograd.grad(F.binary_cross_entropy_with_logits(output,target),risk_fast)
        before=float(output.detach().sigmoid()[0])
        if abs(before-float(scores[point['endpoint']]))>2e-6:
            raise ValueError('zero diagnostic endpoint differs from stored static curve')
        updated=normalized_step(fast.detach(),torch.zeros_like(fast),gradient.detach(),step=.05,radius=1.)
        one_signal=float(model(xe,ae,updated)['logit'].detach().sigmoid()[0])
        current=model.initial_fast(1)
        best=before
        for _ in range(32):
            current=current.detach().requires_grad_()
            logit=model(xe,ae,current)['logit']
            g,=torch.autograd.grad(F.binary_cross_entropy_with_logits(logit,target),current)
            current=normalized_step(current.detach(),torch.zeros_like(current),g.detach(),step=.1,radius=1.)
            value=float(model(xe,ae,current)['logit'].detach().sigmoid()[0])
            best=max(best,value) if disruptive else min(best,value)
        records.append(dict(record,eligible=True,**point,before=before,after_one_signal=one_signal,
            oracle_best_visited=best,
            signal_correct=bool(one_signal>=protocol['threshold']) if disruptive else bool(one_signal<protocol['threshold']),
            oracle_correct=bool(best>=protocol['threshold']) if disruptive else bool(best<protocol['threshold']),
            gradient_cosine=float(F.cosine_similarity(gradient,risk_gradient,dim=-1)[0]),
            risk_gradient_norm=float(risk_gradient.double().norm()),
            signal_gradient_norm=float(gradient.double().norm()),
            lead_ms=round((down-times[point['endpoint']])*1000.,9) if disruptive else None))
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in snapshot.items()):
        raise ValueError('diagnostic retained a parameter update')
    result=dict(scope='POSTHOC ORACLE AND GRADIENT DIAGNOSTIC; NOT ONLINE PERFORMANCE',
        future_labels_used=True,model_weights_changed=False,memory_retained=False,
        source_sha256=PARENT_SHA256,parent_zero_cuda_exact=True,source_bank_windows=len(x),
        reference_protocol_sha256=_hash(previous/'protocol.json'),threshold=protocol['threshold'],
        parameter_count=65536,gate_normalization='unit_l2',signal_step=.05,
        oracle_steps=32,oracle_step=.1,radius=1.,source_collateral_guard=False,
        endpoint_rule='all static120 errors; last strict timely point or first actual safe alarm',
        schedule='latest actual stride100 origin matured before chosen endpoint; fast starts zero in probe',
        records=records)
    _durable_json(args.output,result)
    print(dict(output=str(args.output),errors=len(records),eligible=sum(r['eligible'] for r in records),
        oracle_correct=sum(r.get('oracle_correct',False) for r in records),
        signal_correct=sum(r.get('signal_correct',False) for r in records)),flush=True)


if __name__=='__main__':main()
