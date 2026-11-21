"""Matched source-only ablation: forecast fitting vs risk-through-FFN fitting.

The qualified classifier, threshold, FFN update rule and source scales do not
change. Both arms start from the original head and see the SAME source episodes.
This is an auxiliary-head development screen, not JTEXT online deployment.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import (
    load_parent, load_source_train, classifier_state, assert_classifier_unchanged, DEFAULT_PARENT,
)
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.source import split_shots
from ..ffn_intra_20261005.prepare import build_pairs, full_source, evaluate
from .matched_meta import enable_head_only, maintenance_loss, train_pair
from .prepare import risk_probe


def selection_report(model,pairs,threshold):
    boundary=math.log(threshold/(1-threshold))
    return dict(whole=full_source(model,pairs,step=.1,radius=1.,rule=None,threshold=threshold),
        later=evaluate(model,pairs,step=.1,radius=1.,rule=None,boundary=boundary),
        maintenance=float(np.mean([float(maintenance_loss(model,s,e).detach()) for s,e in pairs])))


def gate(candidate, initial, static):
    """No target-driven admission; require actual source risk + alarm preservation."""
    r=candidate['whole'];m=r['metrics30']
    return bool(candidate['maintenance']<=1.10*initial['maintenance'] and all(
        m['tp']>=ref['whole']['metrics30']['tp'] and m['fp']<=ref['whole']['metrics30']['fp']
        and r['balanced_loss']<ref['whole']['balanced_loss']
        and candidate['later']['mean_loss']<ref['later']['mean_loss'] for ref in (initial,static)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffn-prepared',type=Path,required=True)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=131)
    args=p.parse_args()
    if args.steps<1:p.error('positive source iteration count required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);started=time.monotonic()
    previous=json.loads((args.ffn_prepared/'summary.json').read_text())['protocol']
    parent_hash=_hash(DEFAULT_PARENT)
    if not previous['source_only'] or previous['parent_sha256']!=parent_hash:
        raise ValueError('matched source episode provenance required')
    shots,ids=load_source_train();lookup={s['shot']:s for s in shots}
    fit,selection=split_shots(shots,args.seed)
    selection_ids={s['shot'] for s in selection}
    val=[(lookup[e['shot']],e) for e in previous['validation_episodes']]
    if not {s['shot'] for s,_ in val}<=selection_ids:
        raise ValueError('selection episodes overlap head fitting or are not canonical TRAIN')
    fit_episodes=[build_pairs(fit,rng,1) for _ in range(args.steps)]
    base,state=load_parent(DEFAULT_PARENT);base=base.to(args.device)
    model=FfnModel(base);before=classifier_state(base)
    original={k:v.detach().cpu().clone() for k,v in base.forecast_head.state_dict().items()}
    threshold=float(state['threshold']);boundary=math.log(threshold/(1-threshold))
    reference_risk=risk_probe(model,val[0][0])
    dependencies=[Path(__file__),Path(__file__).with_name('matched_meta.py'),
        Path(__file__).parents[1]/'simple23.py',Path(__file__).parents[1]/'fast_update.py',
        *[Path(__file__).parents[1]/'ffn_intra_20261005'/f for f in ('source.py','model.py','loss.py','stream.py')]]
    code={str(f.resolve()):_hash(f) for f in dependencies}
    protocol=dict(source_only=True,no_east_reads=True,parent_sha256=parent_hash,
        decoder_kind='original_absolute_direct_H25',source_train_ids=ids,
        fit_ids=[s['shot'] for s in fit],selection_ids=sorted(selection_ids),
        validation_episodes=[e for _,e in val],fit_episodes=[[e for _,e in pair] for pair in fit_episodes],
        seed=args.seed,steps=args.steps,lr=.00003,weight_decay=0.,gradient_clip=1.,
        arms=['signal_only','risk_aligned'],initial_head='original qualified source; NOT fast128 prior or anchored head',
        step=.1,radius=1.,stride=26,inner_updates=4,
        burn_in='real 0,26,... start; gradient truncated only before last four updates',
        objective='risk_aligned: later-segment risk + .20 zero-state signal; signal_only: .20 zero-state signal',
        outer_positive='source D >30 to <=125ms auxiliary bag; original source point labels unchanged',
        source_classifier_frozen=True,source_scaler_frozen=True,threshold=threshold,alarm_tuning=False,
        selection='every20 plus final; admit only no TP/FP regressions and both risk losses improve vs original-head adaptation AND static; forecast<=1.10 initial',
        maintenance_tolerance='10% predeclared engineering safeguard, not physical guarantee',
        code_sha256=code,hardware=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
        timing='concurrent full2953 replay; not an isolated or real-time latency benchmark',
        interpretation='source auxiliary development holdout; classifier previously trained on all TRAIN')
    _durable_json(args.output/'protocol.json',protocol)
    static=dict(whole=full_source(model,val,step=.1,radius=1.,rule=None,threshold=threshold,adapt=False),
        later=evaluate(model,val,step=.1,radius=1.,rule=None,boundary=boundary,adapt=False))
    initial=selection_report(model,val,threshold)
    _durable_json(args.output/'baseline.json',dict(static=static,original_head=initial))
    results={}
    for arm in protocol['arms']:
        base.forecast_head.load_state_dict(original);enable_head_only(model)
        optimizer=torch.optim.Adam(base.forecast_head.parameters(),lr=.00003)
        history=[];selections=[];best_step=None;best=None;best_loss=float('inf')
        arm_dir=args.output/arm;arm_dir.mkdir()
        for iteration,pair in enumerate(fit_episodes,1):
            row=train_pair(model,pair,optimizer,mode=arm,step=.1,radius=1.,boundary=boundary)
            history.append(dict(iteration=iteration,**row))
            if iteration%10==0 or iteration==args.steps:
                print(json.dumps(dict(stage='matched_head_fit',arm=arm,iteration=iteration,
                    objective=row['objective'],gradient=row['head_gradient_norm'],elapsed=time.monotonic()-started)),flush=True)
            if iteration%20==0 or iteration==args.steps:
                report=selection_report(model,val,threshold);admitted=gate(report,initial,static)
                selections.append(dict(iteration=iteration,source_gate_passed=admitted,**report))
                current={k:v.detach().cpu().clone() for k,v in base.forecast_head.state_dict().items()}
                torch.save(dict(forecast_head=current,protocol=protocol,arm=arm,iteration=iteration,
                                source_gate_passed=admitted),arm_dir/f'candidate_{iteration:03d}.pt')
                if admitted and report['whole']['balanced_loss']<best_loss:
                    best=current;best_step=iteration;best_loss=report['whole']['balanced_loss']
                print(json.dumps(dict(stage='matched_head_selection',arm=arm,iteration=iteration,
                    source_gate_passed=admitted,whole=report['whole']['balanced_loss'],
                    later=report['later']['mean_loss'],maintenance=report['maintenance'],
                    tp=report['whole']['metrics30']['tp'],fp=report['whole']['metrics30']['fp'])),flush=True)
                _durable_json(arm_dir/'training.json',dict(history=history,selections=selections))
        assert_classifier_unchanged(base,before)
        if not torch.equal(reference_risk,risk_probe(model,val[0][0])):
            raise RuntimeError('auxiliary-head fitting altered static risk output')
        if best is not None:
            torch.save(dict(forecast_head=best,protocol=protocol,arm=arm,iteration=best_step,
                            source_gate_passed=True),arm_dir/'admitted.pt')
        results[arm]=dict(source_gate_passed=best is not None,selected_step=best_step,selections=selections)
    if _hash(DEFAULT_PARENT)!=parent_hash or any(_hash(Path(f))!=h for f,h in code.items()):
        raise RuntimeError('parent or training code changed during experiment')
    _durable_json(args.output/'summary.json',dict(protocol=protocol,static=static,initial=initial,
        results=results,classifier_unchanged=True,static_risk_probe_bit_exact=True,
        elapsed_seconds=time.monotonic()-started,interpretation='source gate only; no EAST online gain established'))


if __name__=='__main__':main()
