"""One source-only change: replace late-bag outer risk with causal whole-shot risk.

Reuses the completed matched experiment's exact source pairs, epochs, optimiser,
head initialisation and admission test. No extra parameter sweep or EAST fit.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from ..simple_online import _hash, _durable_json
from ..redesign_20261004.prepare_prior import (
    load_parent, load_source_train, classifier_state, assert_classifier_unchanged, DEFAULT_PARENT,
)
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.source import split_shots
from .matched_meta import enable_head_only
from .prepare_matched import selection_report, gate
from .prepare import risk_probe
from .whole_meta import train_pair


def publish_admitted(output,best,protocol,iteration,parent_hash,code):
    """An integrity failure must never leave an apparently admitted artifact."""
    if _hash(DEFAULT_PARENT)!=parent_hash or any(_hash(Path(f))!=h for f,h in code.items()):
        raise RuntimeError('parent or training code changed during experiment')
    if best is not None:
        torch.save(dict(forecast_head=best,protocol=protocol,iteration=iteration,source_gate_passed=True),
                   output/'admitted.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    old=json.loads((args.matched/'summary.json').read_text())
    previous=old['protocol'];parent_hash=_hash(DEFAULT_PARENT)
    if (not previous['source_only'] or previous['parent_sha256']!=parent_hash
            or (previous['steps'],previous['lr'],previous['step'],previous['radius'],previous['stride'])
            !=(60,.00003,.1,1.,26)):
        raise ValueError('completed fixed-budget matched experiment required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(previous['seed']);started=time.monotonic()
    shots,ids=load_source_train();lookup={s['shot']:s for s in shots}
    fit,selection=split_shots(shots,previous['seed'])
    if (ids!=previous['source_train_ids'] or [s['shot'] for s in fit]!=previous['fit_ids']
            or sorted(s['shot'] for s in selection)!=previous['selection_ids']):
        raise ValueError('canonical source split changed')
    val=[(lookup[e['shot']],e) for e in previous['validation_episodes']]
    if not {s['shot'] for s,_ in val}<=set(previous['selection_ids']):
        raise ValueError('selection/fit overlap')
    fit_episodes=[[(lookup[e['shot']],e) for e in pair] for pair in previous['fit_episodes']]
    if (len(fit_episodes)!=60 or any(s['shot'] not in previous['fit_ids']
                                   for pair in fit_episodes for s,_ in pair)):
        raise ValueError('exact matched 60 fit pairs required')
    base,state=load_parent(DEFAULT_PARENT);base=base.to(args.device)
    model=FfnModel(base);before=classifier_state(base);enable_head_only(model)
    threshold=float(state['threshold']);boundary=math.log(threshold/(1-threshold))
    reference_risk=risk_probe(model,val[0][0])
    dependencies=[Path(__file__),Path(__file__).with_name('whole_meta.py'),
        Path(__file__).with_name('matched_meta.py'),Path(__file__).with_name('prepare_matched.py'),
        Path(__file__).parents[1]/'simple23.py',Path(__file__).parents[1]/'fast_update.py',
        *[Path(__file__).parents[1]/'ffn_intra_20261005'/f for f in ('prepare.py','source.py','model.py','loss.py','stream.py')]]
    code={str(f.resolve()):_hash(f) for f in dependencies}
    protocol=dict(previous,arms=['risk_aligned_whole'],objective='whole causal source peak risk + .20 unchanged zero-state signal',
        outer_positive='all source endpoints strict lead>30ms; no125ms upper bound, original point labels unchanged',
        inner_updates='actual0..4 final mature events; real burn-in before them, gradients truncated',
        peak='earliest maximizer of emitted float32 probability in whole source evaluation set',
        matched_run=str(args.matched.resolve()),matched_summary_sha256=_hash(args.matched/'summary.json'),
        changed_factor='ONLY outer risk endpoint selection: late bag -> whole shot peak',code_sha256=code,
        preupdate_peaks='kept in loss and evaluation; zero risk gradient to head, never discarded',
        no_east_reads='source-only loader and code path; not a separate filesystem I/O sandbox',
        torch_version=str(torch.__version__))
    _durable_json(args.output/'protocol.json',protocol)
    initial=selection_report(model,val,threshold)
    for name,value in [('whole',initial['whole']['balanced_loss']),('later',initial['later']['mean_loss'])]:
        expected=old['initial'][name]['balanced_loss' if name=='whole' else 'mean_loss']
        if abs(value-expected)>2e-6:
            raise RuntimeError('original-head baseline differs from matched reference')
    static=old['static']
    _durable_json(args.output/'baseline.json',dict(static=static,initial=initial))
    optimizer=torch.optim.Adam(base.forecast_head.parameters(),lr=.00003)
    history=[];selections=[];best=None;best_loss=float('inf');best_step=None
    for iteration,pair in enumerate(fit_episodes,1):
        row=train_pair(model,pair,optimizer,step=.1,radius=1.,boundary=boundary)
        history.append(dict(iteration=iteration,**row))
        if iteration%10==0:
            print(json.dumps(dict(stage='whole_head_fit',iteration=iteration,objective=row['objective'],
                                  elapsed=time.monotonic()-started)),flush=True)
        if iteration%20==0:
            report=selection_report(model,val,threshold);admitted=gate(report,initial,static)
            selections.append(dict(iteration=iteration,source_gate_passed=admitted,**report))
            current={k:v.detach().cpu().clone() for k,v in base.forecast_head.state_dict().items()}
            torch.save(dict(forecast_head=current,protocol=protocol,iteration=iteration,
                            source_gate_passed=admitted),args.output/f'candidate_{iteration:03d}.pt')
            if admitted and report['whole']['balanced_loss']<best_loss:
                best=current;best_step=iteration;best_loss=report['whole']['balanced_loss']
            print(json.dumps(dict(stage='whole_head_selection',iteration=iteration,source_gate_passed=admitted,
                whole=report['whole']['balanced_loss'],later=report['later']['mean_loss'],
                maintenance=report['maintenance'],tp=report['whole']['metrics30']['tp'],
                fp=report['whole']['metrics30']['fp'])),flush=True)
            _durable_json(args.output/'training.json',dict(history=history,selections=selections))
    assert_classifier_unchanged(base,before)
    if not torch.equal(reference_risk,risk_probe(model,val[0][0])):
        raise RuntimeError('auxiliary fitting changed static risk output')
    publish_admitted(args.output,best,protocol,best_step,parent_hash,code)
    _durable_json(args.output/'summary.json',dict(protocol=protocol,static=static,initial=initial,
        selections=selections,selected_step=best_step,source_gate_passed=best is not None,
        classifier_unchanged=True,static_risk_probe_bit_exact=True,
        preupdate_fit_peaks=sum(r['before_first_update'] for h in history for r in h['records']),
        fit_events=len(history)*2,elapsed_seconds=time.monotonic()-started,
        interpretation='source development gate, no EAST adaptation gain established'))


if __name__=='__main__':main()
