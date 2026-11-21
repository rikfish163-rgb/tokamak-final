"""Matched three-head SOURCE diagnostic before any target replay promotion."""
import argparse
import json
import math
from pathlib import Path

import torch

from .. import simple_data
from ..simple_online import _durable_json,_hash
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..redesign_20261004.run import load_parent as load_prior_parent
from ..redesign_20261004.intra import _batch_windows
from ..ffn_intra_20261005.model import FfnModel
from ..ffn_intra_20261005.prepare import DEFAULT_PRIOR,full_source,evaluate
from ..ffn_intra_20261005.diagnose import diagnostic
from .model import AnchoredFfnModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--ffn-prepared',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();output=args.prepared/'source_risk_comparison.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4)
    artifact=torch.load(args.prepared/'decoder.pt',map_location='cpu',weights_only=True)
    protocol=artifact['protocol']
    if (not protocol['source_only'] or not protocol['no_east_reads'] or
            protocol['decoder_kind']!='last_observation_plus_direct_increment' or
            protocol['parent_sha256']!=_hash(DEFAULT_PARENT) or
            protocol['finite_prior_sha256']!=_hash(DEFAULT_PRIOR)):
        raise ValueError('source-only anchored-head provenance mismatch')
    previous=json.loads((args.ffn_prepared/'summary.json').read_text())['protocol']
    selection_episodes=previous['validation_episodes']
    probes=previous['calibration_episodes'][:8]
    if not {e['shot'] for e in selection_episodes}<=set(protocol['selection_ids']):
        raise ValueError('whole-shot selection must be disjoint from head fitting')
    train=set(simple_data.source_roles()['train']);advanced=simple_data.advanced_times()
    shots={}
    for e in [*selection_episodes,*probes]:
        sid=e['shot']
        if sid not in train:raise ValueError('not canonical source TRAIN')
        if sid not in shots:
            shots[sid]=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
    threshold=previous['threshold'];boundary=math.log(threshold/(1-threshold))
    pairs=[(shots[e['shot']],e) for e in selection_episodes]
    results={}
    for name in ('parent_original','finite_fast128_prior','anchored'):
        if name=='finite_fast128_prior':
            wrapper,_,_,_=load_prior_parent(DEFAULT_PARENT,args.device,persistent_site='embedding',prior=DEFAULT_PRIOR)
            model=FfnModel(wrapper.base)
        else:
            base,_=load_parent(DEFAULT_PARENT);base=base.to(args.device)
            if name=='anchored':
                base.forecast_head.load_state_dict(artifact['forecast_head'],strict=True)
                model=AnchoredFfnModel(base,initialize=False)
            else:model=FfnModel(base)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        if not results:
            results['static']=dict(whole=full_source(model,pairs,step=.1,radius=1.,rule=None,threshold=threshold,adapt=False),
                later_segments=evaluate(model,pairs,step=.1,radius=1.,rule=None,boundary=boundary,adapt=False))
        rows=[]
        for e in probes:
            s=shots[e['shot']];o=e['origins'][-1]
            signals={k:s[k] for k in ('values','available')}
            x,a=_batch_windows(simple_data,signals,[o],model,args.device)
            qx,qa=_batch_windows(simple_data,signals,e['queries'],model,args.device)
            truth=torch.as_tensor(s['values'][o+1:o+26],device=args.device)[None]
            observed=torch.as_tensor(s['available'][o+1:o+26],device=args.device)[None]
            row=diagnostic(model,x,a,truth,observed,qx,qa,e['label'],boundary,.1,1.)
            rows.append(dict(shot=s['shot'],**row))
        results[name]=dict(whole=full_source(model,pairs,step=.1,radius=1.,rule=None,threshold=threshold),
            later_segments=evaluate(model,pairs,step=.1,radius=1.,rule=None,boundary=boundary),
            eight_zero_state_probes=rows,
            finite_probe_improvement_count=sum(r['risk_change_uniform_step']<0 for r in rows))
        if not all(torch.equal(v,before[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('source replay changed fixed classifier/decoder')
        print(json.dumps(dict(stage='source_head_comparison',head=name,
            whole=results[name]['whole'],segment_loss=results[name]['later_segments']['mean_loss'],
            probe_improvements=results[name]['finite_probe_improvement_count'])),flush=True)
        _durable_json(args.prepared/'source_risk_progress.json',results)
    a=results['anchored'];f=results['finite_fast128_prior'];s=results['static']
    gate=all(a['whole']['metrics30']['tp']>=r['whole']['metrics30']['tp'] and
             a['whole']['metrics30']['fp']<=r['whole']['metrics30']['fp'] and
             a['whole']['balanced_loss']<r['whole']['balanced_loss'] and
             a['later_segments']['mean_loss']<r['later_segments']['mean_loss'] for r in (f,s))
    _durable_json(output,dict(source_only=True,head_fit_sha256=_hash(args.prepared/'decoder.pt'),
        classifier_unchanged=True,selection_episodes=selection_episodes,probe_episodes=probes,
        probes='same eight zero-delta calibration probes for continuity; some head-fit overlap, not selection evidence',
        risk_selection='32 source head-selection shots, actual from-start burn-in; not independent parent validation',
        step=.1,radius=1.,stride=26,threshold=threshold,results=results,
        source_gate_passed=gate,interpretation='no target promotion unless source gate passes; no EAST claim'))


if __name__=='__main__':main()
