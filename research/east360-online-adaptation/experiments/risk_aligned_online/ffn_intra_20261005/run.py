"""Same-parent static / fixed-loss FFN / source-learned-loss FFN EAST replay."""
import argparse
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import torch

from .. import simple_data
from ..redesign_20261004.run import (DEFAULT_SOURCE,DEFAULT_ORDER,DEFAULT_ROSTER,load_parent,
    load_roster,validate_order,validate_roster_rate_contract,run_arm)
from ..simple_online import _hash,_durable_json
from .model import FfnModel
from .loss import SignalRule
from .prepare import DEFAULT_PRIOR
from .stream import predict_shot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True,help='source-only prepare output directory')
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--prior',type=Path,default=DEFAULT_PRIOR)
    p.add_argument('--order',type=Path,default=DEFAULT_ORDER)
    p.add_argument('--roster',type=Path,default=DEFAULT_ROSTER)
    p.add_argument('--arms',nargs='+',choices=('static','fixed','learned'),default=['static','fixed','learned'])
    p.add_argument('--limit',type=int,default=120)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(131)
    parent,state,gate,prior_protocol=load_parent(args.checkpoint,args.device,
                                               persistent_site='embedding',prior=args.prior)
    model=FfnModel(parent.base)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    prepared=torch.load(args.prepared/'rule.pt',map_location='cpu',weights_only=False)
    prior_sha=_hash(args.prior)
    pp=prepared['protocol']
    if (pp['parent_sha256']!=gate['checkpoint_sha256'] or pp['prior_sha256']!=prior_sha
            or not pp['source_only'] or not pp['classifier_and_decoder_unchanged']):
        raise ValueError('rule and static parent/decoder/source-only contract mismatch')
    rule=SignalRule().to(args.device).eval().requires_grad_(False)
    rule.load_state_dict(prepared['rule'],strict=True)
    order=json.loads(args.order.read_text())
    ids,roster=load_roster(args.roster)
    validate_order(order,ids);validate_roster_rate_contract(roster,'nominal')
    if not 0<=args.limit<=len(order) or len(set(args.arms))!=len(args.arms):
        p.error('invalid limit or duplicated arms')
    target=SimpleNamespace(read_shot=partial(simple_data.read_shot,rate_contract='nominal'),
                           attach_outcome=simple_data.attach_outcome,window=simple_data.window)
    args.output.mkdir(parents=True,exist_ok=False)
    _durable_json(args.output/'order.json',order)
    _durable_json(args.output/'roster.json',roster)
    sources=list(Path(__file__).parent.glob('*.py'))
    sources+=[Path(__file__).parents[1]/n for n in
              ('simple23.py','simple_data.py','simple_train.py','simple_online.py','fast_update.py','io23.py','causal_input.py')]
    sources+=[Path(__file__).parents[1]/'redesign_20261004'/n for n in ('run.py','intra.py')]
    with zipfile.ZipFile(args.output/'code_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sources:
            z.write(f,str(f.relative_to(Path(__file__).parents[1])))
    _durable_json(args.output/'protocol.json',dict(lineage='FFN_INTRA_SOURCE_LEARNED_SIGNAL_LOSS_NOT_V3C',
        source_sha256=gate['checkpoint_sha256'],prior_sha256=prior_sha,
        rule_sha256=_hash(args.prepared/'rule.pt'),source_rule_protocol=pp,
        source_gate=gate,model_config=state['model_config'],threshold=float(state['threshold']),
        order_sha256=_hash(args.order),roster_sha256=_hash(args.roster),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        intra_parameter_count=model.delta_count,step=pp['step'],radius=pp['radius'],stride=pp['stride'],
        source_only_initialization=True,target_label_access='natural reveal after whole-shot durable seal',
        inter_updates=False,alarm_tuning=False,code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        interpretation='development; no independent validation; segment meta-training not physics proof'))
    results={}
    for arm in args.arms:
        def predictor(m,shot,intra,device,stride,step,radius,objective,data_api):
            return predict_shot(m,shot,intra,device,stride,step,radius,
                                rule=rule if arm=='learned' else None,data_api=data_api)
        results[arm]=run_arm(model,order=order,arm='static' if arm=='static' else 'intra',
            threshold=float(state['threshold']),output_dir=args.output/arm,limit=args.limit,device=args.device,
            stride=pp['stride'],step=pp['step'],radius=pp['radius'],data_api=target,predictor=predictor)
        if not all(torch.equal(v,before[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('registered source/decoder changed during stream')
    if _hash(args.checkpoint)!=gate['checkpoint_sha256'] or _hash(args.prior)!=prior_sha:
        raise RuntimeError('source/decoder artifact changed')
    _durable_json(args.output/'comparison.json',results)


if __name__=='__main__':
    main()
