"""Isolate unlabeled EAST intra-shot transfer; no persistent target training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from . import io23
from .simple23 import Simple23,Config
from .simple_online import run_arm,_durable_json,_hash
from .source_gate import qualify_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--order',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gate=qualify_source(args.checkpoint)
    if not gate['qualified']:
        raise ValueError(gate['reasons'])
    order=json.loads(args.order.read_text())
    ids=[int(r['shot']) for r in order]
    if len(ids)!=360 or len(set(ids))!=360 or set(ids)!=set(io23.cohort_ids('EAST')):
        raise ValueError('complete fixed EAST360 order required')
    torch.set_num_threads(4)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    net=Simple23(Config(**c['model_config']),c['source_center'],c['source_scale']).cuda().eval()
    net.load_state_dict(c['model'])
    args.output.mkdir(parents=True)
    _durable_json(args.output/'protocol.json',dict(source_gate=gate,
        checkpoint_sha256=_hash(args.checkpoint),order_sha256=_hash(args.order),
        threshold=c['threshold'],model_config=c['model_config'],scope='DEVELOPMENT_ONLY',
        persistent_updates=False,startup_budget_reporting_only=30,
        code_sha256={n:_hash(Path(__file__).with_name(n)) for n in
                     ('evaluate_transfer_fast.py','simple_online.py','simple23.py','fast_update.py',
                      'simple_data.py','io23.py','simple_train.py')}))
    outputs={}
    for arm in ('static','intra'):
        model=deepcopy(net)
        outputs[arm]=run_arm(model,order=order,arm=arm,threshold=c['threshold'],
                            output_dir=args.output/arm,device='cuda',limit=360,startup_budget=30)
        if any(not torch.equal(v,net.state_dict()[k]) for k,v in model.state_dict().items()):
            raise AssertionError('temporary adaptation modified persistent model')
        print(arm,json.dumps({k:v for k,v in outputs[arm]['metrics_full'].items() if k!='predictions'}),flush=True)
    _durable_json(args.output/'summary.json',outputs)


if __name__=='__main__':
    main()
