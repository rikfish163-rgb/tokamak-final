"""Eight fixed source-calibration episodes: decoder accuracy and gradient scope."""
import argparse
import json
import math
from pathlib import Path

import torch

from .. import simple_data,io23
from ..simple_online import _hash,_durable_json
from ..redesign_20261004.run import load_parent,DEFAULT_SOURCE
from ..redesign_20261004.intra import _batch_windows
from .model import FfnModel
from .prepare import DEFAULT_PRIOR
from .diagnose import diagnostic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads((args.prepared/'summary.json').read_text())['protocol']
    if (_hash(DEFAULT_SOURCE)!=protocol['parent_sha256'] or _hash(DEFAULT_PRIOR)!=protocol['prior_sha256']):
        raise ValueError('source/decoder differs from the prepared rule')
    parent,state,_,_=load_parent(DEFAULT_SOURCE,args.device,persistent_site='embedding',prior=DEFAULT_PRIOR)
    model=FfnModel(parent.base);torch.set_num_threads(4)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    threshold=protocol['threshold'];boundary=math.log(threshold/(1-threshold))
    train=set(simple_data.source_roles()['train'])
    advanced=simple_data.advanced_times()
    rows=[]
    for e in protocol['calibration_episodes'][:8]:
        sid=e['shot']
        if sid not in train:raise ValueError('not a canonical TRAIN shot')
        shot=simple_data.attach_outcome(simple_data.read_shot('J-TEXT',sid),'J-TEXT',advanced)
        origin=e['origins'][-1]
        if min(e['queries'])<=origin+25 or int(shot['is_disrupt'])!=e['label']:
            raise ValueError('diagnostic query must follow maturity with matched source outcome')
        signals={k:shot[k] for k in ('values','available')}
        x,a=_batch_windows(simple_data,signals,[origin],model,args.device)
        qx,qa=_batch_windows(simple_data,signals,e['queries'],model,args.device)
        truth=torch.as_tensor(shot['values'][origin+1:origin+26],device=args.device)[None]
        observed=torch.as_tensor(shot['available'][origin+1:origin+26],device=args.device)[None]
        row=diagnostic(model,x,a,truth,observed,qx,qa,e['label'],boundary,protocol['step'],protocol['radius'])
        rows.append(dict(shot=sid,episode=e,**row))
    if not all(torch.equal(v,before[k]) for k,v in model.state_dict().items()):
        raise RuntimeError('diagnostic changed registered model state')
    _durable_json(args.output,dict(scope='SOURCE_POSTHOC_DIRECTION_NOT_ONLINE',
        selection='first eight predeclared source calibration episodes, four per class; no error selection',
        coordinate='zero FFN delta at final support, NOT full burn-in state or an online gradient',
        parent_sha256=protocol['parent_sha256'],prior_sha256=protocol['prior_sha256'],channels=io23.RAW23,
        records=rows,registered_state_unchanged=True,
        uniform_descent_count=sum(r['uniform_dot']>0 for r in rows),
        positive_box_bound_count=sum(r['boxmax_B']>0 for r in rows),
        finite_uniform_improvement_count=sum(r['risk_change_uniform_step']<0 for r in rows)))
    print(json.dumps({k:v for k,v in json.loads(args.output.read_text()).items() if k not in ('records','channels')}))


if __name__=='__main__':main()
