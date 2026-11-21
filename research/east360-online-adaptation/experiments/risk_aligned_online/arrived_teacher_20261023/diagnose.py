"""Posthoc historical signal description, not physical causal attribution."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import encode


def main(root):
    read=lambda p:json.loads(p.read_text())
    verification=read(root/'verification.json');protocol=read(root/'protocol.json')
    assert verification['status']=='VERIFIED' and verification['completed_sha256']==sha(root/'completed.json')
    rows=read(root/'probes.json');advance=[r for r in rows if r['arm']=='advance']
    selected=sorted((r for r in advance if r['group']=='safe'),key=lambda r:r['actual_delta'],reverse=True)[:3]
    for sid in sorted({r['shot'] for r in advance if r['group']=='disruptive_timely'}):
        group=[r for r in advance if r['shot']==sid and r['group']=='disruptive_timely']
        selected.extend([min(group,key=lambda r:r['actual_delta']),max(group,key=lambda r:r['actual_delta'])])
    torch.set_num_threads(2);model,_=load_model();mapping=read(Path(protocol['reference'])/'input_map.json')
    details=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('no new outcome reads')):
        for r in selected:
            assert r['shot'] in protocol['selected_ids']
            s=apply_input_contract(simple_data.read_shot('EAST',r['shot'],rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            a,b=r['count']-26,r['count']-1;q=r['query']
            h=encode(model,signal,[a,b,q],initial_encoder(model),'cuda')
            probability=model.base.risk_head(h).squeeze(-1).sigmoid().detach().cpu().tolist()
            columns=[]
            for c,name in enumerate(io23.RAW23):
                values=[];levels=[];available=[]
                for e in (a,b,q):
                    mask=signal['available'][e-7:e+1,c] & np.isfinite(signal['values'][e-7:e+1,c])
                    v=signal['values'][e-7:e+1,c][mask]
                    values.append(float(np.median(v)) if len(v) else None);available.append(int(len(v)))
                    levels.append(float(np.median(np.arcsinh((v-float(model.base.center[c]))/float(model.base.scale[c])))) if len(v) else None)
                columns.append(dict(channel=name,valid_points=available,aligned_median=values,
                    source_coordinate_median=levels,
                    observed_delta=None if levels[0] is None or levels[1] is None else levels[1]-levels[0],
                    later_delta=None if levels[1] is None or levels[2] is None else levels[2]-levels[1]))
            order=sorted((c for c in columns if c['observed_delta'] is not None),key=lambda c:abs(c['observed_delta']),reverse=True)
            details.append(dict(shot=r['shot'],query=q,group=r['group'],indices=[a,b,q],
                actual_times=s['times'][[a,b,q]].tolist(),parent_probabilities=probability,
                adapted_query_probability=float(1/(1+np.exp(-r['logit_after']))),
                cosine=r['cosine'],actual_bce_delta=r['actual_delta'],first_order_delta=r['first_order_delta'],
                teacher_fit_before=r['update']['loss_before'],teacher_fit_after=r['update']['loss_after'],
                top_observed_changes=order[:6],all_channels=columns))
    groups={}
    for group in ('safe','disruptive_timely','disruptive_early'):
        part=[r for r in advance if r['group']==group]
        groups[group]=dict(n=len(part),opposite_first_order_and_actual=sum(r['actual_delta']*r['first_order_delta']<0 for r in part),
            auxiliary_loss_decreased=sum(r['update']['loss_after']<r['update']['loss_before'] for r in part),
            first_order_mean=float(np.mean([r['first_order_delta'] for r in part])),
            actual_mean=float(np.mean([r['actual_delta'] for r in part])))
    result=dict(status='DESCRIPTIVE_DIAGNOSTIC',code_sha256=sha(Path(__file__)),
        verification_sha256=sha(root/'verification.json'),groups=groups,cases=details,
        limits='Extremes selected posthoc from historical probes; no new policy, no TEST, no causal plasma diagnosis. '
               'Coordinate changes use existing empirical alignment and source asinh; units remain uncertified. '
               'Early disruptive y125=0 is not equivalent to a false alarm under unrestricted timely lead.')
    _durable_json(root/'mechanism_diagnostic.json',result)
    print(json.dumps(dict(groups=groups,cases=[{k:r[k] for k in ('shot','query','group','parent_probabilities','adapted_query_probability','actual_bce_delta')} for r in details])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
