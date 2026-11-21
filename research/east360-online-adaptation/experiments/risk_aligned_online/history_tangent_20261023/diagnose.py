"""Post-hoc risk-gradient coverage. TEST labels ONLY select diagnostic probes.

These probes never enter preparation, updating, threshold selection or replay.
No claim that gradient overlap is a shared plasma mechanism.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu,assert_unchanged
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ,load_model
from ..warmstart_alignment_20261022.alignment import transform
from .audit import summary
from .learning import ARMS,flat,logit
from .run import NAMES


def main(root):
    p=READ(root/'protocol.json');verified=READ(root/'verification.json')
    assert verified['status']=='VERIFIED' and sha(root/'completed.json')==verified['completed_sha256']
    torch.set_num_threads(2);model,_=load_model();original=state_cpu(model)
    mapping=READ(Path(p['reference'])/'input_map.json')
    geometry=torch.load(root/'geometry.pt',map_location='cuda',weights_only=True)
    q=geometry['bases']['history'];fast=initial_encoder(model)
    full=READ(root/'per_shot.json');rmap={a:{r['shot']:r for r in v['predictions']} for a,v in full.items()}
    probes=[];cases=[]
    for sid in p['contract']['shot_ids']:
        s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal=transform({k:s[k] for k in ('values','available')},mapping)
        disrupt,down=io23.read_outcome('EAST',sid)
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            lead=np.round((down-z['times'])*1000,9) if disrupt else None
            valid=np.ones(len(z['times']),bool) if not disrupt else (down-z['times'])*1000>=-1e-6
            eligible=valid&(lead>30) if disrupt else valid
            groups={'operational_peak':eligible}
            if disrupt:groups['positive_30_125_peak']=eligible&(lead<=125)
            for group,mask in groups.items():
                indexes=np.flatnonzero(mask)
                if not len(indexes):raise ValueError('empty diagnostic interval; do not silently exclude')
                end=int(indexes[z['static'][indexes].argmax()])
                x,m=[torch.as_tensor(a[None],device='cuda') for a in simple_data.window(signal,end,128)]
                score=logit(model,x,m,fast)
                np.testing.assert_allclose(float(score.detach().sigmoid()[0]),z['static'][end],rtol=2e-5,atol=3e-7)
                grad=flat(torch.autograd.grad(score.sum(),tuple(fast.values()))).detach()
                norm=grad.double().square().sum()
                if norm<=0:raise ValueError('zero diagnostic derivative')
                fraction=float((q@grad).double().square().sum()/norm)
                probes.append(dict(shot=sid,group=group,is_disrupt=disrupt,end=end,
                    lead_ms=float(lead[end]) if disrupt else None,parent_risk=float(z['static'][end]),
                    protected_fraction=fraction,risk_gradient_norm=float(norm.sqrt()),
                    static_correct=rmap['static'][sid]['timely'] if disrupt else not rmap['static'][sid]['alarm'],
                    risks={a:float(z[a][end]) for a in NAMES}))
            updates=READ(root/'shots'/f'{sid}_updates.json')
            for arm in ARMS:
                for kind,ids in verified['paired_flips'][arm].items():
                    if sid not in ids:continue
                    decision=arm if kind in ('rescued_timely','added_fp') else 'static'
                    end=int(np.flatnonzero(valid&(z[decision]>=p['threshold']))[0])
                    preceding=[u for u in updates[arm] if u['first_affected_prediction']<=end]
                    cases.append(dict(shot=sid,arm=arm,kind=kind,end=end,
                        lead_ms=float(lead[end]) if disrupt else None,
                        risks={a:float(z[a][end]) for a in NAMES},
                        eligible_max={a:float(z[a][eligible].max()) for a in NAMES},
                        static_result=rmap['static'][sid],new_result=rmap[arm][sid],
                        previous_update=preceding[-1] if preceding else None,updates_before=len(preceding)))
    groups={}
    for group in ('operational_peak','positive_30_125_peak'):
        for disruptive in (False,True):
            selected=[r for r in probes if r['group']==group and r['is_disrupt']==disruptive]
            if not selected:continue
            for correct in (False,True):
                cc=[r for r in selected if r['static_correct']==correct]
                if cc:groups[f'{group}_disruptive{disruptive}_correct{correct}']=dict(n=len(cc),protected_fraction=summary([r['protected_fraction'] for r in cc]))
    assert_unchanged(model,original)
    report=dict(status='POSTHOC_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(root/'verification.json'),diagnostic_code_sha256=sha(Path(__file__)),
        probes=probes,groups=groups,cases=cases,
        caveat='TEST outcomes used AFTER all sealed experiments to choose diagnostic endpoints; never a deployable policy or independent confirmation')
    _durable_json(root/'posthoc_geometry.json',report)
    print(json.dumps(dict(probes=len(probes),cases=len(cases),groups=groups)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
