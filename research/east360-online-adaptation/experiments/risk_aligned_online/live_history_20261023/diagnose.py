"""Posthoc changed-shot probes: history protection versus future risk change.

TEST outcomes choose diagnostic endpoints only AFTER sealed, audited results.
They never enter the original or replayed update. The last nonzero step is
not an attribution of the entire cumulative trajectory.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..history_tangent_20261023.learning import flat, logit
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ, load_model
from ..warmstart_alignment_20261022.alignment import transform
from .learning import ARMS, update_all, HISTORY_TOL


def main(root):
    p,v,full=READ(root/'protocol.json'),READ(root/'verification.json'),READ(root/'per_shot.json')
    assert v['status']=='VERIFIED' and v['completed_sha256']==sha(root/'completed.json')
    probes=[];changes={a:[] for a in ARMS}
    rmap={a:{r['shot']:r for r in full[a]['predictions']} for a in full}
    for sid in p['contract']['shot_ids']:
        disruptive,down=io23.read_outcome('EAST',sid)
        updates=READ(root/'shots'/f'{sid}_updates.json')
        with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as z:
            valid=np.ones(len(z['times']),bool) if not disruptive else (down-z['times'])*1000>=-1e-6
            eligible=valid if not disruptive else valid&(np.round((down-z['times'])*1000,9)>30)
            for a in ARMS:
                difference=np.abs(z[a].astype(float)-z['static'])
                changes[a].append(dict(shot=sid,max_abs=float(difference.max()),mean_abs=float(difference.mean()),
                    changed_points_over1e6=int((difference>1e-6).sum()),
                    static_eligible_peak=float(z['static'][eligible].max()),new_eligible_peak=float(z[a][eligible].max())))
                for kind,ids in v['paired_flips'][a].items():
                    if sid not in ids:continue
                    choose=a if kind in ('rescued_timely','added_fp') else 'static'
                    end=int(np.flatnonzero(valid&(z[choose]>=p['threshold']))[0])
                    preceding=[r for r in updates[a] if r['gradient_steps'] and r['first_affected_prediction']<=end]
                    if not preceding:raise AssertionError('changed decision with no preceding nonzero update')
                    count=preceding[-1]['first_affected_prediction'];lead=(down-z['times'][end])*1000 if disruptive else None
                    probes.append(dict(shot=sid,arm=a,kind=kind,end=end,count=count,is_disrupt=bool(disruptive),
                        lead_ms=float(lead) if lead is not None else None,
                        point_label=int(disruptive and 0<=lead<=125),shot_label=int(disruptive),
                        static_probability=float(z['static'][end]),recorded_probability=float(z[a][end]),
                        static_result=rmap['static'][sid],new_result=rmap[a][sid],
                        nonzero_updates_before=len(preceding)))
    torch.set_num_threads(2);model,_=load_model();parent=state_cpu(model)
    anchors=torch.load(root/'anchors.pt',map_location='cuda',weights_only=True)
    mapping=READ(Path(p['prepared_reference'])/'input_map.json');results=[]
    with patch.object(io23,'read_outcome',side_effect=AssertionError('replayed updates cannot read TEST outcomes')):
        for sid in sorted({r['shot'] for r in probes}):
            cases=[r for r in probes if r['shot']==sid];limit=max(r['count'] for r in cases)
            s=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
            signal=transform({k:s[k] for k in ('values','available')},mapping)
            states={a:initial_encoder(model) for a in ARMS};stored=READ(root/'shots'/f'{sid}_updates.json')
            for index,count in enumerate(range(256,limit+1,128)):
                previous=states;states,receipts=update_all(model,signal,count,states,anchors,device='cuda')
                for a in ARMS:assert receipts[a]==stored[a][index],(sid,count,a)
                for case in [r for r in cases if r['count']==count]:
                    a=case['arm'];before,after=previous[a],states[a]
                    # This future window is diagnostic-only; update_all above
                    # still used exactly its original arrived prefix.
                    x,m=[torch.as_tensor(t[None],device='cuda') for t in simple_data.window(signal,case['end'],128)]
                    z0=logit(model,x,m,before);z1=logit(model,x,m,after).detach()
                    grad=flat(torch.autograd.grad(z0.sum(),tuple(before.values()))).detach().double()
                    step=flat([before[k].detach()-after[k].detach() for k in before]).double()
                    np.testing.assert_allclose(float(z1.sigmoid()[0]),case['recorded_probability'],rtol=2e-5,atol=4e-7)
                    actual=float((z1-z0.detach())[0]);linear=-float(grad@step)
                    errors={}
                    for label_name in ('shot_label','point_label'):
                        target=z0.new_full(z0.shape,case[label_name]).double()
                        errors[label_name]=float(F.binary_cross_entropy_with_logits(z1.double(),target)-
                            F.binary_cross_entropy_with_logits(z0.detach().double(),target))
                    r=receipts[a]
                    results.append(dict(**case,before_last_step_probability=float(z0.detach().sigmoid()[0]),
                        after_last_step_probability=float(z1.sigmoid()[0]),last_step_logit_change=actual,
                        current_linear_logit_change=linear,finite_step_residual=actual-linear,
                        last_step_risk_bce_change=errors,history_step_max=max(r['history_bce_change']),
                        history_parent_max=max(r['history_change_from_parent']),
                        observed_forecast_change=r['loss_after']-r['loss_before'],
                        parameter_delta_norm=r['parameter_delta_norm'],exact_prefix_replay=True,
                        guard_history_kept=bool(a=='guarded' and max(r['history_bce_change'])<=HISTORY_TOL
                            and max(r['history_change_from_parent'])<=HISTORY_TOL)))
            print(json.dumps(dict(diagnosed=sid,cases=len(cases))),flush=True)
    assert_unchanged(model,parent)
    aggregate={a:dict(changed_shots=sum(r['max_abs']>1e-6 for r in rows),
        max_abs=max(r['max_abs'] for r in rows),median_shot_mean_abs=float(np.median([r['mean_abs'] for r in rows]))) for a,rows in changes.items()}
    report=dict(status='POSTHOC_CHANGED_SHOT_DIAGNOSTIC',completed_sha256=sha(root/'completed.json'),
        verification_sha256=sha(root/'verification.json'),code_sha256=sha(Path(__file__)),cases=results,
        full_curve_changes=changes,change_summary=aggregate,
        caveat='TEST labels only select posthoc probes; shot-level desired direction differs from125ms training labels; last-step effect is not whole-trajectory attribution')
    _durable_json(root/'changed_shot_diagnostic.json',report)
    print(json.dumps(dict(cases=len(results),change_summary=aggregate)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
