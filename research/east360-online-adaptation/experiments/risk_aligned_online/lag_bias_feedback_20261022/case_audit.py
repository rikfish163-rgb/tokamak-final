"""Post-hoc sensitivity and exact replay of the single rescued shot.

This diagnoses already-sealed results; no policy or threshold is fitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..paired_feedback_20261022.model import PairedFeedback, initial_encoder
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .learning import adapt_arrived
from .run import MAP_PATH, predict_arms


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    root=p.parse_args().directory
    protocol=json.loads((root/'protocol.json').read_text())
    verification=json.loads((root/'verification.json').read_text())
    assert verification['status']=='VERIFIED'
    full=json.loads((root/'per_shot.json').read_text());threshold=protocol['threshold']
    rescued=verification['paired_comparisons']['static_to_lag_bias']['rescued_timely']
    # A bound for THIS completed experiment, not a general assumptions-based
    # fallback: if results change, update the analysis explicitly.
    assert rescued==[73405]
    sensitivity={}
    for lead in (30.,31.,40.,50.):
        sensitivity[str(lead)]={k:dict(tp=sum(r['is_disrupt'] and r['alarm'] and r['lead_ms']>lead
                                            for r in v['predictions']),
                                      fp=v['fp']) for k,v in full.items()}
    torch.set_num_threads(2);torch.manual_seed(131)
    base,_=load_parent(DEFAULT_PARENT);model=PairedFeedback(base,np.ones(23)).cuda();model.set_stage('frozen')
    mapping=json.loads(MAP_PATH.read_text());sid=rescued[0]
    raw=apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
    signal=transform({k:raw[k] for k in ('values','available')},mapping)
    fresh,updates=predict_arms(model,signal)
    with np.load(root/'shots'/f'{sid}.npz',allow_pickle=False) as trace:
        for k,values in fresh.items():np.testing.assert_array_equal(values,trace[k])
    assert updates==json.loads((root/'shots'/f'{sid}_updates.json').read_text())
    outcome=simple_data.attach_outcome(raw,'EAST',{})
    hit=int(np.flatnonzero(outcome['valid'] & (fresh['lag_bias']>=threshold))[0])
    eligible=outcome['valid'] & (np.round((outcome['down']-outcome['times'])*1000,9)>30.)
    initial=initial_encoder(model);fast=initial_encoder(model)
    last_update=hit//128*128
    for count in range(256,last_update+1,128):
        fast,_=adapt_arrived(model,signal,count,fast,device='cuda',correct=True)
    # Reproduce the exact risk batch containing the newly timely alarm.
    start=hit//32*32;stop=min(start+32,len(signal['values']))
    windows=[simple_data.window(signal,e,128) for e in range(start,stop)]
    x,m=[torch.as_tensor(np.stack(v),device='cuda') for v in zip(*windows)]
    old=model.predict_fast(x,m,initial,mode='risk_state')['risk_state']
    new=model.predict_fast(x,m,fast,mode='risk_state')['risk_state']
    old_logit=model.base.risk_head(old).squeeze(-1)
    new_logit=model.base.risk_head(new).squeeze(-1)
    np.testing.assert_array_equal(new_logit.sigmoid().detach().cpu().numpy(),fresh['lag_bias'][start:stop])
    np.testing.assert_array_equal(old_logit.sigmoid().detach().cpu().numpy(),fresh['static'][start:stop])
    i=hit-start
    dot=(model.base.risk_head.weight.squeeze(0)*(new[i]-old[i])).sum()
    torch.testing.assert_close(dot,new_logit[i]-old_logit[i],rtol=1e-5,atol=1e-6)
    details={k:next(r for r in v['predictions'] if r['shot']==sid) for k,v in full.items()}
    result=dict(scope='POSTHOC diagnostic, not policy selection or independent confirmation',
        shot=sid,records=details,threshold=threshold,first_lag_alarm_index=hit,
        first_lag_alarm_time=float(raw['times'][hit]),down=float(outcome['down']),
        last_update_first_affected_index=last_update,last_observation=last_update-1,
        last_feedback_lead_ms=round(float(outcome['down']-raw['times'][last_update-1])*1000,9),
        static_probability_at_lag_alarm=float(fresh['static'][hit]),
        lag_probability_at_lag_alarm=float(fresh['lag_bias'][hit]),
        lag_margin_at_alarm=float(fresh['lag_bias'][hit])-threshold,
        timely_maxima={k:float(v[eligible].max()) for k,v in fresh.items()},
        internal_state_change_norm=float((new[i]-old[i]).detach().norm()),
        fixed_head_logit_change=float((new_logit[i]-old_logit[i]).detach()),
        state_projection_on_fixed_head=float(dot.detach()),
        cumulative_temporary_parameter_norm=float(sum((fast[k].detach()-v.detach()).double().square().sum()
                                                      for k,v in initial.items()).sqrt()),
        lead_sensitivity_strict_greater=sensitivity,
        replayed_full_traces=3,max_replay_error=0.,case_audit_sha256=sha(Path(__file__)))
    _durable_json(root/'rescued_case.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
