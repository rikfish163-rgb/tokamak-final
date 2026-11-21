"""Matched label-group schedules for wider source-context coverage."""

from collections import Counter

import numpy as np
import torch

from test_joint_stream_20261107 import signal


def test_short_schedule_keeps_recorded_groups_budget_and_native_targets():
    from run_risk_short_context_20261108 import short_notes
    from risk_short_context_20261108 import context_pools,sequence_from_note
    from run_risk_teacher_20261107 import teacher_pools,GROUPS

    shots=[dict(signal(400),shot=1,valid=np.ones(400,bool),y=np.zeros(400,np.float32)),
           dict(signal(400),shot=2,valid=np.ones(400,bool),y=(np.arange(400)>=300).astype(np.float32)),
           dict(signal(200),shot=3,valid=np.ones(200,bool),y=(np.arange(200)>=130).astype(np.float32))]
    old=teacher_pools(shots,steps=4);history=[]
    for group in GROUPS*3:
        i,origins=old[group][0];first=int(origins[0]);q=first+122
        history.append(dict(shot=shots[i]['shot'],origins=[first+32*k for k in range(4)],
                            query=q,label_time=q+25,group=group))
    before=[dict(v) for v in history]
    pools=context_pools(shots,minimum_origin=7)
    notes=short_notes(shots,pools,history,seed=145)
    assert history==before
    assert len(notes)==len(history) and [n['group'] for n in notes]==[n['group'] for n in history]
    assert Counter(n['group'] for n in notes)==Counter(n['group'] for n in history)
    assert notes==short_notes(shots,pools,history,seed=145)
    by_id={s['shot']:s for s in shots}
    for note in notes:
        es,qx,qa,label,rebuilt=sequence_from_note(by_id[note['shot']],note,'cpu')
        assert len(es)==4 and rebuilt==note
        assert label.tolist()==[float(by_id[note['shot']]['y'][note['label_time']])]
        assert not qa[...,2].any()


def test_padded_four_feedback_path_has_finite_risk_credit_without_mutating_risk():
    from copy import deepcopy
    from risk_meta_sequence_20261108 import teacher_sequence
    from risk_short_context_20261108 import sequence_from_note
    from test_risk_meta_step_20261107 import teacher_case

    model,_=teacher_case();before=deepcopy(model.risk.state_dict())
    s=dict(signal(155),shot=7,valid=np.ones(155,bool),y=(np.arange(155)>=140).astype(np.float32))
    note=dict(shot=7,origins=[7,39,71,103],query=129,label_time=154,group='0_to_1')
    es,qx,qa,label,_=sequence_from_note(s,note,'cpu')
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        result=teacher_sequence(model,es,qx,qa,label,credit='last')
        gradients=torch.autograd.grad(result['risk_loss'],tuple(model.decoder.parameters()))
    assert torch.isfinite(result['signal_loss']) and torch.isfinite(result['risk_loss'])
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.detach().square().sum()) for g in gradients)>0
    assert all(torch.equal(value,before[key]) for key,value in model.risk.state_dict().items())
