"""Independently recount decisions; replay saved memory and causal prefixes."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import (
    READ,make_parent,models_for_calibration,signal,transform,REFERENCE,DEFAULT_PARENT,PARENT_SHA256,
)
from ..crossfit_teacher_20261025.fit import load_teachers
from ..simple_online import _durable_json
from ..simple_train import sha
from ..operational_bag_loss_20261025.learning import bag_loss
from .model import ARMS,DiagnosticMemory,RecurrentAdapter,patch_signal
from .learning import predict,prepare_example,risk_from_trace,eligible
from .run import GEOMETRY


def audit(root):
    done=READ(root/'completed.json');p=READ(root/'protocol.json')
    assert done['status']=='completed'
    for name,digest in done['artifact_sha256'].items():assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items():assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items():assert sha(name)==digest,name
    assert sha(DEFAULT_PARENT)==PARENT_SHA256
    assert sha(REFERENCE/'fold0/mapped_fixed/prepared.pt')==p['parent_sha256']
    records=READ(root/'per_shot.json');outcomes={r['shot']:r for r in READ(root/'outcomes.json')}
    checks=0;norm_error=0.
    for sid in p['shot_ids']:
        o=outcomes[sid]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            valid=np.ones(len(z['times']),dtype=bool) if not o['is_disrupt'] else (o['down']-z['times'])*1000>=-1e-6
            for arm in ('off',)+ARMS:
                probability=z[arm][valid];times=z['times'][valid]
                hit=np.flatnonzero(probability>=p['threshold'])
                lead=None if not len(hit) or not o['is_disrupt'] else float(round((o['down']-times[hit[0]])*1000,9))
                r=next(r for r in records[arm]['predictions'] if r['shot']==sid)
                assert r['alarm']==bool(len(hit))
                assert r['timely']==bool(o['is_disrupt'] and lead is not None and lead>30.)
                assert r['lead_ms']==lead
                checks+=1
        update=READ(root/'updates'/f'{sid}.json');main=np.asarray(update['memory']['norm'])
        for arm in ('local','hold','reverse'):
            values=np.asarray(update[arm]['norm']);zero=np.asarray(update[arm]['zero_direction'])
            error=float(np.max(np.abs(values[~zero]-main[~zero])))
            assert error<2e-6;norm_error=max(norm_error,error)
    for arm in ('off',)+ARMS:
        rows=records[arm]['predictions'];result=done['results'][arm]
        assert result['tp']==sum(r['timely'] for r in rows)
        assert result['fp']==sum(not r['is_disrupt'] and r['alarm'] for r in rows)
    assert done['eligible_for360']==eligible(done['results'],done['historical_gate_passed'])
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();deployed=models_for_calibration(parent)['mapped_fixed']
    memory=DiagnosticMemory(64).cuda();memory.load_state_dict(torch.load(root/'memory.pt',map_location='cuda',weights_only=True))
    adapter=RecurrentAdapter(deployed.base).cuda();adapter.memory=memory
    mapping=READ(REFERENCE/'input_map.json');replayed=[]
    def deny(*args,**kwargs):raise AssertionError('audit replay may not read outcomes')
    with patch.object(io23,'read_outcome',side_effect=deny),torch.no_grad():
        for sid in (89272,89284,89584):
            raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
            curves,updates,logits=predict(adapter,s)
            with np.load(root/'shots'/f'{sid}.npz') as z:
                for arm in ('off',)+ARMS:
                    np.testing.assert_array_equal(curves[arm],z[arm])
                    np.testing.assert_array_equal(logits[arm],z[arm+'_logit'])
            assert updates==READ(root/'updates'/f'{sid}.json')
            replayed.append(sid)
        raw=signal(89584);s=transform({k:raw[k] for k in ('values','available')},mapping)
        q,m=patch_signal(s,adapter.base.center,adapter.base.scale)
        a=memory.trace(q,m,mode='memory')
        changed=q.clone();changed[32:]+=10
        b=memory.trace(changed,m,mode='memory')
        torch.testing.assert_close(a['delta'][:32],b['delta'][:32],atol=0,rtol=0)
        torch.testing.assert_close(a['forecast'][:32],b['forecast'][:32],atol=0,rtol=0)
    selected=READ(root/'selection.json');teachers=load_teachers(parent)
    rows={r['shot']:r for r in selected['per_shot']};objective=0.;aux=0.;maxerror=0.
    endpoints={r['shot']:r for r in READ(root/'selected_endpoints.json')}
    with patch.object(io23,'read_outcome',side_effect=deny):
        for teacher in teachers:
            adapter=RecurrentAdapter(teacher['model'].base).cuda();adapter.memory=memory
            for sid in teacher['held']:
                o=READ(GEOMETRY/f'{sid}.json');raw=signal(sid)
                s=transform({k:raw[k] for k in ('values','available')},teacher['mapping'])
                e=prepare_example(adapter,s,raw['times'],o['is_disrupt'],o['down'],p['threshold'])
                assert e['selected_ends']==endpoints[sid]['selected_ends']
                trace=memory.trace(e['means'],e['complete'],mode='memory')
                z=risk_from_trace(e,trace);loss=bag_loss(z.max(),e['flag'],p['threshold'])
                delta=float(loss.detach())-e['baseline'];weight=.5/(4 if e['flag'] else 16)
                assert float(loss.detach())==rows[sid]['loss']
                assert float(trace['forecast_loss'].detach())==rows[sid]['forecast_loss']
                error=abs(float(loss.detach())-rows[sid]['loss']);maxerror=max(maxerror,error)
                objective+=weight*(delta+max(delta,0));aux+=weight*float(trace['forecast_loss'].detach())
    assert objective==selected['objective'] and aux==selected['forecast_loss']
    epochs=[READ(root/'training'/f'epoch{i:02d}.json') for i in range(11)]
    assert selected['epoch']==min(epochs,key=lambda r:r['objective'])['epoch']
    result=dict(status='passed',decisions_recounted=checks,replayed_shots_all5_arms=replayed,
                maximum_control_norm_error=norm_error,selected20_objective_exact=objective,
                selected20_forecast_loss_exact=aux,maximum_historical_replay_error=maxerror,
                future_patch_perturbation_prefix_exact=True,input_files_hashed=len(p['input_sha256']),
                checkpoint_and_parent_unchanged=True,completed_sha256=sha(root/'completed.json'))
    _durable_json(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    audit(parser.parse_args().out)
