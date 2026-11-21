"""Post-hoc error attribution; NEVER used by the online update or selection."""
import argparse
from pathlib import Path
import numpy as np
import torch

from .. import io23
from ..feedback_cone_feasibility_20261024.run import READ,make_parent,models_for_calibration,signal,transform,REFERENCE
from ..simple_online import _durable_json
from ..simple_train import sha
from .model import DiagnosticMemory,RecurrentAdapter,patch_signal,endpoint_corrections
from .learning import batch


def main(root):
    assert READ(root/'audit.json')['status']=='passed'
    protocol=READ(root/'protocol.json');outcomes={r['shot']:r for r in READ(root/'outcomes.json')}
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
    parent,_=make_parent();deployed=models_for_calibration(parent)['mapped_fixed']
    adapter=RecurrentAdapter(deployed.base).cuda();memory=DiagnosticMemory(64).cuda()
    memory.load_state_dict(torch.load(root/'memory.pt',map_location='cuda',weights_only=True));adapter.memory=memory
    mapping=READ(REFERENCE/'input_map.json');rows=[];attributions=[]
    for sid in protocol['shot_ids']:
        raw=signal(sid);s=transform({k:raw[k] for k in ('values','available')},mapping)
        with torch.no_grad():
            q,m=patch_signal(s,adapter.base.center,adapter.base.scale)
            real=memory.trace(q,m,mode='memory')
            reverse=memory.trace(q,m,mode='reverse',reference_norm=real['norm'])
            targets=torch.stack([q[i+1:i+5]-q[i] for i in range(len(q)-4)])
            masks=torch.stack([m[i+1:i+5]&m[i] for i in range(len(q)-4)])
            both=masks&masks.flip(1)
            order_energy=((targets-targets.flip(1)).square()*both).sum()
            target_energy=(targets.square()*both).sum()
            delta=real['delta'].double();rev=reverse['delta'].double()
            norm=delta.norm(dim=1)*rev.norm(dim=1);valid=norm>0
            cosine=(delta[valid]*rev[valid]).sum(1)/norm[valid]
            projected=memory.output(real['delta']).double()
            projected_reverse=memory.output(reverse['delta']).double()
            projected_norm=projected.norm(dim=1)*projected_reverse.norm(dim=1)
            pv=projected_norm>0
            projected_cosine=(projected[pv]*projected_reverse[pv]).sum(1)/projected_norm[pv]
            forecast_error=(real['forecast'][:-4]-targets).square()*masks
            counts=masks.sum((0,1));error=forecast_error.sum((0,1))/counts.clamp_min(1)
            rows.append(dict(shot=sid,reverse_target_difference_ratio=float(order_energy/target_energy.clamp_min(1e-20)),
                             memory_reverse_cosine_median=float(cosine.median()),
                             injected_reverse_cosine_median=float(projected_cosine.median()),
                             main_forecast_mse=float(real['forecast_loss']),
                             channel_mse=[dict(channel=c,mse=float(error[j]) if counts[j]>0 else None,
                                               valid_targets=int(counts[j])) for j,c in enumerate(io23.RAW23)]))
        if sid not in (89272,89284,89584):continue
        o=outcomes[sid]
        with np.load(root/'shots'/f'{sid}.npz') as z:
            if o['is_disrupt']:
                eligible=np.flatnonzero(np.round((o['down']-z['times'])*1000,9)>30.)
                endpoints=[('timely_peak',int(eligible[np.argmax(z['memory'][eligible])]))]
                alarm=np.flatnonzero(z['memory']>=protocol['threshold'])
                if len(alarm):endpoints.append(('first_alarm',int(alarm[0])))
            else:endpoints=[('false_alarm_peak',int(np.argmax(z['memory'])))]
        for kind,e in endpoints:
            # Replay the same32-sized chunk as the sealed prediction.
            lo=e//32*32;ends=list(range(lo,min(lo+32,len(s['values']))));position=e-lo
            x,a=batch(s,ends,'cuda')
            with torch.no_grad():
                h,v=adapter.tokens(x,a)
                d=endpoint_corrections(real['delta'],torch.as_tensor(ends,device='cuda')).detach()
                z0=adapter.risk(h,v,torch.zeros_like(d))[position]
            d.requires_grad_(True);risk=adapter.risk(h,v,d)[position]
            g=torch.autograd.grad(risk,d)[0][position]
            contribution=(g*d[position]).reshape(4,23).sum(0).detach().cpu().numpy()
            with np.load(root/'shots'/f'{sid}.npz') as z:
                assert float(risk.detach())==float(z['memory_logit'][e])
            attributions.append(dict(shot=sid,kind=kind,end=e,time=float(raw['times'][e]),
                lead_ms=None if not o['is_disrupt'] else float(round((o['down']-raw['times'][e])*1000,9)),
                actual_logit_change=float((risk-z0).detach()),first_order_sum=float(contribution.sum()),
                channel_first_order=[dict(channel=io23.RAW23[j],contribution=float(contribution[j]))
                                     for j in np.argsort(-np.abs(contribution))],
                interpretation='local sensitivity of inferred diagnostic-change coordinates, not physical causal effects'))
    weight=memory.output.weight.detach().double().reshape(64,4,23)
    common=weight.mean(1,keepdim=True).expand_as(weight)
    result=dict(scope='POSTHOC_DIAGNOSTIC_ONLY_NO_MODEL_OR_THRESHOLD_CHANGE',shots=rows,attributions=attributions,
                output_horizon_invariant_weight_energy=float(common.square().sum()/weight.square().sum()),
                output_order_specific_weight_energy=float((weight-common).square().sum()/weight.square().sum()),
                energy_interpretation='weight-space diagnostic only; not a fraction of causal outcome benefit',
                completed_sha256=sha(root/'completed.json'))
    _durable_json(root/'posthoc_diagnosis.json',result)
    print(dict(shots=len(rows),attributions=attributions),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
