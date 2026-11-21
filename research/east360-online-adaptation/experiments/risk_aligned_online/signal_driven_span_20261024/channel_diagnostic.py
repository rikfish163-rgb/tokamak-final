"""Post-hoc channel-gradient attribution; NEVER used to choose online updates."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .. import io23, simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import flat, logit, apply_step
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder
from ..simple_online import _durable_json
from ..simple_train import sha
from .learning import projected_step
from .run import (READ, REFERENCE, make_parent, models_for_calibration, signal, transform,
                 DEFAULT_PARENT, PARENT_SHA256, state_cpu, assert_unchanged)


def main(root):
    outcomes = {r['shot']:r for r in READ(root/'outcomes.json')}
    assert READ(root/'completed.json')['status'] == 'completed'
    torch.set_num_threads(2); torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(True); torch.backends.cuda.matmul.allow_tf32=False
    parent, _ = make_parent(); model = models_for_calibration(parent)['mapped_fixed']; original = state_cpu(model)
    mapping = READ(REFERENCE/'input_map.json'); q = torch.load(root/'spaces.pt', weights_only=True)['contrast'].cuda()
    result = []
    def deny(*args, **kwargs):
        raise AssertionError('posthoc diagnostics use sealed outcomes, no new labels')
    with patch.object(io23, 'read_outcome', side_effect=deny):
        for sid in (89272, 89284, 89584):
            with np.load(root/'shots'/f'{sid}.npz') as z:
                flag, down = outcomes[sid]['is_disrupt'], outcomes[sid]['down']
                if flag:
                    valid = np.flatnonzero(np.round((down-z['times'])*1000, 9) > 30.)
                    end = int(valid[-1])
                    why = 'last endpoint with strictly more than30ms lead'
                else:
                    end = int(z['contrast'].argmax()); why = 'candidate maximum risk on safe shot'
                sealed = {a: float(z[a][end]) for a in ('off', 'contrast')}
            count = end//128*128
            assert count >= 256 and count <= end
            raw = signal(sid); mapped = transform({k:raw[k] for k in ('values','available')}, mapping)
            prefix = {k:v[:count] for k,v in mapped.items()}
            x,m,target,tm,origins = forecast_examples(prefix, count, model.cfg, 'cuda')
            valid = target_mask(x,m,target,tm); fast=initial_encoder(model)
            prediction = model.predict_fast(x,m,fast)['forecast']
            loss = model.base.forecast_loss(prediction,target,valid)
            g = flat(torch.autograd.grad(loss,tuple(fast.values()),retain_graph=True)).detach()
            by_channel=[]; channel_losses=[]
            for c in range(23):
                channel_mask=torch.zeros_like(valid); channel_mask[:,:,c]=valid[:,:,c]
                component=model.base.forecast_loss(prediction,target,channel_mask)*channel_mask.sum()/valid.sum()
                by_channel.append(flat(torch.autograd.grad(component,tuple(fast.values()),retain_graph=True)).detach())
                channel_losses.append(float(component.detach()))
            parts=torch.stack(by_channel)
            torch.testing.assert_close(parts.sum(0),g,atol=1e-6,rtol=2e-4)
            xx,mm=[torch.as_tensor(v[None],device='cuda') for v in simple_data.window(mapped,end,128)]
            z0=logit(model,xx,mm,fast)
            jac=flat(torch.autograd.grad(z0.sum(),tuple(fast.values()))).detach()
            step,receipt=projected_step(g,q)
            scale=.01/receipt['projected_gradient_norm']
            influence=-(parts@q.T)@(q@jac)*scale
            predicted_delta=float(-(jac.double()*step.double()).sum())
            np.testing.assert_allclose(float(influence.sum()),predicted_delta,atol=1e-6,rtol=2e-4)
            changed=apply_step(fast,step)
            # Same gradient-enabled math kernel for both logits. The deployed
            # no_grad probabilities are separately reported from sealed curves.
            z1=logit(model,xx,mm,changed)
            named=[dict(channel=name,observations=int(valid[:,:,c].sum()),
                        forecast_loss_contribution=channel_losses[c],
                        linearized_logit_delta=float(influence[c])) for c,name in enumerate(io23.RAW23)]
            result.append(dict(shot=sid,is_disrupt=flag,selection=why,end=end,update_count=count,
                lead_ms=float((down-raw['times'][end])*1000) if flag else None,
                selected_after_outcomes=True,forecast_origins=origins,sealed_probabilities=sealed,
                projected_gradient_norm=receipt['projected_gradient_norm'],
                linearized_logit_delta=predicted_delta,actual_math_logit_delta=float((z1-z0).detach()[0]),
                gradient_sum_relative_error=float((parts.sum(0)-g).norm()/g.norm()),
                channels=sorted(named,key=lambda v:abs(v['linearized_logit_delta']),reverse=True)))
    assert_unchanged(model,original); assert sha(DEFAULT_PARENT)==PARENT_SHA256
    report=dict(scope='POSTHOC_SELECTED_POINTS_NOT_CAUSAL_PHYSICS_OR_VALIDATED_POLICY',
        completed_sha256=sha(root/'completed.json'),code_sha256=sha(Path(__file__)),
        note='Contributions are local parameter-gradient arithmetic, not experimental channel removal or physical causes.',
        points=result)
    _durable_json(root/'channel_diagnostic.json',report)
    for r in result:
        print({**{k:v for k,v in r.items() if k!='channels'},'top6':r['channels'][:6]},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
