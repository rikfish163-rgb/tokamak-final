"""Source-TRAIN numerical evidence for teaching a signal-driven encoder update."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

import run_joint_trajectory_20261107 as j
from joint_motion_20261107 import OrderedTrajectory
from probe_risk_masked_utility_20261107 import PARENT, masked_view
from risk_feature_intra_20261107 import FeatureIntra, update
from risk_meta_step_20261107 import MetaOrderedTrajectory, teacher_step
from run_joint_motion_20261107 import FIT


def run(out):
    j.r.configure()
    base,_ = j.r.load_parent(j.r.DEFAULT_PARENT)
    base.cuda()
    original = deepcopy(base.state_dict())
    decoder = MetaOrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    decoder.load_state_dict(torch.load(FIT/'source_predictor.pt',map_location='cuda',weights_only=True))
    original_decoder = deepcopy(decoder.state_dict())
    ordinary = OrderedTrajectory(base.center,base.scale,dim=64,scenarios=4).cuda().eval()
    ordinary.load_state_dict(decoder.state_dict())
    cases = j.r.read(PARENT)['records']
    roles = j.simple_data.source_roles()
    if any(v['shot'] not in roles['train'] for v in cases):
        raise ValueError('teacher proof may only use source TRAIN')
    advanced = j.simple_data.advanced_times()
    records = []
    for old in cases:
        s = j.simple_data.attach_outcome(j.simple_data.read_shot('J-TEXT',old['shot']),'J-TEXT',advanced)
        o,q = old['origin'],old['query']
        if q != o+26 or old['label_time'] != q+25 or not s['valid'][o-127:q+26].all():
            raise ValueError('a complete earlier signal forecast and later source query are required')
        x,a = j.simple_data.window(s,o,128)
        qx,qa = j.simple_data.window(s,q,128)
        e = {k:torch.as_tensor(v[None],device='cuda') for k,v in
             dict(x=x,a=a,truth=s['values'][o+1:o+26],mask=s['available'][o+1:o+26]).items()}
        qx,qa = torch.as_tensor(qx[None],device='cuda'),torch.as_tensor(qa[None],device='cuda')
        e,qa = masked_view(e,qa)
        labels = torch.as_tensor(s['y'][q+25:q+26],device='cuda')
        model = FeatureIntra(base,decoder)
        model.decoder.requires_grad_(True)
        deployed = FeatureIntra(base,decoder)
        legacy = FeatureIntra(base,ordinary)
        optimizer = torch.optim.AdamW([deployed.weight],lr=1e-4,weight_decay=0.)
        legacy_optimizer = torch.optim.AdamW([legacy.weight],lr=1e-4,weight_decay=0.)
        with sdpa_kernel(SDPBackend.MATH):
            full = teacher_step(model,e,qx,qa,labels)
            parameters = tuple(model.decoder.parameters())
            gradients = torch.autograd.grad(full['risk_loss'],parameters,allow_unused=True)
            stopped = teacher_step(model,e,qx,qa,labels,stop_update=True)
            control = torch.autograd.grad(stopped['risk_loss'],parameters,allow_unused=True)
            deployed_loss = deployed.signal_loss(e)
            deployed_gradient, = torch.autograd.grad(deployed_loss,deployed.weight)
            stats = update(deployed,optimizer,e)
            legacy_loss = legacy.signal_loss(e)
            legacy_gradient, = torch.autograd.grad(legacy_loss,legacy.weight)
            update(legacy,legacy_optimizer,e)
            with torch.no_grad():
                native_logit = deployed.risk(qx,qa)['logit']
                legacy_logit = legacy.risk(qx,qa)['logit']
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('nonfinite actual mixed derivative')
        if any(g is not None and torch.count_nonzero(g) for g in control):
            raise ValueError('a stopped update still teaches the decoder')
        norm = float(torch.stack([g.square().sum() for g in gradients if g is not None]).sum().sqrt())
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('finite nonzero update-mediated source teaching is required')
        torch.testing.assert_close(full['logit'],stopped['logit'],atol=0,rtol=0)
        torch.testing.assert_close(full['gradient'],deployed_gradient,atol=1e-6,rtol=1e-4)
        torch.testing.assert_close(full['weight'],deployed.weight,atol=2e-7,rtol=0)
        torch.testing.assert_close(full['logit'],native_logit,atol=3e-6,rtol=0)
        torch.testing.assert_close(full['gradient'],legacy_gradient,atol=1e-6,rtol=1e-4)
        records.append(dict(shot=old['shot'],group=old['group'],origin=o,query=q,
            label_time=q+25,decoder_outer_gradient_norm=norm,stopped_outer_gradient_norm=0.,
            signal_error=abs(float(full['signal_loss'].detach())-stats['signal_loss']),
            signal_gradient_error=float((full['gradient']-deployed_gradient).abs().max().detach()),
            first_step_weight_error=float((full['weight']-deployed.weight).abs().max().detach()),
            current_risk_logit_error=float((full['logit']-native_logit).abs().max().detach()),
            legacy_signal_gradient_error=float((full['gradient']-legacy_gradient).abs().max().detach()),
            legacy_weight_error=float((deployed.weight-legacy.weight).abs().max().detach()),
            legacy_logit_error=float((native_logit-legacy_logit).abs().max().detach()),
            source_risk_loss=float(full['risk_loss'].detach())))
    for model,expected in [(base,original),(decoder,original_decoder)]:
        if any(not torch.equal(v,expected[k]) for k,v in model.state_dict().items()):
            raise ValueError('a source template changed during the functional proof')
    summary = dict(n=len(records),
        maximum_weight_error=max(v['first_step_weight_error'] for v in records),
        maximum_logit_error=max(v['current_risk_logit_error'] for v in records),
        maximum_legacy_weight_error=max(v['legacy_weight_error'] for v in records),
        maximum_legacy_logit_error=max(v['legacy_logit_error'] for v in records),
        median_outer_gradient_norm=float(np.median([v['decoder_outer_gradient_norm'] for v in records])),
        all_stopped_gradients_zero=True,all_full_gradients_finite_nonzero=True)
    out.mkdir(parents=True,exist_ok=False)
    j.r.write_new(out/'result.json',dict(status='SOURCE_FIRST_STEP_TEACHING_PATH_VERIFIED',
        summary=summary,records=records,source_TRAIN_only=True,no_east_data=True,
        source_templates_unchanged=True,attention_backend='MATH for all paired calls',
        optimizer_scope='zero-state first AdamW step only, not accumulated online optimizer',
        target='query-window risk against native source label at query+25; not native query label',
        no_meta_training_or_east_efficacy_claim=True,
        source_files={str(p):p.read_text() for p in [Path(__file__),
            j.r.ROOT/'risk_meta_step_20261107.py',j.r.ROOT/'meta_energy_20261107.py']}))
    print(summary,flush=True)


if __name__ == '__main__':
    run(j.r.ROOT/'probe_outputs/risk_teacher_numerical_20261107_v1')
