"""Source-trained cross-diagnostic H25 supervision of a disposable input bias.

This is a numerical identifiability experiment, not physical density/unit
calibration. No density history enters the auxiliary predictor. Original risk
and Transformer tensors stay untouched; the auxiliary head never adds a risk
score. Only after real H25 signals mature may a one-dimensional fast bias
change density input BEFORE the original patch embedding.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import fast_meta, io23, simple_data, simple_train
from .redesign_20261004.embedding_model import EmbeddingModel
from .redesign_20261004.run import (DEFAULT_SOURCE, DEFAULT_ORDER, DEFAULT_ROSTER,
    load_parent, load_roster, validate_order, validate_roster_rate_contract, run_arm)
from .simple_online import _durable_json, _hash

D = io23.RAW23.index('ne_nG')
M = io23.RAW23.index('MNM')
FORMAT = 'source-cross-density-h25-v1'


def density_loss(prediction, target, available):
    if (prediction.ndim != 2 or prediction.shape[1] != 25 or target.shape != prediction.shape
            or available.shape != target.shape or available.dtype != torch.bool):
        raise ValueError('matching direct [B,25] density target and bool mask required')
    valid = available & torch.isfinite(target)
    clean = torch.where(valid, target.detach(), torch.zeros_like(target))
    error = F.smooth_l1_loss(prediction, clean, reduction='none').masked_fill(~valid, 0.)
    return error.sum() / valid.sum().clamp_min(1)


class CrossDensityModel(nn.Module):
    temporary_dim = 1
    adaptation_site = 'density source-coordinate bias BEFORE original embedding'

    def __init__(self, parent):
        super().__init__()
        if not isinstance(parent, EmbeddingModel):
            raise ValueError('qualified embedding parent required')
        self.parent = parent.eval().requires_grad_(False)
        # 64 CLS + 22 last values + 22 eight-lag differences + 44 mask flags.
        self.cross_head = nn.Sequential(nn.Linear(152, 64), nn.SiLU(), nn.Linear(64, 25))
        self.cross_head.to(device=parent.center.device, dtype=parent.center.dtype)

    @property
    def base(self): return self.parent.base
    @property
    def cfg(self): return self.parent.cfg
    @property
    def center(self): return self.parent.center
    @property
    def scale(self): return self.parent.scale

    def train(self, mode=True):
        super().train(mode); self.parent.eval(); return self

    def initial_fast(self, batch):
        if type(batch) is not int or batch < 0: raise ValueError('nonnegative batch required')
        return self.center.new_zeros(batch, 1)

    @staticmethod
    def offset(fast):
        if fast.ndim != 2 or fast.shape[1] != 1 or not torch.isfinite(fast).all():
            raise ValueError('finite [B,1] temporary density bias required')
        # Source-asinh numerical trust region, NOT a physical density bound.
        return fast.clamp(-12., 12.)

    def cross_features(self, values, available):
        masked = available.clone(); masked[:, :, [D, M]] = False
        valid = masked & torch.isfinite(values)
        clean = torch.where(valid, values, self.center)
        captured = []
        hook = self.base.risk_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]))
        try:
            # Match the always-differentiable parent attention path. This
            # auxiliary call never returns a replacement risk prediction.
            zero = self.parent.initial_fast(len(values)).detach().requires_grad_(True)
            self.parent(clean, valid, zero)
        finally:
            hook.remove()
        if len(captured) != 1: raise RuntimeError('one source CLS capture required')
        level = torch.asinh((clean - self.center) / self.scale).masked_fill(~valid, 0.)
        others = [i for i in range(23) if i != D]
        last, older = valid[:, -1, others], valid[:, -9, others]
        diff = (level[:, -1, others] - level[:, -9, others]).masked_fill(~(last & older), 0.)
        return torch.cat((captured[0], level[:, -1, others], diff,
                          last.to(level.dtype), older.to(level.dtype)), -1)

    def forward(self, values, available, fast=None):
        if (values.ndim != 3 or values.shape[1:] != (128, 23) or not len(values)
                or values.dtype != self.center.dtype or values.device != self.center.device
                or available.shape != values.shape or available.dtype != torch.bool
                or available.device != values.device):
            raise ValueError('source dtype/device nonempty [B,128,23] and bool mask required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if fast.shape != (len(values), 1) or fast.dtype != values.dtype or fast.device != values.device:
            raise ValueError('matching source [B,1] temporary density bias required')
        bias = self.offset(fast)
        valid = available & torch.isfinite(values)
        clean = torch.where(valid, values, self.center)
        level = torch.asinh((clean[:, :, D] - self.center[D]) / self.scale[D])
        mapped = clean.clone()
        # Exact zero identity: no sinh/asinh round-trip of the original data.
        mapped[:, :, D] = clean[:, :, D] + self.scale[D] * (
            (level + bias).sinh() - level.sinh())
        if not torch.isfinite(mapped).all(): raise FloatingPointError('nonfinite density mapping')
        out = self.parent(mapped, valid, self.parent.initial_fast(len(values)))
        canonical = self.cross_head(self.cross_features(values, valid))
        forecast = out['forecast'].clone()
        # Observed-coordinate inverse: truth remains the actual arrived data.
        # The density prediction C(other signals)-bias cannot copy ne history.
        forecast[:, :, D] = canonical - bias
        return dict(logit=out['logit'], forecast=forecast)

    def intra_forecast_loss(self, prediction, truth, available):
        seen = available[:, :, D] & torch.isfinite(truth[:, :, D])
        raw = torch.where(seen, truth[:, :, D].detach(), self.center[D])
        target = torch.asinh((raw - self.center[D]) / self.scale[D])
        return density_loss(prediction[:, :, D], target, seen)

    def forecast_loss(self, prediction, truth, available):
        return self.base.forecast_loss(prediction, truth, available)


def collect(model, shots, pools, rng, batches):
    rows = []
    for _ in range(batches):
        e = fast_meta.sample(shots, pools, model.cfg, rng, 32, model.center.device)
        feature = model.cross_features(e['x'], e['a']).detach()
        mask = e['mask'][:, :, D] & torch.isfinite(e['truth'][:, :, D])
        target = torch.asinh((torch.where(mask, e['truth'][:, :, D], model.center[D])
                              - model.center[D]) / model.scale[D]).detach()
        if not torch.isfinite(feature).all(): raise FloatingPointError('nonfinite source cross features')
        rows.append((feature, target, mask))
    return tuple(torch.cat([r[i] for r in rows]) for i in range(3))


def train_head(model, features, target, available, *, steps):
    original = {k:v.detach().clone() for k,v in model.parent.state_dict().items()}
    model.parent.requires_grad_(False); model.cross_head.requires_grad_(True)
    optimizer = torch.optim.Adam(model.cross_head.parameters(), lr=1e-3)
    rng = np.random.default_rng(131); history = []
    for step in range(steps):
        indices = torch.as_tensor(rng.integers(len(features), size=64), device=features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = density_loss(model.cross_head(features[indices]), target[indices], available[indices])
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite cross-head loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.cross_head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); history.append(float(loss.detach()))
        if (step + 1) % 128 == 0: print(json.dumps(dict(source_step=step+1, signal_loss=history[-1])), flush=True)
    if any(not torch.equal(model.parent.state_dict()[k],v) for k,v in original.items()):
        raise RuntimeError('original source tensors changed')
    model.cross_head.requires_grad_(False)
    return history


def load_head(path, model):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    p = artifact['protocol']
    if (p.get('format') != FORMAT or p.get('parent_sha256') != _hash(DEFAULT_SOURCE)
            or p.get('source_only') is not True or p.get('east_access') is not False
            or p.get('original_parent_bitwise_unchanged') is not True):
        raise ValueError('source-only cross-head parent contract required')
    model.cross_head.load_state_dict(artifact['head'], strict=True)
    model.cross_head.requires_grad_(False)
    return p


def snapshot(output):
    root = Path(__file__).parent
    files = [root / n for n in ('cross_density.py','fast_meta.py','fast_direction.py','simple23.py',
        'simple_data.py','simple_train.py','simple_online.py','io23.py','causal_input.py','fast_update.py','source_gate.py')]
    files.extend(root / 'redesign_20261004' / n for n in ('run.py','intra.py','embedding_model.py'))
    with zipfile.ZipFile(output / 'code_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for path in files: z.write(path, str(path.relative_to(root)))
    return {str(p.relative_to(root)):_hash(p) for p in files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('source','east'), required=True)
    p.add_argument('--head', type=Path)
    p.add_argument('--limit', type=int, default=120)
    args = p.parse_args()
    if args.output.exists() or args.limit < 1 or args.mode == 'east' and args.head is None:
        p.error('new output, positive limit and source head for EAST required')
    if not torch.cuda.is_available(): raise RuntimeError('local rl_env CUDA required')
    torch.set_num_threads(4); torch.manual_seed(131)
    parent, state, gate, _ = load_parent(DEFAULT_SOURCE,'cuda',persistent_site='embedding')
    model = CrossDensityModel(parent).cuda().eval()
    original = {k:v.detach().clone() for k,v in parent.state_dict().items()}
    args.output.mkdir(parents=True)
    code = snapshot(args.output)
    protocol = dict(format=FORMAT, parent_sha256=gate['checkpoint_sha256'],
        original_parent_bitwise_unchanged=True, threshold=float(state['threshold']),
        source_only=args.mode=='source', east_access=args.mode=='east', jtext_online=False,
        source_classifier_training=False, support110=False, target_scaler_fit=False,
        risk_correction_head=False, code_sha256=code, code_snapshot_sha256=_hash(args.output/'code_snapshot.zip'),
        auxiliary_density_history_masked=True, auxiliary_mnm_masked=True,
        auxiliary_features=152, auxiliary_output='25 density values in original source-asinh coordinate',
        source_objective='SmoothL1 density signals; no risk labels in decoder loss',
        temporary_parameters=1, temporary_offset_bound=[-12.,12.], bound_is_physical=False,
        intra_stride=25, intra_step=1., intra_radius=12., intra_objective='forward_holdout',
        fit_steps=[1,12], acceptance_steps=[13,25], maturity_then_effective=[25,26],
        scope='DEVELOPMENT; numerical cross-diagnostic hypothesis, not physical calibration')
    _durable_json(args.output/'protocol.json', protocol)
    if args.mode == 'source':
        roles, advanced = simple_data.source_roles(), simple_data.advanced_times()
        train = simple_train.load_shots(roles['train'], advanced)
        val = simple_train.load_shots(roles['val'], advanced)
        train_data = collect(model, train, fast_meta.episode_pools(train, model.cfg), np.random.default_rng(131), 256)
        val_data = collect(model, val, fast_meta.episode_pools(val, model.cfg), np.random.default_rng(132), 32)
        history = train_head(model, *train_data, steps=512)
        constant = train_data[1][train_data[2]].median().detach()
        with torch.no_grad():
            learned = float(density_loss(model.cross_head(val_data[0]), val_data[1], val_data[2]))
            baseline = float(density_loss(torch.full_like(val_data[1],constant), val_data[1], val_data[2]))
        protocol.update(train_ids=roles['train'],val_ids=roles['val'],test_access=False,
            training_windows=len(train_data[0]),validation_windows=len(val_data[0]),
            source_steps=512,source_batch=64,seed=131,selection='last fixed512, not best EAST checkpoint')
        screening = dict(source_cross_signal_loss=learned,source_constant_loss=baseline,
            source_screen_passed=learned < baseline,constant=float(constant),
            warning='signal prediction is not a proof of useful risk adaptation')
        torch.save(dict(head=model.cross_head.state_dict(),protocol=protocol),args.output/'head.pt')
        torch.save(dict(features=val_data[0].cpu(),target=val_data[1].cpu(),available=val_data[2].cpu()),
                   args.output/'source_val_features.pt')
        _durable_json(args.output/'training.json',dict(history=history))
        _durable_json(args.output/'validation.json',screening)
        print(json.dumps(screening),flush=True)
    else:
        head_protocol=load_head(args.head,model)
        source_screen=json.loads((args.head.parent/'validation.json').read_text())
        if source_screen.get('source_screen_passed') is not True:
            raise ValueError('cross-head failed source signal screen; do not expand to EAST')
        order=json.loads(DEFAULT_ORDER.read_text()); ids,roster=load_roster(DEFAULT_ROSTER)
        validate_order(order,ids);validate_roster_rate_contract(roster,'nominal')
        if args.limit > len(order):raise ValueError('limit exceeds locked roster')
        protocol.update(head_sha256=_hash(args.head),head_source_protocol=head_protocol,
            natural_outcomes=True,paid_query_budget=False,alarm_tuning=False,inter_updates=False,
            selected_shots=args.limit,roster_shots=len(order),order_sha256=_hash(DEFAULT_ORDER),
            roster_sha256=_hash(DEFAULT_ROSTER),arms=['static','intra'])
        _durable_json(args.output/'protocol.json',protocol)
        _durable_json(args.output/'order.json',order)
        _durable_json(args.output/'roster.json',roster)
        provider=SimpleNamespace(window=simple_data.window,attach_outcome=simple_data.attach_outcome,
            read_shot=lambda device,sid:simple_data.read_shot(device,sid,rate_contract='nominal'))
        results={}
        for arm in ('static','intra'):
            results[arm]=run_arm(model,order=order,arm=arm,threshold=float(state['threshold']),
                output_dir=args.output/arm,limit=args.limit,device='cuda',stride=25,step=1.,radius=12.,
                objective='forward_holdout',data_api=provider)
            _durable_json(args.output/'comparison.json',results)
            print(json.dumps(dict(completed_arm=arm,metrics={k:v for k,v in results[arm]['metrics'].items() if k!='predictions'})),flush=True)
    if any(not torch.equal(parent.state_dict()[k],v) for k,v in original.items()) or _hash(DEFAULT_SOURCE)!=gate['checkpoint_sha256']:
        raise RuntimeError('qualified original source changed')
    _durable_json(args.output/'protocol.json',protocol)


if __name__=='__main__':main()
