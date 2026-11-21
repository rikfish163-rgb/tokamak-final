"""Ended-EAST learning of the ORIGINAL H25 auxiliary head's update direction.

The immutable J-TEXT classifier is kept separately and never overwritten.
Changing this head cannot change a zero-fast risk prediction. During a later
shot, mature real H25 observations induce disposable last-FFN gradients. The
outer ended-shot objective learns whether those updates help timely alarms.
This is inter-trained intra adaptation, not label-free first-shot meta-training.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call

from . import io23, simple_data
from .ended_context_rule import causal_peak_logit
from .persistent_ffn import PersistentFfnModel
from .redesign_20261004.intra import _batch_windows


SIGNAL_CONTRACTS = ('all23', 'without_mnn')


def _signal_loss(base,prediction,truth,available,contract):
    if contract=='all23':
        return base.forecast_loss(prediction,truth,available)
    if (prediction.shape!=truth.shape or available.shape!=truth.shape or available.dtype!=torch.bool
            or truth.ndim!=3 or truth.shape[-1]!=23):
        raise ValueError('batched flat23 forecast targets and boolean masks must match')
    observed=available.clone();observed[...,io23.RAW23.index('mode_number_n')]=False
    return base.forecast_loss(prediction,truth,observed)


class DecoderFfnModel(PersistentFfnModel):
    def __init__(self, base, *, signal_contract='all23'):
        if signal_contract not in SIGNAL_CONTRACTS:
            raise ValueError('explicit auxiliary signal contract required')
        super().__init__(base)
        self.memory.requires_grad_(False)
        self.decoder = deepcopy(self.base.forecast_head).requires_grad_(True)
        self.signal_contract = signal_contract

    def forecast_loss(self, prediction, truth, available):
        # Preserve the exact base objective; only explicit without_mnn changes
        # supervision, never the input or a claim about integer categories.
        return _signal_loss(self.base,prediction,truth,available,self.signal_contract)

    def make_reference(self):
        return DecoderFfnModel(self.base,signal_contract=self.signal_contract)

    def assert_persistent_inactive(self):
        if self.memory.any() or self.memory.grad is not None:
            raise AssertionError('ended-head training changed persistent FFN')

    def forecast_overrides(self):
        return {f'forecast_head.{name}': parameter
                for name, parameter in self.decoder.named_parameters()}

    def prepare_intra_forecast_loss(self, issued, truth, observed):
        # No moving target or candidate-dependent loss weights. The same
        # declared masked SmoothL1 is used before and after a proposed step.
        return self.forecast_loss


class SourceTokenDecoderModel(nn.Module):
    """Original source token gain/bias, not a new layer or final-score head.

    The external fast vector is an OFFSET from the source's learned origin.
    Thus zero offset is exactly the parent even when fast_origin is nonzero;
    live and meta steps both project offsets around zero. No persistent neural
    state exists in this bounded pilot. Only the independent H25 head learns
    from ended EAST experience, through that same original token path.
    """
    adaptation_site='original source token gain/bias before Transformer'

    def __init__(self,base,*,signal_contract='all23'):
        super().__init__()
        if base.cfg.fast_site!='tokens' or signal_contract not in SIGNAL_CONTRACTS:
            raise ValueError('source token configuration and explicit signal contract required')
        self.base=deepcopy(base).requires_grad_(False)
        self.decoder=deepcopy(self.base.forecast_head).requires_grad_(True)
        self.signal_contract=signal_contract
        self.temporary_dim=2*self.base.cfg.dim

    @property
    def cfg(self):return self.base.cfg

    @property
    def center(self):return self.base.center

    @property
    def scale(self):return self.base.scale

    def initial_fast(self,batch):
        if type(batch) is not int or batch<1:raise ValueError('positive integer batch required')
        return self.center.new_zeros(batch,self.temporary_dim)

    def forecast_loss(self,prediction,truth,available):
        return _signal_loss(self.base,prediction,truth,available,self.signal_contract)

    def prepare_intra_forecast_loss(self,issued,truth,observed):return self.forecast_loss

    def make_reference(self):
        return SourceTokenDecoderModel(self.base,signal_contract=self.signal_contract)

    def assert_persistent_inactive(self):
        if any(not name.startswith(('base.','decoder.')) for name,_ in self.named_parameters()):
            raise AssertionError('unexpected persistent parameters in token-only pilot')

    def forward(self,values,available,fast=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError('inference_mode cannot preserve source token gradient-path dispatch')
        fast=self.initial_fast(len(values)) if fast is None else fast
        if (fast.shape!=(len(values),self.temporary_dim) or fast.dtype!=self.center.dtype
                or fast.device!=self.center.device or not torch.isfinite(fast).all()):
            raise ValueError('finite matching source-token offset required')
        overrides={f'forecast_head.{name}':p for name,p in self.decoder.named_parameters()}
        enabled=torch.is_grad_enabled()
        # Match the established FFN control's fixed CUDA attention dispatch.
        # The constant origin leaf is never optimized or retained. A no_grad
        # call otherwise takes fused encoder inference and changes the logits;
        # the CUDA regression checks exact no_grad/gradient/FFN parity.
        with torch.enable_grad():
            origin=self.base.initial_fast(len(values)).detach().requires_grad_(True)
            out=functional_call(self.base,overrides,(values,available,origin+fast),strict=False)
        return out if enabled else {key:value.detach() for key,value in out.items()}


def maintenance_loss(model, shot):
    """Four equally-spaced fully mature H25 origins from an ENDED shot.

    This keeps the head tied to observed signals rather than allowing a risk
    objective to invent arbitrary future diagnostics. .20 is an engineering
    tradeoff, not a physical constant or a proof of prediction faithfulness.
    """
    origins = np.arange(0, len(shot['times'])-25, 100, dtype=int)
    origins = origins[np.asarray(shot['valid'], bool)[origins]]
    if not len(origins):
        raise ValueError('ended shot has no fully mature valid forecast origin')
    selected = np.unique(np.linspace(0, len(origins)-1, min(4, len(origins)), dtype=int))
    origins = origins[selected]
    device = model.center.device
    signals = {k: shot[k] for k in ('values', 'available')}
    x, available = _batch_windows(simple_data, signals, origins, model, device)
    truth = torch.as_tensor(np.stack([shot['values'][o+1:o+26] for o in origins]),
                            device=device, dtype=model.center.dtype)
    observed = torch.as_tensor(np.stack([shot['available'][o+1:o+26] for o in origins]),
                               device=device) & torch.isfinite(truth)
    forecast = model(x, available, model.initial_fast(len(origins)))['forecast']
    return model.forecast_loss(forecast, truth, observed)


def train_pair(model, pair, optimizer, *, boundary, mode):
    if mode not in ('signal_only', 'risk_after_update') or not math.isfinite(boundary):
        raise ValueError('explicit objective mode and finite frozen boundary required')
    allowed = {id(p) for p in model.decoder.parameters()}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if (set(actual) != allowed or len(actual) != len(allowed)
            or {id(p) for p in model.parameters() if p.requires_grad} != allowed):
        raise ValueError('only the ended-EAST auxiliary decoder may be optimized')
    if (len(pair) != 2 or sorted(int(s['is_disrupt']) for s, _ in pair) != [0, 1]
            or any(s.get('device') != 'EAST' or '_sealed_scores' not in s
                   or s.get('horizon') != 125. for s, _ in pair)):
        raise ValueError('one balanced pair of sealed ended EAST shots required')
    base_before = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
    decoder_before = [p.detach().clone() for p in model.decoder.parameters()]
    model.zero_grad(set_to_none=True)
    records = []
    risk_gradient_square = 0.
    for shot, query in pair:
        if not shot['valid'][query] or (shot['is_disrupt'] and
                round(float((shot['down']-shot['times'][query])*1000.), 9) <= 30.):
            raise ValueError('training peak must be operationally eligible')
        logit, info = causal_peak_logit(model, shot, query,
                                       create_graph=mode == 'risk_after_update')
        risk = F.softplus((1-2*int(shot['is_disrupt']))*(logit-boundary))
        signal = maintenance_loss(model, shot)
        if mode == 'risk_after_update' and risk.requires_grad:
            gradients = torch.autograd.grad(risk, tuple(model.decoder.parameters()),
                                            retain_graph=True, allow_unused=True)
            risk_gradient_square += sum(float(g.detach().double().square().sum())
                                        for g in gradients if g is not None)
        loss = .20*signal + (risk if mode == 'risk_after_update' else 0.)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite ended-head objective')
        (loss/2).backward()
        records.append(dict(**info, shot=shot['shot'], label=int(shot['is_disrupt']),
            query=query, adapted_probability=float(logit.detach().sigmoid()),
            risk_loss=float(risk.detach()), maintenance_loss=float(signal.detach()),
            objective=float(loss.detach())))
    norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.decoder.parameters()):
        raise FloatingPointError('nonfinite auxiliary decoder parameter')
    model.assert_persistent_inactive()
    for key, value in model.base.state_dict().items():
        torch.testing.assert_close(value, base_before[key], rtol=0, atol=0)
    delta = sum(float((p.detach().double()-q.double()).square().sum())
                for p, q in zip(model.decoder.parameters(), decoder_before))**.5
    return dict(records=records, gradient_norm=float(norm), decoder_delta_norm=delta,
                risk_gradient_norm=math.sqrt(risk_gradient_square),
                objective=float(np.mean([r['objective'] for r in records])))
