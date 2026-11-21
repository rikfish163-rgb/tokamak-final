"""A direct forecast is judged by the SAME frozen risk network.

This changes inference, not the qualified parent artifact. At time t, the
virtual input contains the last103 measured points and 25 predicted points.
No true future values, future masks, new risk head or autoregressive rollout
are used. The caller must timestamp the alarm at t, not t+25.
"""
import torch
from torch import nn


def complete_context(values, available):
    """Every past row has a finite measured diagnostic, not padded history.

    This does not require all23 diagnostics or infer future availability. A
    measured zero is valid; an unavailable or nonfinite row is not.
    """
    if (values.ndim != 3 or values.shape[1:] != (128, 23) or not len(values)
            or available.shape != values.shape or available.dtype != torch.bool
            or available.device != values.device):
        raise ValueError('W128 flat23 values and observed boolean mask required')
    return (available & torch.isfinite(values)).any(-1).all(-1)


def future_window(values, available, forecast, center, scale, *, mode='forecast'):
    if (values.ndim != 3 or values.shape[1:] != (128, 23) or not len(values)
            or available.shape != values.shape or available.dtype != torch.bool
            or available.device != values.device
            or forecast.shape != (len(values), 25, 23)
            or forecast.dtype != values.dtype or forecast.device != values.device
            or center.shape != (23,) or scale.shape != (23,)
            or center.device != values.device or scale.device != values.device
            or not torch.isfinite(center).all() or not torch.isfinite(scale).all()
            or not (scale > 0).all() or not torch.isfinite(forecast).all()
            or mode not in ('forecast', 'hold')):
        raise ValueError('finite direct H25/source coordinates and W128 flat23 input required')
    actual_mask = available & torch.isfinite(values)
    # Unknown future availability is not fetched. Only channels observed NOW
    # may supply forecast inputs. Explicit missingness remains missing.
    predicted_mask = actual_mask[:, -1:].expand(-1, 25, -1)
    if mode == 'hold':
        predicted = torch.where(actual_mask[:, -1:], values[:, -1:], center).expand(-1, 25, -1)
    else:
        # Forecasts are already in source-asinh coordinates. Convert exactly
        # once to the input's source-file coordinates; no target refit or
        # invented physical unit conversion. Overflow fails explicitly.
        predicted = (center.double() + scale.double()*forecast.double().sinh()).to(values.dtype)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError('direct forecast inverse overflows; no clamp or substitution')
        predicted = torch.where(predicted_mask, predicted, center)
    return (torch.cat((values[:, 25:], predicted), dim=1),
            torch.cat((actual_mask[:, 25:], predicted_mask), dim=1))


class AnticipatoryReadout(nn.Module):
    """Reuse an internal fast/slow model; only its logit readout is changed.

    Signal supervision retains the ORIGINAL direct forecast, not a forecast
    of synthetic inputs. Both neural passes share the same temporary and
    persistent state. A hold-last-signal control isolates window shifting.
    """
    def __init__(self, internal, *, mode='forecast', require_full_context=False):
        super().__init__()
        if mode not in ('forecast', 'hold') or (internal.cfg.window, internal.cfg.horizon) != (128, 25):
            raise ValueError('fixed W128/H25 forecast or hold control required')
        self.internal, self.mode = internal, mode
        self.require_full_context = require_full_context
        self.temporary_dim = internal.temporary_dim
        self.adaptation_site = internal.adaptation_site

    @property
    def cfg(self): return self.internal.cfg

    @property
    def center(self): return self.internal.center

    @property
    def scale(self): return self.internal.scale

    @property
    def base(self): return self.internal.base

    @property
    def memory(self): return self.internal.memory

    def initial_fast(self, batch): return self.internal.initial_fast(batch)

    def persistent_parameters(self): return self.internal.persistent_parameters()

    def summarize_fast(self, tensor): return self.internal.summarize_fast(tensor)

    def forecast_loss(self, prediction, truth, available):
        return self.internal.forecast_loss(prediction, truth, available)

    def forward(self, values, available, fast=None):
        actual = self.internal(values, available, fast)
        eligible = (complete_context(values, available) if self.require_full_context
                    else torch.ones(len(values), device=values.device, dtype=torch.bool))
        if not eligible.any():
            # Explicit eligibility policy: preserve the observed-risk branch,
            # including early alarms. This is not an alarm veto or silent
            # substitution after a failed forecast.
            return dict(logit=actual['logit'], forecast=actual['forecast'],
                        observed_logit=actual['logit'])
        virtual, mask = future_window(values[eligible], available[eligible], actual['forecast'][eligible],
                                      self.center, self.scale, mode=self.mode)
        anticipated = self.internal(virtual, mask, None if fast is None else fast[eligible])
        logits = actual['logit'].clone()
        logits[eligible] = anticipated['logit']
        return dict(logit=logits, forecast=actual['forecast'],
                    observed_logit=actual['logit'])
