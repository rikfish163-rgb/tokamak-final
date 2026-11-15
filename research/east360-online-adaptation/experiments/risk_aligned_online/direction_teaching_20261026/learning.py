import math

import numpy as np
import torch
from torch.nn import functional as F

from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..lifecycle_stream_20261025.model import LifecycleAdapter
from ..operational_bag_loss_20261025.learning import bag_loss


PROFILE_KINDS = {
    'direction_teaching_v1': ('bag', 'direction'),
    'reference_gradient_ablation_v2': ('direction', 'detached'),
    'zero_prior_ablation_v3': ('detached', 'zero_prior'),
}


def prior_policy(kind):
    if kind not in ('detached', 'zero_prior'):
        raise ValueError('explicit zero-prior ablation arm required')
    return 'fixed_zero' if kind == 'zero_prior' else 'learned'


def reference_policy(kind):
    if kind not in ('direction', 'detached', 'zero_prior'):
        raise ValueError('explicit reference ablation arm required')
    return 'joint' if kind == 'direction' else 'stopped'


def local_targets(shot, ends):
    if shot['device'] != 'J-TEXT':
        raise ValueError('source J-TEXT labels only')
    ends = np.asarray(ends)
    keep = shot['valid'][ends].copy()
    if shot['is_disrupt']:
        keep &= (shot['y'][ends] > 0) & (np.round((shot['down']-shot['times'][ends])*1000, 9) > 10)
    return dict(labels=torch.as_tensor(shot['y'][ends], dtype=torch.float32),
                local_mask=torch.as_tensor(keep), direction_mask=torch.as_tensor(keep & (ends >= 16)))


def make_adapter(base, *, zero_prior=False):
    m = LifecycleAdapter(base, readout='full', feedback='progressive').eval()
    m.slow.requires_grad_(False)
    m.memory.requires_grad_(True)
    if zero_prior:
        # With zero diagnostic changes, the zero-initialized matrix remains
        # exactly zero. No learned-prior relaxation can create a fast readout.
        with torch.no_grad():
            m.memory.prior.zero_()
        m.memory.prior.requires_grad_(False)
    return m


def objective(after, before, e, threshold, kind):
    if kind not in ('bag', 'direction', 'detached', 'zero_prior'):
        raise ValueError('explicit teaching objective required')
    bag = bag_loss(after.max(), e['flag'], threshold)
    shift = math.log(threshold/(1-threshold))
    a = F.binary_cross_entropy_with_logits(after-shift, e['labels'], reduction='none')
    b = F.binary_cross_entropy_with_logits(before-shift, e['labels'], reduction='none')
    local = a[e['local_mask']].mean() if e['local_mask'].any() else a.sum()*0
    # Single-factor ablation: identical forward value; no descent through the
    # reference branch that could lower this penalty by worsening the reference.
    # Shared parameters can still change both predictions on the NEXT forward.
    gap = a-(b.detach() if kind in ('detached', 'zero_prior') else b)
    selected = gap[e['direction_mask']]
    direction = .02*F.softplus(selected/.02).mean() if selected.numel() else gap.sum()*0
    total = bag if kind == 'bag' else bag+.5*local+.5*direction
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite direction teaching')
    return total, dict(bag=bag, local=local, direction=direction,
                       direction_count=int(selected.numel()),
                       harmful_writes=int((selected.detach() > 1e-7).sum()))


def loss_from_example(model, e, threshold, kind, *, forecast_weight=0.):
    if forecast_weight not in (0., 1.):
        raise ValueError('declared forecast-objective ablation weight must be zero or one')
    trace = model.memory.trace(e['means'], e['complete'], mode='memory', return_before=True)
    after = endpoint_corrections(trace['delta'], e['ends'])
    before = endpoint_corrections(trace['before_delta'], e['ends'])
    h, v = e['h'], e['valid']

    def risk(delta):
        return torch.cat([model.risk(h[i:i+32], v[i:i+32], delta[i:i+32])
                          for i in range(0, len(h), 32)])

    total, detail = objective(risk(after), risk(before), e, threshold, kind)
    if forecast_weight:
        # Explicit source-only multi-task teaching; deployment still consumes
        # only targets that have actually arrived. Default risk-only is intact.
        total = total + trace['forecast_loss']
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite forecast teaching objective')
    info = {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in detail.items()}
    info.update(writes=trace['updates'], forecast_loss=float(trace['forecast_loss'].detach()))
    if 'unweighted_forecast_loss' in trace:
        info['unweighted_forecast_loss'] = float(trace['unweighted_forecast_loss'].detach())
    return total, info
