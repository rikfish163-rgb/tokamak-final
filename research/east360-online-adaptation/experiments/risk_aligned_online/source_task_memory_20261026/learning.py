"""Risk gradients traverse diagnostic writes; no target labels enter teaching."""
import numpy as np
import torch

from ..delayed_recurrent_memory_20261025.learning import batch
from ..delayed_recurrent_memory_20261025.model import patch_signal, endpoint_corrections
from ..lifecycle_stream_20261025.model import LifecycleAdapter
from ..operational_bag_loss_20261025.learning import bag_loss


def validate_roles(roles, counts=(1353, 160, 220)):
    names = ('train', 'val', 'test')
    for name, count in zip(names, counts):
        if len(roles[name]) != count or len(set(roles[name])) != count:
            raise ValueError('canonical source role count required')
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if set(roles[a]) & set(roles[b]):
                raise ValueError('source role overlap')


def source_endpoints(shot):
    if shot['device'] != 'J-TEXT':
        raise ValueError('J-TEXT offline examples only')
    t = shot['times']
    # Preserve the base model's masked left-padding contract. Real TRAIN shot
    # 1051627 has only52 points; requiring128 would discard a valid discharge.
    keep = shot['valid'].copy()
    if shot['is_disrupt']:
        keep &= np.round((shot['down']-t)*1000., 9) > 10.
    all_ends = np.flatnonzero(keep)
    if not len(all_ends):
        raise ValueError('no eligible source timely endpoint')
    uniform = all_ends[np.linspace(0, len(all_ends)-1, min(64, len(all_ends)), dtype=int)]
    positive = all_ends[shot['y'][all_ends] > 0]
    if len(positive):
        positive = positive[np.linspace(0, len(positive)-1, min(16, len(positive)), dtype=int)]
    return np.unique(np.r_[uniform, positive])


def make_adapter(base, kind):
    if kind not in ('learned', 'fixed'):
        raise ValueError('learned or fixed source task required')
    model = LifecycleAdapter(base, readout='full').eval()
    model.slow.requires_grad_(False)
    model.memory.requires_grad_(kind == 'learned')
    model.memory.output.requires_grad_(True)
    return model


@torch.no_grad()
def prepare_example(model, shot, threshold):
    ends = source_endpoints(shot)
    signal = {k: shot[k] for k in ('values', 'available')}
    means, complete = patch_signal(signal, model.base.center, model.base.scale)
    hs, vs, zs = [], [], []
    for start in range(0, len(ends), 32):
        x, mask = batch(signal, ends[start:start+32], model.base.center.device)
        h, v = model.tokens(x, mask)
        hs.append(h.cpu()); vs.append(v.cpu())
        zs.append(model.base(x, mask)['logit'].cpu())
    z = torch.cat(zs)
    return dict(shot=int(shot['shot']), flag=bool(shot['is_disrupt']),
                means=means.cpu(), complete=complete.cpu(), h=torch.cat(hs), valid=torch.cat(vs),
                ends=torch.as_tensor(ends), baseline=float(bag_loss(z.max(), shot['is_disrupt'], threshold)),
                source_horizon_ms=float(shot['horizon']), baseline_logits=z)


def to_device(example, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in example.items()}


def loss_from_example(model, example, threshold):
    # No no_grad/detach here: the outer loss teaches the actual recurrent rule.
    trace = model.memory.trace(example['means'], example['complete'], mode='memory')
    delta = endpoint_corrections(trace['delta'], example['ends'])
    h, v = example['h'], example['valid']
    z = torch.cat([model.risk(h[i:i+32], v[i:i+32], delta[i:i+32])
                   for i in range(0, len(h), 32)])
    loss = bag_loss(z.max(), example['flag'], threshold)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite sampled shot objective')
    return loss, dict(writes=trace['updates'], max_state_norm=float(trace['norm'].detach().max()),
                     forecast_loss=float(trace['forecast_loss'].detach()),
                     peak_logit=float(z.detach().max()))
