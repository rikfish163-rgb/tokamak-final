"""Remove mean-level fitting from a window-local auxiliary update only."""
import torch
from ..curvature_fast_window_20261023.model import CurvatureLayer, CurvatureAdapter


class CenteredLayer(CurvatureLayer):
    def fast_weights(self, h, valid, *, mode):
        if mode == 'off':
            # Preserve both values AND gradient graph of the static control.
            return super().fast_weights(h, valid, mode=mode)
        if (mode not in ('on', 'reverse') or h.ndim != 3
                or valid.shape != h.shape[:2] or valid.dtype != torch.bool):
            raise ValueError('past observed token mask and declared mode required')
        _, k, v = self.coordinates(h)
        observed = valid.clone(); observed[:, -1] = False
        count = observed.sum(-1); denominator = count.clamp_min(1)
        weight = observed[..., None]
        k = (k - (k*weight).sum(1)[:, None]/denominator[:, None, None])*weight
        v = (v - (v*weight).sum(1)[:, None]/denominator[:, None, None])*weight
        if mode == 'reverse':
            length = h.shape[1]
            positions = torch.arange(length, device=h.device).expand(len(h), -1)
            ordered = torch.where(observed, positions, length).sort(-1).values
            ranks = observed.long().cumsum(-1)-1
            indices = ordered.gather(1, (count[:, None]-1-ranks).clamp(0, length-1)).clamp_max(length-1)
            v = v.gather(1, indices[..., None].expand_as(v))*weight
        origin = self.origin.expand(len(h), -1, -1)
        error = k@origin-v
        gradient = k.transpose(1, 2)@error/denominator[:, None, None]
        trace = k.square().sum((1, 2))/denominator
        eta = self.eta/(1+trace)
        updated = count >= 2
        fast = origin-eta[:, None, None]*gradient*updated[:, None, None]
        info = dict(updated=updated.detach(), observations=count.detach(),
            curvature_trace=trace.detach(), effective_eta=eta.detach(),
            delta_norm=(fast-origin).detach().square().sum((1, 2)).sqrt(),
            loss_before=(error.square().sum((1, 2))/(2*denominator)).detach(),
            loss_after=((k@fast-v).square().sum((1, 2))/(2*denominator)).detach())
        return fast, info


class CenteredAdapter(CurvatureAdapter):
    def __init__(self, base):
        super().__init__(base)
        previous = self.layer
        self.layer = CenteredLayer(base.cfg.dim)
        self.layer.load_state_dict(previous.state_dict())
