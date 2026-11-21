"""Arrived-window feedback retains an explicit frozen-task direction.

Only the auxiliary feedback coordinates change. The actual risk still uses
the original network on REAL observations. No virtual forecast is supplied
to the alarm, and the frozen task coordinate is not an EAST outcome label.
It may reproduce source bias; a complete target-stream ablation is required.
"""
import torch

from .event_dynamics import EventDynamicsFfnModel


def task_basis(risk_weight, scale, events, *, mode):
    if (mode not in ('risk', 'event_risk', 'random_risk') or risk_weight.shape != (1, 64)
            or scale.shape != (192,) or events.shape != (5, 192)
            or not all(torch.isfinite(t).all() for t in (risk_weight, scale, events))
            or not (scale > 0).all()):
        raise ValueError('frozen linear CLS risk head, source feature scale and five state contrasts required')
    # z=(f-mean)/scale; logit=w*f_cls+b. Its directional part in z-space
    # is (w*scale_cls,0,0), without refitting a risk head or any EAST scale.
    raw = torch.cat((risk_weight.detach().cpu().double()[0] * scale.detach().cpu().double()[:64],
                     torch.zeros(128, dtype=torch.float64)))
    if raw.norm() <= 0: raise ValueError('nonzero original risk direction required')
    risk = raw / raw.norm()
    if mode == 'risk': return risk[None].to(events)
    if mode == 'random_risk':
        rng = torch.Generator(device='cpu').manual_seed(131)
        q, _ = torch.linalg.qr(torch.randn(192, 5, dtype=torch.float64, generator=rng), mode='reduced')
        auxiliary = q.T
    else:
        auxiliary = events.detach().cpu().double()
    residual = auxiliary - (auxiliary @ risk)[:, None] * risk
    _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    if singular[4] <= singular[0] * 1e-7:
        raise ValueError('five independent task-orthogonal auxiliary directions required')
    result = torch.cat((risk[None], vh[:5]), 0).to(events)
    torch.testing.assert_close(result @ result.T, torch.eye(6).to(result), atol=2e-6, rtol=2e-6)
    return result.detach()


class TaskStateFfnModel(EventDynamicsFfnModel):
    intra_objective_name = 'direct-window state error including the frozen source risk direction'

    def __init__(self, base, directions, feature_mean, feature_scale, *, mode):
        super().__init__(base, directions, feature_mean, feature_scale, mode='event')
        self.state_directions = task_basis(base.risk_head.weight, self.feature_scale,
                                          self.state_directions, mode=mode)
        self.state_mode = mode
        self.register_buffer('risk_coordinate_scale',
            (base.risk_head.weight[0].detach() * self.feature_scale[:64]).norm().detach())

    def prepare_matured_forecast_loss(self, issued, truth, available, *, values, observed):
        loss, receipt = super().prepare_matured_forecast_loss(issued, truth, available,
                                                            values=values, observed=observed)
        receipt.update(state_dimensions=len(self.state_directions), risk_coordinate_count=1,
            risk_coordinate_scale=float(self.risk_coordinate_scale),
            risk_teacher_is_physical_truth=False,
            teacher_target='source task response to arrived diagnostics, not the shot outcome',
            direction_selection='one original risk direction plus orthogonal auxiliary directions')
        return loss, receipt
