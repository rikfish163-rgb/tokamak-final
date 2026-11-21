"""Protect source event-discriminating directions during signal-only EAST steps.

This is an OGD-like local first-order constraint, not a physical conservation
law and not a six-class EAST pseudo-label. It may preserve wrong source biases.
The final risk head, alarm and source checkpoint are untouched. The random
five-direction control separates event information from generic restriction.
"""
import torch

from .ended_decoder import DecoderFfnModel
from .fast_update import normalized_step


def make_directions(weight, scale, *, mode, seed=131):
    if (weight.shape != (6, 192) or scale.shape != (192,) or mode not in ('event', 'random')
            or not torch.isfinite(weight).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('finite six-class 192-feature head, positive scale and explicit mode required')
    if mode == 'event':
        # A common shift of all six logits leaves their differences unchanged.
        # The five-dimensional row span contains every class-logit contrast.
        contrasts = (weight.double() - weight.double().mean(0)) / scale.double()
        _, singular, vh = torch.linalg.svd(contrasts, full_matrices=False)
        if singular[4] <= singular[0] * 1e-7:
            raise ValueError('event head does not identify five independent contrasts')
        return vh[:5].to(weight).detach()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(192, 5, dtype=torch.float64, generator=generator), mode='reduced')
    return q.T.to(weight).detach()


def project_gradient(gradient, jacobian):
    """Euclidean projection into the nullspace of local state derivatives.

    Double-precision SVD handles dependent/zero constraints explicitly. No
    identity-gradient substitute is used when the projected gradient is zero.
    """
    if (gradient.ndim != 2 or gradient.shape[0] != 1 or jacobian.ndim != 2
            or not len(jacobian) or jacobian.shape[1] != gradient.shape[1]
            or gradient.device != jacobian.device or gradient.dtype != jacobian.dtype
            or not torch.isfinite(gradient).all() or not torch.isfinite(jacobian).all()):
        raise ValueError('matching finite single-shot gradient and state Jacobian required')
    j, g = jacobian.detach().double(), gradient.detach().double()[0]
    _, singular, vh = torch.linalg.svd(j, full_matrices=False)
    rank = int((singular > singular[0] * 1e-7).sum()) if singular[0] > 0 else 0
    basis = vh[:rank]
    projected = g - (g @ basis.T) @ basis
    original_norm = g.norm()
    if projected.norm() <= 64 * torch.finfo(torch.float64).eps * original_norm:
        projected.zero_()
    if not torch.isfinite(projected).all():
        raise FloatingPointError('nonfinite projected gradient')
    receipt = dict(rank=rank, original_norm=float(original_norm),
        projected_norm=float(projected.norm()),
        retained_norm_fraction=float(projected.norm() / original_norm) if original_norm else 0.,
        state_derivative_original_norm=float((j @ g).norm()),
        state_derivative_projected_norm=float((j @ projected).norm()))
    return projected.to(gradient)[None], receipt


def maturity_window(values, observed, truth, available):
    if (values.shape != (1, 128, 23) or truth.shape != (1, 25, 23)
            or observed.shape != values.shape or available.shape != truth.shape
            or observed.dtype != torch.bool or available.dtype != torch.bool
            or values.device != truth.device or values.dtype != truth.dtype):
        raise ValueError('one past128 window and its fully arrived next25 required')
    return (torch.cat((values.detach(), truth.detach()), dim=1)[:, -128:],
            torch.cat((observed.detach(), available.detach()), dim=1)[:, -128:])


class EventProtectedFfnModel(DecoderFfnModel):
    fast_gradient_transform = 'maturity-prefix state-Jacobian nullspace projection'

    def __init__(self, base, directions, *, mode):
        super().__init__(base, signal_contract='all23')
        if (mode not in ('event', 'random') or directions.shape != (5, 192)
                or not torch.isfinite(directions).all()):
            raise ValueError('five finite 192-dimensional state directions and explicit mode required')
        torch.testing.assert_close(directions @ directions.T,
            torch.eye(5, dtype=directions.dtype, device=directions.device), atol=2e-6, rtol=2e-6)
        self.register_buffer('state_directions', directions.detach().clone().to(self.center))
        self.projection_mode = mode

    def state_values(self, values, available, fast):
        """Same 192 features as the source probe, retaining only fast gradients."""
        captured = []

        def capture(module, args):
            captured.append(args[0].reshape(len(values), -1, self.cfg.dim))

        hook = self.base.forecast_head.register_forward_pre_hook(capture)
        try:
            output = self(values, available, fast)
            if len(captured) != 1:
                raise AssertionError('exactly one original forecast-head input required')
            tokens = captured[0]
            valid = (available & torch.isfinite(values)).reshape(len(values), 16, -1).any(-1)
            mean = (tokens[:, :-1] * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
            features = torch.cat((tokens[:, -1], mean, tokens[:, -2]), dim=-1)
            return features @ self.state_directions.T, output['logit']
        finally:
            hook.remove()

    def state_jacobian(self, values, available, fast):
        current = fast.detach().clone().requires_grad_(True)
        if len(values) != 1 or current.shape != (1, self.temporary_dim):
            raise ValueError('one observed prefix and one temporary state required')
        coordinates, risk = self.state_values(values, available, current)
        rows = [torch.autograd.grad(coordinates[0, i], current, retain_graph=True)[0][0]
                for i in range(5)]
        risk_gradient, = torch.autograd.grad(risk.sum(), current)
        return coordinates.detach(), risk.detach(), torch.stack(rows).detach(), risk_gradient.detach()

    def project_intra_gradient(self, gradient, *, values, observed, truth, available, fast, step, radius):
        x, mask = maturity_window(values, observed, truth, available)
        before, risk_before, jacobian, risk_gradient = self.state_jacobian(x, mask, fast)
        projected, receipt = project_gradient(gradient, jacobian)
        # Measure finite-step effects; first-order protection is not claimed
        # to guarantee nonlinear invariance after normalization/radius clipping.
        changes = {}
        for name, direction in (('unprojected', gradient), ('projected', projected)):
            candidate = normalized_step(fast.detach(), torch.zeros_like(fast), direction, step=step, radius=radius)
            coordinates, risk = self.state_values(x, mask, candidate.detach().requires_grad_(True))
            changes[f'state_change_{name}'] = float((coordinates.detach() - before).double().norm())
            changes[f'risk_logit_change_{name}'] = float((risk.detach() - risk_before)[0])
            changes[f'candidate_step_norm_{name}'] = float((candidate - fast).detach().double().norm())
        receipt.update(changes, mode=self.projection_mode,
            context='latest fully arrived window at maturity',
            outcome_access=False, event_pseudo_label_used=False,
            risk_logit_before=float(risk_before[0]),
            risk_direction_original=float((risk_gradient.double() * gradient.double()).sum()),
            risk_direction_projected=float((risk_gradient.double() * projected.double()).sum()))
        return projected, receipt
