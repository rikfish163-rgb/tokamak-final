"""Arrived diagnostic feedback measured in a frozen source state space.

The ORIGINAL direct H25 forecast is encoded only inside the auxiliary loss.
No forecasted signal or event score enters risk inference. Five future prefix
states (5,10,15,20,25 ms on the 1 kHz stream) are compared once all25 samples
arrive. Source-event coordinates are candidate patterns, not physical truth.
"""
import torch
from torch.nn import functional as F

from .ended_decoder import DecoderFfnModel


HORIZONS = (5, 10, 15, 20, 25)


def _check(values, observed, future, available, center):
    if (values.ndim != 3 or values.shape[1:] != (128, 23) or not len(values)
            or observed.shape != values.shape or observed.dtype != torch.bool
            or future.shape != (len(values), 25, 23) or available.shape != future.shape
            or available.dtype != torch.bool or center.shape != (23,)
            or any(t.device != values.device for t in (observed, future, available, center))
            or future.dtype != values.dtype or center.dtype != values.dtype
            or not torch.isfinite(center).all()):
        raise ValueError('matching W128/H25 flat23 observations, masks and source center required')


def _windows(values, observed, future, common, center):
    actual = observed & torch.isfinite(values)
    past = torch.where(actual, values, center)
    future = torch.where(common, future, center)
    # Batch order is [shot0/h5,...,shot0/h25,shot1/h5,...]. Prefix suffixes
    # are actual past, not predictions from a previous forecasting step.
    windows = torch.stack([torch.cat((past[:, h:], future[:, :h]), 1) for h in HORIZONS], 1)
    masks = torch.stack([torch.cat((actual[:, h:], common[:, :h]), 1) for h in HORIZONS], 1)
    return windows.flatten(0, 1), masks.flatten(0, 1)


def arrived_windows(values, observed, truth, available, center):
    _check(values, observed, truth, available, center)
    common = (available & torch.isfinite(truth) &
              (observed[:, -1:] & torch.isfinite(values[:, -1:])))
    windows, masks = _windows(values, observed, truth, common, center)
    return windows, masks, common


def forecast_windows(values, observed, prediction, common, center, scale):
    _check(values, observed, prediction, common, center)
    if (scale.shape != (23,) or scale.device != values.device or scale.dtype != values.dtype
            or not torch.isfinite(scale).all() or not (scale > 0).all()
            or not torch.isfinite(prediction).all()):
        raise ValueError('finite direct forecast and positive source scale required')
    # Algebraic inverse of the parent's asinh normalization, not calibration.
    # Do not clamp an invalid prediction into apparently valid supervision.
    raw = (center.double() + scale.double() * prediction.double().sinh()).to(values.dtype)
    if not torch.isfinite(raw).all():
        raise FloatingPointError('forecast inverse overflow in matured state loss')
    return _windows(values, observed, raw, common, center)


def state_features(model, values, available):
    """Same CLS/valid-patch-mean/last192 features as the source probe.

    Teacher parameters and its FFN memory are frozen, but input derivatives
    must remain enabled to train the *forecast-producing* temporary FFN.
    """
    captured = []
    hook = model.base.forecast_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].reshape(len(values), 17, 64)))
    try:
        model(values, available)
        if len(captured) != 1:
            raise AssertionError('exactly one frozen state encoding required')
        tokens = captured[0]
        valid = (available & torch.isfinite(values)).reshape(len(values), 16, -1).any(-1)
        mean = (tokens[:, :-1] * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return torch.cat((tokens[:, -1], mean, tokens[:, -2]), -1)
    finally:
        hook.remove()


class EventDynamicsFfnModel(DecoderFfnModel):
    intra_objective_name = 'frozen-state error of the direct predicted and arrived H25 windows'

    def __init__(self, base, directions, feature_mean, feature_scale, *, mode):
        super().__init__(base, signal_contract='all23')
        if (mode not in ('event', 'random') or directions.shape != (5, 192)
                or feature_mean.shape != (192,) or feature_scale.shape != (192,)
                or not all(torch.isfinite(t).all() for t in (directions, feature_mean, feature_scale))
                or not (feature_scale > 0).all()):
            raise ValueError('five source192 directions, positive TRAIN scale and explicit mode required')
        torch.testing.assert_close(directions @ directions.T,
            torch.eye(5, dtype=directions.dtype, device=directions.device), atol=2e-6, rtol=2e-6)
        self.register_buffer('state_directions', directions.detach().clone().to(self.center))
        self.register_buffer('feature_mean', feature_mean.detach().clone().to(self.center))
        self.register_buffer('feature_scale', feature_scale.detach().clone().to(self.center))
        self.teacher = DecoderFfnModel(base).requires_grad_(False).eval()
        self.state_mode = mode

    def coordinates(self, values, available):
        features = state_features(self.teacher, values, available)
        return ((features - self.feature_mean) / self.feature_scale) @ self.state_directions.T

    def prepare_matured_forecast_loss(self, issued, truth, available, *, values, observed):
        # One fixed target/context for issued, before and after losses. This
        # teacher is the immutable source, NOT the evolving student itself.
        values, observed = values.detach().clone(), observed.detach().clone()
        truth, available = truth.detach().clone(), available.detach().clone()
        actual, masks, common = arrived_windows(values, observed, truth, available, self.center)
        target = self.coordinates(actual, masks).detach()
        if not torch.isfinite(target).all():
            raise FloatingPointError('nonfinite arrived source-state target')
        receipt = dict(mode=self.state_mode, horizons=list(HORIZONS),
            valid_common_targets=int(common.sum()), state_dimensions=5,
            teacher='immutable source; zero temporary and persistent FFN',
            target_detached=True, outcome_access=False, predicted_signals_used_for_risk=False,
            hard_event_pseudo_labels=False)

        def loss(prediction, _truth, _available):
            if not common.any():
                # Explicitly no observed forecast/target intersection: zero
                # supervision, hence zero gradient and rejection by the runner.
                return prediction.sum() * 0.
            virtual, virtual_mask = forecast_windows(values, observed, prediction, common,
                                                      self.center, self.scale)
            coordinates = self.coordinates(virtual, virtual_mask)
            return F.smooth_l1_loss(coordinates, target)

        return loss, receipt

    def assert_persistent_inactive(self):
        super().assert_persistent_inactive()
        self.teacher.assert_persistent_inactive()
