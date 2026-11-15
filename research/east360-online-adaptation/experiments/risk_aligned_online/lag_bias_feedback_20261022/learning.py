"""Test strictly older forecast bias as a nuisance in subsequent feedback.

This only changes a temporary encoder's auxiliary loss. Neither inputs to
the risk network nor its risk head are bias-corrected. A sustained physical
change can also contribute to older residuals: this is a testable local
stationarity hypothesis, not a physical calibration guarantee.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..deepcoral_prefix_20261022.learning import forecast_examples


def _check(prediction, truth, mask):
    if (prediction.shape != (4, 25, 23) or truth.shape != prediction.shape
            or mask.shape != prediction.shape or mask.dtype != torch.bool):
        raise ValueError('four chronological H25x23 segments and boolean masks required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(truth[mask]).all():
        raise FloatingPointError('nonfinite observed residual')


def split_residuals(prediction, truth, mask):
    """Detached per-channel residual mean from OLD segments only."""
    _check(prediction, truth, mask)
    count = mask[:2].sum((0, 1))
    valid = count > 0
    residual = (prediction[:2].detach() - truth[:2].detach()).masked_fill(~mask[:2], 0.)
    # Zero is a placeholder only for explicitly INELIGIBLE channels; they do
    # not enter either arm's gradient-target denominator or training loss.
    bias = residual.sum((0, 1)) / count.clamp_min(1)
    return bias.detach(), valid


def feedback_loss(prediction, truth, mask, bias, valid, *, correct):
    _check(prediction, truth, mask)
    if (bias.shape != (23,) or valid.shape != (23,) or valid.dtype != torch.bool
            or bias.requires_grad or type(correct) is not bool):
        raise ValueError('detached channel bias, eligibility and explicit arm required')
    selected = mask[2:] & valid[None, None, :]
    if not selected.any():
        raise ValueError('no_shared_observed_targets')
    residual = prediction[2:] - truth[2:].detach()
    if correct:
        residual = residual - bias
    # Select BEFORE computing SmoothL1 so missing values cannot cause NaN
    # gradients even though their loss would subsequently be masked out.
    residual = residual[selected]
    return F.smooth_l1_loss(residual, torch.zeros_like(residual))


def adapt_arrived(model, signal, observed_count, fast, *, device, correct,
                  learning_rate=1e-3):
    if (model.cfg.window != 128 or model.cfg.horizon != 25
            or type(observed_count) is not int or observed_count < 256
            or observed_count % 128 or type(correct) is not bool
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('fixed W128/H25 schedule and positive finite step required')
    x, m, target, tm, origins = forecast_examples(signal, observed_count, model.cfg, device)
    order = np.argsort(origins)
    origins = [origins[i] for i in order]
    x, m, target, tm = [a[order.tolist()] for a in (x, m, target, tm)]
    if len(origins) != 4 or origins[0] < observed_count - 128:
        raise ValueError('all four origins must share the unchanged-fast block')
    tm = tm & torch.isfinite(target)
    clean = torch.where(tm, target.detach(), model.base.center)
    truth = torch.asinh((clean - model.base.center) / model.base.scale)
    prediction = model.predict_fast(x, m, fast)['forecast']
    bias, valid = split_residuals(prediction, truth, tm)
    eligible = tm[2:] & valid[None, None, :]
    count = int(eligible.sum())
    receipt = dict(last_observation=observed_count-1,
        first_affected_prediction=observed_count,
        calibration_origins=origins[:2], gradient_origins=origins[2:],
        calibration_last_target=origins[1]+25, first_gradient_target=origins[2]+1,
        observed_old_targets=int(tm[:2].sum()), observed_gradient_targets=count,
        eligible_channels=int(valid.sum()), gradient_steps=0, parameter_delta_norm=0.)
    if not count:
        receipt['reason'] = 'no_shared_observed_targets'
        return fast, receipt
    loss = feedback_loss(prediction, truth, tm, bias, valid, correct=correct)
    gradients = torch.autograd.grad(loss, tuple(fast.values()), allow_unused=False)
    norm = sum(g.square().sum() for g in gradients).sqrt()
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise FloatingPointError('nonfinite temporal-bias feedback')
    factor = (1. / norm.clamp_min(1.)).detach()
    updated = {k:(v-learning_rate*factor*g).detach().requires_grad_(True)
               for (k,v),g in zip(fast.items(), gradients)}
    delta = sum((updated[k].detach()-v.detach()).double().square().sum()
                for k,v in fast.items()).sqrt()
    with torch.no_grad():
        # Preserve the PRE-UPDATE calibration: re-estimation would change the
        # objective and turn the before/after comparison into a moving target.
        after = feedback_loss(model.predict_fast(x,m,updated)['forecast'],
                              truth,tm,bias,valid,correct=correct)
        raw = (prediction[2:].detach()-truth[2:])[eligible]
        remaining = (prediction[2:].detach()-truth[2:]-bias)[eligible]
        raw_loss = feedback_loss(prediction,truth,tm,bias,valid,correct=False)
        corrected_loss = feedback_loss(prediction,truth,tm,bias,valid,correct=True)
    receipt.update(gradient_steps=1, gradient_norm=float(norm),
        parameter_delta_norm=float(delta), loss_before=float(loss.detach()), loss_after=float(after),
        old_bias_mean_square=float(bias[valid].square().mean()),
        recent_raw_mean_square=float(raw.square().mean()),
        recent_corrected_mean_square=float(remaining.square().mean()),
        recent_raw_loss=float(raw_loss), recent_corrected_loss=float(corrected_loss))
    return updated, receipt
