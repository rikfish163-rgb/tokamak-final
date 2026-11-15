"""Arrived-signal gradients within historical encoder correction directions.

No final-logit adapter, no current-shot labels, and no persistent fast state.
History defines the permitted parameter span, NOT the sign of today's update.
"""
import numpy as np
import torch

from .. import simple_data
from ..deepcoral_prefix_20261022.learning import forecast_examples
from ..history_tangent_20261023.learning import flat, apply_step, logit
from ..increment_forecast_20261023.learning import target_mask
from ..paired_feedback_20261022.model import initial_encoder

ARMS = ('raw', 'global', 'contrast', 'permuted', 'hold', 'reverse')


def history_spaces(gradients, shots, labels):
    if (gradients.ndim != 2 or len(gradients) != len(shots) or shots.shape != labels.shape
            or not torch.isfinite(gradients).all() or set(labels.tolist()) != {0, 1}):
        raise ValueError('finite historical gradients and both explicit shot classes required')
    ids = np.unique(shots); means = []; classes = []
    for sid in ids:
        rows = shots == sid
        if len(np.unique(labels[rows])) != 1:
            raise ValueError('mixed historical outcomes within a shot')
        means.append(gradients[rows].double().mean(0)); classes.append(int(labels[rows][0]))
    means = torch.stack(means); norms = means.norm(dim=1)
    if (norms == 0).any() or any(classes.count(c) < 2 for c in (0, 1)):
        raise ValueError('nonzero historical directions and >=2 shots per class required')
    means /= norms[:, None]
    centered = means.clone(); classes = np.array(classes)
    for c in (0, 1):
        centered[classes == c] -= centered[classes == c].mean(0)
    spaces = {}; spectra = {}
    for name, matrix in (('global', means), ('contrast', centered)):
        _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = int((s > s[0] * 1e-7).sum()) if float(s[0]) > 0 else 0
        if not rank:
            raise ValueError('historical directions do not identify the declared span')
        spaces[name] = vh[:rank].to(gradients).contiguous()
        spectra[name] = dict(rank=rank, singular_values=s.cpu().tolist())
    return spaces, dict(shots=ids.tolist(), labels=classes.tolist(), spectra=spectra,
                        shot_mean_gradient_norms=norms.cpu().tolist())


def projected_step(gradient, basis=None):
    if gradient.ndim != 1 or not torch.isfinite(gradient).all():
        raise ValueError('finite parameter gradient required')
    if basis is not None:
        if (basis.ndim != 2 or basis.shape[1] != len(gradient)
                or not torch.isfinite(basis).all() or basis.device != gradient.device
                or basis.dtype != gradient.dtype):
            raise ValueError('matching finite historical parameter basis required')
        direction = basis.T @ (basis @ gradient)
    else:
        direction = gradient
    norm = float(direction.double().norm())
    # Zero projection is a genuine lack of supported update direction.
    # Do not insert an unprojected gradient to manufacture adaptation.
    step = direction * (.01 / norm) if norm else torch.zeros_like(direction)
    return step, dict(raw_gradient_norm=float(gradient.double().norm()),
        projected_gradient_norm=norm, step_length=.01 if norm else 0.,
        first_order_signal_decrease=float((gradient.double() * step.double()).sum()))


def update_all(model, prefix, count, spaces, device='cuda'):
    if (set(prefix) != {'values', 'available'} or len(prefix['values']) != count
            or count < 256 or count % 128):
        raise ValueError('exact arrived signal prefix at a declared update time required')
    parent = initial_encoder(model)
    x, mask, truth, tm, ends = forecast_examples(prefix, count, model.cfg, device)
    mask_target = target_mask(x, mask, truth, tm)
    if not mask_target.any():
        return ({a: initial_encoder(model) for a in ARMS},
                {a: dict(gradient_steps=0, reason='no_observed_forecast_targets',
                         last_observation=count-1, first_affected_prediction=count) for a in ARMS})
    gradients = {}; losses = {}; targets = {}
    for name in ('real', 'hold', 'reverse'):
        target = truth if name == 'real' else (x[:, -1:].expand_as(truth) if name == 'hold' else truth.flip(1))
        valid = mask_target if name != 'reverse' else target_mask(x, mask, target, tm.flip(1))
        prediction = model.predict_fast(x, mask, parent)['forecast']
        loss = model.base.forecast_loss(prediction, target, valid)
        g = flat(torch.autograd.grad(loss, tuple(parent.values()))).detach()
        if not torch.isfinite(loss) or not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite observed feedback loss or gradient')
        gradients[name] = g; losses[name] = float(loss.detach()); targets[name] = (target, valid)
    states = {}; receipts = {}
    for arm in ARMS:
        mode = arm if arm in ('hold', 'reverse') else 'real'
        basis = None if arm == 'raw' else spaces[arm if arm in ('global', 'permuted') else 'contrast']
        vector, r = projected_step(gradients[mode], basis)
        states[arm] = apply_step(parent, vector)
        with torch.no_grad():
            prediction = model.predict_fast(x, mask, states[arm])['forecast']
            after = model.base.forecast_loss(prediction, *targets[mode])
            true_after = model.base.forecast_loss(prediction, truth, mask_target)
        actual = flat([states[arm][k].detach() - v.detach() for k, v in parent.items()])
        receipts[arm] = dict(r, gradient_steps=int(bool(vector.any())),
            parameter_delta_norm=float(actual.double().norm()), loss_before=losses[mode],
            loss_after=float(after), real_signal_before=losses['real'], real_signal_after=float(true_after),
            last_observation=count-1, first_affected_prediction=count, origins=ends,
            observed_targets=int(mask_target.sum()),
            basis=None if basis is None else (arm if arm in ('global', 'permuted') else 'contrast'))
    return states, receipts


def validate_spaces(spaces):
    if set(spaces) != {'global', 'contrast', 'permuted'}:
        raise ValueError('three declared historical parameter spaces required')
    for q in spaces.values():
        # Real 58,240-coordinate float32 rows have Gram accumulation errors
        # up to 1.06e-5, but cast-row float64 Gram error is only 8.35e-9.
        # Validate the stored rows in double; do not relax orthogonality.
        r = q.double()
        torch.testing.assert_close(r @ r.T, torch.eye(len(r), device=r.device, dtype=r.dtype), atol=2e-6, rtol=2e-6)


def predict(model, signal, spaces, device='cuda'):
    if (set(signal) != {'values', 'available'} or signal['values'].ndim != 2
            or signal['values'].shape[1] != 23 or signal['available'].shape != signal['values'].shape
            or signal['available'].dtype != bool or not len(signal['values'])):
        raise ValueError('nonempty signal-only flat23 input required')
    validate_spaces(spaces)
    states = {a: initial_encoder(model) for a in ('off',) + ARMS}
    curves = {a: [] for a in states}; receipts = {a: [] for a in ARMS}
    n = len(signal['values'])
    for lo in range(0, n, 128):
        hi = min(lo + 128, n)
        for start in range(lo, hi, 32):
            windows = [simple_data.window(signal, t, 128) for t in range(start, min(start+32, hi))]
            x, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
            with torch.no_grad():
                for a, state in states.items():
                    curves[a].append(logit(model, x, mask, state).sigmoid().cpu().numpy())
        if hi >= 256 and hi < n:
            changed, r = update_all(model, {k: v[:hi] for k, v in signal.items()}, hi, spaces, device)
            states.update(changed)
            for a in ARMS:
                receipts[a].append(r[a])
    result = {a: np.concatenate(v) for a, v in curves.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('nonfinite complete risk trajectory')
    return result, receipts
