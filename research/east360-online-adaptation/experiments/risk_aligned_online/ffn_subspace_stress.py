"""SOURCE numerical-convention stress, not a physical disruption simulator.

Use the already documented source stress range from affine_signal_meta: 30%
identity, otherwise gain exp(U[-.35,.35]) and offset U[-1,1] in source-asinh
coordinates. No target statistics determine these bounds. One convention
applies to ALL past windows, matured truths and later anchors in a chain/shot.
Labels are copied because this hypothesis changes the recorded convention,
not the underlying plasma trajectory. It cannot certify real unit equivalence.
"""
import numpy as np
import torch


def sample_convention(rng):
    clean = bool(rng.random() < .3)
    gain = np.ones(23) if clean else np.exp(rng.uniform(-.35, .35, 23))
    bias = np.zeros(23) if clean else rng.uniform(-1., 1., 23)
    return dict(gain=gain.tolist(), bias=bias.tolist(), clean=clean,
                scope='SOURCE_NUMERICAL_STRESS_NOT_PHYSICS')


def map_values(model, values, available, convention):
    if (values.ndim != 3 or values.shape[-1] != 23 or available.shape != values.shape
            or available.dtype != torch.bool):
        raise ValueError('aligned 23-channel windows and boolean availability required')
    gain = torch.as_tensor(convention['gain'], dtype=torch.float64, device=values.device)
    bias = torch.as_tensor(convention['bias'], dtype=torch.float64, device=values.device)
    if (gain.shape != (23,) or bias.shape != (23,) or not torch.isfinite(gain).all()
            or not torch.isfinite(bias).all() or not (gain > 0).all()):
        raise ValueError('finite positive gains and finite offsets for 23 channels required')
    center = model.center.detach().to(device=values.device, dtype=torch.float64)
    scale = model.scale.detach().to(device=values.device, dtype=torch.float64)
    raw = values.double()
    valid = available & torch.isfinite(values)
    level = torch.asinh((torch.where(valid, raw, center) - center) / scale)
    mapped = raw + scale * ((level * gain + bias).sinh() - level.sinh())
    result = torch.where(valid, mapped.to(values.dtype), values).detach()
    if not torch.isfinite(result[valid]).all():
        raise FloatingPointError('nonfinite synthetic SOURCE convention')
    return result


def distort_chain(model, chain, rng):
    if chain.get('device') != 'J-TEXT':
        raise ValueError('numerical stress is authorized only for SOURCE development')
    convention = sample_convention(rng)
    moved = dict(chain); moved['steps'] = []
    for step in chain['steps']:
        e = dict(step)
        for name, mask in [('x', 'a'), ('qx', 'qa'), ('truth', 'mask'),
                           ('segment_x', 'segment_a')]:
            e[name] = map_values(model, step[name], step[mask], convention)
        moved['steps'].append(e)
    return moved, convention


def distort_source_record(model, shot, rng):
    if shot.get('device') != 'J-TEXT':
        raise ValueError('numerical stress is authorized only for SOURCE development')
    convention = sample_convention(rng)
    values = torch.from_numpy(shot['values'])[None]
    available = torch.from_numpy(shot['available'])[None]
    moved = dict(shot, values=map_values(model, values, available, convention)[0].numpy())
    return moved, convention
