"""Local parameter-trajectory diagnostics, not a deployable update policy."""
import math

import torch


def parameter_group(name):
    if name in ('base.cls','base.position'): return 'position_cls'
    if name.startswith('base.embedding.'): return 'embedding'
    stem='base.encoder.layers.0.'
    if name.startswith(stem+'self_attn.'): return 'attention'
    if name.startswith((stem+'linear1.',stem+'linear2.')): return 'ffn'
    if name.startswith((stem+'norm1.',stem+'norm2.')): return 'normalization'
    raise ValueError(f'not a first-encoder parameter: {name}')


def separation(early,late,g_early,g_late,*,learning_rate=.001,clip=1.):
    if (early.ndim!=1 or not all(v.shape==early.shape for v in (late,g_early,g_late))
            or not all(torch.isfinite(v).all() for v in (early,late,g_early,g_late))
            or not all(math.isfinite(v) and v>0 for v in (learning_rate,clip))):
        raise ValueError('finite matching vectors and positive step/clip required')
    a,b,ga,gb=[v.detach().double() for v in (early,late,g_early,g_late)]
    d=a-b;sa=min(1.,clip/max(float(ga.norm()),1e-300));sb=min(1.,clip/max(float(gb.norm()),1e-300))
    diff=learning_rate*(sa*ga-sb*gb)
    before=float(d.square().sum());after=float((d-diff).square().sum())
    linear=float(-2*d.dot(diff));quadratic=float(diff.square().sum())
    return dict(distance_before=math.sqrt(before),distance_after_squared=after,
        distance_ratio=math.sqrt(after/before) if before>0 else None,
        secant_curvature=float(d.dot(ga-gb)/before) if before>0 else None,
        scale_early=sa,scale_late=sb,gradient_norm_early=float(ga.norm()),gradient_norm_late=float(gb.norm()),
        linear_distance_squared_change=linear,quadratic_distance_squared_change=quadratic,
        identity_error=abs(after-(before+linear+quadratic)))


def channel_separation(d,early,late,scale_early,scale_late,learning_rate):
    if d.ndim!=1 or early.ndim!=2 or late.shape!=early.shape or early.shape[1]!=len(d):
        raise ValueError('channel matrices and matching parameter difference required')
    d= d.detach().double();early=early.detach().double();late=late.detach().double()
    return [dict(early_gradient_norm=float(a.norm()),late_gradient_norm=float(b.norm()),
        linear_distance_squared_change=float(-2*learning_rate*d.dot(scale_early*a-scale_late*b)))
        for a,b in zip(early,late)]
