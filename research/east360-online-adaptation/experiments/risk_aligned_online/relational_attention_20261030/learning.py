"""Signal-only temporal-relation targets and first-head Q/K update helpers.

Reference fitting is limited to the exact 256-row CPU records supplied by the
caller. Window selection, shot eligibility, and model-input timing remain the
caller's responsibility; these functions neither read records nor use labels.
"""

import math
from numbers import Real

import torch

from ..io23 import RAW23


_COVARIATES = tuple(RAW23.index(name) for name in ('Bt', 'ip'))
_DIAGNOSTICS = tuple(RAW23.index(name) for name in
                     ('Mir_avg_amp', 'sxr_mean', 'xuv_ratio', 'ne_nG'))
_MEASURED = _COVARIATES + _DIAGNOSTICS


def _observations(values, masks, shape, *, cpu=False):
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.shape != shape or not isinstance(masks, torch.Tensor)
            or masks.dtype != torch.bool or masks.shape != shape
            or masks.device != values.device or (cpu and values.device.type != 'cpu')):
        raise ValueError(f'floating values and matching boolean masks required: {shape}')
    # Unavailable entries are deliberately ignored, including NaN/Inf. Every
    # entry advertised as available, including unused channels, must be finite.
    if bool((masks & ~torch.isfinite(values)).any()):
        raise ValueError('available observations must be finite')


def normalized(net, values, masks):
    """Apply the frozen base's source-asinh and availability transformation."""
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.ndim < 2 or values.shape[-1] != 23
            or not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
            or masks.shape != values.shape or masks.device != values.device):
        raise ValueError('floating values and matching boolean masks ending in 23 required')
    base = net.base
    valid = masks & torch.isfinite(values)
    clean = torch.where(valid, values, base.center)
    return torch.asinh((clean - base.center) / base.scale), valid


@torch.no_grad()
def fit_reference(shots):
    """Fit shot-weighted ridge and residual differences from supplied prefixes.

    Each item must be exactly (256,23) on CPU. Beta is (3,4), scale is the
    four-element diagonal of D, and support includes per-shot joint-valid and
    consecutive-patch counts plus the raw weighted-design rank. The returned
    tensors are detached float64 CPU tensors; the caller controls which shots
    are permitted and when the resulting reference is frozen.
    """
    if not isinstance(shots, (list, tuple)) or not shots:
        raise ValueError('at least one supplied 256-row reference shot is required')
    designs, responses, prepared = [], [], []
    joint_counts = []
    for item in shots:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('each shot must contain normalized values and masks')
        z, mask = item
        _observations(z, mask, (256, 23), cpu=True)
        joint = mask[:, _MEASURED].all(dim=-1)
        count = int(joint.sum())
        if count < 32:
            raise ValueError('each reference shot needs at least 32 joint-valid rows')
        observed = z[joint][:, _MEASURED].to(torch.float64)
        design = torch.cat((torch.ones((count, 1), dtype=torch.float64), observed[:, :2]), dim=1)
        designs.append(design)
        responses.append(observed[:, 2:])
        prepared.append((z, joint))
        joint_counts.append(count)

    weighted = torch.cat([x / math.sqrt(len(x)) for x in designs], dim=0)
    if not torch.isfinite(weighted).all():
        raise ValueError('nonfinite weighted design')
    rank = int(torch.linalg.matrix_rank(weighted))
    if rank < 3:
        raise ValueError('raw weighted reference design must have rank 3')
    gram = torch.stack([x.T @ x / len(x) for x in designs]).mean(dim=0)
    rhs = torch.stack([x.T @ y / len(x) for x, y in zip(designs, responses)]).mean(dim=0)
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise ValueError('nonfinite reference normal equations')
    penalty = torch.diag(gram.new_tensor([0., 1., 1.]))
    beta = torch.linalg.solve(gram + penalty, rhs)
    if not torch.isfinite(beta).all():
        raise ValueError('nonfinite reference regression')

    variances, patch_counts, difference_counts = [], [], []
    for z, joint in prepared:
        selected = torch.where(joint[:, None], z[:, _MEASURED], 0.).to(torch.float64)
        design = torch.cat((torch.ones((256, 1), dtype=torch.float64), selected[:, :2]), dim=1)
        residual = torch.where(joint[:, None], selected[:, 2:] - design @ beta, 0.)
        counts = joint.reshape(32, 8).sum(dim=1)
        patch_valid = counts >= 6
        means = residual.reshape(32, 8, 4).sum(dim=1) / counts.clamp_min(1)[:, None]
        pairs = patch_valid[1:] & patch_valid[:-1]
        difference_count = int(pairs.sum())
        if difference_count == 0:
            raise ValueError('each reference shot needs a valid consecutive-patch difference')
        differences = means[1:][pairs] - means[:-1][pairs]
        variance = differences.square().mean(dim=0)
        if not torch.isfinite(variance).all():
            raise ValueError('nonfinite reference residual scale')
        variances.append(variance)
        patch_counts.append(int(patch_valid.sum()))
        difference_counts.append(difference_count)
    scale = torch.stack(variances).mean(dim=0).clamp_min(1e-4)
    if not torch.isfinite(scale).all():
        raise ValueError('nonfinite reference scale')
    return {
        'beta': beta,
        'scale': scale,
        'support': {
            'shot_count': len(shots),
            'joint_valid_counts': joint_counts,
            'valid_patch_counts': patch_counts,
            'valid_difference_counts': difference_counts,
            'design_rank': rank,
        },
    }


def _reference(ref, device):
    if not isinstance(ref, dict):
        raise ValueError('reference must contain beta and scale tensors')
    beta, scale = ref.get('beta'), ref.get('scale')
    if (not isinstance(beta, torch.Tensor) or not beta.is_floating_point() or beta.shape != (3, 4)
            or not isinstance(scale, torch.Tensor) or not scale.is_floating_point() or scale.shape != (4,)
            or not torch.isfinite(beta).all() or not torch.isfinite(scale).all()
            or bool((scale <= 0).any())):
        raise ValueError('reference requires finite beta and positive finite diagonal scale')
    return beta.to(device=device, dtype=torch.float64), scale.to(device=device, dtype=torch.float64)


@torch.no_grad()
def relation_target(z, m, ref, mode='real'):
    """Return detached (target, eligible edges, valid rows) for arrived windows.

    In reverse mode only the last 25 normalized rows and their masks are
    reversed for target construction. This function does not alter model input.
    """
    if not isinstance(z, torch.Tensor) or z.ndim != 3:
        raise ValueError('normalized windows must have shape (B,128,23)')
    _observations(z, m, (len(z), 128, 23))
    if mode not in ('real', 'reverse', 'unconditioned'):
        raise ValueError('mode must be real, reverse, or unconditioned')
    beta, scale = _reference(ref, z.device)
    if mode == 'unconditioned':
        beta = torch.zeros_like(beta)
    elif mode == 'reverse':
        z = torch.cat((z[:, :-25], z[:, -25:].flip(dims=(1,))), dim=1)
        m = torch.cat((m[:, :-25], m[:, -25:].flip(dims=(1,))), dim=1)
    joint = m[:, :, _MEASURED].all(dim=-1)
    selected = torch.where(joint[..., None], z[:, :, _MEASURED], 0.).to(torch.float64)
    design = torch.cat((torch.ones_like(selected[..., :1]), selected[..., :2]), dim=-1)
    residual = torch.where(joint[..., None], selected[..., 2:] - design @ beta, 0.)
    counts = joint.reshape(len(z), 16, 8).sum(dim=-1)
    patch_valid = counts >= 6
    means = residual.reshape(len(z), 16, 8, 4).sum(dim=2) / counts.clamp_min(1)[..., None]
    differences = torch.cat((torch.zeros_like(means[:, :1]), means[:, 1:] - means[:, :-1]), dim=1)
    diff_valid = torch.cat((torch.zeros_like(patch_valid[:, :1]),
                            patch_valid[:, 1:] & patch_valid[:, :-1]), dim=1)
    differences = torch.where(diff_valid[..., None], differences, 0.)
    if not torch.isfinite(differences).all():
        raise ValueError('nonfinite conditional patch differences')
    scores = torch.einsum('bqc,bkc,c->bqk', differences[:, 12:16], differences, scale.reciprocal())
    if not torch.isfinite(scores).all():
        raise ValueError('nonfinite temporal-relation scores')
    keys = torch.arange(16, device=z.device)
    queries = torch.arange(12, 16, device=z.device)
    chronological = (keys[None, :] >= 1) & (keys[None, :] <= queries[:, None] - 2)
    edges = diff_valid[:, 12:16, None] & diff_valid[:, None, :] & chronological
    row_valid = edges.sum(dim=-1) >= 4
    edges = edges & row_valid[..., None]
    logits = scores.clamp(-4., 4.).masked_fill(~edges, -torch.inf)
    logits = torch.where(row_valid[..., None], logits, 0.)
    target = torch.where(edges, torch.softmax(logits, dim=-1), 0.).to(z.dtype)
    return target, edges, row_valid


@torch.no_grad()
def restrict_target(target, commonmask):
    """Intersect and renormalize, returning the restricted target tensor.

    Rows with fewer than four common keys or no remaining target mass are
    entirely zero, including when the caller supplies an all-zero target. The
    caller must clear those rows in the mask passed to attention_kl as well.
    """
    if (not isinstance(target, torch.Tensor) or not target.is_floating_point()
            or target.ndim != 3 or target.shape[1:] != (4, 16)
            or not isinstance(commonmask, torch.Tensor) or commonmask.dtype != torch.bool
            or commonmask.shape != target.shape or commonmask.device != target.device
            or not torch.isfinite(target).all() or bool((target < 0).any())):
        raise ValueError('finite nonnegative targets and matching boolean common masks required')
    masked = torch.where(commonmask, target, 0.)
    mass = masked.sum(dim=-1, keepdim=True)
    if not torch.isfinite(mass).all():
        raise ValueError('nonfinite common target mass')
    valid = (commonmask.sum(dim=-1) >= 4) & (mass.squeeze(-1) > 0)
    effective = commonmask & valid[..., None]
    denominator = torch.where(valid[..., None], mass, 1.)
    restricted = torch.where(effective, masked / denominator, 0.)
    return restricted


def attention_kl(packed_query, packed_key, target, edge_mask):
    """Per-example KL on the exact auxiliary edge support of attention head 0."""
    if not isinstance(packed_query, torch.Tensor) or packed_query.ndim != 3:
        raise ValueError('packed query and key must have shape (B,17,64)')
    batch = len(packed_query)
    if (not packed_query.is_floating_point() or packed_query.shape != (batch, 17, 64)
            or not isinstance(packed_key, torch.Tensor) or packed_key.shape != packed_query.shape
            or packed_key.dtype != packed_query.dtype or packed_key.device != packed_query.device
            or not torch.isfinite(packed_query).all() or not torch.isfinite(packed_key).all()
            or not isinstance(target, torch.Tensor) or not target.is_floating_point()
            or target.shape != (batch, 4, 16) or target.device != packed_query.device
            or not isinstance(edge_mask, torch.Tensor) or edge_mask.dtype != torch.bool
            or edge_mask.shape != target.shape or edge_mask.device != packed_query.device):
        raise ValueError('finite matching Q/K, targets, and boolean edge masks required')
    p = target.detach()
    if (not torch.isfinite(p).all() or bool((p < 0).any())
            or bool((p.masked_select(~edge_mask) != 0).any())):
        raise ValueError('target must be finite, nonnegative, and zero outside its support')
    row_valid = edge_mask.any(dim=-1)
    mass = p.sum(dim=-1)
    tolerance = max(1e-6, 8 * torch.finfo(p.dtype).eps)
    if not torch.allclose(mass, row_valid.to(mass.dtype), rtol=tolerance, atol=tolerance):
        raise ValueError('target must sum to one on supported rows and zero otherwise')
    p = p.to(dtype=packed_query.dtype)
    q = packed_query[:, 12:16, :16]
    k = packed_key[:, :16, :16]
    logits = q @ k.transpose(1, 2) / 4.
    if not torch.isfinite(logits.masked_select(edge_mask)).all():
        raise ValueError('nonfinite supported attention logits')
    logits = logits.masked_fill(~edge_mask, -torch.inf)
    logits = torch.where(row_valid[..., None], logits, 0.)
    log_attention = torch.log_softmax(logits, dim=-1)
    log_attention = torch.where(edge_mask, log_attention, 0.)
    log_target = p.clamp_min(torch.finfo(p.dtype).tiny).log()
    row_loss = (p * (log_target - log_attention)).sum(dim=-1)
    return row_loss.sum(dim=-1) / row_valid.sum(dim=-1).clamp_min(1)


def bounded_step(g, radius=.05):
    """Normalize the head-0 rows of each Q/K gradient to a fresh bounded step."""
    if (not isinstance(g, torch.Tensor) or not g.is_floating_point() or g.ndim != 2
            or g.shape[1] != 8192 or not torch.isfinite(g).all()):
        raise ValueError('finite floating Q/K gradients with shape (B,8192) required')
    if (isinstance(radius, bool) or not isinstance(radius, Real)
            or not math.isfinite(radius) or radius < 0):
        raise ValueError('radius must be a finite nonnegative scalar')
    head = torch.arange(64, device=g.device) < 16
    selected = torch.where(head[None, None, :, None], g.reshape(len(g), 2, 64, 64), 0.)
    selected = selected.reshape(len(g), 8192).to(torch.float64)
    peak = selected.abs().amax(dim=1, keepdim=True)
    scaled = selected / torch.where(peak > 0, peak, 1.)
    norm = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
    direction = scaled / torch.where(norm > 0, norm, 1.)
    step = (-radius * direction).to(g.dtype)
    if not torch.isfinite(step).all():
        raise ValueError('nonfinite bounded step')
    return step
