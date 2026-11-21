"""Window-relative context bases; original backbone and risk head are retained."""

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def context_trace(memory, means, complete, *, mode="memory", reference_norm=None):
    """Same c8 write algebra, with prefix-stable fixed32 basis evaluation.

    The historical vectorized trace can differ by rounding when the number of
    future rows changes GEMM shapes. Padding independent basis rows to32 fixes
    arithmetic shape; padded rows never enter the recurrence. The legacy arm
    still uses the archived trace to reproduce its original prediction path.
    """
    if (
        mode not in ("memory", "reverse", "local")
        or means.ndim != 2
        or means.shape[1] != 23
        or complete.shape != means.shape
    ):
        raise ValueError("observed patches and declared c8 mode required")
    if mode != "memory" and (
        reference_norm is None or reference_norm.shape != means.shape[:1]
    ):
        raise ValueError("control needs actual context norm at each arrived patch")
    clean = torch.where(complete, means, 0.0)
    features = memory.key_features(clean, complete)
    parts = []
    for lo in range(0, len(means), 32):
        count = min(32, len(means) - lo)
        padded = features.new_zeros(32, 46)
        padded[:count] = features[lo : lo + count]
        k = torch.cat((memory.key(padded).tanh(), padded.new_ones(32, 1)), -1)
        parts.append(torch.einsum("nk,khfc->nhfc", k, memory.kernel)[:count])
    if not parts:
        return means.new_zeros(0, 8)
    basis = torch.cat(parts)
    c = means.new_zeros(8)
    contexts = []
    for j in range(len(means)):
        if j:
            count = min(j, 4)
            origins = torch.arange(j - 1, j - count - 1, -1, device=means.device)
            target = clean[j] - clean[origins]
            mask = complete[j] & complete[origins]
            if mask.any():
                if mode == "reverse":
                    target, mask = target.flip(0), mask & mask.flip(0)
                c = memory.write(
                    torch.zeros_like(c) if mode == "local" else c,
                    basis[origins, torch.arange(count, device=means.device)],
                    target,
                    mask,
                )
                if mode != "memory":
                    c = (
                        c
                        * reference_norm[j]
                        / c.norm().clamp_min(torch.finfo(c.dtype).tiny)
                    )
        contexts.append(c)
    return torch.stack(contexts)


def token_basis(memory, values, available, center, scale):
    if (
        values.ndim != 3
        or values.shape[1:] != (128, 23)
        or available.shape != values.shape
        or available.dtype != torch.bool
    ):
        raise ValueError("causal B,W128,23 windows and boolean mask required")
    mask = available & torch.isfinite(values)
    clean = torch.where(mask, values, center)
    level = torch.asinh((clean - center) / scale).reshape(len(values), 16, 8, 23)
    complete = mask.reshape(len(values), 16, 8, 23).all(2)
    means = torch.where(complete, level.mean(2), 0.0)
    features = memory.key_features(means, complete)
    key = torch.cat(
        (memory.key(features).tanh(), means.new_ones(len(values), 16, 1)), -1
    )
    basis = torch.einsum("btk,khfc->bthfc", key, memory.kernel).flatten(2, 3)
    supported = complete.any(-1)
    # A partial token may be valid for the classifier but has no full-patch
    # measured target/key support. It cannot create a bias-only adaptive write.
    return torch.where(supported[:, :, None, None], basis, 0.0), supported


def context_at(trace, ends):
    if trace.ndim != 2 or trace.shape[1] != 8:
        raise ValueError("c8 trace required")
    ends = torch.as_tensor(ends, dtype=torch.long, device=trace.device)
    if ends.ndim != 1 or (ends < 0).any():
        raise ValueError("nonnegative endpoint vector required")
    index = torch.div(ends, 8, rounding_mode="floor") - 1
    if not len(trace):
        if (index >= 0).any():
            raise ValueError("context trace shorter than requested prefix")
        return trace.new_zeros(len(ends), 8)
    if (index >= len(trace)).any():
        raise ValueError("context trace shorter than requested prefix")
    return torch.where((index >= 0)[:, None], trace[index.clamp_min(0)], 0.0)


def action(output_weight, basis, context, valid, mode):
    if (
        basis.ndim != 4
        or basis.shape[1:] != (16, 92, 8)
        or context.shape != (len(basis), 8)
        or output_weight.shape != (64, 92)
        or valid.shape != (len(basis), 17)
        or valid.dtype != torch.bool
        or mode not in ("per_token", "aligned_last", "tiled_norm", "zero")
    ):
        raise ValueError("fixed c8/token basis and explicit routing required")
    delta = torch.einsum("btdc,bc->btd", basis, context)
    full = F.linear(delta, output_weight) * valid[:, :-1, None]
    zero_direction = torch.zeros(len(basis), dtype=torch.bool, device=basis.device)
    if mode == "per_token":
        shift = full
    elif mode == "zero":
        shift = torch.zeros_like(full)
    elif mode == "aligned_last":
        shift = torch.cat((torch.zeros_like(full[:, :-1]), full[:, -1:]), 1)
    else:
        tile = full[:, -1:, :].expand_as(full) * valid[:, :-1, None]
        norm = tile.norm(dim=(1, 2))
        desired = full.norm(dim=(1, 2))
        zero_direction = (norm == 0) & (desired > 0)
        shift = (
            tile
            * (desired / norm.clamp_min(torch.finfo(norm.dtype).tiny))[:, None, None]
        )
    info = dict(
        norm=shift.norm(dim=(1, 2)),
        full_norm=full.norm(dim=(1, 2)),
        tiled_zero_direction=zero_direction,
    )
    return shift, info


def risk_with_shifts(adapter, h, valid, shift):
    if shift.shape != (len(h), 16, 64) or valid.shape != h.shape[:2]:
        raise ValueError("one shift per data token, without CLS write required")
    last = h[:, -2] + adapter.slow(h[:, -2]) * valid[:, -2, None]
    data = torch.cat((h[:, :-2], last[:, None]), 1)
    data = data + shift * valid[:, :-1, None]
    sequence = torch.cat((data, h[:, -1:]), 1)
    with sdpa_kernel(SDPBackend.MATH):
        sequence = adapter.base.encoder.layers[1](sequence, src_key_padding_mask=~valid)
    gain, bias = adapter.base.initial_fast(len(h)).chunk(2, -1)
    sequence = (
        adapter.base.norm(sequence) * (1 + 0.1 * gain[:, None].tanh())
        + 0.1 * bias[:, None].tanh()
    )
    return adapter.base.risk_head(sequence[:, -1]).squeeze(-1)
