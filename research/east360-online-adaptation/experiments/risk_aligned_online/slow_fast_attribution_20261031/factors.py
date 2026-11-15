"""Numeric intervention on an existing lifecycle risk path.

This is an attribution helper, not a new deployed model or adaptation rule.
"""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def logits_with_components(model, h, valid, delta, slow_weight, output_weight):
    """Return risk logits with explicitly supplied slow and readout matrices.

    ``h`` is the output of the original first encoder block, including CLS.
    The supplied matrices replace only the two linear projections; the frozen
    source's second block, normalization, initial fast state and head are used
    unchanged. All modules must already be in evaluation mode. Autograd may
    flow to the supplied tensors, but no arguments or model state are changed.
    """
    cfg = model.base.cfg
    if (cfg.layers != 2 or len(model.base.encoder.layers) != 2
            or cfg.fast_site != 'tokens' or cfg.input_view != 'source'
            or model.readout != 'full'):
        raise ValueError('two-block token/source model with full readout required')
    if any(module.training for module in model.modules()):
        raise ValueError('all model modules must be in evaluation mode')

    tensors = (h, valid, delta, slow_weight, output_weight)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError('tensor inputs and explicit weight matrices required')
    if (h.ndim != 3 or h.shape[0] == 0 or h.shape[1:] != (17, cfg.dim)
            or valid.shape != h.shape[:2] or valid.dtype != torch.bool
            or delta.shape != (h.shape[0], 92)
            or slow_weight.shape != (cfg.dim, cfg.dim)
            or output_weight.shape != (cfg.dim, 92)):
        raise ValueError('matching 17-token inputs and component matrices required')
    modules = (model.base.encoder.layers[1], model.base.norm, model.base.risk_head)
    if (any(tensor.device != h.device for tensor in tensors)
            or model.base.fast_origin.device != h.device
            or any(tensor.device != h.device for module in modules for tensor in module.parameters())
            or any(tensor.device != h.device for module in modules for tensor in module.buffers())):
        raise ValueError('inputs and relevant model tensors must share a device')
    if (not h.is_floating_point()
            or any(tensor.dtype != h.dtype for tensor in (delta, slow_weight, output_weight))):
        raise ValueError('matching floating-point inputs and component matrices required')
    if not torch.isfinite(slow_weight).all() or not torch.isfinite(output_weight).all():
        raise ValueError('finite component matrices required')

    # Preserve LifecycleAdapter.risk's slow-then-fast ordering and the clone
    # used by RecurrentAdapter.risk before updating the last observed token.
    last = h[:, -2] + F.linear(h[:, -2], slow_weight) * valid[:, -2, None]
    adapted = torch.cat((h[:, :-2], last[:, None], h[:, -1:]), dim=1)
    adapted = adapted.clone()
    adapted[:, -2] = adapted[:, -2] + F.linear(delta, output_weight) * valid[:, -2, None]
    with sdpa_kernel(SDPBackend.MATH):
        adapted = model.base.encoder.layers[1](adapted, src_key_padding_mask=~valid)
    gain, bias = model.base.initial_fast(len(adapted)).chunk(2, -1)
    adapted = model.base.norm(adapted) * (1 + .1 * gain[:, None].tanh()) + .1 * bias[:, None].tanh()
    return model.base.risk_head(adapted[:, -1]).squeeze(-1)
