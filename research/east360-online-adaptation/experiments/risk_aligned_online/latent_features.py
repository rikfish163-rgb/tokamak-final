"""Read the qualified encoder's causal risk features without changing it."""
import numpy as np
import torch

from . import simple_data


def extract_shot(model, shot, device, *, batch_size=256):
    """Use only signals; each row sees its own past window, not the full shot.

    The pre-hook reads the actual risk-head input rather than duplicating the
    encoder's normalization, padding or learned source-origin semantics. It is
    scoped to this call and removed even if inference fails.
    """
    if batch_size < 1 or len(shot['values']) < 1:
        raise ValueError('positive batch and nonempty recorded signal required')
    model.eval()
    features, logits = [], []
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0].detach())

    handle = model.risk_head.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for start in range(0, len(shot['values']), batch_size):
                windows = [simple_data.window(shot, end, model.cfg.window) for end in
                           range(start, min(start+batch_size, len(shot['values'])))]
                values, mask = [torch.as_tensor(np.stack(v), device=device) for v in zip(*windows)]
                captured.clear()
                out = model(values, mask)
                if len(captured) != 1 or captured[0].ndim != 2:
                    raise ValueError('one flat risk representation per endpoint required')
                features.append(captured[0].cpu().numpy())
                logits.append(out['logit'].detach().cpu().numpy())
    finally:
        handle.remove()
    hidden, risk = np.concatenate(features), np.concatenate(logits)
    if not np.isfinite(hidden).all() or not np.isfinite(risk).all():
        raise FloatingPointError('nonfinite qualified encoder output')
    return hidden, risk
