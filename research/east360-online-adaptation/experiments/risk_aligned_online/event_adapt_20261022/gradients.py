"""Source-only task-gradient routing, with an explicit magnitude control.

Not a GradNorm reproduction: no learned task weights/training-rate estimator.
This simpler controlled intervention tests the observed ~300x risk/weighted-
forecast encoder-gradient disparity, without adding a new loss or target.
"""
import torch


def assign_task_gradients(model, losses, weights, *, normalize_shared):
    if len(losses) != 3 or len(weights) != 3 or any(w < 0 for w in weights):
        raise ValueError('risk/forecast/event losses and nonnegative weights required')
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    head_prefixes = ('risk_head.', 'trajectory_embedding.', 'forecast_head.', 'event_head.')
    shared = [i for i, (n, _) in enumerate(named) if not n.startswith(head_prefixes)]
    grads = [torch.autograd.grad(loss, parameters, retain_graph=i < 2, allow_unused=True)
             for i, loss in enumerate(losses)]
    norms = [torch.stack([g[i].square().sum() for i in shared if g[i] is not None]).sum().sqrt()
             if any(g[i] is not None for i in shared) else losses[0].detach().new_zeros(())
             for g in grads]
    if not all(torch.isfinite(n) for n in norms):
        raise FloatingPointError('nonfinite task gradient norm')
    for i, (name, parameter) in enumerate(named):
        if name.startswith('forecast_head.'):
            tasks = [1]  # physical diagnostic target only, never risk encoding
        elif name.startswith('event_head.'):
            tasks = [2]
        elif name.startswith(('risk_head.', 'trajectory_embedding.')):
            tasks = [0]
        else:
            tasks = [0, 1, 2]
        combined = None
        for task in tasks:
            g = grads[task][i]
            if g is None or weights[task] == 0:
                continue  # this declared task has no dependency on this parameter
            if normalize_shared and i in shared:
                # clamp protects the EXACT zero-gradient case; zero remains
                # zero. This does not create a gradient for absent event labels.
                g = g/norms[task].clamp_min(torch.finfo(g.dtype).tiny)
            term = weights[task]*g
            combined = term if combined is None else combined+term
        parameter.grad = combined
    return {f'{name}_shared_norm': float(n.detach())
            for name, n in zip(('risk', 'forecast', 'event'), norms)}
