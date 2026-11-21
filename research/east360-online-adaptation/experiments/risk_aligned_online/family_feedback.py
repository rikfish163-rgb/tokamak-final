"""Patch-dependent, diagnostic-family adaptation BEFORE the first Transformer.

The four families are candidate engineering labels, not independently proven
plasma mechanisms or a shared/private domain split. Every recorded channel is
retained. A single frozen Transformer and its original risk head combine them.
Only per-shot low-rank coefficients change online; zero coefficients reproduce
the qualified source exactly. Auxiliary forecasts directly output the whole
future window of 23 channels and never feed predictions back as measurements.
"""
import numpy as np
import torch
from torch import nn

from .io23 import RAW23

FAMILIES = {
    'operating_density': ('Bt', 'ip', 'v_loop', 'ne_nG', 'qa_proxy'),
    'magnetic': ('MNM', 'Mir_VV', 'Mir_avg_amp', 'Mir_avg_fre', 'mode_number_n', 'n=1 amplitude'),
    'radiation_impurity': ('CIII', 'Z_proxy', 'sxr_kurt', 'sxr_mean', 'sxr_skew', 'sxr_var',
                           'xuv_kurt', 'xuv_ratio', 'xuv_skew', 'xuv_var'),
    'position': ('dx_a', 'dy_a'),
}


class FamilyFeedback(nn.Module):
    def __init__(self, dim=64, *, patch=8, horizon=8, rank=4, grouping='physical'):
        super().__init__()
        if min(dim, patch, horizon, rank) < 1 or grouping not in ('physical', 'shuffled'):
            raise ValueError('positive dimensions and declared diagnostic grouping required')
        self.dim, self.patch, self.horizon, self.rank = dim, patch, horizon, rank
        groups = [tuple(RAW23.index(n) for n in names) for names in FAMILIES.values()]
        if grouping == 'shuffled':
            # Same group sizes, parameter budget and deterministic source-only
            # assignment. No outcome or EAST signal selects this control.
            shuffled = np.random.default_rng(131).permutation(23).tolist()
            groups = [tuple(shuffled[start:stop]) for start, stop in ((0, 5), (5, 11), (11, 21), (21, 23))]
        self.groups = tuple(groups)
        self.state_dim = rank*len(groups)
        self.down = nn.ModuleList(nn.Linear(patch*2*len(g), rank, bias=False) for g in groups)
        self.up = nn.ModuleList(nn.Linear(rank, dim, bias=False) for _ in groups)
        self.forecaster = nn.Sequential(nn.LayerNorm(3*dim), nn.Linear(3*dim, dim),
                                        nn.GELU(), nn.Linear(dim, horizon*23))
        self.controller = nn.Sequential(nn.LayerNorm(3*dim+self.state_dim),
                                        nn.Linear(3*dim+self.state_dim, dim), nn.Tanh(),
                                        nn.Linear(dim, self.state_dim))
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def initial_fast(self, batch):
        return self.forecaster[-1].bias.new_zeros(batch, self.state_dim)

    def forward(self, parent, values, available, fast=None, *, perturb=None):
        if (parent.cfg.dim != self.dim or parent.cfg.patch != self.patch or
                parent.cfg.fast_site != 'tokens' or parent.cfg.input_view != 'source'):
            raise ValueError('matching source-view token parent required')
        if values.ndim != 3 or values.shape[1:] != (parent.cfg.window, 23):
            raise ValueError('exact parent causal window required')
        if available.shape != values.shape or available.dtype != torch.bool:
            raise ValueError('matching boolean availability required')
        fast = self.initial_fast(len(values)) if fast is None else fast
        if fast.shape != (len(values), self.state_dim):
            raise ValueError('one family coefficient vector per shot required')
        if perturb is not None and perturb.shape != fast.shape:
            raise ValueError('declared coefficient perturbation must match state')
        available = available & torch.isfinite(values)
        clean = torch.where(available, values, parent.center)
        # Keep the parent's absolute source coordinates. This is NOT a new
        # cross-device density/unit calibration, or unconditional centering
        # that could remove a sustained precursor.
        level = torch.asinh((clean-parent.center)/parent.scale)
        batch, patches = len(values), parent.cfg.window//self.patch
        parts = []
        coefficients = fast if perturb is None else fast+perturb
        for i, (channels, down, up) in enumerate(zip(self.groups, self.down, self.up)):
            observed = available[..., channels]
            inp = torch.cat((level[..., channels], observed.to(level.dtype)), -1)
            latent = down(inp.reshape(batch, patches, -1)).tanh()
            state = coefficients[:, i*self.rank:(i+1)*self.rank].tanh()
            delta = up(latent*state[:, None])
            valid = observed.reshape(batch, patches, -1).any(-1)
            parts.append(delta*valid[:, :, None])
        family_delta = torch.stack(parts, 1)
        valid = available.reshape(batch, patches, -1).any(-1)
        captured = {}

        def insert(_module, _args, hidden):
            # Only input patches change; CLS, positions, parent normalization,
            # both attention blocks and the single risk head stay untouched.
            return hidden+family_delta.sum(1)

        def observe(_module, _args, hidden):
            signal = hidden[:, :-1]
            count = valid.sum(-1, keepdim=True).clamp_min(1)
            mean = signal.masked_fill(~valid[:, :, None], 0.).sum(1)/count
            last = signal[:, -1]-mean
            pairs = valid[:, 1:] & valid[:, :-1]
            change = (signal[:, 1:]-signal[:, :-1]).abs().masked_fill(~pairs[:, :, None], 0.)
            variation = change.sum(1)/pairs.sum(-1, keepdim=True).clamp_min(1)
            captured['context'] = torch.cat((mean, last, variation), -1)

        fresh = perturb is None and not fast.requires_grad and not bool(fast.detach().count_nonzero())
        # A block hook disables torch's fused encoder path and can change
        # rounding. Exact zero-state compatibility uses the real unhooked
        # parent logit, while the differentiable branch remains fully active.
        # Regression tests verify equality, cleanup and double backward.
        original = parent(values, available)['logit'] if fresh else None
        handles = [parent.embedding.register_forward_hook(insert),
                   parent.encoder.layers[0].register_forward_hook(observe)]
        try:
            out = parent(values, available)
        finally:
            for handle in handles:
                handle.remove()
        context = captured['context']
        return dict(logit=original if fresh else out['logit'], context=context,
                    trajectory=self.forecaster(context).reshape(batch, self.horizon, 23),
                    family_delta=family_delta)

    def update(self, current, gradient, context, *, learned=True):
        if current.shape != gradient.shape or context.shape != (len(current), 3*self.dim):
            raise ValueError('matched state, gradient and current context required')
        norm = gradient.norm(dim=-1, keepdim=True)
        unit = gradient/norm.clamp_min(1e-8)
        modulation = .5*self.controller(torch.cat((context, unit), -1)).tanh() if learned else 0.
        direction = -unit+modulation
        direction = direction/direction.norm(dim=-1, keepdim=True).clamp_min(1.)
        proposal = current+.5*(norm/.01).clamp(max=1.)*direction
        return proposal*(8./proposal.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)


def signal_target23(values, available, truth, mask):
    """All-channel observed future change; scale fixed at the issuing origin.

    No future moments, outcome labels, equilibrium equations or physical
    threshold assumptions. Missing/constant/short-prefix channels are excluded.
    Float64 statistics retain tiny qa_proxy variation and avoid cancellation;
    asinh is applied BEFORE float32 conversion to avoid extreme-ratio overflow.
    """
    if (values.ndim != 3 or values.shape[-1] != 23 or available.shape != values.shape or
            truth.ndim != 3 or truth.shape[-1] != 23 or mask.shape != truth.shape or
            available.dtype != torch.bool or mask.dtype != torch.bool):
        raise ValueError('past and matured future 23-channel windows with boolean masks required')
    data = values.detach().double()
    seen = available & torch.isfinite(data)
    count = seen.sum(1)
    mean = data.masked_fill(~seen, 0.).sum(1)/count.clamp_min(1)
    variance = (data-mean[:, None]).masked_fill(~seen, 0.).square().sum(1)/count.clamp_min(1)
    eligible = (count >= 8) & (variance > 0) & seen[:, -1]
    reference = data[:, -1].masked_fill(~eligible, 0.)
    scale = torch.where(eligible, variance.sqrt(), 1.)
    future = truth.detach().double()
    valid = mask & torch.isfinite(future) & eligible[:, None]
    future = torch.where(valid, future, reference[:, None])
    target = torch.asinh((future-reference[:, None])/scale[:, None]).to(values.dtype)
    if not torch.isfinite(target).all():
        raise FloatingPointError('nonfinite matured all-channel target')
    return target, valid
