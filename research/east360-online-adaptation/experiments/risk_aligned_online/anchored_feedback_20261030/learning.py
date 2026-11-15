"""Mean-anchored feedback targets and deterministic fit-shot donors."""

import torch

from ..context_vector_20261030.learning import ContextTarget, _floating


def make_reference_context(context, train_mean):
    """Keep each detached origin and replace feedback with the fit-only mean."""
    if not isinstance(context, torch.Tensor) or context.ndim != 2:
        raise ValueError('context must have shape (B,358)')
    if (not isinstance(train_mean, torch.Tensor) or train_mean.device != context.device
            or train_mean.dtype != context.dtype):
        raise ValueError('context and train_mean must share floating dtype and device')
    _floating(context, (len(context), 358), 'context')
    _floating(train_mean, (358,), 'train_mean')
    return torch.cat((context.detach()[:, :128],
                      train_mean.detach()[None, 128:].expand(len(context), -1)), dim=1)


class AnchoredTarget(ContextTarget):
    """Use the unchanged ContextTarget parameters for both sides of the anchor."""

    def forward(self, context, support, train_mean):
        weight = self.network[0].weight
        if isinstance(context, torch.Tensor) and (context.device != weight.device
                                                  or context.dtype != weight.dtype):
            raise ValueError('context and target must share floating dtype and device')
        reference = make_reference_context(context, train_mean)
        if (not isinstance(support, torch.Tensor) or support.dtype != torch.bool
                or support.shape != (len(context),) or support.device != context.device):
            raise ValueError('support must be a same-device boolean tensor of shape (B,)')
        delta = .5 * (self.network(context.detach()) - self.network(reference))
        if not torch.isfinite(delta).all():
            raise FloatingPointError('anchored target must be finite')
        return torch.where(support[:, None], delta, torch.zeros_like(delta))


def nearest_donor_indices(contexts, shot_ids, fit_ids):
    """Choose another fit shot by origin MSE, breaking ties by source row index.

    Origins are already scaled by the caller. Feedback coordinates, labels,
    and clocks do not enter the distance or the candidate pool.
    """
    if (not isinstance(contexts, torch.Tensor) or contexts.ndim != 2
            or contexts.device.type != 'cpu'):
        raise ValueError('contexts must be a CPU floating tensor of shape (B,358)')
    _floating(contexts, (len(contexts), 358), 'contexts')
    shot_ids = shot_ids.tolist() if isinstance(shot_ids, torch.Tensor) else list(shot_ids)
    fit_ids = fit_ids.tolist() if isinstance(fit_ids, torch.Tensor) else list(fit_ids)
    if len(shot_ids) != len(contexts):
        raise ValueError('one shot ID per context row required')
    fit = set(fit_ids)
    pool = [index for index, shot in enumerate(shot_ids) if shot in fit]
    origins = contexts.detach()[:, :128].double()
    donors = []
    for row, shot in enumerate(shot_ids):
        candidates = [index for index in pool if shot_ids[index] != shot]
        if not candidates:
            raise ValueError(f'no different fit-shot donor candidate for row {row}')
        distances = (origins[candidates] - origins[row]).square().mean(dim=1)
        if not torch.isfinite(distances).all():
            raise ValueError('donor distances must be finite')
        donors.append(candidates[int(distances.argmin())])
    return torch.tensor(donors, dtype=torch.int64)
