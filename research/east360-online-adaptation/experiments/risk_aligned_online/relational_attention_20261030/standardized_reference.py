"""Covariate-standardized ridge variant of the frozen relation reference.

Only reference fitting changes. Returned beta is in the original normalized
signal coordinates, and scale is the same residual-increment diagonal D used
by learning.relation_target. Numerical scale invariance does not establish a
physical nuisance/hazard decomposition.
"""

import math

import torch

from .learning import _MEASURED, _observations


# Absolute threshold in the supplied normalized covariate coordinates. Affine
# unit invariance applies when both original and transformed columns pass it.
MIN_COVARIATE_STD = 1e-12


@torch.no_grad()
def fit_reference_standardized(shots):
    """Fit only the supplied, exact 256-row CPU prefixes with equal shot weights.

    Each valid row has weight 1/(shot_count * shot_valid_count). Both covariate
    moments and ridge use these weights; variance includes between-shot mean
    differences. Ridge penalizes the two standardized slopes with coefficient
    one and leaves the intercept unpenalized. Rank is checked after scaling.

    All returned tensors are detached float64 CPU tensors. Extra fields are
    covariate_center, covariate_scale, regression_standardized_beta, and
    condition_eigenvalues: ascending eigenvalue lists for raw_covariance,
    standardized_covariance, standardized_design, and regularized_design.
    The caller chooses permitted shots; this helper performs no data reads.
    """
    if not isinstance(shots, (list, tuple)) or not shots:
        raise ValueError('at least one supplied 256-row reference shot is required')
    observed_shots, prepared, joint_counts = [], [], []
    for item in shots:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('each shot must contain normalized values and masks')
        z, mask = item
        _observations(z, mask, (256, 23), cpu=True)
        joint = mask[:, _MEASURED].all(dim=-1)
        count = int(joint.sum())
        if count < 32:
            raise ValueError('each reference shot needs at least 32 joint-valid rows')
        observed_shots.append(z[joint][:, _MEASURED].to(torch.float64))
        prepared.append((z, joint))
        joint_counts.append(count)

    center = torch.stack([observed[:, :2].mean(dim=0) for observed in observed_shots]).mean(dim=0)
    centered = [observed[:, :2] - center for observed in observed_shots]
    covariance = torch.stack([u.T @ u / len(u) for u in centered]).mean(dim=0)
    if not torch.isfinite(center).all() or not torch.isfinite(covariance).all():
        raise ValueError('nonfinite covariate moments')
    covariate_scale = covariance.diagonal().sqrt()
    if not torch.isfinite(covariate_scale).all():
        raise ValueError('nonfinite covariate standard deviation')
    if bool((covariate_scale <= MIN_COVARIATE_STD).any()):
        raise ValueError(f'covariate standard deviation must exceed {MIN_COVARIATE_STD:g}')

    designs = [torch.cat((u.new_ones((len(u), 1)), u / covariate_scale), dim=1)
               for u in centered]
    weighted = torch.cat([x / math.sqrt(len(shots) * len(x)) for x in designs], dim=0)
    if not torch.isfinite(weighted).all():
        raise ValueError('nonfinite standardized design')
    rank = int(torch.linalg.matrix_rank(weighted))
    if rank < 3:
        raise ValueError('standardized weighted reference design must have rank 3')
    gram = torch.stack([x.T @ x / len(x) for x in designs]).mean(dim=0)
    rhs = torch.stack([x.T @ observed[:, 2:] / len(x)
                       for x, observed in zip(designs, observed_shots)]).mean(dim=0)
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise ValueError('nonfinite standardized normal equations')
    regularized = gram + torch.diag(gram.new_tensor([0., 1., 1.]))
    beta_std = torch.linalg.solve(regularized, rhs)
    slopes = beta_std[1:] / covariate_scale[:, None]
    beta = torch.cat((beta_std[:1] - center[None] @ slopes, slopes), dim=0)
    if not torch.isfinite(beta_std).all() or not torch.isfinite(beta).all():
        raise ValueError('nonfinite reference regression coefficients')

    variances, patch_counts, difference_counts = [], [], []
    for z, joint in prepared:
        # Match the frozen reference's residuals, patch masks, differences,
        # equal-shot averaging, and scale floor in the original coordinates.
        selected = torch.where(joint[:, None], z[:, _MEASURED], 0.).to(torch.float64)
        design = torch.cat((selected.new_ones((256, 1)), selected[:, :2]), dim=1)
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

    eigenvalues = {name: torch.linalg.eigvalsh(matrix) for name, matrix in (
        ('raw_covariance', covariance),
        ('standardized_covariance', gram[1:, 1:]),
        ('standardized_design', gram),
        ('regularized_design', regularized),
    )}
    if any(not torch.isfinite(value).all() for value in eigenvalues.values()):
        raise ValueError('nonfinite conditioning eigenvalues')
    return {
        'beta': beta,
        'scale': scale,
        'support': {
            'shot_count': len(shots),
            'joint_valid_counts': joint_counts,
            'valid_patch_counts': patch_counts,
            'valid_difference_counts': difference_counts,
            'design_rank': rank,
            'design_rank_coordinates': 'standardized',
            'min_covariate_std': MIN_COVARIATE_STD,
        },
        'covariate_center': center,
        'covariate_scale': covariate_scale,
        'regression_standardized_beta': beta_std,
        'condition_eigenvalues': {name: value.tolist() for name, value in eigenvalues.items()},
    }
