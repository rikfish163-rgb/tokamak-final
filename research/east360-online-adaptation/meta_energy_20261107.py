"""Twice-differentiable joint energy away from coincident trajectories."""

import torch


def joint_energy_no_self(encoded, probabilities, target, available):
    """Score each batch row without constructing mathematically zero self-pairs.

    Inputs have shapes [B, K, H, C], [B, K], [B, H, C], and [B, H, C].
    Nonempty rows require every observed target/scenario and distinct-scenario
    pair to be non-coincident: the Euclidean norm is not smooth at zero.
    Entirely unobserved rows return a graph-connected zero.
    """
    masked_scenarios = torch.where(available[:, None], encoded, 0.0)
    masked_target = torch.where(available, target, 0.0)
    counts = available.sum(dim=(1, 2))
    nonempty = counts > 0

    # The null branch must not evaluate zero norms or multiply raw NaNs by zero.
    null_scenarios = torch.where(~nonempty[:, None, None, None], masked_scenarios, 0.0)
    null_target = torch.where(~nonempty[:, None, None], masked_target, 0.0)
    null_probabilities = torch.where(~nonempty[:, None], probabilities, 0.0)
    result = 0.0 * (
        null_scenarios.square().sum(dim=(1, 2, 3))
        + null_target.square().sum(dim=(1, 2))
        + null_probabilities.square().sum(dim=1)
    )

    indices = nonempty.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return result

    scenarios = masked_scenarios[indices]
    observed_target = masked_target[indices]
    weights = probabilities[indices]
    normalizer = counts[indices].to(encoded.dtype).sqrt()

    target_difference = (scenarios - observed_target[:, None]).flatten(start_dim=2)
    if target_difference.eq(0).all(dim=-1).any():
        raise ValueError("meta energy helper requires non-coincident target and scenarios")

    left, right = torch.triu_indices(
        scenarios.shape[1], scenarios.shape[1], offset=1, device=encoded.device
    )
    pair_difference = (scenarios[:, left] - scenarios[:, right]).flatten(start_dim=2)
    if pair_difference.eq(0).all(dim=-1).any():
        raise ValueError("meta energy helper requires non-coincident distinct scenarios")

    target_distances = torch.linalg.vector_norm(target_difference, dim=-1)
    target_term = (weights * target_distances).sum(dim=1) / normalizer
    pair_distances = torch.linalg.vector_norm(pair_difference, dim=-1)
    spread_term = (weights[:, left] * weights[:, right] * pair_distances).sum(dim=1)
    score = target_term - spread_term / normalizer
    return result.index_add(0, indices, score)
