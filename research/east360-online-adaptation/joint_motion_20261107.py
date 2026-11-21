"""Ordered changes in the existing joint trajectory space."""

import torch

from joint_trajectory_20261107 import JointTrajectory, joint_energy_score


def components(encoded, probabilities, target_encoded, available, *, lag=8):
    """Return level and lagged-change energy scores for each sample."""
    if not isinstance(lag, int) or not 0 < lag < encoded.shape[2]:
        raise ValueError("lag must be an integer satisfying 0 < lag < H")

    level = joint_energy_score(encoded, probabilities, target_encoded, available)
    masked_encoded = torch.where(available[:, None], encoded, 0.0)
    masked_target = torch.where(available, target_encoded, 0.0)
    pair_available = available[:, lag:] & available[:, :-lag]
    change = joint_energy_score(
        masked_encoded[:, :, lag:] - masked_encoded[:, :, :-lag],
        probabilities,
        masked_target[:, lag:] - masked_target[:, :-lag],
        pair_available,
    )
    return level, change


class OrderedTrajectory(JointTrajectory):
    """Use the unchanged joint trajectory model with an ordered-change loss."""

    def __init__(self, center, scale, *, dim=64, scenarios=4, motion_weight=1.0):
        super().__init__(center, scale, dim=dim, scenarios=scenarios)
        weight = torch.as_tensor(motion_weight, dtype=torch.get_default_dtype()).detach().clone()
        if weight.ndim != 0 or not torch.isfinite(weight) or weight < 0:
            raise ValueError("motion_weight must be a finite nonnegative scalar")
        self.register_buffer("motion_weight", weight)

    def loss(self, encoded, probabilities, target, available):
        level, change = components(encoded, probabilities, target, available)
        return level + self.motion_weight * change
