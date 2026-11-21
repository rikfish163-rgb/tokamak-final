"""Joint trajectory scenarios and their masked energy score."""

import torch
from torch import nn
from torch.nn import functional as F


class JointTrajectory(nn.Module):
    """Predict whole-trajectory scenarios from the final temporal token."""

    def __init__(self, center, scale, *, dim=64, scenarios=4):
        super().__init__()
        center = torch.as_tensor(center).detach().clone()
        scale = torch.as_tensor(scale).detach().clone()
        if center.shape != (23,) or scale.shape != (23,):
            raise ValueError("center and scale must each contain 23 channels")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("center and scale must be finite")
        if not (scale > 0).all():
            raise ValueError("scale must be positive")
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.scenarios = scenarios
        self.temporal = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=2 * dim,
            dropout=0.0,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.trajectory = nn.Linear(dim, scenarios * 25 * 23)
        self.probability = nn.Linear(dim, scenarios)
        nn.init.zeros_(self.probability.weight)
        nn.init.zeros_(self.probability.bias)

    def loss(self, encoded, probabilities, target_encoded, available):
        return joint_energy_score(encoded, probabilities, target_encoded, available)

    def forward(self, tokens, token_valid, current, current_available):
        tokens = torch.where(token_valid.unsqueeze(-1), tokens, 0.0)
        key_padding_mask = ~token_valid
        # An entirely masked row uses the already-zeroed CLS as a null key.
        last_padding = key_padding_mask[:, -1:] & token_valid.any(dim=1, keepdim=True)
        key_padding_mask = torch.cat((key_padding_mask[:, :-1], last_padding), dim=1)
        pooled = self.norm(
            self.temporal(tokens, src_key_padding_mask=key_padding_mask)[:, -1]
        )
        changes = self.trajectory(pooled).reshape(-1, self.scenarios, 25, 23)
        current = torch.where(current_available, current, self.center)
        anchor = torch.asinh((current - self.center) / self.scale)[:, None, None, :]

        ordinary_before = anchor[..., :17] + changes[..., :17]
        ordinary_between = anchor[..., 18:22] + changes[..., 18:22]
        variance_raw = self.scale[[17, 22]] * F.softplus(changes[..., [17, 22]])
        variance_encoded = torch.asinh(
            (variance_raw - self.center[[17, 22]]) / self.scale[[17, 22]]
        )
        raw = torch.cat(
            (
                self.center[:17] + self.scale[:17] * torch.sinh(ordinary_before),
                variance_raw[..., :1],
                self.center[18:22] + self.scale[18:22] * torch.sinh(ordinary_between),
                variance_raw[..., 1:],
            ),
            dim=-1,
        )
        encoded = torch.cat(
            (
                ordinary_before,
                variance_encoded[..., :1],
                ordinary_between,
                variance_encoded[..., 1:],
            ),
            dim=-1,
        )
        return {
            "raw": raw,
            "encoded": encoded,
            "probabilities": torch.softmax(self.probability(pooled), dim=-1),
            "available": current_available[:, None, :].expand(-1, 25, -1),
        }


def joint_energy_score(encoded, probabilities, target_encoded, available):
    """Return the masked joint energy score for each batch element."""
    masked_scenarios = torch.where(available[:, None], encoded, 0.0)
    masked_target = torch.where(available, target_encoded, 0.0)
    normalizer = available.sum(dim=(1, 2)).clamp_min(1).to(encoded.dtype).sqrt()

    target_distances = torch.linalg.vector_norm(
        (masked_scenarios - masked_target[:, None]).flatten(start_dim=2), dim=-1
    ) / normalizer[:, None]
    pairwise_distances = torch.linalg.vector_norm(
        (masked_scenarios[:, :, None] - masked_scenarios[:, None, :]).flatten(
            start_dim=3
        ),
        dim=-1,
    ) / normalizer[:, None, None]
    target_term = (probabilities * target_distances).sum(dim=1)
    pairwise_weights = probabilities[:, :, None] * probabilities[:, None, :]
    spread_term = (pairwise_weights * pairwise_distances).sum(dim=(1, 2))
    return target_term - 0.5 * spread_term
