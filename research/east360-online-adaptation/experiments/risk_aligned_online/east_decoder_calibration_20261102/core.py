"""Small direct-forecast change; source classifier and forecast prior stay fixed."""

from copy import deepcopy

import torch


class LowRankDecoder(torch.nn.Module):
    def __init__(self, prior, basis):
        super().__init__()
        if not isinstance(prior, torch.nn.Linear) or (
            prior.in_features,
            prior.out_features,
        ) != (1088, 575):
            raise ValueError("fixed direct H25 source forecast prior required")
        if (
            basis.shape != (4, 1088)
            or basis.dtype != prior.weight.dtype
            or basis.device != prior.weight.device
            or not torch.isfinite(basis).all()
        ):
            raise ValueError("matching finite rank4 diagnostic basis required")
        if not torch.allclose(
            basis @ basis.T,
            torch.eye(4, device=basis.device, dtype=basis.dtype),
            atol=1e-4,
            rtol=0,
        ):
            raise ValueError("orthonormal diagnostic basis required")
        self.prior = deepcopy(prior).eval().requires_grad_(False)
        self.register_buffer("basis", basis.detach().clone())
        self.delta = torch.nn.Parameter(basis.new_zeros(575, 4))

    def forward(self, x):
        return self.prior(x) + (x @ self.basis.T / 2) @ self.delta.T


def training_eligible(record):
    if not record["disruptive"]:
        if record["label"] != 0:
            raise ValueError("safe shot must have zero point labels")
        return True
    lead = record["actual_lead_ms"]
    if record["label"] != int(lead <= 125):
        raise ValueError("EAST125 canonical point label mismatch")
    return 30 < lead <= 125
