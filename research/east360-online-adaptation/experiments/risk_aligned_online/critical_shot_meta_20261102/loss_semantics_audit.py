"""Deterministic mathematical audit, no experiment or model modification.

The intentional real-minus-hold objective is a directional contrast, not a
reconstruction error guaranteed to stop at an accurate diagnostic forecast.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ..channel_feedback_20261102.loss import channel_losses
from ..centered_gate_stream_20261101.model import loss as reconstruction
from ..history_gradient_metric_20261102.run import ROOT, sha, write


def case(model, y, p):
    truth, hold = y.sinh(), torch.zeros_like(y)
    mask = torch.ones_like(y, dtype=torch.bool)
    prediction = p.clone().requires_grad_(True)
    contrast = channel_losses(model, prediction, truth, hold, mask).sum()
    (g,) = torch.autograd.grad(contrast, prediction, retain_graph=True)
    direct = reconstruction(model, prediction, truth, mask)
    (direct_g,) = torch.autograd.grad(direct, prediction)
    return dict(
        contrast=float(contrast.detach()),
        reconstruction=float(direct.detach()),
        contrast_gradient_norm=float(g.norm()),
        reconstruction_gradient_norm=float(direct_g.norm()),
    ), g


def main():
    model = SimpleNamespace(
        base=SimpleNamespace(
            center=torch.zeros(23, dtype=torch.float64),
            scale=torch.ones(23, dtype=torch.float64),
        )
    )
    y = torch.zeros(1, 25, 23, dtype=torch.float64)
    y[0, :, 0] = torch.linspace(-0.2, 0.2, 25, dtype=torch.float64)
    perfect, g_perfect = case(model, y, y)
    zero, g_zero = case(model, y, torch.zeros_like(y))
    assert perfect["reconstruction"] < 1e-30
    assert perfect["reconstruction_gradient_norm"] < 1e-15
    assert perfect["contrast_gradient_norm"] > 0
    torch.testing.assert_close(g_zero, g_perfect, atol=1e-15, rtol=1e-12)
    y2 = torch.zeros_like(y)
    y2[0, :, 0] = torch.tensor([0.2, -0.2] * 12 + [0.2], dtype=torch.float64)
    saturated, g_saturated = case(model, y2, y2 * 10)
    assert saturated["reconstruction"] > 0
    assert saturated["contrast_gradient_norm"] == 0
    assert saturated["reconstruction_gradient_norm"] > 0
    result = dict(
        status="PASS_MATHEMATICAL_CHARACTERIZATION_NOT_METHOD_VALIDATION",
        perfect_prediction=perfect,
        zero_prediction=zero,
        tenfold_wrong_saturated=saturated,
        gradient_equal_at_zero_and_perfect=True,
        interpretation="The contrast intentionally cancels the common prediction term in the quadratic region, and can stop when both Huber terms saturate. It is not the ordinary innovation/reconstruction loss.",
        new_neural_updates=0,
        raw_data_reads=0,
        code_sha256=sha(Path(__file__)),
    )
    output = ROOT / "probe_outputs/critical_shot_heads20_v1/loss_semantics_audit.json"
    write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
