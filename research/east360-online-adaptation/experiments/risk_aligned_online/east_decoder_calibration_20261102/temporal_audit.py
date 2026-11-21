"""CPU-only parameter decomposition and a synthetic check of the real loss.

These fractions are coordinate-dependent squared parameter norms, not physical
energy, output contribution, gradient importance, or a diagnosis of efficacy.
No model or raw-signal replay is performed here.
"""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from ..gate_loss_audit_20261101.core import centered_loss
from ..history_gradient_metric_20261102.run import sha, write


def temporal_split(delta):
    if delta.shape != (575, 4) or not torch.isfinite(delta).all():
        raise ValueError("finite direct-H25 rank4 delta required")
    values = delta.detach().double().reshape(25, 23, 4)
    dc = values.mean(0, keepdim=True).expand_as(values)
    ac = values - dc
    total = float(values.square().sum())
    return (
        dc.reshape(575, 4),
        ac.reshape(575, 4),
        dict(
            quantity="squared_parameter_norm_not_physical_energy",
            norm=total**0.5,
            dc_fraction=float(dc.square().sum()) / total if total else None,
            ac_fraction=float(ac.square().sum()) / total if total else None,
        ),
    )


def invariance_fixture():
    """DC depends on W, so the test covers the cancelled chain-rule term too."""
    generator = torch.Generator().manual_seed(131)

    def random(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)

    inputs = random(2, 4)
    weights = random(4, 4).requires_grad_(True)
    base = random(2, 25, 23)
    decoder = random(4, 23)
    truth = random(2, 25, 23)
    coefficient = inputs @ weights @ decoder
    shifted = base + coefficient[:, None, :]
    # W also changes temporal shape: a nonzero shared gradient must survive.
    wave = torch.linspace(-1, 1, 25, dtype=torch.float64)[None, :, None]
    original = base + 0.2 * wave * coefficient[:, None, :]
    shifted = shifted + 0.2 * wave * coefficient[:, None, :]
    complete = torch.ones_like(truth, dtype=torch.bool)
    partial = torch.rand(truth.shape, generator=generator) > 0.35
    partial[:, :, 2] = False
    partial[0, :, 3] = False
    partial[0, 7, 3] = True
    result = {}
    for name, mask in (("complete", complete), ("partial", partial)):
        before = centered_loss(original, truth, mask)
        after = centered_loss(shifted, truth, mask)
        g0 = torch.autograd.grad(before, weights, retain_graph=True)[0]
        g1 = torch.autograd.grad(after, weights, retain_graph=True)[0]
        if not g0.norm() > 0:
            raise AssertionError("nontrivial shape gradient required")
        result[name] = dict(
            loss_abs_difference=float((after - before).detach().abs()),
            state_gradient_max_abs_difference=float((g1 - g0).abs().max()),
            uncentered_loss_abs_difference=float(
                (
                    F.smooth_l1_loss(shifted[mask], truth[mask])
                    - F.smooth_l1_loss(original[mask], truth[mask])
                )
                .detach()
                .abs()
            ),
        )
    return dict(scope="synthetic_algebra_not_neural_replay", masks=result)


def main(checkpoint, output):
    heads = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(heads) != {"forecast_only", "mean_meta", "worst_meta"}:
        raise ValueError("three fixed all-ended20 decoder corrections required")
    result = dict(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha(checkpoint),
        audit_code_sha256=sha(Path(__file__)),
        centered_loss_code_sha256=sha(
            Path(__file__).parents[1] / "gate_loss_audit_20261101/core.py"
        ),
        parameters={name: temporal_split(values)[2] for name, values in heads.items()},
        invariance=invariance_fixture(),
        proof_of_physical_mechanism=False,
        proof_of_failure_cause=False,
    )
    write(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
