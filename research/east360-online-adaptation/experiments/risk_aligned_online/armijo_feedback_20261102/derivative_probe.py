"""Fixed actual-neural three-update path and piecewise directional derivative."""

import json
from pathlib import Path

import torch

from .. import io23
from ..adaptivity_capacity_20261102.model import source_basis
from ..critical_shot_meta_20261102.prepare import load_inputs, read_example
from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import sha, write
from ..lora_stream_20261102.core import forward
from .core import propose

PLAN = io23.ROOT / "ARMIJO_FEEDBACK_PLAN_20261102.md"


def rollout(model, example, coefficient, *, create_graph):
    state = coefficient.new_zeros(4, 64)
    receipts = []
    for maturity in (159, 287, 415):
        origin = maturity - 25
        observed, available = targets(example["signal"], origin, maturity)
        truth, mask = [
            torch.as_tensor(v, device=model.base.center.device)
            for v in (observed["real"], available)
        ]
        point = take(example["pack"], origin, origin + 1)
        state, receipt = propose(
            model, point, state, truth, mask, coefficient, create_graph=create_graph
        )
        receipts.append(receipt)
    logit = forward(model, take(example["pack"], 416, 448), state)["logit"][0]
    return logit, state, receipts


def branches(receipts):
    return [(r["accepted"], r["alpha"]) for r in receipts]


def main():
    root = io23.ROOT / "probe_outputs/armijo_derivative_probe_v1"
    if root.exists():
        raise FileExistsError(root)
    configure()
    model, _, cold, parent, mapping = load_inputs()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    root.mkdir(parents=True)
    files = [Path(__file__), Path(propose.__code__.co_filename), PLAN]
    write(
        root / "protocol.json",
        dict(
            shot=69854,
            points=448,
            query=416,
            maturities=[159, 287, 415],
            coefficient="constant0.01",
            source_sha256=parent["source_sha256"],
            cold_state_sha256=sha(cold),
            code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
            scope="Numerical ended-history case, not a new EAST performance evaluation.",
        ),
    )
    example = read_example(model, source_basis(model), mapping, parent, 69854, 448)
    table = model.base.center.new_full((25, 23), 0.01).requires_grad_(True)
    logit, state, receipts = rollout(model, example, table, create_graph=True)
    (gradient,) = torch.autograd.grad(logit, table)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-5
    numeric, numeric_state, numeric_receipts = rollout(
        model, example, table.detach(), create_graph=False
    )
    assert float(logit.detach()) == float(numeric.detach()) and torch.equal(
        state.detach(), numeric_state
    )
    assert receipts == numeric_receipts
    generator = torch.Generator(device="cpu").manual_seed(131)
    directions = [gradient.detach().double().cpu()] + [
        torch.randn(25, 23, generator=generator, dtype=torch.float64) for _ in range(2)
    ]
    checks = []
    for index, direction in enumerate(directions):
        direction = (direction / direction.norm()).to(table.device, table.dtype)
        analytic = float((gradient.double() * direction.double()).sum())
        for step in (0.005, 0.0025):
            values = []
            pattern = []
            for sign in (-1, 1):
                value, _, events = rollout(
                    model,
                    example,
                    table.detach() + sign * step * direction,
                    create_graph=False,
                )
                values.append(float(value.detach()))
                pattern.append(branches(events))
            stable = pattern == [branches(receipts), branches(receipts)]
            assert stable, (
                "changed Armijo branch",
                index,
                step,
                pattern,
                branches(receipts),
            )
            measured = (values[1] - values[0]) / (2 * step)
            tolerance = 2e-4 + 0.03 * abs(analytic)
            assert abs(measured - analytic) <= tolerance, (
                index,
                step,
                measured,
                analytic,
                tolerance,
            )
            checks.append(
                dict(
                    direction=index,
                    step=step,
                    analytic=analytic,
                    measured=measured,
                    absolute_error=abs(measured - analytic),
                    tolerance=tolerance,
                    alpha_branches_stable=stable,
                )
            )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        checks=checks,
        query416_logit=float(logit.detach()),
        head_gradient_norm=float(gradient.double().norm()),
        state256_after=state.detach().flatten().cpu().tolist(),
        receipts=receipts,
        accepted=sum(r["accepted"] for r in receipts),
        evaluations=sum(r["evaluations"] for r in receipts),
        gradient_and_numeric_forward_exact=True,
        source_tensors_unchanged=True,
        source_sha256=parent["source_sha256"],
        cold_state_sha256=sha(cold),
        plan_sha256=sha(PLAN),
        auditor_sha256=sha(Path(__file__)),
        raw_signal_shots_read=[69854],
        outcome_reads=1,
        suffix_shots_read=0,
        scope="Three explicit actual-neural matured updates on one fixed historical prefix; finite differences inside identical discrete alpha branches. No full stream or global differentiability claim.",
    )
    write(root / "completed.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "checks",
                    "query416_logit",
                    "head_gradient_norm",
                    "accepted",
                    "evaluations",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
