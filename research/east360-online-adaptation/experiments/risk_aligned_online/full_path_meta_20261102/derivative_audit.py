"""Real-model directional differences for a fixed, ended-history query."""

import json
from pathlib import Path

import torch

from .. import io23
from ..adaptivity_capacity_20261102.model import source_basis
from ..critical_shot_meta_20261102.prepare import load_inputs, read_example
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import sha, write
from ..innovation_metric_20261102.prepare import replay_query as truncated_query
from ..innovation_metric_20261102.stream import simulate
from .prepare import ARM, PLAN, replay_full_query


def main():
    root = io23.ROOT / "probe_outputs/full_path_preflight_v1"
    # Preserve the first arithmetic PASS record; this lint-clean rerun changes
    # no model, query, direction, numerical criterion or scientific parameter.
    report = root / "directional_derivative_v2.json"
    if report.exists():
        raise FileExistsError(report)
    configure()
    model, _, cold, parent, mapping = load_inputs()
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    example = read_example(model, basis, mapping, parent, 69854, 448)
    coefficient = model.base.center.new_full((25, 23), 0.01).requires_grad_(True)
    heads = {ARM: (coefficient.detach(), "real", "metric")}
    simulation = simulate(
        model, example["signal"], basis, heads, cached=example["pack"]
    )
    query = 416
    replay = replay_full_query(model, example, simulation, ARM, coefficient, query)
    (gradient,) = torch.autograd.grad(replay["after"], coefficient)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-5
    # Same numerical trajectory, renamed only to call the immutable TBPTT1
    # replay. Its rejected last proposal has no current-coefficient risk path.
    prior_name = "metric_operational"
    prior = {
        k: {prior_name: v[ARM]}
        for k, v in simulation.items()
        if k in ("logits", "tapes")
    }
    truncated = truncated_query(model, example, prior, prior_name, coefficient, query)
    (truncated_gradient,) = torch.autograd.grad(
        truncated["after"], coefficient, allow_unused=True
    )
    assert not truncated["receipt"]["accepted"] and truncated_gradient is None
    assert float(truncated["after"].detach()) == float(replay["after"].detach())
    generator = torch.Generator(device="cpu").manual_seed(131)
    directions = [gradient.detach().double().cpu()]
    directions += [
        torch.randn(25, 23, generator=generator, dtype=torch.float64) for _ in range(2)
    ]
    checks = []
    accepted = [r["accepted"] for r in replay["receipts"]]
    for index, direction in enumerate(directions):
        direction = (direction / direction.norm()).to(
            coefficient.device, coefficient.dtype
        )
        analytic = float((gradient.double() * direction.double()).sum())
        for step in (0.01, 0.005):
            values, branches = [], []
            for sign in (-1, 1):
                table = coefficient.detach() + sign * step * direction
                numeric = simulate(
                    model,
                    example["signal"],
                    basis,
                    {ARM: (table, "real", "metric")},
                    cached=example["pack"],
                )
                values.append(float(numeric["logits"][ARM][query]))
                branches.append(
                    [
                        t["receipt"]["accepted"]
                        for t in numeric["tapes"][ARM]
                        if t["maturity"] < query
                    ]
                )
            stable = branches == [accepted, accepted]
            assert stable, ("acceptance branch changed", index, step, branches)
            measured = (values[1] - values[0]) / (2 * step)
            # A stated float32 numerical check, not a formal roundoff bound.
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
                    stable_acceptance_branch=stable,
                )
            )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result = dict(
        status="PASS",
        shot=69854,
        points=448,
        fixed_query=query,
        checks=checks,
        full_head_gradient_norm=float(gradient.double().norm()),
        truncated_gradient=None,
        truncated_gradient_path_present=False,
        accepted_path=accepted,
        full_and_truncated_forward_risk_exact=True,
        source_tensors_unchanged=True,
        raw_signal_shots_read=[69854],
        outcome_reads=1,
        suffix_shots_read=0,
        source_sha256=parent["source_sha256"],
        cold_state_sha256=sha(cold),
        plan_sha256=sha(PLAN),
        auditor_sha256=sha(Path(__file__)),
        scope="Single fixed historical query and three directional tests at two step sizes; not full training verification or an EAST gain result.",
    )
    write(report, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
