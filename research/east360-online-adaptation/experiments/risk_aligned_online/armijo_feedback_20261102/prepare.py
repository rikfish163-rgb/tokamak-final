"""Matched ended20 training under the actual bounded Armijo update rule."""

import argparse
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

from .. import io23
from ..adaptivity_capacity_20261102.model import source_basis
from ..critical_shot_meta_20261102.objectives import (
    last_update,
    margin_loss,
    select_critical,
)
from ..critical_shot_meta_20261102.prepare import load_inputs, read_example
from ..critical_shot_meta_20261102.stream import take
from ..dual_history_ttt_20261101.stream import targets
from ..east_context_operator_20261031.run import configure
from ..history_gradient_metric_20261102.run import read, sha, write
from ..innovation_metric_20261102.core import innovation_loss
from ..lora_stream_20261102.core import forward
from ..risk_bridge_forecaster_20261102.core import auxiliary
from .core import propose
from .stream import simulate

PLAN = io23.ROOT / "ARMIJO_FEEDBACK_PLAN_20261102.md"
ARMS = ("meta", "forecast")
PREFLIGHT = io23.ROOT / "probe_outputs/armijo_preflight_v1"
MANUAL = io23.ROOT / "probe_outputs/armijo_derivative_probe_v1"


def snapshot(root, parent, cold, scope):
    root.mkdir(parents=True)
    files = {Path(__file__), PLAN}
    files.update(
        Path(m.__file__)
        for name, m in list(sys.modules.items())
        if name.startswith("experiments.risk_aligned_online.")
        and getattr(m, "__file__", None)
    )
    protocol = dict(
        scope=scope,
        ids=parent["ids"],
        source_sha256=parent["source_sha256"],
        cold_state_sha256=sha(cold),
        input_map_sha256=parent["input_map_sha256"],
        heads=ARMS,
        forecast_parameters_per_head=575,
        steps=128,
        optimizer="Adam",
        learning_rate=0.001,
        gradient_clip=1.0,
        seed=131,
        initial_head="zero",
        risk_meta_gradient_horizon="all_matured_updates_before_fixed_query",
        diagnostic_incoming_state_detached=True,
        first_update=159,
        first_possible_effect=160,
        update_interval=128,
        diagnostic_forecast_points=25,
        metric_learning_rate=1.0,
        relative_metric_damping=0.01,
        absolute_metric_damping=1e-8,
        metric_increment_hidden_rms_cap=0.05,
        total_hidden_rms_cap=0.25,
        state_anchor=0.01,
        armijo_c=1e-4,
        alpha_reduction=0.5,
        max_proposal_evaluations=11,
        outcome_authority="HDF5",
        lead_strictly_greater_ms=30,
        reset_each_shot=True,
        warmup_cost_retained=True,
        independent_confirmation=False,
        plan_sha256=sha(PLAN),
        code_sha256={str(f.relative_to(io23.ROOT)): sha(f) for f in files},
    )
    write(root / "protocol.json", protocol)
    with zipfile.ZipFile(
        root / "code_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(io23.ROOT)))


def diagnostic(model, example, tape, coefficient):
    observed, available = targets(example["signal"], tape["origin"], tape["maturity"])
    truth, mask = [
        torch.as_tensor(v, device=model.base.center.device)
        for v in (observed["real"], available)
    ]
    point = take(example["pack"], tape["origin"], tape["origin"] + 1)
    signal = innovation_loss(
        model,
        auxiliary(model, point, tape["before"].detach(), coefficient),
        truth,
        mask,
    )
    return signal


def replay_query(model, example, simulation, arm, coefficient, query):
    if arm not in ARMS:
        raise ValueError("known auxiliary training task required")
    maturity = last_update(query)
    if maturity is None:
        return None
    tapes = [t for t in simulation["tapes"][arm] if t["maturity"] < query]
    assert [t["maturity"] for t in tapes] == list(range(159, maturity + 1, 128))
    state = coefficient.new_zeros(4, 64)
    receipts = []
    for tape in tapes:
        assert (
            tape["origin"] == tape["maturity"] - 25
            and tape["first_affected"] == tape["maturity"] + 1
        )
        assert torch.equal(state.detach(), tape["before"])
        observed, available = targets(
            example["signal"], tape["origin"], tape["maturity"]
        )
        truth, mask = [
            torch.as_tensor(v, device=model.base.center.device)
            for v in (observed["real"], available)
        ]
        point = take(example["pack"], tape["origin"], tape["origin"] + 1)
        state, receipt = propose(
            model, point, state, truth, mask, coefficient, create_graph=True
        )
        assert torch.equal(state.detach(), tape["after"]), (
            "state replay",
            arm,
            query,
            tape["maturity"],
        )
        assert receipt == tape["receipt"], (
            "receipt replay",
            arm,
            query,
            tape["maturity"],
        )
        receipts.append(receipt)
    lo = query // 32 * 32
    later = take(example["pack"], lo, min(lo + 32, len(example["times"])))
    after = forward(model, later, state)["logit"][query - lo]
    assert float(after.detach()) == float(simulation["logits"][arm][query]), (
        "query replay",
        arm,
        query,
    )
    with torch.no_grad():
        before = forward(model, later, tapes[-1]["before"].detach())["logit"][
            query - lo
        ]
    return dict(
        state=state,
        after=after,
        before=before,
        signal=diagnostic(model, example, tapes[-1], coefficient),
        maturity=maturity,
        receipts=receipts,
    )


def episode(model, example, simulation, arm, coefficient, threshold):
    if arm not in ARMS:
        raise ValueError("known auxiliary training task required")
    query = select_critical(
        example["times"],
        simulation["logits"][arm],
        example["disruptive"],
        example["down"],
    )
    replay = replay_query(model, example, simulation, arm, coefficient, query)
    if replay is None:
        before = after = coefficient.new_tensor(float(simulation["logits"][arm][query]))
        signal = coefficient.sum() * 0
        if simulation["tapes"][arm]:
            signal = diagnostic(
                model, example, simulation["tapes"][arm][0], coefficient
            )
    else:
        before, after, signal = replay["before"], replay["after"], replay["signal"]
    label = int(example["disruptive"])
    risk_before = margin_loss(before, threshold, label).detach()
    risk_after = margin_loss(after, threshold, label)
    regularizer = 1e-4 * coefficient.square().mean()
    objective = (
        (risk_after - risk_before) / 0.1 + 0.20 * signal + regularizer
        if arm == "meta"
        else signal + regularizer
    )
    events = replay["receipts"] if replay is not None else []
    return objective, dict(
        shot=example["shot"],
        query=query,
        supported=replay is not None,
        maturity=replay["maturity"] if replay is not None else None,
        label=label,
        risk_before=float(risk_before),
        risk_after=float(risk_after.detach()),
        signal=float(signal.detach()),
        replay_updates=len(events),
        accepted_path_updates=sum(e["accepted"] for e in events),
        proposal_evaluations=sum(e["evaluations"] for e in events),
    )


def run_preflight(model, basis, mapping, parent):
    manual = read(MANUAL / "completed.json")
    assert manual["status"] == "PASS" and manual["plan_sha256"] == sha(PLAN)
    assert manual["source_sha256"] == parent["source_sha256"]
    example = read_example(model, basis, mapping, parent, 69854, 448)
    table = model.base.center.new_full((25, 23), 0.01)
    heads = {a: (table, "real") for a in ARMS}
    cached = simulate(model, example["signal"], basis, heads, cached=example["pack"])
    actual = simulate(model, example["signal"], basis, heads)
    checks = []
    for arm in ARMS:
        assert np.array_equal(cached["logits"][arm], actual["logits"][arm])
        assert np.array_equal(cached["probability"][arm], actual["probability"][arm])
        assert [t["receipt"] for t in cached["tapes"][arm]] == manual["receipts"]
        for query in (160, 287, 288, 415, 416):
            variable = table.clone().requires_grad_(True)
            result = replay_query(model, example, cached, arm, variable, query)
            (gradient,) = torch.autograd.grad(
                result["after"], variable, allow_unused=True
            )
            accepted = sum(e["accepted"] for e in result["receipts"])
            if accepted:
                assert gradient is not None and torch.isfinite(gradient).all()
                norm = float(gradient.double().norm())
            else:
                assert gradient is None
                norm = None
            if query == 416:
                assert float(result["after"].detach()) == manual["query416_logit"]
                assert (
                    result["state"].detach().flatten().cpu().tolist()
                    == manual["state256_after"]
                )
            checks.append(
                dict(
                    arm=arm,
                    query=query,
                    accepted_path_updates=accepted,
                    head_gradient_norm=norm,
                )
            )
    return dict(
        status="PASS",
        checks=checks,
        cached_stream_bitwise_equal=True,
        full_state_and_receipt_replay_exact=True,
        independently_written_three_update_probe_exact=True,
        manual_derivative_sha256=sha(MANUAL / "completed.json"),
        outcome_reads=1,
        suffix_shots_read=0,
    )


def main(root, *, preflight=False):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    if not preflight:
        assert read(PREFLIGHT / "completed.json")["status"] == "PASS"
        assert read(PREFLIGHT / "protocol.json")["plan_sha256"] == sha(PLAN)
    configure()
    started = time.monotonic()
    model, configuration, cold, parent, mapping = load_inputs()
    assert all(not p.requires_grad for p in model.parameters())
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    basis = source_basis(model)
    snapshot(
        root,
        parent,
        cold,
        "ARMIJO_MATCHED_PREFLIGHT"
        if preflight
        else "ENDED20_ARMIJO_FULL_PATH_MATCHED_TRAINING",
    )
    if preflight:
        result = run_preflight(model, basis, mapping, parent)
    else:
        examples = []
        for sid in parent["ids"]:
            examples.append(read_example(model, basis, mapping, parent, sid))
            print(
                json.dumps(
                    dict(stage="cache", shot=sid, points=len(examples[-1]["times"]))
                ),
                flush=True,
            )
        groups = {
            flag: [e for e in examples if e["disruptive"] == flag]
            for flag in (True, False)
        }
        assert len(groups[True]) == 8 and len(groups[False]) == 12
        heads = {
            a: torch.nn.Parameter(model.base.center.new_zeros(25, 23)) for a in ARMS
        }
        optimizers = {a: torch.optim.Adam([b], lr=0.001) for a, b in heads.items()}
        rng, history = np.random.default_rng(131), []
        for step in range(128):
            selected = [
                groups[flag][int(rng.integers(len(groups[flag])))]
                for flag in (True, False)
            ]
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            record = dict(
                step=step + 1,
                sample_shots=[e["shot"] for e in selected],
                episodes={a: [] for a in ARMS},
            )
            for example in selected:
                simulation = simulate(
                    model,
                    example["signal"],
                    basis,
                    {a: (b.detach(), "real") for a, b in heads.items()},
                    cached=example["pack"],
                )
                for arm, b in heads.items():
                    objective, receipt = episode(
                        model,
                        example,
                        simulation,
                        arm,
                        b,
                        float(configuration["threshold"]),
                    )
                    if not torch.isfinite(objective):
                        raise FloatingPointError(
                            "nonfinite matched Armijo outer objective"
                        )
                    (objective / 2).backward()
                    record["episodes"][arm].append(
                        dict(**receipt, objective=float(objective.detach()))
                    )
            record["gradient_norms"] = {}
            for arm, b in heads.items():
                norm = torch.nn.utils.clip_grad_norm_([b], 1.0, error_if_nonfinite=True)
                record["gradient_norms"][arm] = float(norm)
                optimizers[arm].step()
            assert all(p.grad is None for p in model.parameters())
            history.append(record)
            if (step + 1) % 16 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="train",
                            step=step + 1,
                            seconds=time.monotonic() - started,
                            objective={
                                a: float(
                                    np.mean(
                                        [r["objective"] for r in record["episodes"][a]]
                                    )
                                )
                                for a in ARMS
                            },
                        )
                    ),
                    flush=True,
                )
        write(
            root / "heads.json",
            {a: b.detach().cpu().tolist() for a, b in heads.items()},
        )
        write(root / "history.json", history)
        write(
            root / "training_cohort.json",
            [
                dict(
                    shot=e["shot"],
                    disruptive=e["disruptive"],
                    down=e["down"],
                    points=len(e["times"]),
                )
                for e in examples
            ],
        )
        result = dict(
            status="COMPLETED",
            head_optimizer_steps=128,
            number_heads=2,
            ids=parent["ids"],
            outcome_reads=20,
            suffix_shots_read=0,
            heads_sha256=sha(root / "heads.json"),
            history_sha256=sha(root / "history.json"),
            plan_sha256=sha(PLAN),
            support_counts={
                a: sum(e["supported"] for r in history for e in r["episodes"][a])
                for a in ARMS
            },
            online_gain_measured=False,
            independent_future_confirmation=False,
        )
    assert all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in original.items())
    result.update(source_tensors_unchanged=True, seconds=time.monotonic() - started)
    write(root / "completed.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    main(args.output, preflight=args.preflight)
